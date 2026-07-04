"""
LoRA SFT training script (TRL SFTTrainer) + evaluation (BLEU, chrF)
Compares:
  (A) Base model (no SFT)
  (B) LoRA-SFT model (adapter or merged)

Assumes your dataset on disk has columns: "English", "Qanjobal"
and you already saved train/eval splits with load_from_disk().

Install:
  pip install -U torch transformers datasets accelerate peft trl evaluate sacrebleu
Notes:
- This script uses a chat-template prompt for translation.
- For evaluation, we generate translations from the model and compute BLEU/chrF/COMET.
- Saves LoRA adapter by default; optional merge for inference.
"""

import os
import math
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import numpy as np


# -------------------------
# Device & dtype
# -------------------------
USE_CUDA = torch.cuda.is_available()
USE_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
MPS_BUILT = hasattr(torch.backends, "mps") and getattr(torch.backends.mps, "is_built", lambda: True)()
USE_CPU = not (USE_CUDA or (USE_MPS and MPS_BUILT))

BF16_OK = USE_CUDA and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
FP16_OK = USE_CUDA

dtype = torch.bfloat16 if BF16_OK else (torch.float16 if FP16_OK else torch.float32)
device = "cuda" if USE_CUDA else ("mps" if (USE_MPS and MPS_BUILT) else "cpu")

print(f"Device: {device} | dtype: {dtype} | CUDA: {USE_CUDA} | bf16_ok: {BF16_OK}")


# -------------------------
# Paths / hyperparams
# -------------------------
MODEL_ID = "./mayan_continued_pretrain_llama_3_2_1"     # base pretrained model (local path or HF id)
TRAIN_SPLIT_PATH = "data/train_split"
EVAL_SPLIT_PATH  = "data/eval_split"

OUT_DIR_ADAPTER = "out_sft_lora_adapter-cpt-llama"     # where LoRA adapter checkpoints go
SAVE_LOC_ADAPTER = "model/llama_model_qanjobal_lora_adapter_pretrained"
SAVE_LOC_MERGED  = "model/llama_model_qanjobal_lora_merged_pretrained"  # optional merged full model

EPOCHS = 5
MAX_LEN = 512
BATCH = 4
GRAD_ACCUM = 4

# LoRA-friendly LR tends to be higher than full-FT.
# Start here; tune later.
LR = 2e-4

# Generation settings for eval
GEN_MAX_NEW_TOKENS = 196
GEN_TEMPERATURE = 0.0     # greedy when 0.0 and do_sample=False
GEN_DO_SAMPLE = False
EVAL_MAX_EXAMPLES = None  # set e.g. 500 for quicker eval


# -------------------------
# Tokenizer + chat template
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

GENERIC_TEMPLATE = """{% if bos_token %}{{ bos_token }}{% endif -%}
{% for m in messages -%}
{{ m['role'].upper() }}: {{ m['content'] }}
{% endfor -%}
ASSISTANT:"""

if not getattr(tokenizer, "chat_template", None):
    tokenizer.chat_template = GENERIC_TEMPLATE


def build_prompt(english_text: str) -> str:
    """Prompt used for generation (no gold answer appended)."""
    messages = [{
        "role": "user",
        "content": (
            "Task: Translate this text from English to Qanjobal:\n"
            f"User content:\n{english_text}\n"
        ),
    }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_train_text(english_text: str, qanjobal_text: str) -> str:
    """Full SFT training text: prompt + gold completion."""
    return build_prompt(english_text) + qanjobal_text


def fmt(ex):
    return {
        "text": build_train_text(
            str(ex["English"]).strip(),
            str(ex["Qanjobal"]).strip(),
        )
    }


# -------------------------
# Load & format datasets
# -------------------------
train_ds = load_from_disk(TRAIN_SPLIT_PATH)
eval_ds  = load_from_disk(EVAL_SPLIT_PATH)

train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(fmt, remove_columns=eval_ds.column_names)

print(train_ds)
print(eval_ds)


# -------------------------
# Load base model
# -------------------------
# For Gemma/Gemma2, attention implementation varies; "eager" is safest.
ATTN_IMPL = "eager"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    attn_implementation=ATTN_IMPL,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

# Optional memory saver
if hasattr(base_model, "gradient_checkpointing_enable"):
    base_model.gradient_checkpointing_enable()


LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
)

model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()


# -------------------------
# TRL SFT config + trainer
# -------------------------
cfg = SFTConfig(
    output_dir=OUT_DIR_ADAPTER,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,

    logging_steps=20,
    max_length=MAX_LEN,
    packing=False,

    bf16=BF16_OK,
    fp16=(FP16_OK and not BF16_OK),

    # TRL/Accelerate will choose device automatically; keep this simple:
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    report_to="tensorboard",
    eval_strategy="steps",
eval_steps=50,
metric_for_best_model="eval_loss",
    
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    eval_strategy="steps",
        eval_steps=50,
        eval_split="eval",        # ← tells trainer to split from train set
        eval_split_size=0.1,
    # If you want periodic eval during training:
    # compute_metrics=... (but we do full generation-based eval after)
)

# -------------------------
# Helpers: generation + metrics
# -------------------------
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
#comet = evaluate.load("comet")  # will download a COMET model on first run (internet)

@torch.inference_mode()
def generate_translations(m, english_list, batch_size=8):
    """
    Generate translations for a list of English strings.
    Uses tokenizer + model.generate directly (faster than pipeline for large evals).
    """
    m.eval()
    preds = []

    for i in range(0, len(english_list), batch_size):
        batch_eng = english_list[i:i+batch_size]
        prompts = [build_prompt(x) for x in batch_eng]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(device)

        out_ids = m.generate(
            **enc,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=GEN_DO_SAMPLE,
            temperature=GEN_TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated portion after the prompt
        # Find prompt lengths to slice
        prompt_lens = (enc["attention_mask"].sum(dim=1)).tolist()
        for seq, pl in zip(out_ids, prompt_lens):
            gen_part = seq[pl:]
            text = tokenizer.decode(gen_part, skip_special_tokens=True).strip()
            preds.append(text)

    return preds


def compute_all_metrics(preds, refs):
    """
    refs: list[str] gold Qanjobal
    preds: list[str] model outputs
    """
    # BLEU expects list of predictions and list of list-of-references
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    chrf_res = chrf.compute(predictions=preds, references=refs)

    # COMET expects sources + predictions + references
    '''
    comet_res = comet.compute(
        predictions=preds,
        references=refs,
        sources=[""] * len(refs),   # source optional; you can pass English here if you want
    )
    '''

    return {
        "bleu": float(bleu_res["score"]),
        "chrf": float(chrf_res["score"]),
        #"comet": float(comet_res["mean_score"]),
    }


def eval_model_on_split(m, raw_eval_split_path, max_examples=None):
    """
    Loads the *raw* eval split (with English/Qanjobal columns),
    generates predictions, then computes metrics.
    """
    raw_eval = load_from_disk(raw_eval_split_path)
    if max_examples is not None:
        raw_eval = raw_eval.select(range(min(max_examples, len(raw_eval))))

    english = [str(x).strip() for x in raw_eval["English"]]
    refs    = [str(x).strip() for x in raw_eval["Qanjobal"]]

    preds = generate_translations(m, english, batch_size=max(1, BATCH))
    return compute_all_metrics(preds, refs)


# -------------------------
# Evaluate BASE model (no SFT)
# -------------------------
# Important: evaluate the same underlying base weights, without LoRA adapters.
# We use a fresh base model instance to avoid any adapter wrapping.
base_eval_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    attn_implementation=ATTN_IMPL,
).to(device)
base_eval_model.config.pad_token_id = tokenizer.pad_token_id

print("\n=== Evaluating BASE model (before SFT) ===")
base_metrics = eval_model_on_split(base_eval_model, EVAL_SPLIT_PATH, max_examples=EVAL_MAX_EXAMPLES)
print(base_metrics)

# Free memory
del base_eval_model
if USE_CUDA:
    torch.cuda.empty_cache()


# -------------------------
# Train LoRA SFT
# -------------------------
print("\n=== Training LoRA SFT ===")
trainer.train()

# Save adapter (recommended)
os.makedirs(SAVE_LOC_ADAPTER, exist_ok=True)
trainer.model.save_pretrained(SAVE_LOC_ADAPTER)
tokenizer.save_pretrained(SAVE_LOC_ADAPTER)
print(f"\nSaved LoRA adapter to: {SAVE_LOC_ADAPTER}")

# -------------------------
# Evaluate LoRA-SFT model
# -------------------------
# Move model to device for generation eval
trainer.model.to(device)

print("\n=== Evaluating LoRA-SFT model ===")
sft_metrics = eval_model_on_split(trainer.model, EVAL_SPLIT_PATH, max_examples=EVAL_MAX_EXAMPLES)
print(sft_metrics)

# -------------------------
# Compare
# -------------------------
print("\n=== COMPARISON (SFT - BASE) ===")
for k in ["bleu", "chrf"]:
    print(f"{k}: {sft_metrics[k]:.4f}  (base {base_metrics[k]:.4f})  Δ {sft_metrics[k]-base_metrics[k]:+.4f}")

# -------------------------
# Optional: merge adapter into base weights for a single standalone model
# -------------------------
# This makes inference simpler (no PEFT dependency at runtime), and speed identical to base.
do_merge = True
if do_merge:
    os.makedirs(SAVE_LOC_MERGED, exist_ok=True)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(SAVE_LOC_MERGED)
    tokenizer.save_pretrained(SAVE_LOC_MERGED)
    print(f"\nSaved MERGED full model to: {SAVE_LOC_MERGED}")
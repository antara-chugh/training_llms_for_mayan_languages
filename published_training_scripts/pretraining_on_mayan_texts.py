import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# =========================
# Config
# =========================
BASE_MODEL = "meta-llama/Llama-3.2-1B"   # BASE (not instruct)
TRAIN_PATH = "mayan_train.jsonl"
TEST_PATH  = "mayan_test.jsonl"
OUTPUT_DIR = "./mayan_continued_pretrain_llama_3_2_1"

TEXT_FIELD = "text"
SEED = 42

BLOCK_SIZE = 1024        # 2048 if model + VRAM allow
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-5
EPOCHS = 1

MAX_TEST_EXAMPLES = 2000  # cap eval cost (set None for full test)

# =========================
# Helpers
# =========================
def safe_exp(x):
    return float("inf") if x > 50 else math.exp(x)

def cap_dataset(ds, max_n):
    if max_n is None:
        return ds
    return ds.select(range(min(max_n, len(ds))))

# =========================
# Load datasets
# =========================
train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
test_ds  = load_dataset("json", data_files=TEST_PATH, split="train")

test_ds = cap_dataset(test_ds, MAX_TEST_EXAMPLES)

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# Tokenize
# =========================
def tokenize_fn(batch):
    return tokenizer(batch[TEXT_FIELD], truncation=False)

tok_train = train_ds.map(tokenize_fn, batched=True, remove_columns=[TEXT_FIELD])
tok_test  = test_ds.map(tokenize_fn, batched=True, remove_columns=[TEXT_FIELD])

# =========================
# Pack into fixed-length blocks
# =========================
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE

    if total_len == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_train = tok_train.map(group_texts, batched=True)
lm_test  = tok_test.map(group_texts, batched=True)

# =========================
# Model dtype
# =========================
use_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 8
)
dtype = torch.bfloat16 if use_bf16 else (
    torch.float16 if torch.cuda.is_available() else torch.float32
)

# =========================
# Model
# =========================
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# =========================
# Training args
# =========================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    seed=SEED,

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    logging_steps=25,
    save_steps=500,
    save_total_limit=2,

    bf16=use_bf16,
    fp16=(torch.cuda.is_available() and not use_bf16),
    optim="adamw_torch",

    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=lm_train,
    eval_dataset=lm_test,
    data_collator=collator,
)

# =========================
# TEST EVAL — BEFORE TRAINING
# =========================
baseline_metrics = trainer.evaluate()
baseline_loss = float(baseline_metrics["eval_loss"])
baseline_ppl = safe_exp(baseline_loss)

print("\n=== TEST SET (before pretraining) ===")
print("Test loss:", baseline_loss)
print("Test perplexity:", baseline_ppl)

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# TEST EVAL — AFTER TRAINING
# =========================
after_metrics = trainer.evaluate()
after_loss = float(after_metrics["eval_loss"])
after_ppl = safe_exp(after_loss)

print("\n=== TEST SET (after pretraining) ===")
print("Test loss:", after_loss)
print("Test perplexity:", after_ppl)

# =========================
# COMPARISON
# =========================
delta = after_ppl - baseline_ppl
pct = (delta / baseline_ppl) * 100 if baseline_ppl > 0 else float("nan")

print("\n=== COMPARISON ===")
print(f"Δ perplexity: {delta:.4f}")
print(f"% change: {pct:.2f}%")

# =========================
# SAVE
# =========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nSaved model to:", OUTPUT_DIR)

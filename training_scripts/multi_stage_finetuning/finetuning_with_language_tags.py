"""
Experiment G2 — Unstructured Mayan SFT with Language Tags
English -> Q'anjob'al

G2: Base -> SFT(all Mayan parallel, flat mix, WITH lang tags) -> SFT(Q'anjob'al)


"""

import os, json, math, torch, numpy as np
from pathlib import Path
from datasets import Dataset, load_from_disk, interleave_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, TrainerCallback,
)
from torch.utils.data import DataLoader
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import evaluate
import matplotlib.pyplot as plt

# ── Device ────────────────────────────────────────────────────────────────────

USE_CUDA = torch.cuda.is_available()
BF16_OK  = USE_CUDA and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
FP16_OK  = USE_CUDA and not BF16_OK
dtype    = torch.bfloat16 if BF16_OK else (torch.float16 if FP16_OK else torch.float32)
device   = "cuda" if USE_CUDA else "cpu"
print(f"Device: {device} | dtype: {dtype}")

# ── Paths ─────────────────────────────────────────────────────────────────────

SPLITS_DIR          = "mayan_data/splits"
QANJOBAL_TRAIN_PATH = "data/train_split"
QANJOBAL_TEST_PATH  = "data/eval_split"   # held-out test — final eval ONLY

VAL_SPLIT_SIZE = 0.1
VAL_SPLIT_SEED = 42

BASE_MODEL = "atara15/continued_pretrain_llama_3_2_1""

# ── Hyperparameters ───────────────────────────────────────────────────────────

EPOCHS             = 5
MAX_LEN            = 512
BATCH              = 4
GRAD_ACCUM         = 4
LR                 = 2e-4
GEN_MAX_NEW_TOKENS = 196
GEN_DO_SAMPLE      = False
GEN_TEMPERATURE    = 0.0
BLOCK_SIZE         = 512
EVAL_MAX_EXAMPLES  = None

# ── Language metadata ─────────────────────────────────────────────────────────

LANGUAGE_META = {
    "acr": {"name": "Achi",        "branch": "Quichean"},
    "agu": {"name": "Awakateko",   "branch": "Mamean_Ixilean"},
    "cac": {"name": "Chuj",        "branch": "Qanjobalan_Chujean"},
    "itz": {"name": "Itza'",       "branch": "Yucatecan"},
    "ixl": {"name": "Ixil",        "branch": "Mamean_Ixilean"},
    "kek": {"name": "Q'eqchi'",    "branch": "Qeqchi"},
    "mam": {"name": "Mam",         "branch": "Mamean"},
    "poc": {"name": "Poqomam",     "branch": "Poqom"},
    "poh": {"name": "Poqomchi'",   "branch": "Poqom"},
    "quc": {"name": "K'iche'",     "branch": "Quichean_Proper"},
    "qum": {"name": "Sipakapense", "branch": "Quichean_Proper"},
    "ttc": {"name": "Tektitek",    "branch": "Mamean"},
    "tzj": {"name": "Tz'utujil",   "branch": "Quichean_Proper"},
}

# ── Tokenizer ─────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

GENERIC_TEMPLATE = """{% if bos_token %}{{ bos_token }}{% endif -%}
{% for m in messages -%}
{{ m['role'].upper() }}: {{ m['content'] }}
{% endfor -%}
ASSISTANT:"""
if not getattr(tokenizer, "chat_template", None):
    tokenizer.chat_template = GENERIC_TEMPLATE


# ── Prompts ───────────────────────────────────────────────────────────────────

def build_prompt_mayan_tagged(english: str, lang_name: str, branch: str) -> str:
    """Mayan prompt WITH language family tags — the G2 difference vs G."""
    messages = [{
        "role": "user",
        "content": (
            f"Language family: Mayan | Branch: {branch} | Language: {lang_name}\n"
            f"Task: Translate this text from English to {lang_name}:\n"
            f"User content:\n{english}\n"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def build_prompt_qanjobal(english: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Task: Translate this text from English to Qanjobal:\n"
            f"User content:\n{english}\n"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Format functions ──────────────────────────────────────────────────────────

def make_mayan_tagged_fmt(lang_code: str):
    """Per-language formatter that injects lang name and branch into prompt."""
    meta = LANGUAGE_META[lang_code]
    def fmt(ex):
        english = str(ex["english"]).strip() if ex.get("english") else ""
        mayan   = str(ex["mayan"]).strip()   if ex.get("mayan")   else ""
        return {"text": build_prompt_mayan_tagged(english, meta["name"], meta["branch"]) + mayan}
    return fmt

def fmt_qanjobal(ex):
    return {
        "text": build_prompt_qanjobal(str(ex["English"]).strip())
                + str(ex["Qanjobal"]).strip()
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_splits(splits_dir):
    datasets = {}
    for lang_code in LANGUAGE_META:
        path = os.path.join(splits_dir, lang_code, "train")
        if os.path.exists(path):
            ds = load_from_disk(path)
            datasets[lang_code] = ds
            print(f"  {lang_code} ({LANGUAGE_META[lang_code]['name']}): {len(ds)}")
        else:
            print(f"  {lang_code}: not found, skipping")
    return datasets


def make_qanjobal_sft_datasets():
    raw = load_from_disk(QANJOBAL_TRAIN_PATH)

    def is_valid(ex):
        return (ex.get("English") and ex.get("Qanjobal")
                and str(ex["English"]).strip() != ""
                and str(ex["Qanjobal"]).strip() != "")

    raw   = raw.filter(is_valid)
    split = raw.train_test_split(test_size=VAL_SPLIT_SIZE, seed=VAL_SPLIT_SEED)
    train_raw, val_raw = split["train"], split["test"]
    print(f"  Q'anjob'al train: {len(train_raw)} | val (carved): {len(val_raw)}")

    return (
        train_raw.map(fmt_qanjobal, remove_columns=train_raw.column_names),
        val_raw.map(fmt_qanjobal,   remove_columns=val_raw.column_names),
    )


def assemble_flat_mayan_tagged(lang_datasets):
    """
    Mix all Mayan data with per-language tags injected.
    Each language gets its own formatter so the tag is accurate.
    """
    ds_list   = []
    lang_codes = []
    for lang_code, ds in lang_datasets.items():
        fmt = make_mayan_tagged_fmt(lang_code)
        ds_list.append(ds.map(fmt, remove_columns=ds.column_names))
        lang_codes.append(lang_code)

    weights = [len(ds) ** 0.7 for ds in ds_list]
    probs   = [w / sum(weights) for w in weights]

    mixed = interleave_datasets(
        ds_list, probabilities=probs, seed=42, stopping_strategy="all_exhausted"
    )
    print(f"Flat Mayan mix (tagged): {len(mixed)} examples from {len(ds_list)} languages")
    return mixed


# ── Model utils ───────────────────────────────────────────────────────────────

def build_lora_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="eager"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

def load_peft_model(adapter_path):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model

def make_sft_config(output_dir, epochs, lr, use_eval=True):
    return SFTConfig(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=lr, logging_steps=20, max_length=MAX_LEN, packing=False,
        bf16=BF16_OK, fp16=FP16_OK,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=50 if use_eval else None,
        save_strategy="steps", save_steps=50, save_total_limit=5,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        save_safetensors=True, report_to="tensorboard",
    )


# ── Perplexity ────────────────────────────────────────────────────────────────

def safe_ppl(loss):
    return float("inf") if loss > 50 else math.exp(loss)

@torch.no_grad()
def compute_qanjobal_ppl(model):
    ds    = load_from_disk(QANJOBAL_TEST_PATH)
    col   = "Qanjobal" if "Qanjobal" in ds.column_names else "mayan"
    texts = [str(t).strip() for t in ds[col] if t]
    enc   = tokenizer(texts, truncation=False, add_special_tokens=False)
    ids   = sum(enc["input_ids"], [])
    total = (len(ids) // BLOCK_SIZE) * BLOCK_SIZE
    if total == 0: return float("inf")
    chunks   = [ids[i:i+BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
    block_ds = Dataset.from_dict({
        "input_ids":      chunks,
        "attention_mask": [[1]*BLOCK_SIZE]*len(chunks),
        "labels":         chunks,
    })
    block_ds.set_format("torch")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader   = DataLoader(block_ds, batch_size=4, collate_fn=collator)
    model.eval()
    tl, tt = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(**batch)
        n     = (batch["labels"] != -100).sum().item()
        tl += out.loss.item() * n; tt += n
    return safe_ppl(tl / tt) if tt > 0 else float("inf")


class PPLTracker:
    def __init__(self): self.epochs = []; self.eval_ppl = []

def make_ppl_callback(tracker, model_getter):
    class _CB(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = round(state.epoch)
            ppl   = compute_qanjobal_ppl(model_getter())
            tracker.epochs.append(epoch); tracker.eval_ppl.append(ppl)
            print(f"    Epoch {epoch} Q'anjob'al test PPL: {ppl:.2f}")
    return _CB()


# ── Log extraction ────────────────────────────────────────────────────────────

def find_trainer_state(stage_dir):
    candidates = [
        os.path.join(stage_dir, "trainer_state.json"),
        *sorted(Path(stage_dir).glob("checkpoint-*/trainer_state.json")),
    ]
    for c in candidates:
        if os.path.exists(str(c)): return str(c)
    return None

def extract_logs(path):
    with open(path) as f: state = json.load(f)
    train_steps, train_losses, eval_steps, eval_losses = [], [], [], []
    for e in state["log_history"]:
        step = e.get("step", 0)
        if "loss" in e:     train_steps.append(step); train_losses.append(e["loss"])
        if "eval_loss" in e: eval_steps.append(step); eval_losses.append(e["eval_loss"])
    return {"train_steps": train_steps, "train_losses": train_losses,
            "eval_steps": eval_steps,   "eval_losses": eval_losses}


# ── Evaluation ────────────────────────────────────────────────────────────────

bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

@torch.inference_mode()
def evaluate_translation(model, split_path, max_examples=None):
    model.eval()
    raw = load_from_disk(split_path)
    if max_examples: raw = raw.select(range(min(max_examples, len(raw))))
    english = [str(x).strip() for x in raw["English"]]
    refs    = [str(x).strip() for x in raw["Qanjobal"]]
    preds   = []
    for i in range(0, len(english), BATCH):
        prompts = [build_prompt_qanjobal(e) for e in english[i:i+BATCH]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN).to(device)
        out_ids = model.generate(
            **enc, max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=GEN_DO_SAMPLE, temperature=GEN_TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        for seq, pl in zip(out_ids, prompt_lens):
            preds.append(tokenizer.decode(seq[pl:], skip_special_tokens=True).strip())
    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    chrf = chrf_metric.compute(predictions=preds, references=refs)
    return {"bleu": float(bleu["score"]), "chrf": float(chrf["score"])}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(stage_logs, ppl_trackers, plots_dir, title):
    os.makedirs(plots_dir, exist_ok=True)
    colors = plt.cm.tab10.colors
    n = len(stage_logs)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4*n), squeeze=False)
    fig.suptitle(f"{title} — Loss Curves", fontsize=13, fontweight="bold")
    for i, (label, logs) in enumerate(stage_logs.items()):
        c = colors[i % len(colors)]
        ax_tr, ax_ev = axes[i]
        if logs.get("train_losses"):
            ax_tr.plot(logs["train_steps"], logs["train_losses"], color=c, lw=1.5)
        ax_tr.set_title(f"{label} — Train Loss")
        ax_tr.set_xlabel("Step"); ax_tr.set_ylabel("Loss"); ax_tr.grid(True, alpha=0.3)
        if logs.get("eval_losses"):
            best_i = logs["eval_losses"].index(min(logs["eval_losses"]))
            ax_ev.plot(logs["eval_steps"], logs["eval_losses"],
                       color=c, lw=1.5, ls="--", marker="o", ms=3, label="Val loss")
            ax_ev.axvline(logs["eval_steps"][best_i], color=c, ls=":", alpha=0.7,
                          label=f"Best @ step {logs['eval_steps'][best_i]}")
            ax_ev.legend(fontsize=8)
        ax_ev.set_title(f"{label} — Val Loss (carved from train)")
        ax_ev.set_xlabel("Step"); ax_ev.set_ylabel("Loss"); ax_ev.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(plots_dir, "loss_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"  Saved: {p}")

    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.suptitle(f"{title} — Q'anjob'al Test PPL", fontsize=13, fontweight="bold")
    offset = 0; xticks_pos, xticks_lab = [], []
    for i, (label, tracker) in enumerate(ppl_trackers.items()):
        c = colors[i % len(colors)]
        epochs_abs = [offset + e for e in tracker.epochs]
        if tracker.eval_ppl:
            ax.plot(epochs_abs, tracker.eval_ppl, color=c, lw=1.5, marker="o", ms=4, label=label)
        if offset > 0: ax.axvline(offset, color="gray", ls=":", alpha=0.4)
        for e in tracker.epochs:
            xticks_pos.append(offset + e); xticks_lab.append(f"{label}\nE{e}")
        if tracker.epochs: offset += max(tracker.epochs)
    ax.set_xlabel("Epoch (cumulative)"); ax.set_ylabel("Perplexity (test, diagnostic)")
    ax.set_xticks(xticks_pos); ax.set_xticklabels(xticks_lab, fontsize=7, rotation=45)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "perplexity_over_epochs.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(); print(f"  Saved: {p2}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    exp_name  = "G2: Unstructured Mayan SFT + Lang Tags"
    out_dir   = "experiments-llama/G2_unstructured_tagged"
    print(f"\n{'='*65}\n{exp_name}\n{'='*65}")
    print(f"Test set: {QANJOBAL_TEST_PATH} (never used during training)")

    print("\nLoading Mayan splits...")
    lang_datasets = load_all_splits(SPLITS_DIR)

    print("\nPreparing Q'anjob'al datasets...")
    qanjobal_train_ds, val_ds = make_qanjobal_sft_datasets()

    os.makedirs(out_dir, exist_ok=True)
    stage_logs   = {}
    ppl_trackers = {}

    # ── Stage 1: Flat Mayan mix WITH lang tags ────────────────────────────────
    print("\n--- Stage 1: All Mayan (flat mix, with lang tags) ---")
    mayan_mix     = assemble_flat_mayan_tagged(lang_datasets)
    mayan_model   = build_lora_model(BASE_MODEL)
    mayan_dir     = out_dir + "/mayan_sft"
    mayan_tracker = PPLTracker()

    mayan_trainer = SFTTrainer(
        model=mayan_model,
        args=make_sft_config(mayan_dir, epochs=1, lr=LR, use_eval=True),
        train_dataset=mayan_mix,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    mayan_trainer.add_callback(
        make_ppl_callback(mayan_tracker, lambda: mayan_trainer.model)
    )
    mayan_trainer.train()

    mayan_best = mayan_dir + "/best"
    mayan_trainer.model.save_pretrained(mayan_best)
    tokenizer.save_pretrained(mayan_best)
    state_path = find_trainer_state(mayan_dir)
    stage_logs["Mayan SFT (tagged)"]   = extract_logs(state_path) if state_path else {}
    ppl_trackers["Mayan SFT (tagged)"] = mayan_tracker
    del mayan_model; torch.cuda.empty_cache()

    # ── Stage 2: Q'anjob'al SFT ───────────────────────────────────────────────
    print(f"\n--- Stage 2: Q'anjob'al SFT ({len(qanjobal_train_ds)} train | {len(val_ds)} val) ---")
    sft_model   = load_peft_model(mayan_best)
    sft_dir     = out_dir + "/qanjobal_sft"
    sft_tracker = PPLTracker()

    sft_trainer = SFTTrainer(
        model=sft_model,
        args=make_sft_config(sft_dir, epochs=EPOCHS, lr=LR*0.5, use_eval=True),
        train_dataset=qanjobal_train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    sft_trainer.add_callback(
        make_ppl_callback(sft_tracker, lambda: sft_trainer.model)
    )
    sft_trainer.train()

    sft_best = sft_dir + "/best"
    sft_trainer.model.save_pretrained(sft_best)
    tokenizer.save_pretrained(sft_best)
    state_path = find_trainer_state(sft_dir)
    stage_logs["Q'anjob'al SFT"]   = extract_logs(state_path) if state_path else {}
    ppl_trackers["Q'anjob'al SFT"] = sft_tracker

    # ── Final eval on test set ─────────────────────────────────────────────────
    print(f"\nEvaluating on test set: {QANJOBAL_TEST_PATH}")
    final = load_peft_model(sft_best)
    final.to(device)
    metrics = evaluate_translation(final, QANJOBAL_TEST_PATH, EVAL_MAX_EXAMPLES)
    print(f"\n{exp_name} TEST results: {metrics}")
    del final; torch.cuda.empty_cache()

    plot_curves(stage_logs, ppl_trackers, out_dir + "/plots", exp_name)

    os.makedirs("experiments-llama", exist_ok=True)
    results_path = "experiments-llama/results_test.json"
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f: existing = json.load(f)
    existing[exp_name] = metrics
    with open(results_path, "w") as f: json.dump(existing, f, indent=2)

    print(f"\nG2 — BLEU: {metrics['bleu']:.2f} | chrF: {metrics['chrf']:.2f}")
    print(f"Saved to experiments-llama/results_test.json")


if __name__ == "__main__":
    main()
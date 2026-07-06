"""
stage2_qanjobal_only.py
=======================
Stage 2 — Experiment 2: Translation SFT on Q'anjob'al pairs ONLY.
Starts from the Stage 1 LangID adapter (langid_sft/best).

Train: Q'anjob'al train split only
Eval:  test splits for ALL Mayan languages + Q'anjob'al test split

Usage:
    python stage2_qanjobal_only.py
    python stage2_qanjobal_only.py --adapter langid_sft/best

Outputs:
    stage2_qanjobal_only/best/                  <- final LoRA adapter
    stage2_qanjobal_only/translation_eval.json  <- per-language BLEU + chrF
    stage2_qanjobal_only/plots/                 <- training curves
"""

import os, json, random, argparse, torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────

CPT_MODEL        = "atara15/continued_pretrain_llama_3_2_1"
STAGE1_ADAPTER   = "atara15/llama_cpt_model_finetuned_with_classification"
SPLITS_DIR       = "mayan_data/splits"
OUTPUT_DIR       = "stage2_qanjobal_only"

QANJOBAL_TRAIN_PATH = "data/train_split"
QANJOBAL_EVAL_PATH  = "data/test_split"

EPOCHS             = 5
MAX_LEN            = 512
BATCH              = 4
GRAD_ACCUM         = 4
LR                 = 2e-4
GEN_MAX_NEW_TOKENS = 196

USE_CUDA = torch.cuda.is_available()
BF16_OK  = USE_CUDA and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
FP16_OK  = USE_CUDA and not BF16_OK

dtype  = torch.bfloat16 if BF16_OK else (torch.float16 if FP16_OK else torch.float32)
device = "cuda" if USE_CUDA else "cpu"
print(f"Device: {device} | dtype: {dtype}")

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Language registry (for eval) ──────────────────────────────────────────────

LANGUAGE_META = {
    "acr": "Achi",
    "agu": "Awakatec",
    "cac": "Chuj",
    "itz": "Itza'",
    "ixl": "Ixil",
    "kek": "Q'eqchi'",
    "mam": "Mam",
    "poc": "Poqomam",
    "poh": "Poqomchi'",
    "quc": "K'iche'",
    "qum": "Sipakapense",
    "ttc": "Tektitek",
    "tzj": "Tz'utjuil",
}

# ── Tokenizer ──────────────────────────────────────────────────────────────────

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(CPT_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    GENERIC_TEMPLATE = (
        "{% if bos_token %}{{ bos_token }}{% endif -%}"
        "{% for m in messages -%}"
        "{{ m['role'].upper() }}: {{ m['content'] }}\n"
        "{% endfor -%}"
        "ASSISTANT:"
    )
    if not getattr(tok, "chat_template", None):
        tok.chat_template = GENERIC_TEMPLATE
    return tok

# ── Prompt builders ───────────────────────────────────────────────────────────

def _apply_template(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def build_translation_prompt_train(tokenizer, english: str, mayan: str, lang_name: str) -> str:
    content = (
        f"Task: Translate this text from English to {lang_name}:\n"
        f"User content:\n{english}\n"
    )
    return _apply_template(tokenizer, content) + mayan

def build_translation_prompt_eval(tokenizer, english: str, lang_name: str) -> str:
    content = (
        f"Task: Translate this text from English to {lang_name}:\n"
        f"User content:\n{english}\n"
    )
    return _apply_template(tokenizer, content)

# ── Dataset ────────────────────────────────────────────────────────────────────

def _get_cols(ds, is_qanjobal=False):
    if is_qanjobal:
        eng = "English" if "English" in ds.column_names else "english"
        may = "Qanjobal" if "Qanjobal" in ds.column_names else "mayan"
    else:
        eng = "english" if "english" in ds.column_names else "English"
        may = "mayan" if "mayan" in ds.column_names else ds.column_names[0]
    return eng, may

def load_data(tokenizer) -> tuple:
    all_train = []
    eval_records = []

    # ── Training: Q'anjob'al only ──────────────────────────────────────────────
    ds = load_from_disk(QANJOBAL_TRAIN_PATH)
    eng_col, may_col = _get_cols(ds, is_qanjobal=True)
    count = 0
    for e, m in zip(ds[eng_col], ds[may_col]):
        eng, may = str(e).strip(), str(m).strip()
        if eng and may:
            all_train.append({
                "text": build_translation_prompt_train(tokenizer, eng, may, "Q'anjob'al"),
            })
            count += 1
    print(f"  [kjb] Q'anjob'al train: {count}")

    # ── Eval: all Mayan test splits ────────────────────────────────────────────
    for lang_code, lang_name in LANGUAGE_META.items():
        test_path = os.path.join(SPLITS_DIR, lang_code, "test")
        if os.path.exists(test_path):
            ds = load_from_disk(test_path)
            eng_col, may_col = _get_cols(ds)
            count = 0
            for e, m in zip(ds[eng_col], ds[may_col]):
                eng, may = str(e).strip(), str(m).strip()
                if eng and may:
                    eval_records.append({
                        "lang": lang_name,
                        "english": eng,
                        "reference": may,
                    })
                    count += 1
            print(f"  [{lang_code}] {lang_name} test: {count}")
        else:
            print(f"  [{lang_code}] test not found at {test_path}, skipping")

    # Q'anjob'al test
    if os.path.exists(QANJOBAL_EVAL_PATH):
        ds = load_from_disk(QANJOBAL_EVAL_PATH)
        eng_col, may_col = _get_cols(ds, is_qanjobal=True)
        count = 0
        for e, m in zip(ds[eng_col], ds[may_col]):
            eng, may = str(e).strip(), str(m).strip()
            if eng and may:
                eval_records.append({
                    "lang": "Q'anjob'al",
                    "english": eng,
                    "reference": may,
                })
                count += 1
        print(f"  [kjb] Q'anjob'al test: {count}")

    random.shuffle(all_train)
    train_ds = Dataset.from_list(all_train)

    # SFT eval subset for loss tracking (held-out slice of Q'anjob'al train)
    sft_eval_size = min(300, len(all_train) // 10)
    sft_eval_ds = Dataset.from_list(random.sample(all_train, sft_eval_size))

    print(f"\n  Total — train: {len(train_ds)} | sft_eval: {len(sft_eval_ds)} | translation_eval: {len(eval_records)}")
    return train_ds, sft_eval_ds, eval_records

# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(tokenizer, adapter_path: str):
    print(f"\nLoading base model: {CPT_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        CPT_MODEL, torch_dtype=dtype, attn_implementation="eager"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading & merging Stage 1 adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

# ── Translation eval ──────────────────────────────────────────────────────────

bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

@torch.inference_mode()
def evaluate_translation_all(model, tokenizer, eval_records: list) -> dict:
    model.eval()
    per_lang = defaultdict(lambda: {"preds": [], "refs": []})

    for i in range(0, len(eval_records), BATCH):
        batch = eval_records[i:i+BATCH]
        prompts = [
            build_translation_prompt_eval(tokenizer, r["english"], r["lang"])
            for r in batch
        ]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LEN,
        ).to(device)
        out_ids = model.generate(
            **enc, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        for j, (seq, pl) in enumerate(zip(out_ids, prompt_lens)):
            pred = tokenizer.decode(seq[pl:], skip_special_tokens=True).strip()
            lang = batch[j]["lang"]
            per_lang[lang]["preds"].append(pred)
            per_lang[lang]["refs"].append(batch[j]["reference"])

    results = {}
    print(f"\n{'Language':<20} {'BLEU':>8} {'chrF':>8} {'N':>6}")
    print("-" * 46)
    for lang_name in sorted(per_lang.keys()):
        preds = per_lang[lang_name]["preds"]
        refs  = per_lang[lang_name]["refs"]
        b = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
        c = chrf_metric.compute(predictions=preds, references=refs)
        results[lang_name] = {
            "bleu": round(float(b["score"]), 4),
            "chrf": round(float(c["score"]), 4),
            "n": len(preds),
        }
        print(f"  {lang_name:<18} {b['score']:>8.2f} {c['score']:>8.2f} {len(preds):>6}")

    all_preds = [p for v in per_lang.values() for p in v["preds"]]
    all_refs  = [r for v in per_lang.values() for r in v["refs"]]
    b_all = bleu_metric.compute(predictions=all_preds, references=[[r] for r in all_refs])
    c_all = chrf_metric.compute(predictions=all_preds, references=all_refs)
    results["_overall"] = {
        "bleu": round(float(b_all["score"]), 4),
        "chrf": round(float(c_all["score"]), 4),
        "n": len(all_preds),
    }
    print("-" * 46)
    print(f"  {'OVERALL':<18} {b_all['score']:>8.2f} {c_all['score']:>8.2f} {len(all_preds):>6}")
    return results

# ── Plotting ──────────────────────────────────────────────────────────────────

def find_trainer_state(output_dir: str):
    candidates = [
        os.path.join(output_dir, "trainer_state.json"),
        *sorted(Path(output_dir).glob("checkpoint-*/trainer_state.json")),
    ]
    for c in candidates:
        if os.path.exists(str(c)):
            return str(c)
    return None

def plot_results(output_dir, trans_results, state_path):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    train_steps, train_losses, eval_steps, eval_losses = [], [], [], []
    if state_path:
        with open(state_path) as f:
            state = json.load(f)
        for e in state["log_history"]:
            step = e.get("step", 0)
            if "loss"      in e: train_steps.append(step); train_losses.append(e["loss"])
            if "eval_loss" in e: eval_steps.append(step);  eval_losses.append(e["eval_loss"])

    if train_losses or eval_losses:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Stage 2 (Q'anjob'al Only) — Training Curves", fontsize=13, fontweight="bold")
        if train_losses:
            axes[0].plot(train_steps, train_losses, color="steelblue", lw=1.5)
        axes[0].set_title("Train Loss"); axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss"); axes[0].grid(True, alpha=0.3)
        if eval_losses:
            axes[1].plot(eval_steps, eval_losses, color="darkorange", lw=1.5, ls="--", marker="o", ms=3)
        axes[1].set_title("Eval Loss"); axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss"); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(plots_dir, "training_curves.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p}")

    lang_results = {k: v for k, v in trans_results.items() if not k.startswith("_")}
    if lang_results:
        names = sorted(lang_results.keys())
        bleus = [lang_results[n]["bleu"] for n in names]
        chrfs = [lang_results[n]["chrf"] for n in names]

        fig2, ax = plt.subplots(figsize=(max(12, len(names) * 1.2), 6))
        x = np.arange(len(names))
        w = 0.35
        bars1 = ax.bar(x - w/2, bleus, w, label="BLEU", color="steelblue", alpha=0.85)
        bars2 = ax.bar(x + w/2, chrfs, w, label="chrF", color="seagreen", alpha=0.85)
        for bar, val in zip(bars1, bleus):
            ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width()/2., bar.get_height()),
                        ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars2, chrfs):
            ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width()/2., bar.get_height()),
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title("Per-Language Translation Scores (Q'anjob'al-Only Training)", fontweight="bold")
        ax.set_ylabel("Score"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        p2 = os.path.join(plots_dir, "per_language_translation.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p2}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=STAGE1_ADAPTER)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Stage 1 adapter : {args.adapter}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print(f"Experiment      : Q'anjob'al only")

    tokenizer = load_tokenizer()
    train_ds, sft_eval_ds, eval_records = load_data(tokenizer)
    model = build_model(tokenizer, args.adapter)

    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=20,
        max_length=MAX_LEN,
        packing=False,
        bf16=BF16_OK,
        fp16=FP16_OK,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=train_ds, eval_dataset=sft_eval_ds,
        processing_class=tokenizer,
    )

    print("\nTraining Stage 2 (Q'anjob'al Only Translation SFT)...")
    trainer.train()

    # ── Save best adapter ──────────────────────────────────────────────────────
    best_dir = os.path.join(OUTPUT_DIR, "best")
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nAdapter saved: {best_dir}")

    # ── Translation eval (all languages) ───────────────────────────────────────
    print("\nRunning per-language translation eval (all Mayan)...")
    trans_results = evaluate_translation_all(trainer.model, tokenizer, eval_records)
    out_path = os.path.join(OUTPUT_DIR, "translation_eval.json")
    with open(out_path, "w") as f:
        json.dump(trans_results, f, indent=2)
    print(f"Saved: {out_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    state_path = find_trainer_state(OUTPUT_DIR)
    plot_results(OUTPUT_DIR, trans_results, state_path)

    overall = trans_results.get("_overall", {})
    qanjobal = trans_results.get("Q'anjob'al", {})
    print(f"\n{'='*55}")
    print(f"Stage 2 (Q'anjob'al Only) complete.")
    print(f"  Q'anjob'al BLEU : {qanjobal.get('bleu', 'N/A')}")
    print(f"  Q'anjob'al chrF : {qanjobal.get('chrf', 'N/A')}")
    print(f"  Overall BLEU    : {overall.get('bleu', 'N/A')}")
    print(f"  Overall chrF    : {overall.get('chrf', 'N/A')}")
    print(f"  Adapter         : {best_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
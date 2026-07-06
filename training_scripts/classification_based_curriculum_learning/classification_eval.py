"""
eval_only.py
============
Eval-only script for Stage 1 LangID adapter.
Loads a saved LoRA adapter, runs classification accuracy + translation BLEU/chrF,
and generates plots. No training.

Usage:
    python eval_only.py
    python eval_only.py --adapter langid_sft/best
    python eval_only.py --adapter langid_sft/checkpoint-4900
"""

import os, json, random, argparse, torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────

CPT_MODEL  = "atara15/continued_pretrain_llama_3_2_1"
SPLITS_DIR = "mayan_data/splits"
OUTPUT_DIR = "langid_sft"

QANJOBAL_TRAIN_PATH = "data/train_split"
QANJOBAL_EVAL_PATH  = "data/test_split"

MAX_LEN            = 256
BATCH              = 4
SNIPPET_MAX_CHARS  = 300
GEN_MAX_NEW_TOKENS = 196

USE_CUDA = torch.cuda.is_available()
BF16_OK  = USE_CUDA and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
FP16_OK  = USE_CUDA and not BF16_OK

dtype  = torch.bfloat16 if BF16_OK else (torch.float16 if FP16_OK else torch.float32)
device = "cuda" if USE_CUDA else "cpu"
print(f"Device: {device} | dtype: {dtype}")

# ── Language registry ──────────────────────────────────────────────────────────

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
    "kjb": "Q'anjob'al",
}

CHOICES = ", ".join(sorted(LANGUAGE_META.values()))

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

def build_langid_prompt(tokenizer, snippet: str, lang_name: str) -> str:
    content = (
        f"What language is this text written in?\n"
        f"Pick one of: {CHOICES}\n\n"
        f"Text:\n{snippet.strip()[:SNIPPET_MAX_CHARS]}\n"
    )
    return _apply_template(tokenizer, content) + lang_name

def build_translation_prompt(tokenizer, english: str) -> str:
    content = (
        "Task: Translate this text from English to Qanjobal:\n"
        f"User content:\n{english}\n"
    )
    return _apply_template(tokenizer, content)

# ── Dataset ────────────────────────────────────────────────────────────────────

def load_lang_snippets(tokenizer) -> tuple:
    all_train, all_eval = [], []

    for lang_code, lang_name in LANGUAGE_META.items():
        if lang_code == "kjb":
            continue

        path = os.path.join(SPLITS_DIR, lang_code, "train")
        if not os.path.exists(path):
            print(f"  [{lang_code}] not found at {path}, skipping")
            continue

        ds   = load_from_disk(path)
        col  = "mayan" if "mayan" in ds.column_names else ds.column_names[0]
        texts = [str(t).strip() for t in ds[col] if str(t).strip()]

        random.shuffle(texts)
        split = max(1, int(0.9 * len(texts)))
        all_train.extend(
            {"text": build_langid_prompt(tokenizer, t, lang_name)}
            for t in texts[:split]
        )
        all_eval.extend(
            {"text": build_langid_prompt(tokenizer, t, lang_name)}
            for t in texts[split:]
        )
        print(f"  [{lang_code}] {lang_name}: {split} train, {len(texts)-split} eval")

    if os.path.exists(QANJOBAL_TRAIN_PATH):
        qds  = load_from_disk(QANJOBAL_TRAIN_PATH)
        col  = "Qanjobal" if "Qanjobal" in qds.column_names else "mayan"
        texts = [str(t).strip() for t in qds[col] if str(t).strip()]

        random.shuffle(texts)
        split = max(1, int(0.9 * len(texts)))
        all_train.extend(
            {"text": build_langid_prompt(tokenizer, t, "Q'anjob'al")}
            for t in texts[:split]
        )
        all_eval.extend(
            {"text": build_langid_prompt(tokenizer, t, "Q'anjob'al")}
            for t in texts[split:]
        )
        print(f"  [kjb] Q'anjob'al: {split} train, {len(texts)-split} eval")
    else:
        print(f"  [kjb] not found at {QANJOBAL_TRAIN_PATH}, skipping")

    random.shuffle(all_train)
    random.shuffle(all_eval)
    train_ds = Dataset.from_list(all_train)
    eval_ds  = Dataset.from_list(all_eval)
    print(f"\n  Total — train: {len(train_ds)} | eval: {len(eval_ds)}")
    return train_ds, eval_ds

# ── Load model from adapter ───────────────────────────────────────────────────

def load_model_from_adapter(adapter_path: str, tokenizer):
    print(f"\nLoading base model: {CPT_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        CPT_MODEL, torch_dtype=dtype, attn_implementation="eager"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(device)
    model.eval()
    return model

# ── Classification eval ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_accuracy(model, tokenizer, eval_ds) -> dict:
    per_lang = defaultdict(lambda: {"correct": 0, "total": 0})
    model.eval()

    for ex in eval_ds:
        full_text = ex["text"]
        marker    = "ASSISTANT:"
        split_idx = full_text.rfind(marker)
        if split_idx == -1:
            continue
        prompt   = full_text[:split_idx + len(marker)]
        expected = full_text[split_idx + len(marker):].strip()

        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN
        ).to(device)
        out = model.generate(
            **enc, max_new_tokens=12, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        matched_lang = next(
            (name for name in LANGUAGE_META.values() if expected.startswith(name)),
            None,
        )
        if matched_lang:
            per_lang[matched_lang]["total"] += 1
            if generated.startswith(matched_lang):
                per_lang[matched_lang]["correct"] += 1

    total_c = sum(v["correct"] for v in per_lang.values())
    total_t = sum(v["total"]   for v in per_lang.values())
    overall = total_c / total_t if total_t > 0 else 0.0

    print(f"\n{'Language':<20} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 48)
    results = {"overall_accuracy": overall, "per_language": {}}
    for lang_name, counts in sorted(per_lang.items()):
        if counts["total"] == 0:
            continue
        acc = counts["correct"] / counts["total"]
        results["per_language"][lang_name] = {
            "correct": counts["correct"], "total": counts["total"], "accuracy": acc,
        }
        print(f"  {lang_name:<18} {counts['correct']:>8} {counts['total']:>8} {acc:>8.3f}")
    print("-" * 48)
    print(f"  {'OVERALL':<18} {total_c:>8} {total_t:>8} {overall:>8.3f}")
    return results

# ── Translation eval ──────────────────────────────────────────────────────────

bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

@torch.inference_mode()
def evaluate_translation(model, tokenizer, split_path: str) -> dict:
    model.eval()
    ds      = load_from_disk(split_path)
    eng_col = "English"  if "English"  in ds.column_names else "english"
    may_col = "Qanjobal" if "Qanjobal" in ds.column_names else "mayan"
    english = [str(x).strip() for x in ds[eng_col]]
    refs    = [str(x).strip() for x in ds[may_col]]
    preds   = []

    for i in range(0, len(english), BATCH):
        prompts = [build_translation_prompt(tokenizer, e) for e in english[i:i+BATCH]]
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
        for seq, pl in zip(out_ids, prompt_lens):
            preds.append(tokenizer.decode(seq[pl:], skip_special_tokens=True).strip())

    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    chrf = chrf_metric.compute(predictions=preds, references=refs)
    return {"bleu": round(float(bleu["score"]), 4), "chrf": round(float(chrf["score"]), 4)}

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

def plot_results(output_dir, eval_results, state_path):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Training curves from trainer_state (if available)
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
        fig.suptitle("LangID SFT — Training Curves", fontsize=13, fontweight="bold")
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

    # Per-language accuracy bar chart
    per_lang = eval_results.get("per_language", {})
    if per_lang:
        names  = list(per_lang.keys())
        accs   = [per_lang[n]["accuracy"] for n in names]
        colors = [
            "seagreen" if a >= 0.8 else "darkorange" if a >= 0.5 else "tomato"
            for a in accs
        ]
        fig2, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 5))
        bars = ax.bar(names, accs, color=colors, alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.annotate(f"{acc:.2f}",
                        (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.axhline(eval_results["overall_accuracy"], color="black", ls="--", lw=1.5,
                   label=f"Overall: {eval_results['overall_accuracy']:.3f}")
        ax.set_ylim(0, 1.1)
        ax.set_title("Per-Language Classification Accuracy", fontweight="bold")
        ax.set_ylabel("Accuracy"); ax.set_xlabel("Language")
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        p2 = os.path.join(plots_dir, "per_language_accuracy.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p2}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eval-only for Stage 1 LangID adapter")
    parser.add_argument("--adapter", type=str, default="langid_sft/best",
                        help="Path to saved LoRA adapter")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Adapter    : {args.adapter}")
    print(f"Output dir : {OUTPUT_DIR}")

    tokenizer = load_tokenizer()
    _, eval_ds = load_lang_snippets(tokenizer)
    model = load_model_from_adapter(args.adapter, tokenizer)

    # ── Classification ─────────────────────────────────────────────────────────
    print("\nRunning full classification accuracy eval...")
    cls_results = evaluate_accuracy(model, tokenizer, eval_ds)
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(cls_results, f, indent=2)
    print(f"Saved: {out_path}")

    # ── Translation ────────────────────────────────────────────────────────────
    print(f"\nRunning translation eval on {QANJOBAL_EVAL_PATH}...")
    print("(Pre-Stage-2 baseline — expect low scores.)")
    trans_results = evaluate_translation(model, tokenizer, QANJOBAL_EVAL_PATH)
    print(f"  BLEU : {trans_results['bleu']}")
    print(f"  chrF : {trans_results['chrf']}")
    out_path = os.path.join(OUTPUT_DIR, "translation_eval.json")
    with open(out_path, "w") as f:
        json.dump(trans_results, f, indent=2)
    print(f"Saved: {out_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    state_path = find_trainer_state(OUTPUT_DIR)
    plot_results(OUTPUT_DIR, cls_results, state_path)

    print(f"\n{'='*55}")
    print(f"Eval complete.")
    print(f"  Classification accuracy : {cls_results['overall_accuracy']:.3f}")
    print(f"  Translation BLEU        : {trans_results['bleu']}")
    print(f"  Translation chrF        : {trans_results['chrf']}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
"""
classification_sft.py
================
Stage 1: Generative Language ID SFT on the CPT checkpoint.

Reuses all functions from train_langid_sft.py unchanged:
  - load_tokenizer, build_model, load_lang_snippets, make_prompt
  - AccuracyTracker, AccuracyCallback, evaluate_accuracy
  - find_trainer_state, plot_results

What this file adds:
  - Translation eval (BLEU + chrF) on QANJOBAL_EVAL_PATH after training,
    so we can see the baseline translation quality before Stage 2.
  - Saves adapter to langid_sft/best/ for Stage 2 to pick up.

Outputs
-------
  langid_sft/best/             <- LoRA adapter, pass to stage2_translation.py
  langid_sft/plots/            <- loss + accuracy curves + per-language bar
  langid_sft/results.json      <- per-language classification accuracy
  langid_sft/translation_eval.json  <- BLEU + chrF before translation training
"""

import os, json, random, torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reuse everything from train_langid_sft ─────────────────────────────────────

from trl import SFTConfig, SFTTrainer

CPT_MODEL  = "atara15/continued_pretrain_llama_3_2_1"
SPLITS_DIR = "mayan_data/splits"
OUTPUT_DIR = "langid_sft"
 
QANJOBAL_TRAIN_PATH = "data/train_split"
QANJOBAL_EVAL_PATH  = "data/test_split"
 
MAX_LEN            = 256
BATCH              = 4
GRAD_ACCUM         = 4
LR                 = 2e-4
EPOCHS             = 3
SNIPPET_MAX_CHARS  = 300
GEN_MAX_NEW_TOKENS = 196

USE_CUDA = torch.cuda.is_available()
BF16_OK  = USE_CUDA and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
FP16_OK  = USE_CUDA and not BF16_OK

dtype    = torch.bfloat16 if BF16_OK else (torch.float16 if FP16_OK else torch.float32)
device   = "cuda" if USE_CUDA else "cpu"
print(f"Device: {device} | dtype: {dtype}")
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
 
# ── Language registry ──────────────────────────────────────────────────────────
LANGUAGE_META = {
    
 "acr": "Achi",
 "agu" : "Awakatec",
 "cac": "Chuj",
 "itz" : "Itza'",
 "ixl" : "Ixil",
 "kek" : "Q'eqchi'",
 "mam" : "Mam",
 "poc" : "Poqomam",
 "poh" : "Poqomchi'",
 "quc" : "K'iche'",
 "qum" : "Sipakapense",
 "ttc" : "Tektitek",
 "tzj" : "Tz'utjuil",
 "kjb": "Q'anjob'al"
}
 
CHOICES = ", ".join(sorted(LANGUAGE_META.values()))
 
 
# ── Tokenizer ──────────────────────────────────────────────────────────────────
 
def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(CPT_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # required for correct loss masking in causal LM
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
 
 
# ── Prompt builders ────────────────────────────────────────────────────────────
 
def _apply_template(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
 
def build_langid_prompt(tokenizer, snippet: str, lang_name: str) -> str:
    """
    Full training string = prompt + answer.
    The model learns to generate lang_name after the ASSISTANT: marker.
    Follows the same pattern as build_chuj_prompt / build_qanjobal_prompt.
    """
    content = (
        f"What language is this text written in?\n"
        f"Pick one of: {CHOICES}\n\n"
        f"Text:\n{snippet.strip()[:SNIPPET_MAX_CHARS]}\n"
    )
    return _apply_template(tokenizer, content) + lang_name
 
def build_translation_prompt(tokenizer, english: str) -> str:
    """Prompt-only (no answer) — used during translation inference."""
    content = (
        "Task: Translate this text from English to Qanjobal:\n"
        f"User content:\n{english}\n"
    )
    return _apply_template(tokenizer, content)
 
 
# ── Dataset ────────────────────────────────────────────────────────────────────
 
def load_lang_snippets(tokenizer) -> tuple:
    """
    Build (train_ds, eval_ds) with a single 'text' column containing the
    full prompt+answer string for each snippet.
    90/10 split per language, then concatenate and shuffle globally.
    """
    all_train, all_eval = [], []
 
    # Mayan languages from splits directory
    for lang_code, lang_name in LANGUAGE_META.items():
        if lang_code == "kjb":
            continue   # handled separately below
 
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
 
    # Q'anjob'al
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
 
 
# ── Model ──────────────────────────────────────────────────────────────────────
 
def build_model(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        CPT_MODEL, torch_dtype=dtype, attn_implementation="eager"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
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
 
 
# ── Accuracy tracking ──────────────────────────────────────────────────────────
 
class AccuracyTracker:
    def __init__(self):
        self.steps      = []
        self.accuracies = []
 
class AccuracyCallback(TrainerCallback):
    """
    Every `eval_every` steps, sample `sample_size` eval examples, generate
    one answer, and check if it starts with the expected language name.
    Lightweight proxy — full eval runs once at the end of training.
    """
    def __init__(self, tracker, eval_ds, tokenizer, sample_size=64, eval_every=50):
        self.tracker     = tracker
        self.eval_ds     = eval_ds
        self.tokenizer   = tokenizer
        self.sample_size = sample_size
        self.eval_every  = eval_every
 
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every != 0:
            return
        indices = random.sample(range(len(self.eval_ds)),
                                min(self.sample_size, len(self.eval_ds)))
        correct = 0
        model.eval()
        with torch.no_grad():
            for idx in indices:
                full_text = self.eval_ds[idx]["text"]
                marker    = "ASSISTANT:"
                split_idx = full_text.rfind(marker)
                if split_idx == -1:
                    continue
                prompt   = full_text[:split_idx + len(marker)]
                expected = full_text[split_idx + len(marker):].strip()
 
                enc = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=MAX_LEN
                ).to(device)
                out = model.generate(
                    **enc, max_new_tokens=12, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
 
                if any(
                    expected.startswith(name) and generated.startswith(name)
                    for name in LANGUAGE_META.values()
                ):
                    correct += 1
 
        acc = correct / len(indices)
        self.tracker.steps.append(state.global_step)
        self.tracker.accuracies.append(acc)
        print(f"    Step {state.global_step} — LangID accuracy (sample): {acc:.3f}")
        model.train()
 
 
# ── Full classification eval ───────────────────────────────────────────────────
 
@torch.no_grad()
def evaluate_accuracy(model, tokenizer, eval_ds) -> dict:
    """Generate language name for every eval example; report per-language accuracy."""
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
            None
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
 
 
# ── Translation eval ───────────────────────────────────────────────────────────
 
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
 
@torch.inference_mode()
def evaluate_translation(model, tokenizer, split_path: str) -> dict:
    """BLEU + chrF on Q'anjob'al split. Run after Stage 1 as a pre-Stage-2 baseline."""
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
 
 
# ── Logging + plotting ─────────────────────────────────────────────────────────
 
def find_trainer_state(output_dir: str):
    candidates = [
        os.path.join(output_dir, "trainer_state.json"),
        *sorted(Path(output_dir).glob("checkpoint-*/trainer_state.json")),
    ]
    for c in candidates:
        if os.path.exists(str(c)):
            return str(c)
    return None
 
def plot_results(output_dir, tracker, eval_results, state_path):
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
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LangID SFT — Training Diagnostics", fontsize=13, fontweight="bold")
 
    if train_losses:
        axes[0].plot(train_steps, train_losses, color="steelblue", lw=1.5)
    axes[0].set_title("Train Loss"); axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss"); axes[0].grid(True, alpha=0.3)
 
    if eval_losses:
        axes[1].plot(eval_steps, eval_losses, color="darkorange",
                     lw=1.5, ls="--", marker="o", ms=3)
    axes[1].set_title("Eval Loss"); axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss"); axes[1].grid(True, alpha=0.3)
 
    if tracker.accuracies:
        axes[2].plot(tracker.steps, tracker.accuracies,
                     color="seagreen", lw=1.5, marker="o", ms=4)
    axes[2].set_ylim(0, 1); axes[2].set_title("LangID Accuracy (sample proxy)")
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("Accuracy"); axes[2].grid(True, alpha=0.3)
 
    plt.tight_layout()
    p = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")
 
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
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    print(f"CPT model  : {CPT_MODEL}")
    print(f"Output dir : {OUTPUT_DIR}")
 
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()
 
    print("\nBuilding LangID dataset...")
    train_ds, eval_ds = load_lang_snippets(tokenizer)
 
    print("\nBuilding model (fresh LoRA on CPT checkpoint)...")
    model = build_model(tokenizer)
 
    tracker = AccuracyTracker()
    acc_cb  = AccuracyCallback(tracker, eval_ds, tokenizer, sample_size=64)
 
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
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        report_to="tensorboard",
    )
 
    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.add_callback(acc_cb)
 
    print("\nTraining Stage 1 (LangID SFT)...")
    trainer.train(resume_from_checkpoint="langid_sft/checkpoint-4900")
    
 
    # ── Save best adapter ──────────────────────────────────────────────────────
    best_dir = os.path.join(OUTPUT_DIR, "best")
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nAdapter saved: {best_dir}")
    
 
    # ── Classification accuracy eval ───────────────────────────────────────────
    print("\nRunning full classification accuracy eval...")
    cls_results = evaluate_accuracy(trainer.model, tokenizer, eval_ds)
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(cls_results, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/results.json")
 
    # ── Translation eval (baseline before Stage 2) ─────────────────────────────
    print(f"\nRunning translation eval on {QANJOBAL_EVAL_PATH}...")
    print("(Pre-Stage-2 baseline — expect low scores.)")
    trans_results = evaluate_translation(trainer.model, tokenizer, QANJOBAL_EVAL_PATH)
    print(f"  BLEU : {trans_results['bleu']}")
    print(f"  chrF : {trans_results['chrf']}")
    with open(os.path.join(OUTPUT_DIR, "translation_eval.json"), "w") as f:
        json.dump(trans_results, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/translation_eval.json")
 
    # ── Plots ──────────────────────────────────────────────────────────────────
    state_path = find_trainer_state(OUTPUT_DIR)
    plot_results(OUTPUT_DIR, tracker, cls_results, state_path)
 
    print(f"\n{'='*55}")
    print(f"Stage 1 complete.")
    print(f"  Classification accuracy : {cls_results['overall_accuracy']:.3f}")
    print(f"  Translation BLEU        : {trans_results['bleu']}")
    print(f"  Translation chrF        : {trans_results['chrf']}")
    print(f"  Adapter for Stage 2     : {best_dir}")
    print(f"{'='*55}")
    print(f"\nNext: python stage2_translation.py --adapter {best_dir}")
 
 
if __name__ == "__main__":
    main()
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# =========================
# Config
# =========================
ROOT_DIR = "./mayan_data_raw"      
OUT_DIR = "./out_mayan_data_json"     

SPLITS = ["train", "test", "dev", "valid", "val"]
SPANISH_FILE = "data.es"

MODEL_NAME = "Helsinki-NLP/opus-mt-es-en"  # free local es->en
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INCLUDE_LANG_FIELD = False  

import re

LANG_TAG_PATTERN = re.compile(r"^#[^#]+#")

def strip_lang_tag(text: str) -> str:
    return LANG_TAG_PATTERN.sub("", text).strip()
# =========================
# Model
# =========================
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

@torch.inference_mode()
def translate_es_to_en(batch_es: List[str]) -> List[str]:
    inputs = tokenizer(
        batch_es,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)
    outputs = model.generate(**inputs, num_beams=4)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# =========================
# Helpers
# =========================
def read_lines(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def append_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def clear_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.jsonl"):
        p.unlink()

def find_split_dirs(root: Path) -> List[Path]:
    """Return all directories named train/test/dev/valid/val under root."""
    dirs = []
    for split in SPLITS:
        for d in root.rglob(split):
            if d.is_dir() and d.name == split:
                dirs.append(d)
    # de-dup
    uniq = sorted({str(d) for d in dirs})
    return [Path(x) for x in uniq]

def find_mayan_files(split_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all data.<lang> in split_dir except data.es.
    Returns list of (lang_code, path).
    """
    pairs = []
    for p in split_dir.glob("data.*"):
        if p.name == SPANISH_FILE:
            continue
        lang_code = p.suffix.lstrip(".")  # data.acr -> acr
        if lang_code:
            pairs.append((lang_code, p))
    return sorted(pairs, key=lambda x: x[0])

def translate_all(spanish_lines: List[str], desc: str) -> List[str]:
    english_lines: List[str] = []
    for i in tqdm(range(0, len(spanish_lines), BATCH_SIZE), desc=desc, leave=False):
        batch_es = spanish_lines[i:i + BATCH_SIZE]
        batch_en = translate_es_to_en(batch_es)
        english_lines.extend(batch_en)
    return english_lines

# =========================
# Main
# =========================
def build_mayan_english(root_dir: str, out_dir: str) -> None:
    root = Path(root_dir)
    out = Path(out_dir)
    clear_outputs(out)

    split_dirs = find_split_dirs(root)
    if not split_dirs:
        raise RuntimeError(f"No split dirs found under {root}. Expected folders named: {SPLITS}")

    total_pairs_written = 0

    for split_dir in split_dirs:
        spanish_path = split_dir / SPANISH_FILE
        if not spanish_path.exists():
            continue

        mayan_files = find_mayan_files(split_dir)
        if not mayan_files:
            continue

        # Read Spanish once for this split dir
        spanish_lines = [strip_lang_tag(x) for x in read_lines(spanish_path)]

        # Translate Spanish->English once (cached)
        english_lines = translate_all(spanish_lines, desc=f"Translate {split_dir.parent.name}/{split_dir.name}")

        if len(english_lines) != len(spanish_lines):
            print(f"[SKIP] Translation mismatch: {spanish_path}")
            continue

        # For each Mayan language file: append to its lang.jsonl
        for lang_code, mayan_path in mayan_files:
            mayan_lines = read_lines(mayan_path)

            if len(mayan_lines) != len(english_lines):
                print(f"[SKIP] Misaligned: {mayan_path} ({len(mayan_lines)}) vs {spanish_path} ({len(spanish_lines)})")
                continue

            out_path = out / f"{lang_code}.jsonl"

            for i in tqdm(
                range(0, len(english_lines), BATCH_SIZE),
                desc=f"Write {split_dir.parent.name}/{split_dir.name}:{lang_code}",
                leave=False
            ):
                batch_mayan = mayan_lines[i:i + BATCH_SIZE]
                batch_en = english_lines[i:i + BATCH_SIZE]

                records = []
                for mayan, en in zip(batch_mayan, batch_en):
                    rec = {"mayan": mayan, "english": en}
                    if INCLUDE_LANG_FIELD:
                        rec["lang"] = lang_code
                    records.append(rec)

                append_jsonl(out_path, records)

            total_pairs_written += len(english_lines)

    print(f"Done. Total (mayan, english) pairs written (counting each language file): {total_pairs_written}")
    print(f"Outputs in: {out.resolve()}")

if __name__ == "__main__":
    build_mayan_english(ROOT_DIR, OUT_DIR)
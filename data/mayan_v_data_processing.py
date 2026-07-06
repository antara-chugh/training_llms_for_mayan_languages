import os
import json
import glob
import argparse
from pathlib import Path
from datasets import Dataset

# ── Config ────────────────────────────────────────────────────────────────────

MAYAN_JSON_DIR = "out_mayan_data_json"
SPLITS_DIR     = "mayan_data/splits"
TEST_SIZE      = 0.1   # 10% held out per language
SEED           = 42

LANGUAGE_META = {
    "acr": {"name": "Achi",        "branch": "Quichean",           "stage": 1},
    "agu": {"name": "Awakateko",   "branch": "Mamean_Ixilean",     "stage": 1},
    "cac": {"name": "Chuj",        "branch": "Qanjobalan_Chujean", "stage": 2},
    "itz": {"name": "Itza'",       "branch": "Yucatecan",          "stage": 1},
    "ixl": {"name": "Ixil",        "branch": "Mamean_Ixilean",     "stage": 1},
    "kek": {"name": "Q'eqchi'",    "branch": "Qeqchi",             "stage": 1},
    "mam": {"name": "Mam",         "branch": "Mamean",             "stage": 1},
    "poc": {"name": "Poqomam",     "branch": "Poqom",              "stage": 1},
    "poh": {"name": "Poqomchi'",   "branch": "Poqom",              "stage": 1},
    "quc": {"name": "K'iche'",     "branch": "Quichean_Proper",    "stage": 1},
    "qum": {"name": "Sipakapense", "branch": "Quichean_Proper",    "stage": 1},
    "ttc": {"name": "Tektitek",    "branch": "Mamean",             "stage": 1},
    "tzh": {"name": "Tzeltal",     "branch": "Cholan_Tzeltalan",   "stage": 2},
    "tzj": {"name": "Tz'utujil",   "branch": "Quichean_Proper",    "stage": 1},
}

DISTANCE_FROM_TARGET = {
    "cac": 1, "tzh": 2,
    "acr": 3, "quc": 3, "tzj": 3, "qum": 3,
    "poc": 3, "poh": 3, "kek": 3,
    "mam": 3, "ttc": 3, "agu": 3, "ixl": 3,
    "itz": 4,
}

# ── Load + split ──────────────────────────────────────────────────────────────

def load_language_file(path: str, lang_code: str) -> Dataset:
    with open(path) as f:
        content = f.read().strip()
        if content.startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]

    normalized = []
    for r in records:
        mayan   = r.get("mayan") or r.get("Mayan") or r.get(lang_code) or ""
        english = r.get("english") or r.get("English") or ""
        if mayan and english:
            normalized.append({
                "mayan":     str(mayan).strip(),
                "english":   str(english).strip(),
                "lang_code": lang_code,
                "lang_name": LANGUAGE_META.get(lang_code, {}).get("name", lang_code),
                "branch":    LANGUAGE_META.get(lang_code, {}).get("branch", "unknown"),
                "distance":  DISTANCE_FROM_TARGET.get(lang_code, 99),
            })

    return Dataset.from_list(normalized)


def prepare_splits():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    summary = {}

    for json_file in sorted(glob.glob(os.path.join(MAYAN_JSON_DIR, "*.jsonl"))):
        lang_code = Path(json_file).stem
        if lang_code not in LANGUAGE_META:
            print(f"Skipping {lang_code} — not in LANGUAGE_META")
            continue

        lang_name = LANGUAGE_META[lang_code]["name"]
        print(f"\nProcessing {lang_code} ({lang_name})...")

        ds = load_language_file(json_file, lang_code)
        print(f"  Total examples: {len(ds)}")

        # Split
        split = ds.train_test_split(test_size=TEST_SIZE, seed=SEED)
        train_ds = split["train"]
        test_ds  = split["test"]

        # Save
        lang_dir = os.path.join(SPLITS_DIR, lang_code)
        os.makedirs(lang_dir, exist_ok=True)
        train_ds.save_to_disk(os.path.join(lang_dir, "train"))
        test_ds.save_to_disk(os.path.join(lang_dir, "test"))

        summary[lang_code] = {
            "name":     lang_name,
            "branch":   LANGUAGE_META[lang_code]["branch"],
            "stage":    LANGUAGE_META[lang_code]["stage"],
            "distance": DISTANCE_FROM_TARGET.get(lang_code, 99),
            "total":    len(ds),
            "train":    len(train_ds),
            "test":     len(test_ds),
        }
        print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")
        print(f"  Saved to {lang_dir}/")

    # Save summary
    summary_path = os.path.join(SPLITS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Summary saved to {summary_path}")
    print(f"\n{'Code':<6} {'Language':<15} {'Branch':<25} {'Train':>6} {'Test':>6}")
    print("-" * 60)
    for code, info in summary.items():
        print(f"{code:<6} {info['name']:<15} {info['branch']:<25} {info['train']:>6} {info['test']:>6}")

    total_train = sum(v["train"] for v in summary.values())
    total_test  = sum(v["test"]  for v in summary.values())
    print("-" * 60)
    print(f"{'TOTAL':<6} {'':<15} {'':<25} {total_train:>6} {total_test:>6}")

    return summary


if __name__ == "__main__":
    prepare_splits()
    print("\nDone. These splits will be reused across all experiments.")
    print("Do NOT re-run this script — it would regenerate test sets.")
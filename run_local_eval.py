"""
run_local_eval.py — Step 5: Local evaluation (BLEU + chrF++) using a train sample.

Uses a subset of the training set as validation: references = gold translations,
predictions = current translator (dummy baseline or your model) on the same sources.
Writes refs.txt and preds.txt to data/processed/, then runs evaluate.py to print
the competition score √(BLEU × chrF++). Use this to check your model before submitting.

Usage:
  python run_local_eval.py [--n 500]
  (default: first 500 rows of train; use --n to change)
"""

import argparse
import os
import subprocess
import sys

from config import RAW_DIR, PROCESSED_DIR, TRAIN_SOURCE_COL, TRAIN_TARGET_COL
from run_baseline import get_translator, find_file, load_table


def main():
    parser = argparse.ArgumentParser(description="Run local eval: refs from train sample, preds from translator")
    parser.add_argument("--n", type=int, default=500, help="Number of train rows to use (default 500)")
    parser.add_argument("--model", action="store_true", help="Use NLLB from data/models/ if available")
    args = parser.parse_args()

    translate_fn = get_translator(use_model=args.model)

    raw = RAW_DIR
    train_path = find_file(raw, ["train.csv", "train.tsv", "training.csv"])
    if not train_path:
        print("No train file in data/raw. Run download_data.py and extract first.")
        sys.exit(1)

    train = load_table(train_path)
    src_col = TRAIN_SOURCE_COL if TRAIN_SOURCE_COL in train.columns else train.columns[1]
    tgt_col = TRAIN_TARGET_COL if TRAIN_TARGET_COL in train.columns else (train.columns[2] if len(train.columns) > 2 else train.columns[1])

    n = min(args.n, len(train))
    train_sub = train.head(n)
    sources = train_sub[src_col].astype(str).tolist()
    refs = train_sub[tgt_col].astype(str).tolist()
    preds = translate_fn(sources)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    refs_path = os.path.join(PROCESSED_DIR, "refs.txt")
    preds_path = os.path.join(PROCESSED_DIR, "preds_baseline.txt")
    with open(refs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(refs))
    with open(preds_path, "w", encoding="utf-8") as f:
        f.write("\n".join(preds))
    print(f"Wrote {n} lines to {refs_path} and {preds_path}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_py = os.path.join(script_dir, "evaluate.py")
    result = subprocess.run(
        [sys.executable, evaluate_py, "--references", refs_path, "--predictions", preds_path]
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

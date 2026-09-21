"""
explore_data.py — Step 4: Load and inspect competition data.

Reads train/test (and sample submission if present) from data/raw/, using
common Kaggle filenames. Prints shapes, column names, sample rows, and basic
stats (e.g. mean character length for source/target). Use this to confirm
column names before running the rest of the pipeline (config.py, run_baseline.py).

Run after: download_data.py and extracting .zip files in data/raw/.

Usage:
  python explore_data.py
"""

import os
import pandas as pd

from config import RAW_DIR, TRAIN_SOURCE_COL, TRAIN_TARGET_COL, TEST_ID_COL, TEST_SOURCE_COL


def find_file(dirpath, names):
    """
    Return the path of the first existing file in dirpath whose name is in the list.
    Used to support multiple possible filenames (e.g. train.csv vs train.tsv).
    """
    for name in names:
        p = os.path.join(dirpath, name)
        if os.path.isfile(p):
            return p
    return None


def load_table(path, sep=None):
    """
    Load a CSV or TSV into a pandas DataFrame. Infers separator from extension if not given.
    Uses UTF-8 and skips/warns on bad lines.
    """
    if path is None:
        return None
    if sep is None:
        sep = "," if path.endswith(".csv") else "\t"
    return pd.read_csv(path, sep=sep, nrows=None, encoding="utf-8", on_bad_lines="warn")


def main():
    raw = RAW_DIR
    if not os.path.isdir(raw):
        print(f"Directory not found: {raw}. Run download_data.py and extract zips first.")
        return

    # Resolve filenames: Kaggle often uses train.csv, test.csv, sample_submission.csv (or .tsv)
    train_path = find_file(raw, ["train.csv", "train.tsv", "training.csv"])
    test_path = find_file(raw, ["test.csv", "test.tsv"])
    sample_path = find_file(raw, ["sample_submission.csv", "sample_submission.tsv"])

    print("=== Deep Past — Data exploration ===\n")

    # --- Train set ---
    if train_path:
        train = load_table(train_path)
        print(f"Train: {train_path}")
        print(f"  Shape: {train.shape}")
        print(f"  Columns: {list(train.columns)}")
        print(train.head(3).to_string())
        print()
        # Map to config column names or fall back to first/second column
        src = TRAIN_SOURCE_COL in train.columns or (train.columns[0] if len(train.columns) >= 1 else None)
        tgt = TRAIN_TARGET_COL in train.columns or (train.columns[1] if len(train.columns) >= 2 else None)
        if src is not None and tgt is not None:
            sc = TRAIN_SOURCE_COL if TRAIN_SOURCE_COL in train.columns else train.columns[0]
            tc = TRAIN_TARGET_COL if TRAIN_TARGET_COL in train.columns else train.columns[1]
            print(f"  Source col: '{sc}', Target col: '{tc}'")
            print(f"  Source length (chars) — mean: {train[sc].astype(str).str.len().mean():.1f}")
            print(f"  Target length (chars) — mean: {train[tc].astype(str).str.len().mean():.1f}")
    else:
        print("No train file found (train.csv / train.tsv). Check data/raw/.")

    # --- Test set ---
    if test_path:
        test = load_table(test_path)
        print(f"\nTest: {test_path}")
        print(f"  Shape: {test.shape}")
        print(f"  Columns: {list(test.columns)}")
        print(test.head(3).to_string())
    else:
        print("\nNo test file found (test.csv / test.tsv).")

    # --- Sample submission (defines required submission format) ---
    if sample_path:
        sample = load_table(sample_path)
        print(f"\nSample submission: {sample_path}")
        print(f"  Shape: {sample.shape}")
        print(f"  Columns: {list(sample.columns)}")
        print(sample.head(3).to_string())
    else:
        print("\nNo sample_submission file found. Check Evaluation tab for submission format.")

    print("\nDone. Update config.py column names if your files use different names.")


if __name__ == "__main__":
    main()

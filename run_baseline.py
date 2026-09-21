"""
run_baseline.py — Step 6: Produce a baseline submission so the pipeline is end-to-end.

Loads the test set from data/raw/, runs a placeholder “translation” (same phrase per row),
and writes a submission CSV to data/submissions/submission_baseline.csv in the format
required by Kaggle (id + translation column). Upload that file to the competition’s
Submit Predictions page to get a score.

To improve: replace dummy_translate() with a real model (e.g. fine-tuned NLLB, mBART,
or MarianMT on the competition train set). You can keep this script as the submission
builder and only swap the translation function.

Usage:
  python run_baseline.py              # placeholder only
  python run_baseline.py --model     # use NLLB from data/models/ if present
"""

import argparse
import os
import pandas as pd

from config import (
    RAW_DIR, SUBMISSIONS_DIR, MODEL_DIR,
    TEST_ID_COL, TEST_SOURCE_COL,
    SUBMISSION_ID_COL, SUBMISSION_PRED_COL,
)


def find_file(dirpath, names):
    """Return path of first existing file in dirpath with one of the given names."""
    for name in names:
        p = os.path.join(dirpath, name)
        if os.path.isfile(p):
            return p
    return None


def load_table(path, sep=None):
    """Load CSV/TSV into DataFrame; separator inferred from extension if not given."""
    if path is None:
        return None
    sep = sep or ("," if path.endswith(".csv") else "\t")
    return pd.read_csv(path, sep=sep, encoding="utf-8", on_bad_lines="warn")


def dummy_translate(sources):
    """Placeholder: one fixed English phrase per input."""
    return ["Translated text placeholder."] * len(sources)


def _load_nllb_pipeline(model_path, batch_size=8):
    """Load NLLB translation pipeline from a local path. Returns None on failure."""
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        return None
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        return None
    pipe = pipeline(
        "translation",
        model=model_path,
        tokenizer=model_path,
        src_lang="arb_Arab",
        tgt_lang="eng_Latn",
        device=0 if torch.cuda.is_available() else -1,
    )
    return pipe, batch_size


def model_translate(sources, pipe, batch_size=8):
    """Run NLLB pipeline on sources in batches."""
    out = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        result = pipe(batch)
        out.extend([r["translation_text"] for r in result])
    return out


def get_translator(use_model=False):
    """
    Return a callable translate(sources) -> list of strings.
    If use_model is True, try to load NLLB from MODEL_DIR; otherwise or on failure, use dummy.
    """
    if not use_model:
        return dummy_translate
    if os.path.isdir(MODEL_DIR):
        for name in os.listdir(MODEL_DIR):
            path = os.path.join(MODEL_DIR, name)
            if os.path.isfile(os.path.join(path, "config.json")):
                loaded = _load_nllb_pipeline(path)
                if loaded is not None:
                    pipe, batch_size = loaded
                    return lambda srcs, p=pipe, bs=batch_size: model_translate(srcs, p, bs)
    return dummy_translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true", help="Use NLLB from data/models/ if available")
    args = parser.parse_args()

    translate_fn = get_translator(use_model=args.model)
    if translate_fn is dummy_translate and args.model:
        print("NLLB not found in", MODEL_DIR, "- run save_nllb_for_kaggle.py first. Using placeholder.")

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    raw = RAW_DIR
    test_path = find_file(raw, ["test.csv", "test.tsv"])
    if not test_path:
        print("No test.csv/test.tsv in data/raw. Download and extract data first.")
        return

    test = load_table(test_path)
    src_col = TEST_SOURCE_COL if TEST_SOURCE_COL in test.columns else test.columns[1]
    id_col = TEST_ID_COL if TEST_ID_COL in test.columns else test.columns[0]

    sources = test[src_col].astype(str).tolist()
    preds = translate_fn(sources)

    out = pd.DataFrame({SUBMISSION_ID_COL: test[id_col], SUBMISSION_PRED_COL: preds})
    path = os.path.join(SUBMISSIONS_DIR, "submission_baseline.csv")
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {path} with {len(out)} rows.")


if __name__ == "__main__":
    main()

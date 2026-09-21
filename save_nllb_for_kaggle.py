"""
save_nllb_for_kaggle.py — Save NLLB model for use on Kaggle (no internet).

Downloads the NLLB-200 distilled 600M model from Hugging Face and saves it
with save_pretrained() to a folder. Zip that folder and upload it as a
**Kaggle Dataset**, then add that dataset as an **Input** to your submission
notebook so the notebook can load the model without internet.

Run once (with internet). Requires: pip install transformers torch accelerate.

Usage:
  python save_nllb_for_kaggle.py [--out dir]
  (default output: data/models/nllb-200-distilled-600M)
  Then zip the folder and create a new Dataset on Kaggle; add it to the notebook.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Save NLLB model for Kaggle dataset")
    parser.add_argument("--out", default=None, help="Output directory (default: data/models/nllb-200-distilled-600M)")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForSeq2SeqLM
        # Use NLLB tokenizer explicitly to avoid AutoTokenizer mapping issues (AttributeError in some versions)
        try:
            from transformers import NllbTokenizer
        except ImportError:
            try:
                from transformers import NllbTokenizerFast as NllbTokenizer
            except ImportError:
                from transformers import AutoTokenizer as NllbTokenizer
    except ImportError:
        print("Install: pip install transformers torch")
        sys.exit(1)

    model_name = "facebook/nllb-200-distilled-600M"
    out_dir = args.out or os.path.join(
        os.path.dirname(__file__), "data", "models", "nllb-200-distilled-600M"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {model_name} (this may take a few minutes)...")
    tokenizer = NllbTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"Saving to {out_dir}...")
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print("Done. Next: zip this folder and create a Kaggle Dataset; add it as Input to your notebook.")

if __name__ == "__main__":
    main()

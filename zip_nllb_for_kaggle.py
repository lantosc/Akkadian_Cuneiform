"""
zip_nllb_for_kaggle.py — Zip the NLLB model folder for Kaggle Dataset upload.

PowerShell's Compress-Archive can fail on large folders (>2 GB) with "Stream was too long."
This script zips the folder using Python so the archive is complete. Run after save_nllb_for_kaggle.py.

Usage:
  python zip_nllb_for_kaggle.py [--source dir] [--output zip_path]
  Default: source = data/models/nllb-200-distilled-600M, output = data/models/nllb-200-distilled-600M.zip
"""

import argparse
import os
import zipfile

def main():
    parser = argparse.ArgumentParser(description="Zip NLLB model folder for Kaggle")
    parser.add_argument("--source", default=None, help="Model folder to zip")
    parser.add_argument("--output", default=None, help="Output zip path")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    default_src = os.path.join(base, "data", "models", "nllb-200-distilled-600M")
    default_out = os.path.join(base, "data", "models", "nllb-200-distilled-600M.zip")

    src = os.path.abspath(args.source or default_src)
    out = os.path.abspath(args.output or default_out)

    if not os.path.isdir(src):
        print("Source folder not found:", src)
        print("Run save_nllb_for_kaggle.py first.")
        return 1
    if not os.path.isfile(os.path.join(src, "config.json")):
        print("Source folder does not look like a transformers model (no config.json).")
        return 1

    # Remove existing zip so we don't append to a broken one
    if os.path.isfile(out):
        os.remove(out)

    print("Zipping", src, "->", out)
    print("This may take several minutes (~2.3 GB)...")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src):
            for f in files:
                path = os.path.join(root, f)
                arcname = os.path.join(os.path.basename(src), os.path.relpath(path, src))
                zf.write(path, arcname)
    size_gb = os.path.getsize(out) / (1024 ** 3)
    print("Done. Zip size: {:.2f} GB".format(size_gb))
    print("Upload", out, "to Kaggle → Datasets → New Dataset.")
    return 0

if __name__ == "__main__":
    exit(main())

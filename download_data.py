"""
download_data.py — Step 3: Download competition data via the Kaggle API.

Run this after joining the Deep Past Initiative competition and setting up the
Kaggle CLI. It downloads the competition dataset into data/raw/.

Requirements:
  - Kaggle account and competition joined (accept rules).
  - API token: kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<You>\\.kaggle\\).
  - pip install kaggle (see requirements.txt).

Usage:
  python download_data.py

After running, extract any .zip files in data/raw/ (e.g. unzip *.zip on Unix,
or use your system’s unzip tool on Windows).
"""

import os
import shutil
import subprocess
import sys

from config import RAW_DIR, KAGGLE_COMPETITION


def main():
    # Ensure raw data directory exists
    os.makedirs(RAW_DIR, exist_ok=True)

    # Find kaggle CLI (console script from pip; "python -m kaggle" is not supported)
    kaggle_cmd = shutil.which("kaggle")
    if not kaggle_cmd:
        script_dir = os.path.dirname(os.path.abspath(sys.executable))
        kaggle_cmd = os.path.join(script_dir, "kaggle.exe" if os.name == "nt" else "kaggle")
        if not os.path.isfile(kaggle_cmd):
            print("Kaggle CLI not found. Run: pip install kaggle")
            sys.exit(1)

    cmd = [
        kaggle_cmd, "competitions", "download",
        "-c", KAGGLE_COMPETITION,
        "-p", RAW_DIR,
    ]
    print(f"Downloading {KAGGLE_COMPETITION} into {RAW_DIR}...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Download failed. Check:")
        print("  1. You have joined the competition on Kaggle.")
        print("  2. Credentials: kaggle.json in ~/.kaggle/ (Windows: %USERPROFILE%\\.kaggle\\)")
        print("     OR set KAGGLE_API_TOKEN in your shell (see README Step 1).")
        print("  3. pip install kaggle")
        sys.exit(1)

    print("Download finished. Extract any .zip files in data/raw/ (e.g. unzip *.zip).")


if __name__ == "__main__":
    main()

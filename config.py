"""
config.py — Paths and settings for Deep Past Initiative (Kaggle).

This module is the single source of truth for:
  - Project and data directory paths (raw downloads, processed data, submissions).
  - Kaggle competition slug used by the download script.
  - Column names for train/test/submission files (check Kaggle Data/Evaluation tabs).
  - Random seed for reproducible train/val splits.

Adjust DATA_DIR (or PROJECT_ROOT) if you store data elsewhere (e.g. another drive).
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Project root: directory containing this config file (kaggle_deep_past/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories (all under PROJECT_ROOT/data/ by default)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")           # Downloaded zips and extracted CSVs/TSVs
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # Tokenized / cleaned data (optional)
SUBMISSIONS_DIR = os.path.join(DATA_DIR, "submissions")  # Prediction CSVs for Kaggle upload
MODEL_DIR = os.path.join(DATA_DIR, "models")  # Local model cache (e.g. NLLB from save_nllb_for_kaggle.py)

# -----------------------------------------------------------------------------
# Kaggle
# -----------------------------------------------------------------------------
# Competition identifier for the Kaggle API (used in download_data.py)
KAGGLE_COMPETITION = "deep-past-initiative-machine-translation"

# -----------------------------------------------------------------------------
# Column names (train / test / submission)
# -----------------------------------------------------------------------------
# Deep Past competition: source = Akkadian transliteration, target = English translation.
# oare_id (or id in test) is the phrase/sample identifier, not the text to translate.
# Train: transliteration (Akkadian) → translation (English).
TRAIN_SOURCE_COL = "transliteration"
TRAIN_TARGET_COL = "translation"
# Test: id (or oare_id) + transliteration only; we predict translation.
TEST_ID_COL = "id"              # or "oare_id" if test.csv uses that (script falls back to first column)
TEST_SOURCE_COL = "transliteration"
# Submission CSV: id + translation (our predicted English).
SUBMISSION_ID_COL = "id"
SUBMISSION_PRED_COL = "translation"  # Or "target" depending on competition template

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
# Random seed for train/validation splits and any stochastic steps
SEED = 42

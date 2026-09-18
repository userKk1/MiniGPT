# config.py

from pathlib import Path


# =========================
# Project paths
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = BASE_DIR / "results"


# =========================
# Dataset
# =========================

DATASET_NAME = "python_code"

TARGET_CORPUS_SIZE_MB = 10


# =========================
# Reproducibility
# =========================

SEED = 42
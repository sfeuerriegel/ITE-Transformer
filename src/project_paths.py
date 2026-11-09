"""Shared repository paths for notebooks and command-line workflows."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("RWE_DATA_ROOT", PROJECT_ROOT / "DATA")).expanduser().resolve()
SHARED_DATA_DIR = DATA_ROOT / "shared"
COHORT_DATA_DIR = DATA_ROOT / "cohorts"
RESULTS_ROOT = PROJECT_ROOT / "results"

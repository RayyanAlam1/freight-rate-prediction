"""File locations and loaders. All paths are relative to the repository root."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TRAIN_FILE = DATA / "train_test.csv"
VALIDATION_FILE = DATA / "validation.csv"
TEMPLATE_FILE = DATA / "validation_predictions_template.csv"
DECEMBER_FILE = DATA / "december_chart_inputs.csv"

MODEL_FILE = ROOT / "models" / "rate_model.joblib"
PREDICTIONS_FILE = ROOT / "validation_predictions.csv"


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_FILE, parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    return pd.read_csv(VALIDATION_FILE, parse_dates=["date"])

"""Validation scheme built to mimic how validation.csv differs from the training data.

validation.csv differs from train_test.csv in two ways, and each fold reproduces both:

1. Time. It is the two months right after the training data (Nov-Dec), and those are the
   last two months of a quarter, where rates ramp up. So every fold trains on everything
   before a cut-off and validates on what follows. The two "analog" folds cut at the same
   point of a quarter as the real task: train through the quarter's first month, predict
   its other two. October is kept as a third, most-recent check.

2. New cities. 8 of validation's 72 cities (11%) never appear in training. So each fold
   hides 7 of the 64 training cities (11%): every training row touching them is removed,
   which makes the validation rows touching them genuine new-city rows.

Scores are reported on clean labels, per segment (seen / new cities), and blended at the
new-city share actually present in validation.csv.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import SEED
from .features import TARGET

# (name, first validation day, last validation day, mirrors the real task's calendar position)
BLOCKS = (
    ("May-Jun", "2025-05-01", "2025-06-30", True),
    ("Aug-Sep", "2025-08-01", "2025-09-30", True),
    ("Oct", "2025-10-01", "2025-10-31", False),
)
N_NEW_CITIES = 7

# A candidate model: (prepared training rows with labels, prepared rows to predict) -> rates.
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class Fold:
    name: str
    analog: bool
    train: pd.Index
    valid: pd.Index
    new_city: np.ndarray  # per validation row: touches a city hidden from training
    hidden_cities: tuple[str, ...]


def make_folds(frame: pd.DataFrame, seed: int = SEED) -> list[Fold]:
    cities = np.array(sorted(set(frame["pickup"]) | set(frame["delivery"])))
    shuffled = np.random.default_rng(seed).permutation(cities)  # disjoint hidden sets per fold
    folds = []
    for number, (name, start, end, analog) in enumerate(BLOCKS):
        hidden = shuffled[number * N_NEW_CITIES : (number + 1) * N_NEW_CITIES]
        touches_hidden = frame["pickup"].isin(hidden) | frame["delivery"].isin(hidden)
        in_block = frame["date"].between(start, end)
        folds.append(
            Fold(
                name=name,
                analog=analog,
                train=frame.index[(frame["date"] < start) & ~touches_hidden],
                valid=frame.index[in_block],
                new_city=touches_hidden[in_block].to_numpy(),
                hidden_cities=tuple(hidden),
            )
        )
    return folds


def cross_validate(fit_predict: FitPredict, frame: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Out-of-fold predictions for every validation row of every fold."""
    parts = []
    for fold in folds:
        valid = frame.loc[fold.valid]
        predicted = fit_predict(frame.loc[fold.train], valid.drop(columns=[TARGET, "label_ok"]))
        parts.append(
            pd.DataFrame(
                {
                    "fold": fold.name,
                    "analog": fold.analog,
                    "date": valid["date"],
                    "new_city": fold.new_city,
                    "label_ok": valid["label_ok"],
                    "actual": valid[TARGET],
                    "predicted": predicted,
                }
            )
        )
    return pd.concat(parts)


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted - actual
    return {
        "MAE": float(error.abs().mean()),
        "MAPE": float(100 * (error.abs() / actual).mean()),
        "RMSE": float(np.sqrt((error**2).mean())),
    }


def blended_metrics(rows: pd.DataFrame, new_city_share: float) -> dict[str, float]:
    """Clean-label metrics with new-city rows weighted as they occur in validation.csv."""
    rows = rows[rows["label_ok"]]
    seen_rows, new_rows = rows[~rows["new_city"]], rows[rows["new_city"]]
    seen = regression_metrics(seen_rows["actual"], seen_rows["predicted"])
    if new_rows.empty:
        return seen
    new = regression_metrics(new_rows["actual"], new_rows["predicted"])
    share = new_city_share
    return {
        "MAE": (1 - share) * seen["MAE"] + share * new["MAE"],
        "MAPE": (1 - share) * seen["MAPE"] + share * new["MAPE"],
        "RMSE": float(np.sqrt((1 - share) * seen["RMSE"] ** 2 + share * new["RMSE"] ** 2)),
        "MAPE seen": seen["MAPE"],
        "MAPE new": new["MAPE"],
    }


def summarise(oof: pd.DataFrame, new_city_share: float) -> pd.DataFrame:
    """One row per fold plus the headline: the mean of the two analog folds."""
    table = pd.DataFrame({name: blended_metrics(rows, new_city_share) for name, rows in oof.groupby("fold", sort=False)}).T
    analog_folds = oof.loc[oof["analog"], "fold"].unique()
    table.loc["HEADLINE (mean of analog folds)"] = table.loc[analog_folds].mean()
    return table

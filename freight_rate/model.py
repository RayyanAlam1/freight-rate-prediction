"""Rate model: a log-linear backbone plus gradient-boosted residuals.

Why two parts. The validation loads are later in time than anything in training, and tree
models cannot extrapolate: they repeat the last level they saw. So the effects that must
carry forward in time (the upward trend, the quarter-end ramp, the market_index
elasticity, the distance and equipment scaling) are fitted by a small linear model on
log(rate). LightGBM then learns what is left: regional price differences from the
coordinates, non-linear shapes and interactions.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import SEED
from .features import EQUIPMENT, TARGET, label_is_clean

RAMP_DAYS = 30  # rates start climbing about this many days before a quarter ends

RESIDUAL_FEATURES = [
    "distance",
    "straight_line",
    "equipment",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "day_of_week",
    "days_to_quarter_end",
]
GBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    random_state=SEED,
    deterministic=True,  # same trees on every run and machine
    force_row_wise=True,
    verbose=-1,
)


def backbone_design(frame: pd.DataFrame, weight_fill: float, trend: bool = True, ramp: bool = True) -> pd.DataFrame:
    """Columns of the linear model on log(rate). `trend` and `ramp` exist for the ablations."""
    weight = frame["weight"].fillna(weight_fill) / 1e4
    design = pd.DataFrame(
        {
            "log_distance": frame["log_distance"],
            "log_distance_sq": frame["log_distance"] ** 2,
            "log_market_index": frame["log_market_index"],
            "weight": weight,
            "weight_sq": weight**2,
        },
        index=frame.index,
    )
    if trend:
        design["years_since_start"] = frame["years_since_start"]
    # 0 until RAMP_DAYS before a quarter end, then rising linearly to 1 on its last day
    ramp_position = np.clip(RAMP_DAYS - frame["days_to_quarter_end"], 0, None) / RAMP_DAYS
    for equipment in EQUIPMENT:
        is_equipment = (frame["equipment"] == equipment).astype(float)
        if equipment != EQUIPMENT[0]:
            design[f"is_{equipment}"] = is_equipment
        if ramp:
            design[f"ramp_{equipment}"] = ramp_position * is_equipment  # steeper for Flatbed and Reefer
    return design


class RateModel:
    """Fit on prepared training rows (see features.prepare); predict rates in dollars."""

    def __init__(
        self,
        residual_features: list[str] | None = None,
        gbm_params: dict | None = None,
        trend: bool = True,
        ramp: bool = True,
        clean_labels: bool = True,
    ):
        self.residual_features = residual_features or RESIDUAL_FEATURES
        self.gbm_params = gbm_params or GBM_PARAMS
        self.trend = trend
        self.ramp = ramp
        self.clean_labels = clean_labels

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        return backbone_design(frame, self.weight_fill_, self.trend, self.ramp)

    def fit(self, train: pd.DataFrame) -> "RateModel":
        if self.clean_labels:
            train = train[label_is_clean(train)]
        log_rate = np.log(train[TARGET])
        self.weight_fill_ = float(train["weight"].median())
        design = self._design(train)
        self.backbone_ = LinearRegression().fit(design, log_rate)
        self.coefficients_ = pd.Series(self.backbone_.coef_, index=design.columns)
        residual = log_rate - self.backbone_.predict(design)
        self.residual_model_ = lgb.LGBMRegressor(**self.gbm_params).fit(train[self.residual_features], residual)
        return self

    def predict_backbone(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.backbone_.predict(self._design(frame)))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        log_rate = self.backbone_.predict(self._design(frame))
        return np.exp(log_rate + self.residual_model_.predict(frame[self.residual_features]))

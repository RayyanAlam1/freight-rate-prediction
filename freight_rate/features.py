"""Cleaning and feature engineering shared by training, validation and prediction.

Everything here is stateless: a row's features depend only on that row and, for the
market_index fill, on the other rows of the same day. Nothing is learned from labels, so
one code path cleans training rows, validation rows and the December chart rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

TARGET = "posted_rate"
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
TREND_ORIGIN = pd.Timestamp("2025-01-01")

# A label counts as corrupted when it is more than this factor away from a rough
# distance-and-equipment expectation. Clean labels sit within 0.82x-1.25x of it and
# corrupted ones beyond 0.47x / 2.13x, so any threshold inside the gap gives the same split.
OUTLIER_RATIO = 1.6


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    half = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(half))


def days_to_quarter_end(dates: pd.Series) -> pd.Series:
    return ((dates + pd.offsets.QuarterEnd(0)) - dates).dt.days


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Fix the known data-quality issues and add the model features."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])

    # weight: negatives are sign flips (same magnitudes as the positives). Missing values
    # stay missing: LightGBM handles them natively and the linear backbone fills its own.
    out["weight"] = out["weight"].abs()

    # market_index is one daily series plus small per-row noise, so a missing value is
    # recovered almost exactly from the same day's other loads.
    out["market_index"] = out["market_index"].fillna(out.groupby("date")["market_index"].transform("median"))
    out["log_market_index"] = np.log(out["market_index"])

    out["log_distance"] = np.log(out["distance"])
    # distance is floored at 70 miles; the straight-line distance still shows a true short hop.
    out["straight_line"] = haversine_miles(out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"])
    out["equipment"] = pd.Categorical(out["equipment"], categories=EQUIPMENT)

    out["day_of_week"] = out["date"].dt.dayofweek
    out["days_to_quarter_end"] = days_to_quarter_end(out["date"])
    out["years_since_start"] = (out["date"] - TREND_ORIGIN).dt.days / 365.0
    return out


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city, gathered from every frame that carries coordinates."""
    parts = [
        frame[[end, f"{end}_lat", f"{end}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for frame in frames
        for end in ("pickup", "delivery")
    ]
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def complete_chart_inputs(chart: pd.DataFrame, coordinates: pd.DataFrame, daily_market_index: pd.Series) -> pd.DataFrame:
    """Add the columns the December chart inputs lack, so they can go through prepare().

    Coordinates come from the city table. market_index is the given day's median across
    the validation loads: an input known for that day, not a label.
    """
    out = chart.copy()
    out["date"] = pd.to_datetime(out["date"])
    for end in ("pickup", "delivery"):
        out[f"{end}_lat"] = out[end].map(coordinates["lat"])
        out[f"{end}_lon"] = out[end].map(coordinates["lon"])
    out["market_index"] = out["date"].map(daily_market_index)
    added = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index"]
    if out[added].isna().any().any():
        raise ValueError("chart inputs reference a city or date with no coordinates / market_index")
    return out


def label_is_clean(train: pd.DataFrame) -> pd.Series:
    """True where posted_rate is plausible, False for the ~1.4% of corrupted labels.

    Fits log(rate) on distance and equipment only, twice: the second pass refits without
    the rows the first pass rejected.
    """
    log_distance = np.log(train["distance"])
    design = pd.DataFrame(
        {
            "log_distance": log_distance,
            "log_distance_sq": log_distance**2,
            "reefer": (train["equipment"] == "Reefer").astype(float),
            "flatbed": (train["equipment"] == "Flatbed").astype(float),
        }
    )
    target = np.log(train[TARGET])
    clean = pd.Series(True, index=train.index)
    for _ in range(2):
        model = LinearRegression().fit(design[clean], target[clean])
        clean = (target - model.predict(design)).abs() < np.log(OUTLIER_RATIO)
    return clean

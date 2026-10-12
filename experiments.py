"""Experiments behind the validation and modelling decisions.

Run:  python experiments.py        (about 3 minutes; output is saved in reports/experiments.txt)

Part 1  Split comparison: why a random split would have picked the wrong model.
Part 2  Model comparison: baselines, gradient boosting alone, and the final model.
Part 3  Ablations: what each cleaning step and feature group is worth.
Part 4  Light tuning of the gradient-boosting settings.

Every number comes from the forward folds in freight_rate/validation.py unless it says
"random". Scores are on clean labels; the headline is the mean of the two folds that
mirror the real task (see validation.py).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from freight_rate import SEED
from freight_rate.data import load_train, load_validation
from freight_rate.features import TARGET, label_is_clean, prepare
from freight_rate.model import GBM_PARAMS, RESIDUAL_FEATURES, RateModel
from freight_rate.validation import FitPredict, Fold, blended_metrics, cross_validate, make_folds, summarise

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", None, "display.float_format", lambda v: f"{v:,.2f}")

HEADLINE = "HEADLINE (mean of analog folds)"
COORDINATES = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


# --------------------------------------------------------------------------- candidates
def rate_model(extra_features: tuple[str, ...] = (), drop_features: tuple[str, ...] = (), **options) -> FitPredict:
    """The final model (linear backbone + LightGBM residuals), with optional variations."""
    features = [f for f in RESIDUAL_FEATURES if f not in drop_features] + list(extra_features)

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return RateModel(residual_features=features, **options).fit(train).predict(test)

    return fit_predict


def backbone_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return RateModel().fit(train).predict_backbone(test)


def gbm_only(features: list[str], **overrides) -> FitPredict:
    """LightGBM on log(rate) with no linear backbone, trained on clean labels."""
    params = {**GBM_PARAMS, **overrides}

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        train = train[label_is_clean(train)]
        model = lgb.LGBMRegressor(**params).fit(train[features], np.log(train[TARGET]))
        return np.exp(model.predict(test[features]))

    return fit_predict


def median_rate_per_mile(by_lane: bool) -> FitPredict:
    """Naive baseline: typical rate per mile (per equipment, optionally per lane) x distance."""

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        train = train[label_is_clean(train)]
        rate_per_mile = train[TARGET] / train["distance"]
        by_equipment = rate_per_mile.groupby(train["equipment"].astype(str)).median()
        estimate = test["equipment"].astype(str).map(by_equipment)
        if by_lane:
            lane = lambda frame: frame["pickup"] + ">" + frame["delivery"] + "|" + frame["equipment"].astype(str)
            estimate = lane(test).map(rate_per_mile.groupby(lane(train)).median()).fillna(estimate)
        return (estimate * test["distance"]).to_numpy()

    return fit_predict


def with_raw_weight(candidate: FitPredict) -> FitPredict:
    """Undo the weight sign fix, to measure what it is worth."""
    return lambda train, test: candidate(train.assign(weight=train["weight_raw"]), test.assign(weight=test["weight_raw"]))


# --------------------------------------------------------------------------- scoring
def random_split_oof(fit_predict: FitPredict, frame: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold predictions from an ordinary shuffled K-fold, for comparison only."""
    parts = []
    for train_rows, valid_rows in KFold(n_splits, shuffle=True, random_state=SEED).split(frame):
        valid = frame.iloc[valid_rows]
        predicted = fit_predict(frame.iloc[train_rows], valid.drop(columns=[TARGET, "label_ok"]))
        parts.append(pd.DataFrame({"date": valid["date"], "new_city": False, "label_ok": valid["label_ok"],
                                   "actual": valid[TARGET], "predicted": predicted}))
    return pd.concat(parts)


def score(candidates: dict[str, FitPredict], frame: pd.DataFrame, folds: list[Fold], new_city_share: float) -> pd.DataFrame:
    rows = {}
    for name, candidate in candidates.items():
        table = summarise(cross_validate(candidate, frame, folds), new_city_share)
        rows[name] = {
            "MAPE %": table.loc[HEADLINE, "MAPE"],
            "MAE $": table.loc[HEADLINE, "MAE"],
            "RMSE $": table.loc[HEADLINE, "RMSE"],
            "MAPE seen cities": table.loc[HEADLINE, "MAPE seen"],
            "MAPE new cities": table.loc[HEADLINE, "MAPE new"],
            "MAPE Oct check": table.loc["Oct", "MAPE"],
        }
    return pd.DataFrame(rows).T


def part(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


# --------------------------------------------------------------------------- experiments
def split_comparison(frame: pd.DataFrame, folds: list[Fold], new_city_share: float) -> None:
    part("PART 1  The same three models scored by a random split and by the forward folds")
    frame = frame.assign(month=frame["date"].dt.month)
    candidates = {
        "1: final model (no quote_signal)": rate_model(),
        "2: same + quote_signal + month": rate_model(("quote_signal", "month")),
        "3: LightGBM alone, no trend handling": gbm_only(RESIDUAL_FEATURES),
    }
    rows, by_month = {}, {}
    for name, candidate in candidates.items():
        forward_oof = cross_validate(candidate, frame, folds)
        rows[name] = {
            "random 5-fold MAPE %": blended_metrics(random_split_oof(candidate, frame), 0)["MAPE"],
            "forward folds MAPE %": summarise(forward_oof, new_city_share).loc[HEADLINE, "MAPE"],
        }
        by_month[name] = {
            month.strftime("%b"): blended_metrics(month_rows, new_city_share)["MAPE"]
            for month, month_rows in forward_oof.groupby(forward_oof["date"].dt.to_period("M"))
        }
    print(pd.DataFrame(rows).T)
    print("\nForward-fold MAPE % by validation month")
    print("(quote_signal regime: May mirrors, Jun tracks, Aug noise, Sep tracks, Oct mirrors; validation.csv is noise)")
    print(pd.DataFrame(by_month).T)


def model_comparison(frame: pd.DataFrame, folds: list[Fold], new_city_share: float) -> None:
    part("PART 2  Model comparison (forward folds, clean labels)")
    with_time = RESIDUAL_FEATURES + ["years_since_start"]
    candidates = {
        "Baseline: median rate/mile per equipment x distance": median_rate_per_mile(by_lane=False),
        "Baseline: median rate/mile per lane x distance": median_rate_per_mile(by_lane=True),
        "Linear backbone alone": backbone_only,
        "LightGBM alone": gbm_only(RESIDUAL_FEATURES),
        "LightGBM alone + time index": gbm_only(with_time),
        "LightGBM alone, linear leaves + time index": gbm_only(with_time, linear_tree=True),
        "FINAL: linear backbone + LightGBM residuals": rate_model(),
    }
    print(score(candidates, frame, folds, new_city_share))

    oof = cross_validate(rate_model(), frame, folds)
    oof = oof[oof["label_ok"]]
    signed = 100 * (oof["predicted"] / oof["actual"] - 1)
    print("\nFinal model, signed error by validation month (%; + means over-prediction):")
    print(signed.groupby(oof["date"].dt.strftime("%b"), sort=False).mean().round(2).to_dict())


def ablations(frame: pd.DataFrame, folds: list[Fold], new_city_share: float) -> None:
    part("PART 3  Ablations of the final model: change one thing at a time")
    candidates = {
        "FINAL model": rate_model(),
        "data: keep the corrupted labels in training": rate_model(clean_labels=False),
        "data: leave negative weights as they are": with_raw_weight(rate_model()),
        "time: no trend in the backbone": rate_model(trend=False),
        "time: no quarter-end ramp anywhere": rate_model(drop_features=("days_to_quarter_end",), ramp=False),
        "time: no market_index in the residual model": rate_model(drop_features=("market_index",)),
        "place: no coordinates": rate_model(drop_features=tuple(COORDINATES)),
        "place: city names instead of coordinates": rate_model(("pickup_name", "delivery_name"), tuple(COORDINATES)),
        "place: no straight-line distance": rate_model(drop_features=("straight_line",)),
        "signal: add quote_signal": rate_model(("quote_signal",)),
    }
    names = frame.assign(pickup_name=frame["pickup"].astype("category"), delivery_name=frame["delivery"].astype("category"))
    print(score(candidates, names, folds, new_city_share))


def tuning(frame: pd.DataFrame, folds: list[Fold], new_city_share: float) -> None:
    part("PART 4  Light tuning of the residual LightGBM (default: 31 leaves, 600 trees, learning rate 0.05)")
    candidates = {
        f"leaves={leaves:3d}  trees={trees:4d}  learning_rate={rate}": rate_model(
            gbm_params={**GBM_PARAMS, "num_leaves": leaves, "n_estimators": trees, "learning_rate": rate}
        )
        for leaves in (15, 31, 63)
        for trees, rate in ((600, 0.05), (1500, 0.02))
    }
    table = score(candidates, frame, folds, new_city_share)
    print(table)
    print(f"\nspread of headline MAPE across the grid: {table['MAPE %'].min():.3f} to {table['MAPE %'].max():.3f}")


def main() -> None:
    raw = load_train()
    train = prepare(raw)
    train["label_ok"] = label_is_clean(train)
    train["weight_raw"] = raw["weight"]
    validation = load_validation()
    new_cities = (set(validation["pickup"]) | set(validation["delivery"])) - (set(train["pickup"]) | set(train["delivery"]))
    new_city_share = float((validation["pickup"].isin(new_cities) | validation["delivery"].isin(new_cities)).mean())
    print(f"training rows: {len(train):,} ({(~train['label_ok']).sum()} corrupted labels excluded from fitting and scoring)")
    print(f"share of validation.csv rows touching a city absent from training: {new_city_share:.1%}")

    folds = make_folds(train)
    print("\nForward folds (each trains on everything before its validation block):")
    for fold in folds:
        print(f"  {fold.name:8s} train rows {len(fold.train):6,d} | validation rows {len(fold.valid):6,d} "
              f"({fold.new_city.mean():.0%} touch a hidden city) | hidden: {', '.join(fold.hidden_cities)}")

    split_comparison(train, folds, new_city_share)
    model_comparison(train, folds, new_city_share)
    ablations(train, folds, new_city_share)
    tuning(train, folds, new_city_share)

    final = RateModel().fit(train)
    part("Final model fitted on all training data: backbone coefficients (log scale)")
    print(final.coefficients_.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()

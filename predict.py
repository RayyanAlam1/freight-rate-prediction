"""Write the two prediction files the scorer expects.

Run:  python predict.py        (after train.py)

Outputs
  validation_predictions.csv        load_id,predicted_rate for all 12,000 validation loads
  data/december_chart_inputs.csv    the provided file with its predicted_rate column filled
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from freight_rate.data import (
    DECEMBER_FILE,
    MODEL_FILE,
    PREDICTIONS_FILE,
    ROOT,
    TEMPLATE_FILE,
    load_train,
    load_validation,
)
from freight_rate.features import TARGET, city_coordinates, complete_chart_inputs, label_is_clean, prepare
from freight_rate.model import RateModel


def predict_validation(model: RateModel, validation: pd.DataFrame) -> pd.DataFrame:
    """Fill the template, matching on load_id so the template's row order is kept."""
    predicted = pd.Series(model.predict(prepare(validation)), index=validation["load_id"])
    template = pd.read_csv(TEMPLATE_FILE)
    template["predicted_rate"] = template["load_id"].map(predicted).round(2)
    return template[["load_id", "predicted_rate"]]


def predict_december(model: RateModel, train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Fill predicted_rate in the chart inputs, leaving every other column as provided."""
    chart = pd.read_csv(DECEMBER_FILE)
    inputs = complete_chart_inputs(
        chart.drop(columns="predicted_rate"),
        coordinates=city_coordinates(train, validation),
        daily_market_index=validation.groupby("date")["market_index"].median(),
    )
    chart["predicted_rate"] = model.predict(prepare(inputs)).round(2)
    return chart


def check_validation_predictions(predictions: pd.DataFrame, validation: pd.DataFrame, train: pd.DataFrame) -> None:
    """Fail loudly if the submission file is malformed or implausible."""
    template = pd.read_csv(TEMPLATE_FILE)
    rate = predictions["predicted_rate"]
    checks = {
        "exactly the columns load_id,predicted_rate": list(predictions.columns) == ["load_id", "predicted_rate"],
        "12,000 rows": len(predictions) == 12_000,
        "no missing values": not predictions.isna().any().any(),
        "no duplicate load_id": not predictions["load_id"].duplicated().any(),
        "same ids in the same order as the template": predictions["load_id"].tolist() == template["load_id"].tolist(),
        "every validation load is covered": set(predictions["load_id"]) == set(validation["load_id"]),
        "all rates numeric, finite and positive": bool(np.isfinite(rate).all() and (rate > 0).all()),
    }

    # Plausibility: predicted rate per mile should stay close to what clean training loads
    # show. 10% of headroom, because validation is later (rates trend up) and can combine
    # extremes training never paired, e.g. a 70-mile Reefer load two days before a quarter end.
    clean = train[label_is_clean(train)]
    seen = clean[TARGET] / clean["distance"]
    per_mile = rate.to_numpy() / validation.set_index("load_id").loc[predictions["load_id"], "distance"].to_numpy()
    low, high = 0.9 * seen.min(), 1.1 * seen.max()
    checks[f"rate per mile within 10% of the training range ({low:.2f}-{high:.2f})"] = bool(
        per_mile.min() >= low and per_mile.max() <= high
    )
    beyond = int(((per_mile < seen.min()) | (per_mile > seen.max())).sum())

    for name, passed in checks.items():
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit("validation_predictions.csv failed its sanity checks")
    quantiles = [0, 0.01, 0.5, 0.99, 1]
    print(f"  predicted rate       min/1%/median/99%/max: {rate.quantile(quantiles).round(0).tolist()}")
    print(f"  training rate, clean min/1%/median/99%/max: {clean[TARGET].quantile(quantiles).round(0).tolist()}")
    print(f"  predicted rate/mile  min/median/max: {per_mile.min():.2f} / {np.median(per_mile):.2f} / {per_mile.max():.2f}"
          f"   (training, clean: {seen.min():.2f} / {seen.median():.2f} / {seen.max():.2f})")
    print(f"  loads predicted beyond the exact training rate/mile range: {beyond}")


def main() -> None:
    # joblib files can run code when loaded: only ever load the one train.py wrote locally.
    model: RateModel = joblib.load(MODEL_FILE)
    train, validation = load_train(), load_validation()

    predictions = predict_validation(model, validation)
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    print(f"wrote {PREDICTIONS_FILE.relative_to(ROOT)}")
    check_validation_predictions(pd.read_csv(PREDICTIONS_FILE), validation, train)

    december = predict_december(model, train, validation)
    december.to_csv(DECEMBER_FILE, index=False)
    rate = december["predicted_rate"]
    print(f"wrote {DECEMBER_FILE.relative_to(ROOT)}: 31 days, ${rate.min():,.2f} to ${rate.max():,.2f}, "
          f"first week ${rate[:7].mean():,.2f} -> last week ${rate[-7:].mean():,.2f}")


if __name__ == "__main__":
    main()

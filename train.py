"""Fit the final model on all labelled data and save it.

Run:  python train.py
"""
from __future__ import annotations

import joblib
import numpy as np

from freight_rate.data import MODEL_FILE, ROOT, load_train
from freight_rate.features import TARGET, label_is_clean, prepare
from freight_rate.model import RateModel


def main() -> RateModel:
    train = prepare(load_train())
    clean = label_is_clean(train)
    model = RateModel().fit(train)  # drops the corrupted labels itself

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    fitted = model.predict(train[clean])
    in_sample = 100 * np.mean(np.abs(fitted / train.loc[clean, TARGET] - 1))
    print(f"trained on {clean.sum():,} loads ({(~clean).sum()} corrupted labels dropped), "
          f"{train['date'].min().date()} to {train['date'].max().date()}")
    print(f"in-sample MAPE {in_sample:.2f}% (a fit check only; see experiments.py for the validated 1.42%)")
    print("backbone coefficients on log(rate):")
    print(model.coefficients_.to_string(float_format=lambda value: f"{value:8.4f}"))
    print(f"saved {MODEL_FILE.relative_to(ROOT)}")
    return model


if __name__ == "__main__":
    main()

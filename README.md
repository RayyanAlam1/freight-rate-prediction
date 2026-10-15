# Freight rate prediction

The task is to predict `posted_rate` for the Nov-Dec 2025 loads in `data/validation.csv` using the Jan-Oct loads in `data/train_test.csv`.

## Files

- `analysis.ipynb` - EDA, validation setup and model comparison
- `model.py` - the final model. Trains on all the data and writes the predictions
- `validation_predictions.csv` - predictions for the 12,000 validation loads
- `data/december_chart_inputs.csv` - December inputs with `predicted_rate` filled in
- `scorer_results/candidate_december.png` - chart made by `score.py`

## How to run

```
pip install -r requirements.txt
python model.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

I used Python 3.12. `model.py` takes a few seconds, the notebook runs top to bottom in about a minute.

## Approach

The script implements a two-stage hybrid regression model

**Data issues**

- About 1.4% of the rates are 3-5x too high or too low (rate per mile outside 1.5-4). I drop them from training.
- Negative weights are sign flips, so `abs()`. Missing `market_index` is filled with the same day's median.
- `quote_signal` follows the rate in some months, is inverted in others and is noise in August. In validation it behaves like August, so I don't use it.
- 8 cities in validation are not in the training data, so the model uses lat/lon and not city names.

**Validation**

Time based. Train on everything before a date and test on the next block (May-Jun, Aug-Sep, Oct). In each fold I also remove 7 cities from the training part, to copy the unseen cities in validation.

A random split is misleading here. It says adding `quote_signal` helps (1.15% vs 1.31% MAPE), the time split says it hurts (1.63% vs 1.47%).

**Model**

Linear regression on log(rate) for the things that have to extrapolate in time (trend, quarter-end ramp, market index, plus distance, weight and equipment), then LightGBM on the residuals for location and anything non-linear.

| model | MAPE | MAE | RMSE |
|---|---|---|---|
| median rate per mile x distance | 8.57% | 177 | 250 |
| linear only | 2.60% | 55 | 74 |
| LightGBM only | 3.58% | 85 | 110 |
| LightGBM + time index | 2.02% | 47 | 68 |
| linear + LightGBM on residuals | 1.47% | 35 | 51 |

Average over the three folds, scored on clean labels.

## Notes

- The task doesn't say which metric to use, so I looked at MAPE, MAE and RMSE. The final model is best on all three.
- The December chart file has no coordinates or `market_index`. Coordinates come from the city, `market_index` is that day's median in `validation.csv`.
- Holiday effects can't be learned from Jan-Oct data, and the fold results show a bias of up to about 1% in single months, so something similar is possible in Nov-Dec.

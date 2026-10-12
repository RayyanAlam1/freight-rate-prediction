# Freight Rate Prediction

Solution to the Machine Learning Engineer take-home assessment: predict the posted rate of
freight loads for November and December 2025 from loads observed January to October 2025.

**Model:** a small linear model on log(rate) that carries the time effects forward (trend,
quarter-end ramp, market elasticity), plus LightGBM on its residuals for regional and
non-linear effects.

**Validated error:** 1.42% MAPE, $33.49 MAE, $48.54 RMSE on forward-in-time folds with
hidden cities. The next best alternative, LightGBM alone, scores 1.94%.

## Run it

Tested with Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

`run.py` is the single entry point. It takes a few seconds and runs three steps:

| Step | Script | Output |
|---|---|---|
| Train on all labelled data | `train.py` | `models/rate_model.joblib` |
| Predict and sanity-check | `predict.py` | `validation_predictions.csv`, and `predicted_rate` filled in `data/december_chart_inputs.csv` |
| Provided scorer (unmodified) | `score.py` | `scorer_results/candidate_december.png` |

Each script also runs on its own. The scorer command from the assessment is:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

To reproduce the analysis behind the decisions:

```bash
python eda.py            # data audit; writes figures to reports/eda/ (about 1 minute)
python experiments.py    # split comparison, model comparison, ablations, tuning (about 1 minute)
```

Their saved outputs are in `reports/eda/audit_output.txt` and `reports/experiments.txt`.

## What the data looks like

| Finding | Evidence | What the solution does |
|---|---|---|
| 1.4% of labels are corrupted | 677 rates are 0.17x-0.44x or 2.2x-5.4x of what distance and equipment imply; clean ones are all within 0.82x-1.25x | Dropped from training; scores are on clean labels |
| `quote_signal` changes meaning by month | It tracks the rate in Jan-Mar, Jun, Sep (r = +0.997), mirrors it in Apr-May, Jul, Oct (r = -0.99), and is noise in Aug. In validation it is noise on all 61 days | Not used |
| Rates ramp into every quarter end | +2.4% (Dry Van), +5.1% (Reefer), +8.0% (Flatbed) by the last day, then a reset. December is a quarter end | Ramp feature per equipment type |
| Rates trend up about 7.5% a year | Trees cannot extrapolate this into Nov-Dec | Linear trend in the backbone |
| 8 validation cities never appear in training | 12.1% of validation loads touch one; city price effects cluster by region | Coordinates instead of city names |
| Negative and missing `weight`, missing `market_index` | 0.6% to 0.8% of training rows each, and about twice that or more in validation | `abs()` for the sign flips; missing weight is left for LightGBM and median-filled in the backbone; `market_index` is filled from the same day |

## How it was validated

`validation.csv` differs from the training data in two ways, and every fold reproduces both
(`freight_rate/validation.py`):

- **Time.** The real task is "train through the first month of a quarter, predict its other
  two". Two folds cut at that same point (predict May-Jun from Jan-Apr, and Aug-Sep from
  Jan-Jul); October is a third, most-recent check. Nothing is ever predicted from later data.
- **New cities.** Each fold hides 7 of the 64 training cities, matching the 8 of 72 that are
  new in validation, and scores new-city loads separately.

A random 5-fold split ranks the models wrongly here: it prefers adding `quote_signal`
(1.15% vs 1.31%), while the forward folds show that making things worse (1.67% vs 1.42%).

| Model | MAPE | MAE | RMSE |
|---|---|---|---|
| Median rate per mile by equipment x distance | 8.67% | $174.16 | $239.32 |
| Median rate per mile by lane x distance | 5.81% | $133.89 | $197.54 |
| Linear backbone alone | 2.59% | $55.34 | $74.39 |
| LightGBM alone | 2.83% | $66.03 | $89.99 |
| LightGBM alone + time index | 1.94% | $45.08 | $66.23 |
| **Linear backbone + LightGBM residuals (final)** | **1.42%** | **$33.49** | **$48.54** |

## Repository layout

```
run.py                  single entry point: train, predict, score
train.py                fit the final model on all labelled data
predict.py              write both prediction files and sanity-check them
score.py                provided scorer, unmodified
eda.py                  data audit and figures
experiments.py          every number behind the modelling decisions
freight_rate/
    data.py             file locations and loaders
    features.py         cleaning and feature engineering (one code path for all files)
    validation.py       forward folds with hidden cities, metrics
    model.py            RateModel: linear backbone + LightGBM residuals
data/                   provided files; december_chart_inputs.csv is filled in place
reports/                saved experiment output and EDA figures
scorer_results/         December chart produced by score.py
```

## Assumptions and limits

- The assessment does not name its metric, so MAE, MAPE and RMSE are all reported and the
  model was chosen because it leads on all three.
- The December chart inputs have no `market_index` or coordinates. Coordinates are looked up
  by city; `market_index` is that day's median across the validation loads. These are inputs
  for the day, not labels, and nothing is fitted on them.
- Forward-fold errors show a level offset of up to about 0.9% in individual months, so a
  similar offset is possible in Nov-Dec. Holiday effects cannot be learned from Jan-Oct data.

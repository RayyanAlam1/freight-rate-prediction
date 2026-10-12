"""Exploratory data audit for the freight-rate assessment.

Run:  python eda.py

Prints the audit tables behind the data-quality, feature and validation decisions and
writes five figures to reports/eda/. Nothing in this file feeds the model: it fits its own,
richer regression (with a price effect per city) purely to make patterns visible.

validation.csv is only ever used here for label-free checks (it has no labels): feature
distributions, data-quality rates and whether quote_signal still relates to the other
features the way it does in the training period.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from freight_rate import SEED
from freight_rate.data import DATA, ROOT
from freight_rate.features import days_to_quarter_end, haversine_miles

FIGURES = ROOT / "reports" / "eda"

EPOCH = pd.Timestamp("2025-01-01")
# A label is called corrupted when it is off from the structural expectation by more than
# this factor. Clean rows fall inside 0.90x-1.13x and corrupted ones outside 0.44x-2.24x,
# so any threshold in the empty gap between gives the same split.
OUTLIER_RATIO = 1.6

# Chart palette: colour-blind-safe categorical slots plus neutral chrome.
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", lambda v: f"{v:,.4f}")


# --------------------------------------------------------------------------- helpers
def section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def new_figure(nrows: int = 1, ncols: int = 1, figsize=(10, 4.2), **kwargs):
    figure, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, **kwargs)
    for axis in np.atleast_1d(axes).ravel():
        axis.set_facecolor(SURFACE)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(AXIS)
        axis.tick_params(colors=MUTED, labelsize=9)
    return figure, axes


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"[figure] reports/eda/{name}")


def daily_corr(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    return frame.groupby("date")[[left, right]].corr().xs(left, level=1)[right]


# --------------------------------------------------------------------------- structural model
def structural_design(frame: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    """Design matrix of a deliberately simple log-linear rate model (no quote_signal)."""
    weight = frame["weight"].abs()
    weight = weight.fillna(weight.median()) / 1e4
    market = frame["market_index"].fillna(frame.groupby("date")["market_index"].transform("median"))
    log_distance = np.log(frame["distance"])
    core = pd.DataFrame(
        {
            "log_distance": log_distance,
            "log_distance_sq": log_distance**2,
            "log_market_index": np.log(market),
            "weight": weight,
            "weight_sq": weight**2,
            "reefer": (frame["equipment"] == "Reefer").astype(float),
            "flatbed": (frame["equipment"] == "Flatbed").astype(float),
            "years": (frame["date"] - EPOCH).dt.days / 365.0,
        }
    )
    # One symmetric effect per city (pickup and delivery effects are ~identical, r=0.99).
    # Centring makes each coefficient a deviation from the average city, so a city the
    # model has never seen is treated as average rather than as the reference city.
    pickup = pd.get_dummies(frame["pickup"]).reindex(columns=cities, fill_value=0)
    delivery = pd.get_dummies(frame["delivery"]).reindex(columns=cities, fill_value=0)
    city = (pickup + delivery).astype(float).sub(2 / len(cities)).add_prefix("city_")
    return pd.concat([core, city], axis=1)


def add_structural_expectation(train: pd.DataFrame, val: pd.DataFrame) -> pd.Series:
    """Add expected rate columns to both frames; return the fitted coefficients."""
    cities = sorted(set(train["pickup"]) | set(train["delivery"]))
    target = np.log(train["posted_rate"])
    keep = pd.Series(True, index=train.index)
    for _ in range(2):  # second pass refits without the corrupted labels found by the first
        model = LinearRegression().fit(structural_design(train[keep], cities), target[keep])
        log_ratio = target - model.predict(structural_design(train, cities))
        keep = log_ratio.abs() < np.log(OUTLIER_RATIO)

    train["expected_rate"] = np.exp(target - log_ratio)
    train["ratio"] = np.exp(log_ratio)
    train["clean"] = keep
    val["expected_rate"] = np.exp(model.predict(structural_design(val, cities)))
    for frame in (train, val):
        frame["expected_rpm"] = frame["expected_rate"] / frame["distance"]
    return pd.Series(model.coef_, index=structural_design(train.head(1), cities).columns)


# --------------------------------------------------------------------------- audits
def overview(train: pd.DataFrame, val: pd.DataFrame, template: pd.DataFrame, december: pd.DataFrame) -> None:
    section("1. Schema, ranges, missing values, duplicates")
    print(f"train-test : {train.shape}  {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"validation : {val.shape}  {val['date'].min().date()} to {val['date'].max().date()}")
    print(f"december   : {december.shape}  columns={list(december.columns)}")
    print("columns missing from the December chart inputs:",
          sorted(set(val.columns) - set(december.columns) - {"load_id"}))
    print("\nmissing values (count):")
    print(pd.DataFrame({"train": train.isna().sum(), "validation": val.isna().sum()}).query("train > 0 or validation > 0"))
    features = [c for c in val.columns if c != "load_id"]
    print(f"\nduplicate load_id      : train {train['load_id'].duplicated().sum()}, validation {val['load_id'].duplicated().sum()}")
    print(f"duplicate feature rows : train {train.duplicated(subset=features).sum()}, validation {val.duplicated(subset=features).sum()}")
    print("template ids match validation.csv in order:", bool((template["load_id"].values == val["load_id"].values).all()))
    print("\nnumeric summary (train):")
    print(train.describe(percentiles=[0.01, 0.5, 0.99]).T[["min", "1%", "50%", "99%", "max"]])


def weight_audit(train: pd.DataFrame, val: pd.DataFrame) -> None:
    section("2. weight: negative, missing and capped values")
    rows = []
    for name, frame in (("train", train), ("validation", val)):
        weight = frame["weight"]
        rows.append(
            {
                "set": name,
                "negative %": 100 * (weight < 0).mean(),
                "missing %": 100 * weight.isna().mean(),
                "min positive": weight[weight > 0].min(),
                "max": weight.max(),
                "at 47,500 cap %": 100 * (weight.abs() == 47_500).mean(),
                "median positive": weight[weight > 0].median(),
                "median |negative|": weight[weight < 0].abs().median(),
            }
        )
    print(pd.DataFrame(rows).set_index("set").T)
    print("-> negatives have the same magnitude distribution as positives: a sign flip, fixed with abs().")
    print("-> market_index missing %: train {:.2f}, validation {:.2f}".format(
        100 * train["market_index"].isna().mean(), 100 * val["market_index"].isna().mean()))


def geography_audit(train: pd.DataFrame, val: pd.DataFrame) -> set[str]:
    section("3. Cities, lanes and distance")
    both = pd.concat([train, val])
    coords = pd.concat(
        [
            both[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
            both[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        ]
    ).drop_duplicates()
    print(f"cities: {coords['city'].nunique()}, distinct (city, lat, lon): {len(coords)} -> one coordinate per city")

    new_cities = (set(val["pickup"]) | set(val["delivery"])) - (set(train["pickup"]) | set(train["delivery"]))
    touches_new = val["pickup"].isin(new_cities) | val["delivery"].isin(new_cities)
    print(f"cities in validation but never in train ({len(new_cities)}): {sorted(new_cities)}")
    print(f"validation rows touching a new city: {touches_new.sum():,} ({touches_new.mean():.1%})")

    train_lanes = set(train["pickup"] + ">" + train["delivery"])
    unseen_lane = ~(val["pickup"] + ">" + val["delivery"]).isin(train_lanes)
    print(f"validation rows on a lane never seen in train: {unseen_lane.sum():,} ({unseen_lane.mean():.1%})")

    for name, frame in (("train", train), ("validation", val)):
        straight = haversine_miles(frame["pickup_lat"], frame["pickup_lon"], frame["delivery_lat"], frame["delivery_lon"])
        ratio = frame["distance"] / straight
        print(f"{name:10s} distance / straight-line: median {ratio.median():.3f}, 1%-99% "
              f"{ratio.quantile(0.01):.3f}-{ratio.quantile(0.99):.3f}, max {ratio.max():.1f}; "
              f"rows at the 70-mile floor: {(frame['distance'] == 70).sum()}")
    lane_distance = train.groupby(train["pickup"] + ">" + train["delivery"])["distance"]
    spread = ((lane_distance.max() - lane_distance.min()) / lane_distance.median()).median()
    print(f"distance is recorded per load, not per lane: median within-lane spread {spread:.1%}")
    return new_cities


def label_audit(train: pd.DataFrame) -> None:
    section("4. Target: posted_rate and corrupted labels")
    rate_per_mile = train["posted_rate"] / train["distance"]
    print("posted_rate quantiles :", train["posted_rate"].quantile([0, 0.01, 0.5, 0.99, 0.999, 1]).round(0).tolist())
    print("rate/mile quantiles   :", rate_per_mile.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).round(2).tolist())
    clean = train[train["clean"]]
    error = np.log(clean["ratio"])
    print(f"structural model (distance, equipment, weight, market_index, city, linear trend) on clean rows: "
          f"residual std {error.std():.4f} log-points, median abs error {100 * np.median(np.abs(np.expm1(error))):.2f}%")

    low, high = train["ratio"] < 1 / OUTLIER_RATIO, train["ratio"] > OUTLIER_RATIO
    print(f"\nactual / expected rate: clean rows span {clean['ratio'].min():.2f}x-{clean['ratio'].max():.2f}x")
    print(f"  too low  : {low.sum():4d} rows ({low.mean():.2%}), {train.loc[low, 'ratio'].min():.2f}x-{train.loc[low, 'ratio'].max():.2f}x")
    print(f"  too high : {high.sum():4d} rows ({high.mean():.2%}), {train.loc[high, 'ratio'].min():.2f}x-{train.loc[high, 'ratio'].max():.2f}x")
    print("  corrupted labels per month:", (~train["clean"]).groupby(train["date"].dt.month).sum().to_dict())

    figure, axis = new_figure(figsize=(10, 4))
    bins = np.logspace(np.log10(0.15), np.log10(6.5), 140)
    axis.hist(train["ratio"], bins=bins, color=BLUE, linewidth=0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks([0.2, 0.3, 0.5, 1, 2, 3, 5], ["0.2x", "0.3x", "0.5x", "1x", "2x", "3x", "5x"])
    axis.minorticks_off()
    axis.set_xlabel("Posted rate / rate expected from distance, equipment, weight, market and city", color=INK_2)
    axis.set_ylabel("Loads (log scale)", color=INK_2)
    axis.set_title("1.4% of training labels are corrupted, and they separate cleanly", loc="left", fontsize=13,
                   fontweight="bold", color=INK)
    top = axis.get_ylim()[1]
    notes = (
        (0.29, "center", f"{low.sum()} loads\n{train.loc[low, 'ratio'].min():.2f}x-{train.loc[low, 'ratio'].max():.2f}x"),
        (1.2, "left", f"{len(clean):,} clean loads\n{clean['ratio'].min():.2f}x-{clean['ratio'].max():.2f}x"),
        (3.4, "center", f"{high.sum()} loads\n{train.loc[high, 'ratio'].min():.2f}x-{train.loc[high, 'ratio'].max():.2f}x"),
    )
    for x, align, text in notes:
        axis.text(x, top * 0.5, text, ha=align, va="top", fontsize=9.5, color=INK_2)
    save(figure, "02_label_outliers.png")


def quote_signal_audit(train: pd.DataFrame, val: pd.DataFrame) -> None:
    section("5. quote_signal: three regimes")
    clean = train[train["clean"]].assign(rate_per_mile=lambda d: d["posted_rate"] / d["distance"])
    truth = daily_corr(clean, "quote_signal", "rate_per_mile")
    regime = pd.cut(truth, [-1.01, -0.5, 0.5, 1.01], labels=["mirrors the rate", "noise", "tracks the rate"]).astype(str)
    runs = (regime != regime.shift()).cumsum()
    table = (
        pd.DataFrame({"date": truth.index, "regime": regime.values, "corr": truth.values, "run": runs.values})
        .groupby("run")
        .agg(start=("date", "min"), end=("date", "max"), days=("date", "size"), regime=("regime", "first"),
             corr_with_actual_rate_per_mile=("corr", "median"))
    )
    print("Train, using labels: daily correlation between quote_signal and the actual rate per mile")
    print(table.to_string(index=False))
    train_regime = train["date"].map(regime)
    print("\nshare of training rows per regime:", train_regime.value_counts(normalize=True).round(3).to_dict())

    tracks = clean[clean["date"].map(regime) == "tracks the rate"]
    mirrors = clean[clean["date"].map(regime) == "mirrors the rate"]
    print(f"tracks regime : quote_signal / rate-per-mile = {(tracks['quote_signal'] / tracks['rate_per_mile']).median():.4f} "
          f"+/- {(tracks['quote_signal'] / tracks['rate_per_mile']).std():.4f}")
    slope, intercept = np.polyfit(mirrors["rate_per_mile"], mirrors["quote_signal"], 1)
    print(f"mirrors regime: quote_signal = {intercept:.3f} {slope:+.3f} x rate-per-mile (a reflection around ~2.07 $/mile)")

    print("\nLabel-free view (works on validation): how quote_signal relates to the other features")
    rows = []
    groups = [(name, train[train_regime == name]) for name in ("tracks the rate", "mirrors the rate", "noise")]
    groups += [("VALIDATION Nov", val[val["date"].dt.month == 11]), ("VALIDATION Dec", val[val["date"].dt.month == 12])]
    for name, frame in groups:
        by_equipment = frame.groupby("equipment")["quote_signal"].median()
        rows.append(
            {
                "rows": len(frame),
                "corr with expected rate/mile": frame["quote_signal"].corr(frame["expected_rpm"]),
                "corr with log(distance)": frame["quote_signal"].corr(np.log(frame["distance"])),
                "median Dry Van": by_equipment["Dry Van"],
                "median Flatbed": by_equipment["Flatbed"],
                "median Reefer": by_equipment["Reefer"],
                "mean": frame["quote_signal"].mean(),
                "std": frame["quote_signal"].std(),
            }
        )
    print(pd.DataFrame(rows, index=[name for name, _ in groups]).T)
    val_daily = daily_corr(val, "quote_signal", "expected_rpm")
    print(f"validation daily correlation with expected rate/mile: min {val_daily.min():.2f}, "
          f"median {val_daily.median():.2f}, max {val_daily.max():.2f} over {len(val_daily)} days")
    print("-> in validation quote_signal behaves exactly like the August 'noise' regime.")

    train_daily = daily_corr(train, "quote_signal", "expected_rpm")
    figure, axis = new_figure(figsize=(10.5, 4.4))
    axis.axhline(0, color=AXIS, linewidth=1)
    axis.plot(train_daily.index, train_daily.values, color=BLUE, linewidth=2, label="Train (Jan-Oct)")
    axis.plot(val_daily.index, val_daily.values, color=ORANGE, linewidth=2, label="Validation (Nov-Dec)")
    axis.set_ylim(-1.45, 1.45)
    axis.set_yticks([-1, -0.5, 0, 0.5, 1])
    axis.set_ylabel("Daily correlation of quote_signal with\nthe expected rate per mile", color=INK_2)
    axis.set_title("quote_signal tracks the rate, mirrors it, or is pure noise, depending on the month",
                   loc="left", fontsize=13, fontweight="bold", color=INK)
    height = {"tracks the rate": 1.2, "mirrors the rate": -1.25, "noise": 0.35}
    for _, run in table.iterrows():
        if run["days"] >= 20:
            middle = run["start"] + (run["end"] - run["start"]) / 2
            axis.text(middle, height[run["regime"]], run["regime"].capitalize(), ha="center", fontsize=9.5, color=INK_2)
    axis.text(val_daily.index[len(val_daily) // 2], 0.35, "Noise on all 61 days", ha="center", fontsize=9.5, color=INK_2)
    axis.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK_2)
    save(figure, "01_quote_signal_regimes.png")


def time_audit(train: pd.DataFrame, val: pd.DataFrame, coefficients: pd.Series) -> None:
    section("6. Time structure: market_index, trend and the quarter-end ramp")
    market_train = train.groupby("date")["market_index"].median()
    market_val = val.groupby("date")["market_index"].median()
    within_day = (train["market_index"] - train["date"].map(market_train)).std()
    print(f"market_index is a daily series: within-day std {within_day:.3f} vs between-day std {market_train.std():.3f}")
    print("  -> missing values can be filled with that day's median")
    print("  by weekday (Mon..Sun), train:", train.groupby(train["date"].dt.dayofweek)["market_index"].median().round(3).tolist())
    print(f"  daily level: train {market_train.min():.2f}-{market_train.max():.2f}, "
          f"validation {market_val.min():.2f}-{market_val.max():.2f} (inside the training range)")
    print(f"  elasticity of rate to market_index: {coefficients['log_market_index']:.3f}")
    print(f"  linear trend in rate: {100 * np.expm1(coefficients['years']):+.1f}% per year")

    clean = train[train["clean"]].assign(residual=lambda d: 100 * np.log(d["ratio"]), to_quarter_end=lambda d: days_to_quarter_end(d["date"]))
    print("\nmedian residual (%) of the structural model by month and equipment:")
    print(clean.groupby([clean["date"].dt.month, "equipment"])["residual"].median().unstack().round(2).T)
    buckets = pd.cut(clean["to_quarter_end"], [-1, 2, 5, 10, 15, 20, 25, 30, 45, 92])
    print("\nmedian residual (%) by days until quarter end:")
    print(clean.groupby([buckets, "equipment"], observed=True)["residual"].median().unstack().round(2).T)
    print("-> rates ramp up over the last ~25 days of every quarter and reset on day one of the next;")
    print("   Flatbed and Reefer ramp harder than Dry Van. December is a quarter end.")

    figure, (left, right) = new_figure(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    daily = clean.groupby("date")["residual"].median()
    left.axhline(0, color=AXIS, linewidth=1)
    left.plot(daily.index, daily.values, color=BLUE, linewidth=2)
    for quarter_end in pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]):
        peak = daily[quarter_end - pd.Timedelta(days=4) : quarter_end].max()
        left.text(quarter_end, peak + 0.3, f"{quarter_end:%b %d}", ha="center", fontsize=9.5, color=INK_2)
    left.set_ylim(top=daily.max() + 1)
    left.set_ylabel("Median residual vs structural model (%)", color=INK_2)
    left.set_title("Rates ramp into every quarter end, then reset", loc="left", fontsize=13, fontweight="bold", color=INK)

    last_days = clean[clean["to_quarter_end"] <= 45]
    for equipment, color in (("Flatbed", ORANGE), ("Reefer", AQUA), ("Dry Van", BLUE)):
        curve = last_days[last_days["equipment"] == equipment].groupby("to_quarter_end")["residual"].median()
        curve = curve.rolling(5, center=True, min_periods=1).mean()
        right.plot(curve.index, curve.values, color=color, linewidth=2, label=equipment)
        right.text(-1, curve.loc[0], equipment, color=INK_2, fontsize=9.5, va="center")
    right.axhline(0, color=AXIS, linewidth=1)
    right.set_xlim(45, -9)
    right.set_xlabel("Days until quarter end", color=INK_2)
    right.set_title("The ramp is steeper for Flatbed and Reefer", loc="left", fontsize=13, fontweight="bold", color=INK)
    right.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_2)
    save(figure, "03_quarter_end_ramp.png")

    figure, axis = new_figure(figsize=(10.5, 4))
    axis.plot(market_train.index, market_train.values, color=BLUE, linewidth=2, label="Train (Jan-Oct)")
    axis.plot(market_val.index, market_val.values, color=ORANGE, linewidth=2, label="Validation (Nov-Dec)")
    axis.set_ylabel("Daily median market_index", color=INK_2)
    axis.set_title("market_index: a weekly cycle on a shifting level; validation sits low but in range",
                   loc="left", fontsize=13, fontweight="bold", color=INK)
    axis.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK_2)
    save(figure, "04_market_index.png")


def city_audit(train: pd.DataFrame, val: pd.DataFrame, coefficients: pd.Series, new_cities: set[str]) -> None:
    section("7. City effects are regional, so coordinates generalise to the new cities")
    effect = 100 * coefficients.filter(like="city_").rename(lambda name: name.removeprefix("city_"))
    both = pd.concat([train, val])
    location = both.groupby("pickup")[["pickup_lat", "pickup_lon"]].first().set_axis(["lat", "lon"], axis=1)
    known = location.join(effect.rename("effect"), how="inner")
    print(f"city effect on rate: {known['effect'].min():+.1f}% to {known['effect'].max():+.1f}% "
          f"(std {known['effect'].std():.1f}%); correlation with latitude {known['effect'].corr(known['lat']):.2f}")
    print("cheapest:", known["effect"].nsmallest(5).round(1).to_dict())
    print("dearest :", known["effect"].nlargest(5).round(1).to_dict())

    figure, axis = new_figure(figsize=(10.5, 5))
    axis.grid(False)
    colormap = LinearSegmentedColormap.from_list("blue_red", [BLUE, "#f0efec", RED])
    limit = float(np.abs(known["effect"]).max())
    points = axis.scatter(known["lon"], known["lat"], c=known["effect"], cmap=colormap,
                          norm=TwoSlopeNorm(0, -limit, limit), s=70, edgecolors=AXIS, linewidths=0.6)
    unseen = location.loc[sorted(new_cities)]
    axis.scatter(unseen["lon"], unseen["lat"], s=95, facecolors="none", edgecolors=INK, linewidths=1.4,
                 label="New in validation (no training labels)")
    for name, row in unseen.iterrows():
        axis.annotate(name, (row["lon"], row["lat"]), xytext=(7, 5), textcoords="offset points", fontsize=8.5, color=INK_2)
    bar = figure.colorbar(points, ax=axis, pad=0.02, shrink=0.85)
    bar.set_label("City effect on rate (%)", color=INK_2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=9)
    axis.set_xlabel("Longitude (as given in the data)", color=INK_2)
    axis.set_ylabel("Latitude", color=INK_2)
    axis.set_title("City price effects cluster by region, so coordinates can place the 8 new validation cities",
                   loc="left", fontsize=13, fontweight="bold", color=INK)
    axis.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK_2)
    save(figure, "05_city_effects.png")


def drift_audit(train: pd.DataFrame, val: pd.DataFrame, new_cities: set[str]) -> None:
    section("8. Train vs validation drift (adversarial validation: can a classifier tell them apart?)")
    both = pd.concat([train.assign(is_val=0), val.assign(is_val=1)], ignore_index=True)
    both["equipment_code"] = both["equipment"].astype("category").cat.codes
    both["weight_abs"] = both["weight"].abs()
    both["day_of_week"] = both["date"].dt.dayofweek
    structural = ["distance", "equipment_code", "weight_abs", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    seen_only = ~(both["pickup"].isin(new_cities) | both["delivery"].isin(new_cities))
    experiments = [
        ("load features only (distance, equipment, weight, coordinates)", structural, slice(None)),
        ("  same, without rows touching a new city", structural, seen_only),
        ("load features + market_index", structural + ["market_index"], slice(None)),
        ("load features + quote_signal", structural + ["quote_signal"], slice(None)),
    ]
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, columns, rows in experiments:
        subset = both.loc[rows]
        classifier = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=SEED)
        scores = cross_val_predict(classifier, subset[columns], subset["is_val"], cv=folds, method="predict_proba")[:, 1]
        print(f"AUC {roc_auc_score(subset['is_val'], scores):.3f}  {name}")
    print("(0.5 = indistinguishable. The loads themselves do not drift apart from the new cities;")
    print(" the drift is in the two market signals, and in time itself.)")

    print("\nfeature medians and 5%-95% ranges:")
    rows = {}
    for column in ("distance", "weight", "market_index", "quote_signal"):
        for name, frame in (("train", train), ("validation", val)):
            values = frame[column].abs() if column == "weight" else frame[column]
            rows[(column, name)] = values.quantile([0.05, 0.5, 0.95]).round(3).tolist()
    print(pd.DataFrame(rows, index=["p5", "median", "p95"]).T)
    print("equipment mix train:", train["equipment"].value_counts(normalize=True).round(3).to_dict())
    print("equipment mix val  :", val["equipment"].value_counts(normalize=True).round(3).to_dict())


def main() -> None:
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    december = pd.read_csv(DATA / "december_chart_inputs.csv")

    overview(train, val, template, december)
    weight_audit(train, val)
    new_cities = geography_audit(train, val)
    coefficients = add_structural_expectation(train, val)
    label_audit(train)
    quote_signal_audit(train, val)
    time_audit(train, val, coefficients)
    city_audit(train, val, coefficients, new_cities)
    drift_audit(train, val, new_cities)


if __name__ == "__main__":
    main()

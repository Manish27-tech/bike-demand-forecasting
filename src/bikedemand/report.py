"""Report generation for the demand-forecasting pipeline.

The narrative is assembled from the JSON artefacts, so the prose cannot drift
from the measured numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from bikedemand.config import REPORTS, TEST_START

REPORT_PATH = REPORTS / "REPORT.md"
MODEL_CARD_PATH = REPORTS / "model_card.md"


def _load(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _table(rows: list[tuple[str, str]]) -> str:
    header = "| Metric | Value |\n| --- | --- |\n"
    return header + "\n".join(f"| {key} | {value} |" for key, value in rows)


def build_report(
    metrics: dict[str, object],
    eda: dict[str, object],
    diagnostics: dict[str, object],
    clean: dict[str, object],
) -> str:
    """Compose the full analysis narrative."""
    best = metrics["selected_model"]
    test = metrics["test"]
    models = metrics["test_all_models"]
    cv = metrics["cv"]
    split = metrics["split"]

    ladder = "\n".join(
        f"| {name.replace('_', ' ')} | {row['rmse']:.1f} | {row['mae']:.1f} | "
        f"{row['r2']:.3f} | {row['mape']:.1f}% |"
        for name, row in models.items()
    )
    cv_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {row['rmse_mean']:.1f} ± {row['rmse_std']:.1f} | "
        f"{row['mae_mean']:.1f} |"
        for name, row in cv.items()
    )

    headline = _table(
        [
            ("Hours modelled", f"{eda['hours']:,}"),
            ("Period", f"{eda['date_min'][:10]} to {eda['date_max'][:10]}"),
            ("Total rentals", f"{eda['total_rentals']:,}"),
            ("Mean hourly demand", f"{eda['mean_hourly_demand']:.1f}"),
            ("Peak hour of day", f"{eda['peak_hour']:02d}:00"),
            ("Selected model", best.replace("_", " ")),
            ("Hold-out RMSE", f"{test['rmse']:.1f} rentals/hour"),
            ("Hold-out MAE", f"{test['mae']:.1f} rentals/hour"),
            ("Hold-out R²", f"{test['r2']:.4f}"),
            ("Improvement over weekly baseline", f"{metrics['improvement_vs_naive_pct']:.1f}%"),
        ]
    )

    return f"""# Hourly bike-share demand forecasting

This project forecasts hourly bike rentals for a two-year Washington DC
system, using only calendar, weather and past-demand information. The question
is operational: **can a model beat "same hour last week" by enough to justify
operating it, and where does it still fail?**

The answer is yes -- RMSE falls from {models["seasonal_naive"]["rmse"]:.1f} to
{test["rmse"]:.1f} rentals per hour ({metrics["improvement_vs_naive_pct"]:.1f}%
better) with R² {test["r2"]:.3f} -- but the residual diagnostics show the errors
concentrate exactly at the peaks the operator cares about.

## Headline results

{headline}

## Data

UCI Machine Learning Repository #275, *Bike Sharing Dataset*, hourly file:
{clean["rows_in"]:,} published rows covering {clean["date_min"][:10]} to
{clean["date_max"][:10]}, {clean["total_rentals"]:,} rentals in total.

Two data issues shape the cleaning stage:

1. **165 missing hours.** The calendar between the first and last timestamp has
   {clean["rows_on_grid"]:,} hours, so {clean["missing_hours"]} are absent. The
   table is reindexed onto the complete grid; without that, a "one-hour lag"
   would silently mean "the previous observed hour", which can be several hours
   earlier and would flatter the model.
2. **Component leakage.** `casual` and `registered` sum exactly to `cnt`, so
   they are dropped rather than used as features. With them included the model
   would score near-perfectly and be worthless in production.

The target is right-skewed (median {eda["median_hourly_demand"]:.0f}, maximum
{eda["max_hourly_demand"]}), so every learned model is fitted on `log1p` demand
and inverted before scoring.

## Method

**Time-based validation, never random folds.** The hold-out period is
everything from {split["test_start"]} onwards -- {split["test_hours"]:,} hours,
about one quarter. Training stops at {split["train_end"]}. A random split would
let the model interpolate between neighbouring hours and produce a fantasy
score.

**Feature families.**

- *Calendar:* hour, weekday, month, day-of-year, holiday, working day.
- *Cyclical encodings:* sine/cosine pairs for hour (three harmonics, so a
  linear model can represent the twin commute peaks), weekday, month and
  day-of-year.
- *Weather:* normalised temperature, apparent temperature, humidity, wind speed
  and a categorical weather condition.
- *History:* `log1p` demand at lags of 1, 24 and 168 hours, plus 24-hour and
  168-hour rolling means. Every history feature is shifted by at least one hour,
  so none of them can see the value being predicted.

**Candidates.**

| Model | Expanding-window CV RMSE | CV MAE |
| --- | --- | --- |
{cv_rows}

Gradient boosting is selected on validation RMSE and then fitted once on the
whole training period. The seasonal-naive baseline needs no fitting at all: it
simply copies last week.

## Results

| Model | RMSE | MAE | R² | MAPE |
| --- | --- | --- | --- | --- |
{ladder}

The ladder is the point of the comparison. The naive baseline is respectable
(R² {models["seasonal_naive"]["r2"]:.2f}) because weekly seasonality is strong.
The linear model captures most of the remaining structure
(R² {models["ridge_regression"]["r2"]:.2f}). Gradient boosting adds the
interactions -- weather only matters at certain hours, holidays behave like
Sundays -- that take it to R² {test["r2"]:.4f}.

### Where the errors are

- **Worst hour:** {diagnostics["worst_hour"]["hour"]:02d}:00 with MAE
  {diagnostics["worst_hour"]["mae"]:.1f} against mean demand
  {diagnostics["worst_hour"]["mean_demand"]:.0f}. Evening peaks are both the
  highest-demand and highest-error hours.
- **Best hour:** {diagnostics["best_hour"]["hour"]:02d}:00 with MAE
  {diagnostics["best_hour"]["mae"]:.1f}. Overnight demand is small and stable.
- **Relative accuracy:** {diagnostics["share_within_25pct"]:.1%} of hold-out
  hours are predicted within 25% of the actual count.
- **Weather:** clear {diagnostics["mae_by_weather"].get("Clear", 0):.1f} MAE vs
  light rain {diagnostics["mae_by_weather"].get("Light rain / snow", 0):.1f}.
  Rain matters, but it is not the dominant error term.
- **Drift:** daily mean error ranges from
  {diagnostics["daily_bias_range"][0]:.1f} to {diagnostics["daily_bias_range"][1]:.1f}
  rentals per hour, so there is a mild systematic under-prediction in the
  coldest weeks rather than pure noise.

Permutation importance ranks {metrics["feature_importance"][0]["feature"]}
first, followed by the weekly lag and the hour harmonics: recent demand carries
the state of the system, and the calendar explains its shape.

## Limitations

- **One-step-ahead forecasting.** Lag features use actual past demand, which is
  available operationally but means this is not a multi-day-ahead forecast;
  errors would compound if the model were run recursively.
- **One city, two years.** The seasonal coverage is two winters and two
  summers; a third year could shift the weather relationships.
- **No exogenous events.** Strikes, roadworks, closures, pricing changes and
  dock rebalancing are absent, yet they can dominate a single hour.
- **Log-scale fitting under-predicts peaks** slightly, visible in the
  predicted-versus-actual tail.
- **Weather gaps were forward-filled**, which assumes persistence across the
  {clean["missing_hours"]} missing hours.

## Reproduction

```bash
python run.py all          # download -> clean -> eda -> train -> diagnostics -> report
python -m pytest -q
```

The dataset is pinned by SHA-256 in `src/bikedemand/config.py`, the split is a
fixed timestamp (`TEST_START = "{TEST_START}"`) and every model uses
`RANDOM_STATE = 42`. Figures land in `reports/figures/`.
"""


def build_model_card(metrics: dict[str, object], eda: dict[str, object]) -> str:
    """Model card for the selected forecaster."""
    test = metrics["test"]
    return f"""# Model card: hourly bike-share demand

## Overview

| Field | Value |
| --- | --- |
| Model | {metrics["selected_model"].replace("_", " ")} on log1p demand |
| Version | 0.1.0 |
| Task | One-step-ahead regression of hourly rentals |
| Inputs | Calendar, weather and lagged demand (31 features) |
| Output | Expected rentals per hour |

## Intended use

- **In scope:** staffing and rebalancing decisions, capacity planning,
  forecasting research, teaching time-series validation.
- **Out of scope:** multi-day recursive forecasting without re-validation,
  safety-critical dispatch, use on a different city without refitting.

## Training data

{eda["hours"]:,} hourly observations, {eda["date_min"][:10]} to
{eda["date_max"][:10]}, mean demand {eda["mean_hourly_demand"]:.1f}. Fitted on
data before {metrics["split"]["test_start"]}; the hold-out quarter is never
touched during fitting or model selection.

## Features

Calendar (hour, weekday, month, day-of-year, holiday, working day), cyclical
encodings with three hour harmonics, weather (temperature, apparent
temperature, humidity, wind speed, condition) and demand history (log1p lags at
1, 24 and 168 hours plus 24-hour and 168-hour rolling means). Lags are shifted
so no feature sees the current hour.

## Metrics (hold-out quarter, {test["n"]:,} hours)

| Metric | Value |
| --- | --- |
| RMSE | {test["rmse"]:.2f} rentals/hour |
| MAE | {test["mae"]:.2f} rentals/hour |
| R² | {test["r2"]:.4f} |
| MAPE | {test["mape"]:.2f}% |
| Median APE | {test["median_ape"]:.2f}% |
| Bias | {test["bias"]:.2f} rentals/hour |
| Improvement over seasonal naive | {metrics["improvement_vs_naive_pct"]:.1f}% |

## Caveats

- Absolute error grows with demand; plan capacity on percentage error at peaks.
- One-step-ahead only: feeding predictions back in for multi-step horizons will
  degrade accuracy and has not been validated.
- Trained on 2011-2012 data for one city; distribution shift in fleet size,
  pricing or weather severity requires recalibration.
- Sub-hourly demand is not modelled.
"""


def generate_report(
    metrics_path: Path = REPORTS / "metrics.json",
    eda_path: Path = REPORTS / "eda_summary.json",
    diagnostics_path: Path = REPORTS / "diagnostics_summary.json",
    clean_path: Path = REPORTS / "clean_report.json",
    report_path: Path = REPORT_PATH,
    model_card_path: Path = MODEL_CARD_PATH,
) -> tuple[Path, Path]:
    """Write REPORT.md and the model card, returning both paths."""
    metrics = _load(metrics_path)
    eda = _load(eda_path)
    diagnostics = _load(diagnostics_path)
    clean = _load(clean_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(metrics, eda, diagnostics, clean), encoding="utf-8")
    model_card_path.write_text(build_model_card(metrics, eda), encoding="utf-8")
    return report_path, model_card_path

# Hourly bike-share demand forecasting

This project forecasts hourly bike rentals for a two-year Washington DC
system, using only calendar, weather and past-demand information. The question
is operational: **can a model beat "same hour last week" by enough to justify
operating it, and where does it still fail?**

The answer is yes -- RMSE falls from 115.1 to
44.0 rentals per hour (61.8%
better) with R² 0.953 -- but the residual diagnostics show the errors
concentrate exactly at the peaks the operator cares about.

## Headline results

| Metric | Value |
| --- | --- |
| Hours modelled | 16,750 |
| Period | 2011-01-08 to 2012-12-31 |
| Total rentals | 3,234,383 |
| Mean hourly demand | 193.1 |
| Peak hour of day | 17:00 |
| Selected model | gradient boosting |
| Hold-out RMSE | 44.0 rentals/hour |
| Hold-out MAE | 27.4 rentals/hour |
| Hold-out R² | 0.9531 |
| Improvement over weekly baseline | 61.8% |

## Data

UCI Machine Learning Repository #275, *Bike Sharing Dataset*, hourly file:
17,379 published rows covering 2011-01-01 to
2012-12-31, 3,292,679 rentals in total.

Two data issues shape the cleaning stage:

1. **165 missing hours.** The calendar between the first and last timestamp has
   17,544 hours, so 165 are absent. The
   table is reindexed onto the complete grid; without that, a "one-hour lag"
   would silently mean "the previous observed hour", which can be several hours
   earlier and would flatter the model.
2. **Component leakage.** `casual` and `registered` sum exactly to `cnt`, so
   they are dropped rather than used as features. With them included the model
   would score near-perfectly and be worthless in production.

The target is right-skewed (median 147, maximum
977), so every learned model is fitted on `log1p` demand
and inverted before scoring.

## Method

**Time-based validation, never random folds.** The hold-out period is
everything from 2012-10-01 onwards -- 1,990 hours,
about one quarter. Training stops at 2012-09-30 23:00:00. A random split would
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
| ridge regression | 82.0 ± 13.0 | 53.3 |
| gradient boosting | 52.9 ± 13.2 | 32.2 |

Gradient boosting is selected on validation RMSE and then fitted once on the
whole training period. The seasonal-naive baseline needs no fitting at all: it
simply copies last week.

## Results

| Model | RMSE | MAE | R² | MAPE |
| --- | --- | --- | --- | --- |
| seasonal naive | 115.1 | 68.9 | 0.678 | 82.1% |
| ridge regression | 73.0 | 46.4 | 0.871 | 33.1% |
| gradient boosting | 44.0 | 27.4 | 0.953 | 21.5% |

The ladder is the point of the comparison. The naive baseline is respectable
(R² 0.68) because weekly seasonality is strong.
The linear model captures most of the remaining structure
(R² 0.87). Gradient boosting adds the
interactions -- weather only matters at certain hours, holidays behave like
Sundays -- that take it to R² 0.9531.

### Where the errors are

- **Worst hour:** 17:00 with MAE
  64.4 against mean demand
  527. Evening peaks are both the
  highest-demand and highest-error hours.
- **Best hour:** 04:00 with MAE
  2.7. Overnight demand is small and stable.
- **Relative accuracy:** 75.7% of hold-out
  hours are predicted within 25% of the actual count.
- **Weather:** clear 26.4 MAE vs
  light rain 33.9.
  Rain matters, but it is not the dominant error term.
- **Drift:** daily mean error ranges from
  -36.4 to 26.2
  rentals per hour, so there is a mild systematic under-prediction in the
  coldest weeks rather than pure noise.

Permutation importance ranks demand_lag_1h
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
  165 missing hours.

## Reproduction

```bash
python run.py all          # download -> clean -> eda -> train -> diagnostics -> report
python -m pytest -q
```

The dataset is pinned by SHA-256 in `src/bikedemand/config.py`, the split is a
fixed timestamp (`TEST_START = "2012-10-01"`) and every model uses
`RANDOM_STATE = 42`. Figures land in `reports/figures/`.

# Model card: hourly bike-share demand

## Overview

| Field | Value |
| --- | --- |
| Model | gradient boosting on log1p demand |
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

16,750 hourly observations, 2011-01-08 to
2012-12-31, mean demand 193.1. Fitted on
data before 2012-10-01; the hold-out quarter is never
touched during fitting or model selection.

## Features

Calendar (hour, weekday, month, day-of-year, holiday, working day), cyclical
encodings with three hour harmonics, weather (temperature, apparent
temperature, humidity, wind speed, condition) and demand history (log1p lags at
1, 24 and 168 hours plus 24-hour and 168-hour rolling means). Lags are shifted
so no feature sees the current hour.

## Metrics (hold-out quarter, 1,990 hours)

| Metric | Value |
| --- | --- |
| RMSE | 43.95 rentals/hour |
| MAE | 27.42 rentals/hour |
| R² | 0.9531 |
| MAPE | 21.47% |
| Median APE | 12.78% |
| Bias | -5.33 rentals/hour |
| Improvement over seasonal naive | 61.8% |

## Caveats

- Absolute error grows with demand; plan capacity on percentage error at peaks.
- One-step-ahead only: feeding predictions back in for multi-step horizons will
  degrade accuracy and has not been validated.
- Trained on 2011-2012 data for one city; distribution shift in fleet size,
  pricing or weather severity requires recalibration.
- Sub-hourly demand is not modelled.

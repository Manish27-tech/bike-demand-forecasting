# Methodology

How the forecast is built, and why each guard exists. The short version of every
choice is in [`DECISIONS.md`](DECISIONS.md); this document explains the
mechanics.

## 1. Data provenance

UCI Machine Learning Repository #275, *Bike Sharing Dataset*, hourly file
`hour.csv` (inside `bike+sharing+dataset.zip`). The archive is downloaded by
`bikedemand.data.download()` and verified against two SHA-256 pins in
`config.py` — one for the zip, one for the extracted CSV. A checksum mismatch
aborts the pipeline rather than silently training on a different file.

The published table has 17,379 rows covering 2011-01-01 to 2012-12-31, of which
17,544 hours exist on the calendar. The 165 absent hours are real gaps, not a
truncated file.

## 2. Cleaning

`clean_hourly()` performs four operations, in this order:

1. **Grid repair.** Build `pd.date_range` over the full span and reindex.
   Missing rows are inserted with `observed = 0`. Weather columns are
   forward-filled (persistence is the least-wrong assumption over gaps of a few
   hours); the target stays `NaN`.
2. **Calendar rebuild.** `hr`, `weekday`, `mnth`, `yr`, `day_of_year`,
   `is_weekend` are recomputed from `timestamp` rather than trusted from the
   file, so the two can never disagree. `season` is recomputed with the
   dataset's coding (1 = Jan–Mar, 4 = Oct–Dec) and asserted against the
   published column.
3. **Leakage removal.** `casual` and `registered` are dropped. They sum to
   `cnt` exactly; including them would make the task trivial and the model
   useless.
4. **Target domain.** `cnt` is cast to float and asserted non-negative where
   observed.

The clean stage reports `rows_in`, `rows_on_grid`, `missing_hours`,
`observed_rows`, `leakage_columns_dropped` and the rental totals to
`reports/clean_report.json`.

## 3. Features

Thirty-one numeric features in four families:

| Family | Columns | Count |
| --- | --- | --- |
| Calendar and weather | `hr`, `weekday`, `mnth`, `day_of_year`, `yr`, `holiday`, `workingday`, `is_weekend`, `season`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed` | 14 |
| Cyclical | hour (3 harmonics), weekday, month, day-of-year as sin/cos pairs | 12 |
| Lags | `log1p(cnt)` at 1, 24 and 168 hours | 3 |
| Rolling | 24-hour and 168-hour mean of `log1p(cnt)`, ending one hour earlier | 2 |

### Why `log1p` history

History features are stored on the log scale and the same transform is applied
to the target. Two reasons came out of building it:

- **Scale.** A 50-rental error at 04:00 is a large relative miss; at 17:00 it is
  noise. Log space makes the loss roughly proportional to relative error.
- **Linearity.** With raw lags, ridge regression extrapolates beyond the training
  range and predicts up to 4,071 rentals per hour on the hold-out quarter
  (R² −0.82). On log features the same model predicts a maximum of 930 and
  scores R² 0.871. The tree model is barely affected, which is why the
  comparison stays fair.

### Leakage guards

- Lags are `shift(k)` with `k ≥ 1`; the current hour is never visible.
- Rolling means use `shift(1)` before `rolling(...)`, so the window is strictly
  past.
- The last 168 hours of the training period carry the lags that the first
  hold-out hour needs; because both splits are adjacent in time, this is
  legitimate (past demand is genuinely known at forecast time).
- `tests/test_features.py` verifies all of the above, including an injected
  9,000-rental spike that must not appear in any earlier row's features.

Rows with fewer than 168 hours of history (the first week) and rows with no
observed target (the 165 gaps) are dropped, leaving 16,750 usable hours.

## 4. Validation design

**Hard time split.** The boundary is `TEST_START = 2012-10-01` in
`config.py`. Training is everything before it (14,760 hours); the hold-out is the
final quarter (1,990 hours). `split_by_time` asserts that the training maximum
precedes the test minimum.

**Expanding-window CV for model selection.** `TimeSeriesSplit(n_splits=4)` on
the training period only. Each fold validates on the hours immediately after its
training window, so no fold ever validates on the past.

**Nothing about the hold-out informs selection.** Both candidates are fitted and
scored on it *after* the winner is chosen, purely to publish a like-for-like
table.

## 5. Models

| Candidate | Specification |
| --- | --- |
| Seasonal naive | `expm1(demand_lag_168h)` — same hour last week, no parameters |
| Ridge | `StandardScaler` + `Ridge(alpha=1.0)` inside `TransformedTargetRegressor(log1p/expm1)` |
| Gradient boosting | `HistGradientBoostingRegressor(learning_rate=0.06, max_iter=600, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1)` |

`RANDOM_STATE = 42` everywhere. Selection rule: lowest mean validation RMSE.
Gradient boosting won at 52.9 ± 13.2 against ridge's 82.0 ± 13.0.

## 6. Metrics

All reported in rentals per hour on the count scale, after inverting `log1p`:

- **RMSE** — headline; punishes the peak-hour misses.
- **MAE** — interpretable average miss.
- **R²** — share of variance explained, comparable across models.
- **MAPE / median APE** — relative error, computed on positive demand only.
- **Bias** — mean signed error; detects systematic over/under-prediction.
- **Lift over seasonal naive** — `(rmse_naive − rmse_model) / rmse_naive`.

Diagnostics add error by hour, error by weather category, daily mean residual
(with a ±1 SD band) and the share of hours within 25% of the actual count.

## 7. Diagnostics

Seven figures, all rendered from `reports/metrics.json` and
`data/processed/test_predictions.csv`:

1. RMSE/MAE by model (the ladder).
2. Predicted vs actual hexbin with the 1:1 line.
3. Daily mean residual over the hold-out quarter.
4. MAE and bias by hour of day.
5. MAE by weather category with segment volumes.
6. Permutation importance on the hold-out quarter (R² drop, 5 repeats).
7. An actual-vs-predicted week (2012-11-05 to 2012-11-11).

## 8. Reproducibility

```bash
python run.py all               # full pipeline, ~2 minutes
python -m pytest -q             # 49 tests, synthetic fixtures, no network
python scripts/build_notebook.py
python scripts/check_artifacts.py
```

Every stage writes its own JSON so the report, notebook and model card are
generated from the same numbers. Figures are committed; raw data, processed
splits and the fitted model are git-ignored.

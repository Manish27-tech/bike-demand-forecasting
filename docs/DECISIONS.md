# Decision log

Short records of the choices that shaped the pipeline, including the options
that were rejected. Kept in the repository so a reviewer can challenge the
reasoning rather than reconstruct it.

## D1. Hard time split at 2012-10-01

**Context.** 17,379 hourly rows over two years.

**Decision.** Train on everything before 2012-10-01; score once on the final
quarter (1,990 hours). `split_by_time` asserts the boundary.

**Why.** Hourly demand is strongly autocorrelated. A random split puts 16:00 and
17:00 of the same day on opposite sides, so the model can all but interpolate
the answer. A random-split RMSE here would be a number about nothing.

**Rejected.** Random 80/20 (leaks neighbouring hours), a fixed random seed for a
"representative" quarter (chooses the weather regime by luck).

## D2. Repair the hourly grid before building lags

**Context.** The published table is missing 165 of the 17,544 calendar hours.

**Decision.** Reindex onto a complete hourly grid, mark inserted rows
`observed = 0`, forward-fill weather, leave the target `NaN`.

**Why.** `shift(1)` on a table with holes means "the previous *observed* hour",
which can be several hours earlier. Lags silently become multi-hour lags
exactly where the data is hardest, and the model looks better for it.

**Rejected.** Dropping the affected days (loses 7 days of signal), interpolating
the target (fabricates the very quantity being predicted).

## D3. Drop `casual` and `registered`

**Context.** Both are published alongside the total.

**Decision.** Remove them before any modelling; record the drop in
`clean_report.json`.

**Why.** `casual + registered == cnt` exactly. With them, R² would approach 1
and the model would be a decomposition, not a forecast. Their availability at
forecast time is also zero in practice.

**Rejected.** Keeping them as "auxiliary targets" (adds two sub-problems without
improving the deliverable).

## D4. Model on `log1p` demand

**Context.** Demand ranges 1–977, median 147.

**Decision.** Wrap every learned model in
`TransformedTargetRegressor(log1p/expm1)`; store history features on the same
log scale.

**Why.** Squared error on raw counts is dominated by a handful of peak hours;
log space weights relative error instead. It also keeps the linear model
physically plausible: below, with raw lags, ridge predicted 4,071 rentals/hour
on the hold-out quarter.

**Cost.** Log-scale fitting under-predicts the very tallest peaks, visible in
`10_predicted_vs_actual.png`. Accepted, because the percentage error at peaks is
the operationally relevant number and it improves.

**Rejected.** Raw-count fitting (peak-dominated), square-root transform
(similar to log, but makes the residual scale harder to explain), Poisson/Tweedie
GLM (a reasonable alternative that was not pursued here).

## D5. Three-model ladder, not one model

**Context.** It is easy to publish a single RMSE with no reference point.

**Decision.** Always report seasonal naive, ridge and gradient boosting on the
same hold-out.

**Why.** The naive baseline (RMSE 115.1, R² 0.68) shows how much of the score is
just weekly seasonality. Ridge (73.0, R² 0.87) shows what a linear model adds.
Gradient boosting (44.0, R² 0.95) shows what interactions add. Without the ladder
the headline is uninterpretable.

**Rejected.** Comparing against a naive "yesterday" baseline only (weaker and
flatters the model).

## D6. Add hour harmonics rather than extra calendar dummies

**Context.** A single `hour_sin`/`hour_cos` pair cannot express two peaks.

**Decision.** Add second and third harmonics in `features.py`.

**Why.** Ridge improved from RMSE 273 to 73 once the log transform and harmonics
were both in place, and the residual pattern stopped being a pure shape error.
Trees are unaffected, so the feature set stays honest for both candidates.

**Rejected.** One-hot hour (24 columns, no circularity), piecewise time features
(more parameters to justify).

## D7. Select on expanding-window CV, report the hold-out once

**Context.** Model selection needs a validation signal that the test quarter
does not provide.

**Decision.** `TimeSeriesSplit(n_splits=4)` inside the training period; lowest
mean RMSE wins; the hold-out is scored afterwards.

**Why.** Selecting on the hold-out would turn the headline number into a
training number. Four folds keep each validation window long enough to be
meaningful while staying inside 2011-01 to 2012-09.

**Rejected.** A single 80/20 time split (no variance estimate), leave-one-week-out
(expensive, correlated folds).

## D8. Permutation importance on the hold-out

**Context.** `HistGradientBoostingRegressor` has no `feature_importances_`.

**Decision.** `permutation_importance(scoring="r2", n_repeats=5)` on the
hold-out quarter.

**Why.** It answers the question a reader actually asks — how much worse does
the forecast get without this column — and it works for both model families.
Impurity-based importance would not be available and is biased anyway.

**Rejected.** Coefficients from ridge (scale-dependent and only valid for a
linear model), SHAP (extra dependency for little gain here).

## D9. Report absolute error by hour *and* relative accuracy

**Context.** The single RMSE hides a 23× spread in MAE across hours of the day.

**Decision.** Publish MAE by hour, bias by hour, MAE by weather, and the share of
hours predicted within 25% of actual.

**Why.** The evening peak has MAE 64 against MAE 2.7 at 04:00. An operator
staffing bikes cares about the former; a reader comparing models cares about the
latter. The percentage-within-band figure (75.7%) is the fair summary.

**Rejected.** Reporting MAPE alone (explodes on low-demand nights).

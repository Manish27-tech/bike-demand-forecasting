# Hourly bike-share demand forecasting

[![ci](https://github.com/Manish27-tech/bike-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Manish27-tech/bike-demand-forecasting/actions/workflows/ci.yml)

End-to-end forecast of hourly bike rentals on the UCI *Bike Sharing* dataset:
grid repair, leakage removal, calendar and weather features, expanding-window
validation, a three-model ladder and an error breakdown by hour and weather.

**The headline:** gradient boosting cuts hold-out RMSE from 115.1 (same hour
last week) to 44.0 rentals per hour — 61.8% better, R² 0.953 — but the largest
absolute errors land on the 17:00 evening peak, exactly where the operational
decision is made.

## Results at a glance

| Metric | Value |
| --- | --- |
| Hours modelled | 16,750 (2011-01-08 to 2012-12-31) |
| Total rentals | 3,234,383 |
| Mean hourly demand | 193.1 |
| Missing hours repaired | 165 |
| Selected model | histogram gradient boosting on `log1p` demand |
| Hold-out RMSE / MAE | 43.95 / 27.42 rentals per hour |
| Hold-out R² / MAPE | 0.9531 / 21.5% |
| Improvement over seasonal naive | 61.8% |
| Share of hours within 25% of actual | 75.7% |
| Worst hour | 17:00, MAE 64.4 against mean demand 527 |

## Why the analysis looks like this

- **Hard time split at 2012-10-01.** A random split lets the model interpolate
  between neighbouring hours, which flatters it enormously. The last quarter is
  never seen during fitting or model selection.
- **Complete hourly grid.** The published table is missing 165 hours. Without
  reindexing, "the previous hour" silently becomes "the previous *observed*
  hour", and lag features leak days of hindsight.
- **`casual` + `registered` are dropped.** They sum exactly to `cnt`; keeping
  them yields a near-perfect and worthless model.
- **Log-scale target.** Demand ranges 1–977 with a long right tail. Fitting on
  `log1p` and inverting before scoring keeps the loss from being dominated by
  the busiest hours — and stops the linear baseline extrapolating to 4,000
  rentals per hour.
- **Three-model ladder.** Seasonal naive establishes what the calendar alone is
  worth; ridge shows what a linear model can add; gradient boosting shows what
  interactions add. A single score would hide that ordering.

## Project layout

```
src/bikedemand/
  config.py        paths, dataset SHA-256 pins, split date, palette
  data.py          download + integrity verification
  clean.py         hourly grid repair, calendar rebuild, leakage removal
  features.py      cyclical encodings, weather, log1p lags and rolling means
  eda.py           8 demand figures + summary JSON
  model.py         time split, expanding-window CV, three-model comparison
  diagnostics.py   error by hour/weather, residuals, 7 diagnostic figures
  report.py        REPORT.md and model card from the JSON artefacts
  cli.py           stage runner
tests/             49 fast tests (synthetic fixtures, no network)
reports/           REPORT.md, model card, metrics, figures 01-15
notebooks/         executed walkthrough
docs/              methodology, data dictionary, decision log
```

## Quickstart

```bash
python -m pip install -r requirements-dev.txt

python run.py all        # download -> clean -> eda -> train -> diagnostics -> report
python -m pytest -q      # 49 tests
python scripts/build_notebook.py
python scripts/check_artifacts.py
```

Individual stages: `python run.py download|clean|eda|train|diagnostics|report`.
The dataset (~1.1 MB `hour.csv`, inside a 1.5 MB zip) is downloaded from UCI and
verified against a pinned SHA-256; `data/raw/` is git-ignored.

## Docs

- [`reports/REPORT.md`](reports/REPORT.md) - full analysis and limitations
- [`reports/model_card.md`](reports/model_card.md) - intended use and caveats
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) - design choices in detail
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) - every raw and derived column
- [`docs/DECISIONS.md`](docs/DECISIONS.md) - decision log and rejected options

## Data citation

Fanaee-T, H. and Gama, J. (2013). *Bike Sharing*. UCI Machine Learning
Repository. <https://doi.org/10.24432/C5W894>

## Limitations

One city over two years, one-step-ahead forecasts only, no events or pricing
changes, forward-filled weather across gaps, and mild under-prediction of the
tallest peaks. See the report for the full list.

## Licence

MIT

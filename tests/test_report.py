"""Tests that the narrative follows the measured artefacts."""

from __future__ import annotations

import json

from bikedemand import report


def _bundle():
    metrics = {
        "selected_model": "gradient_boosting",
        "improvement_vs_naive_pct": 61.83,
        "split": {
            "test_start": "2012-10-01",
            "train_hours": 14_760,
            "test_hours": 1_990,
            "train_end": "2012-09-30 23:00:00",
            "test_end": "2012-12-31 23:00:00",
        },
        "test": {
            "rmse": 43.95,
            "mae": 27.42,
            "r2": 0.9531,
            "mape": 21.47,
            "median_ape": 12.78,
            "bias": -5.33,
            "n": 1_990,
        },
        "test_all_models": {
            "seasonal_naive": {"rmse": 115.14, "mae": 68.95, "r2": 0.678, "mape": 82.1},
            "ridge_regression": {"rmse": 72.97, "mae": 46.4, "r2": 0.871, "mape": 33.1},
            "gradient_boosting": {"rmse": 43.95, "mae": 27.42, "r2": 0.953, "mape": 21.5},
        },
        "cv": {
            "ridge_regression": {"rmse_mean": 81.98, "rmse_std": 12.98, "mae_mean": 53.32},
            "gradient_boosting": {"rmse_mean": 52.85, "rmse_std": 13.23, "mae_mean": 32.2},
        },
        "feature_importance": [
            {"feature": "demand_lag_1h", "importance_mean": 0.4, "importance_std": 0.01},
            {"feature": "demand_lag_168h", "importance_mean": 0.2, "importance_std": 0.01},
        ],
    }
    eda = {
        "hours": 16_750,
        "date_min": "2011-01-08 00:00:00",
        "date_max": "2012-12-31 23:00:00",
        "total_rentals": 3_234_383,
        "mean_hourly_demand": 193.1,
        "median_hourly_demand": 147.0,
        "max_hourly_demand": 977,
        "peak_hour": 17,
        "quietest_hour": 4,
    }
    diagnostics = {
        "overall_mae": 27.42,
        "worst_hour": {"hour": 17, "mae": 64.43, "mean_demand": 527.1},
        "best_hour": {"hour": 4, "mae": 2.67},
        "mae_by_weather": {"Clear": 26.45, "Light rain / snow": 33.86},
        "daily_bias_range": [-36.42, 26.21],
        "share_within_25pct": 0.7573,
    }
    clean = {
        "rows_in": 17_379,
        "rows_on_grid": 17_544,
        "missing_hours": 165,
        "date_min": "2011-01-01 00:00:00",
        "date_max": "2012-12-31 23:00:00",
        "total_rentals": 3_292_679,
    }
    return metrics, eda, diagnostics, clean


def test_report_contains_headline_numbers():
    text = report.build_report(*_bundle())
    assert "43.95" in text or "44.0" in text
    assert "0.9531" in text
    assert "61.8%" in text
    assert "165" in text


def test_report_explains_the_leakage_decision():
    text = report.build_report(*_bundle())
    assert "casual" in text and "registered" in text
    assert "log1p" in text


def test_model_card_lists_scope_and_caveats():
    metrics, eda, _, _ = _bundle()
    card = report.build_model_card(metrics, eda)
    assert "Intended use" in card
    assert "Out of scope" in card
    assert "Caveats" in card


def test_generate_report_writes_files(tmp_path):
    metrics, eda, diag, clean = _bundle()
    paths = [tmp_path / name for name in ("metrics.json", "eda.json", "diag.json", "clean.json")]
    for path, payload in zip(paths, (metrics, eda, diag, clean), strict=True):
        path.write_text(json.dumps(payload), encoding="utf-8")

    report_path, card_path = report.generate_report(
        metrics_path=paths[0],
        eda_path=paths[1],
        diagnostics_path=paths[2],
        clean_path=paths[3],
        report_path=tmp_path / "REPORT.md",
        model_card_path=tmp_path / "model_card.md",
    )
    assert report_path.stat().st_size > 2_000
    assert card_path.stat().st_size > 1_000

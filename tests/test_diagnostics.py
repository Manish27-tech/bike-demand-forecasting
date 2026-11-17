"""Tests for the forecast error breakdown and its figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bikedemand import diagnostics


def _predictions(rows: int = 240, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2012-10-01", periods=rows, freq="h")
    profile = np.clip(np.sin(np.pi * (timestamps.hour.to_numpy() - 6) / 12), 0, None)
    truth = 100 + 60 * profile + rng.normal(0, 10, rows)
    truth = truth.clip(min=1)
    predicted = truth * 0.9 + rng.normal(0, 8, rows)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hour": timestamps.hour,
            "weekday": timestamps.dayofweek,
            "weather_label": rng.choice(["Clear", "Mist / cloud", "Light rain / snow"], rows),
            "workingday": (timestamps.dayofweek < 5).astype(int),
            "y_true": truth.round(2),
            "y_pred": predicted.round(2),
        }
    )
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame["absolute_error"] = frame["residual"].abs()
    return frame


def _metrics() -> dict[str, object]:
    return {
        "test": {"rmse": 20.0, "mae": 15.0, "r2": 0.9},
        "test_all_models": {
            "seasonal_naive": {"rmse": 40.0, "mae": 30.0, "r2": 0.6, "mape": 50.0},
            "ridge_regression": {"rmse": 30.0, "mae": 22.0, "r2": 0.8, "mape": 35.0},
            "gradient_boosting": {"rmse": 20.0, "mae": 15.0, "r2": 0.9, "mape": 20.0},
        },
        "feature_importance": [
            {"feature": "demand_lag_1h", "importance_mean": 0.4, "importance_std": 0.01},
            {"feature": "hour_cos", "importance_mean": 0.2, "importance_std": 0.01},
        ],
        "improvement_vs_naive_pct": 50.0,
    }


def test_error_by_hour_covers_all_hours():
    table = diagnostics.error_by_hour(_predictions())
    assert len(table) == 24
    assert (table["mae"] >= 0).all()


def test_error_by_segment_suppresses_tiny_groups():
    predictions = _predictions(200)
    predictions.loc[:5, "weather_label"] = "Heavy rain / snow"
    table = diagnostics.error_by_segment(predictions, "weather_label", min_rows=50)
    assert "Heavy rain / snow" not in table.index


def test_summary_reports_worst_and_best_hours():
    summary = diagnostics.summarise_diagnostics(_predictions(), _metrics())
    assert 0 <= summary["worst_hour"]["hour"] <= 23
    assert 0 <= summary["best_hour"]["hour"] <= 23
    assert summary["overall_mae"] > 0
    assert 0 <= summary["share_within_25pct"] <= 1


def test_all_diagnostic_figures_render(tmp_path):
    predictions = _predictions(360)
    metrics = _metrics()
    diagnostics.plot_model_comparison(metrics["test_all_models"], tmp_path)
    diagnostics.plot_predicted_vs_actual(predictions, tmp_path)
    diagnostics.plot_residuals_over_time(predictions, tmp_path)
    diagnostics.plot_error_by_hour(predictions, tmp_path)
    diagnostics.plot_error_by_weather(predictions, tmp_path)
    diagnostics.plot_importance(metrics["feature_importance"], tmp_path)
    diagnostics.plot_forecast_week(predictions, tmp_path)

    files = sorted(tmp_path.glob("*.png"))
    assert len(files) == 7
    for path in files:
        assert path.stat().st_size > 5_000

"""Tests for forecasting metrics, time-ordered validation and model factories."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bikedemand.model import (
    MODEL_FACTORIES,
    compute_importance,
    evaluate_baseline,
    regression_metrics,
    select_model,
    train_and_evaluate,
)


def test_perfect_predictions_score_zero_error():
    y = np.array([10.0, 20.0, 30.0])
    metrics = regression_metrics(y, y)
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["mape"] == 0
    assert metrics["r2"] == 1.0


def test_metrics_are_in_demand_units():
    y = np.array([100.0, 200.0])
    predictions = np.array([110.0, 180.0])
    metrics = regression_metrics(y, predictions)
    assert metrics["mae"] == 15
    assert metrics["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2), abs=1e-3)
    assert metrics["bias"] == -5


def test_mape_ignores_zero_demand():
    y = np.array([0.0, 100.0])
    predictions = np.array([50.0, 110.0])
    metrics = regression_metrics(y, predictions)
    assert metrics["mape"] == pytest.approx(10.0)


def test_baseline_scores_the_weekly_lag(features_frame):
    scores = evaluate_baseline(features_frame)
    assert scores["n"] == len(features_frame)
    assert scores["rmse"] > 0


def test_select_model_picks_lowest_rmse():
    scores = {
        "a": {"rmse_mean": 60.0, "mae_mean": 40.0},
        "b": {"rmse_mean": 50.0, "mae_mean": 45.0},
    }
    assert select_model(scores) == "b"


def test_model_factories_fit_small_data(features_frame):
    from bikedemand.features import feature_matrix

    X, y = feature_matrix(features_frame.head(400))
    for name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y), name
        assert np.isfinite(predictions).all(), name
        assert predictions.min() >= 0, name


def test_importance_table_is_sorted_and_complete(features_frame):
    from bikedemand.features import feature_matrix

    X, y = feature_matrix(features_frame.head(400))
    model = MODEL_FACTORIES["ridge_regression"]().fit(X, y)
    table = compute_importance(model, X, y, n_repeats=2)
    assert len(table) == X.shape[1]
    values = [row["importance_mean"] for row in table]
    assert values == sorted(values, reverse=True)


def test_train_and_evaluate_persists_artefacts(cleaned_frame, tmp_path):
    frame, _ = cleaned_frame
    artifact = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"

    payload = train_and_evaluate(
        frame,
        artifact_path=artifact,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        test_start="2012-06-01",
    )

    assert artifact.exists() and metrics_path.exists() and predictions_path.exists()
    assert payload["selected_model"] in {"ridge_regression", "gradient_boosting"}
    assert payload["split"]["test_hours"] > 0
    assert payload["split"]["train_end"] < payload["split"]["test_start"]
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert stored["test"]["n"] == payload["split"]["test_hours"]
    predictions = pd.read_csv(predictions_path)
    assert set(predictions.columns) >= {"timestamp", "y_true", "y_pred", "residual"}
    assert len(predictions) == payload["split"]["test_hours"]

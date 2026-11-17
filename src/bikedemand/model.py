"""Forecasting models with time-based validation.

Three candidates are compared on the same hold-out quarter:

1. **Seasonal naive** -- demand at the same hour one week earlier. This is the
   baseline any forecasting model has to beat to be worth deploying.
2. **Ridge regression** on scaled calendar and weather features, which tests
   whether a linear model is sufficient.
3. **Gradient boosting** on the full feature set, which can capture interactions
   such as "rain only hurts at weekends".

All learned models predict ``log1p`` demand and are inverted back to counts, so
the reported RMSE/MAE are in rentals per hour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bikedemand.config import (
    ARTIFACTS,
    DATA_PROCESSED,
    RANDOM_STATE,
    REPORTS,
    TARGET,
    TEST_START,
)
from bikedemand.features import (
    NUMERIC_FEATURES,
    build_features,
    feature_matrix,
    seasonal_naive,
    split_by_time,
)

MODEL_ARTIFACT = ARTIFACTS / "demand_model.joblib"
METRICS_PATH = REPORTS / "metrics.json"
PREDICTIONS_PATH = DATA_PROCESSED / "test_predictions.csv"


def build_ridge() -> TransformedTargetRegressor:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_gradient_boosting() -> TransformedTargetRegressor:
    pipeline = Pipeline(
        [
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=600,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


MODEL_FACTORIES = {
    "ridge_regression": build_ridge,
    "gradient_boosting": build_gradient_boosting,
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Error metrics in rentals per hour, plus percentage errors on positive demand."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    positive = y_true > 0
    absolute_percentage = np.abs(residual[positive] / y_true[positive]) * 100
    return {
        "rmse": round(float(np.sqrt(np.mean(residual**2))), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(float(absolute_percentage.mean()), 2),
        "median_ape": round(float(np.median(absolute_percentage)), 2),
        "bias": round(float(residual.mean()), 3),
        "n": int(len(y_true)),
    }


def evaluate_baseline(test: pd.DataFrame) -> dict[str, float]:
    """Score the seasonal-naive prediction that ships no parameters at all."""
    return regression_metrics(test[TARGET].to_numpy(), seasonal_naive(test))


def cross_validate_time_ordered(train: pd.DataFrame, folds: int = 4) -> dict[str, dict[str, float]]:
    """Expanding-window validation on the training period only.

    ``TimeSeriesSplit`` never validates on a row that precedes its training
    rows, which is the minimum requirement for an honest forecast estimate.
    """
    X, y = feature_matrix(train)
    splitter = TimeSeriesSplit(n_splits=folds)
    results: dict[str, dict[str, float]] = {}
    for name, factory in MODEL_FACTORIES.items():
        fold_rmse: list[float] = []
        fold_mae: list[float] = []
        for train_index, valid_index in splitter.split(X):
            model = factory()
            model.fit(X.iloc[train_index], y.iloc[train_index])
            predicted = model.predict(X.iloc[valid_index])
            metrics = regression_metrics(y.iloc[valid_index].to_numpy(), predicted)
            fold_rmse.append(metrics["rmse"])
            fold_mae.append(metrics["mae"])
        results[name] = {
            "rmse_mean": round(float(np.mean(fold_rmse)), 3),
            "rmse_std": round(float(np.std(fold_rmse)), 3),
            "mae_mean": round(float(np.mean(fold_mae)), 3),
            "folds": folds,
        }
    return results


def select_model(scores: dict[str, dict[str, float]]) -> str:
    """Cheapest-to-beat criterion: lowest mean validation RMSE."""
    return min(scores, key=lambda name: scores[name]["rmse_mean"])


def compute_importance(
    model: TransformedTargetRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
) -> list[dict[str, float]]:
    """Permutation importance scored by R2 on the hold-out period."""
    result = permutation_importance(
        model,
        X,
        y,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    table = [
        {
            "feature": str(name),
            "importance_mean": round(float(mean), 5),
            "importance_std": round(float(std), 5),
        }
        for name, mean, std in zip(
            X.columns, result.importances_mean, result.importances_std, strict=True
        )
    ]
    return sorted(table, key=lambda row: row["importance_mean"], reverse=True)


def train_and_evaluate(
    frame: pd.DataFrame,
    artifact_path: Path = MODEL_ARTIFACT,
    metrics_path: Path = METRICS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    test_start: str = TEST_START,
) -> dict[str, Any]:
    """Fit the candidates, score the hold-out quarter and persist everything."""
    import joblib

    features = build_features(frame)
    train, test = split_by_time(features, test_start)
    X_train, y_train = feature_matrix(train)
    X_test, y_test = feature_matrix(test)

    cv_scores = cross_validate_time_ordered(train)
    best_name = select_model(cv_scores)

    # Every candidate is fitted and scored so the comparison is like-for-like.
    test_scores: dict[str, dict[str, float]] = {"seasonal_naive": evaluate_baseline(test)}
    fitted: dict[str, Any] = {}
    for name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(X_train, y_train)
        fitted[name] = model
        test_scores[name] = regression_metrics(y_test.to_numpy(), model.predict(X_test))

    best_model = fitted[best_name]
    predictions = best_model.predict(X_test)
    importance = compute_importance(best_model, X_test, y_test)

    naive_rmse = test_scores["seasonal_naive"]["rmse"]
    payload = {
        "selected_model": best_name,
        "selection_metric": "mean expanding-window validation RMSE",
        "target_transform": "log1p",
        "cv": cv_scores,
        "test": test_scores[best_name],
        "test_all_models": test_scores,
        "improvement_vs_naive_pct": round(
            (naive_rmse - test_scores[best_name]["rmse"]) / naive_rmse * 100, 2
        ),
        "split": {
            "test_start": test_start,
            "train_hours": int(len(train)),
            "test_hours": int(len(test)),
            "train_end": str(train["timestamp"].max()),
            "test_end": str(test["timestamp"].max()),
        },
        "feature_importance": importance,
        "features": list(NUMERIC_FEATURES),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "features": list(NUMERIC_FEATURES),
            "target": TARGET,
            "trained_until": str(train["timestamp"].max()),
            "test_metrics": payload["test"],
        },
        artifact_path,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    output = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "hour": test["hr"].to_numpy(),
            "weekday": test["weekday"].to_numpy(),
            "weather_label": test["weather_label"].to_numpy(),
            "workingday": test["workingday"].to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_pred": np.round(predictions, 3),
            "residual": np.round(predictions - y_test.to_numpy(), 3),
        }
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions_path, index=False)

    return payload


def load_model(path: Path = MODEL_ARTIFACT) -> dict[str, Any]:
    """Load the persisted artefact."""
    import joblib

    return joblib.load(path)


def forecast(artifact: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Predict hourly demand for a feature frame."""
    return artifact["model"].predict(features[artifact["features"]])

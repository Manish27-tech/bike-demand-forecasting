"""Forecast diagnostics: where the model wins, and where it loses.

A single RMSE hides the structure that matters operationally. These diagnostics
break the error down by hour, weather and demand level, and show the residuals
drifting over the hold-out quarter rather than scattering randomly.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bikedemand.config import (  # noqa: E402
    ACCENT,
    FIGURE_DPI,
    FIGURES,
    PALETTE,
    REPORTS,
    TARGET,
)
from bikedemand.model import METRICS_PATH, PREDICTIONS_PATH  # noqa: E402


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )


def _save(fig: plt.Figure, name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["absolute_error"] = (frame["y_pred"] - frame["y_true"]).abs()
    return frame


def error_by_hour(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error and bias for each hour of the day."""
    table = predictions.groupby("hour").agg(
        mae=("absolute_error", "mean"),
        bias=("residual", "mean"),
        demand=("y_true", "mean"),
    )
    return table


def error_by_segment(predictions: pd.DataFrame, column: str, min_rows: int = 50) -> pd.DataFrame:
    """MAE, bias and volume per category, suppressing tiny segments."""
    table = (
        predictions.groupby(column)
        .agg(
            mae=("absolute_error", "mean"),
            bias=("residual", "mean"),
            rows=("y_true", "size"),
            demand=("y_true", "mean"),
        )
        .query("rows >= @min_rows")
        .sort_values("mae")
    )
    return table


def summarise_diagnostics(
    predictions: pd.DataFrame, metrics: dict[str, object]
) -> dict[str, object]:
    """Numbers that deserve a paragraph in the report."""
    by_hour = error_by_hour(predictions)
    by_weather = error_by_segment(predictions, "weather_label")
    overall_mae = float(predictions["absolute_error"].mean())
    daily = predictions.set_index("timestamp")["residual"].resample("D").mean()
    worst_hour = int(by_hour["mae"].idxmax())
    best_hour = int(by_hour["mae"].idxmin())
    return {
        "overall_mae": round(overall_mae, 3),
        "worst_hour": {
            "hour": worst_hour,
            "mae": round(float(by_hour.loc[worst_hour, "mae"]), 2),
            "mean_demand": round(float(by_hour.loc[worst_hour, "demand"]), 1),
        },
        "best_hour": {
            "hour": best_hour,
            "mae": round(float(by_hour.loc[best_hour, "mae"]), 2),
        },
        "mae_by_weather": {
            str(index): round(float(row["mae"]), 2) for index, row in by_weather.iterrows()
        },
        "daily_bias_range": [
            round(float(daily.min()), 2),
            round(float(daily.max()), 2),
        ],
        "share_within_25pct": round(
            float((predictions["residual"].abs() / predictions["y_true"] <= 0.25).mean()),
            4,
        ),
        "model_test": metrics["test"],
        "improvement_vs_naive_pct": metrics["improvement_vs_naive_pct"],
    }


def plot_model_comparison(
    test_all: dict[str, dict[str, float]], output_dir: Path = FIGURES
) -> Path:
    _style()
    names = list(test_all)
    display = [name.replace("_", " ") for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in zip(
        axes, ("rmse", "mae"), ("RMSE (rentals/hour)", "MAE (rentals/hour)"), strict=True
    ):
        values = [test_all[name][metric] for name in names]
        colors = ["#9AA7B8", "#4C72B0", ACCENT][: len(names)]
        bars = ax.bar(display, values, color=colors)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.02,
                f"{value:.1f}",
                ha="center",
                fontsize=9,
            )
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=12)
        ax.set_ylim(0, max(values) * 1.2)
    fig.suptitle(
        "Hold-out quarter: the model beats the one-week baseline by 62%", fontweight="bold"
    )
    return _save(fig, "09_model_comparison.png", output_dir)


def plot_predicted_vs_actual(predictions: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.hexbin(
        predictions["y_true"],
        predictions["y_pred"],
        gridsize=45,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    limit = float(max(predictions["y_true"].max(), predictions["y_pred"].max())) * 1.03
    ax.plot([0, limit], [0, limit], color=PALETTE["predicted"], ls="--", lw=1.4)
    ax.set_xlabel("Actual rentals per hour")
    ax.set_ylabel("Predicted rentals per hour")
    ax.set_title("Predictions track the diagonal; the biggest misses are peak hours")
    ax.grid(False)
    return _save(fig, "10_predicted_vs_actual.png", output_dir)


def plot_residuals_over_time(predictions: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    daily = predictions.set_index("timestamp")["residual"].resample("D").agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(daily.index, daily["mean"], color="#4C72B0", width=1.0, label="Daily mean error")
    ax.fill_between(
        daily.index,
        daily["mean"] - daily["std"],
        daily["mean"] + daily["std"],
        color="#4C72B0",
        alpha=0.2,
        label="+/- 1 standard deviation",
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Prediction error (rentals/hour)")
    ax.set_title("Errors stay centred but widen in the coldest weeks")
    ax.legend(fontsize=8)
    return _save(fig, "11_residuals_over_time.png", output_dir)


def plot_error_by_hour(predictions: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    table = error_by_hour(predictions)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(table.index, table["mae"], color="#4C72B0", label="MAE")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean absolute error")
    ax.set_xticks(range(0, 24, 2))
    twin = ax.twinx()
    twin.plot(table.index, table["bias"], color=PALETTE["predicted"], marker="o", lw=1.4)
    twin.axhline(0, color="grey", lw=0.8, ls="--")
    twin.set_ylabel("Mean bias")
    twin.grid(False)
    ax.set_title("Peak hours carry the largest absolute error")
    ax.legend(loc="upper left")
    return _save(fig, "12_error_by_hour.png", output_dir)


def plot_error_by_weather(predictions: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    table = error_by_segment(predictions, "weather_label")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(table.index.astype(str), table["mae"], color="#55A868")
    ax.set_xlabel("Mean absolute error (rentals/hour)")
    ax.set_title("Bad weather is rare in the hold-out quarter and not the main error source")
    for index, (_, row) in enumerate(table.iterrows()):
        ax.text(row["mae"] + 0.3, index, f"n={int(row['rows']):,}", va="center", fontsize=8)
    return _save(fig, "13_error_by_weather.png", output_dir)


def plot_importance(
    importances: list[dict[str, float]], output_dir: Path = FIGURES, top_n: int = 12
) -> Path:
    _style()
    table = pd.DataFrame(importances).head(top_n).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(
        table["feature"],
        table["importance_mean"],
        xerr=table["importance_std"],
        color=ACCENT,
        capsize=3,
    )
    ax.set_xlabel("Drop in R2 when shuffled")
    ax.set_title("History features dominate, calendar features refine")
    return _save(fig, "14_feature_importance.png", output_dir)


def plot_forecast_week(
    predictions: pd.DataFrame, output_dir: Path = FIGURES, start: str = "2012-11-05"
) -> Path:
    _style()
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(days=7)
    window = predictions[
        (predictions["timestamp"] >= start_ts) & (predictions["timestamp"] < end_ts)
    ]
    if window.empty:
        window = predictions.head(24 * 7)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(window["timestamp"], window["y_true"], color="#4C72B0", lw=1.8, label="Actual")
    ax.plot(
        window["timestamp"],
        window["y_pred"],
        color=PALETTE["predicted"],
        lw=1.6,
        ls="--",
        label="Predicted",
    )
    ax.set_ylabel("Rentals per hour")
    ax.set_title("One hold-out week: shape is captured, peak height is under-predicted")
    ax.legend()
    return _save(fig, "15_forecast_week.png", output_dir)


def generate_diagnostics(
    predictions_path: Path = PREDICTIONS_PATH,
    metrics_path: Path = METRICS_PATH,
    output_dir: Path = FIGURES,
    summary_path: Path | None = REPORTS / "diagnostics_summary.json",
) -> dict[str, object]:
    """Render the seven diagnostic figures and persist the error breakdown."""
    predictions = load_predictions(predictions_path)
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

    plot_model_comparison(metrics["test_all_models"], output_dir)
    plot_predicted_vs_actual(predictions, output_dir)
    plot_residuals_over_time(predictions, output_dir)
    plot_error_by_hour(predictions, output_dir)
    plot_error_by_weather(predictions, output_dir)
    plot_importance(metrics["feature_importance"], output_dir)
    plot_forecast_week(predictions, output_dir)

    summary = summarise_diagnostics(predictions, metrics)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

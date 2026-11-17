"""Exploratory analysis: eight figures and a summary payload.

The figures are ordered as an argument: when does demand happen, what moves it,
and what does that imply for a forecasting model?
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
    TEST_START,
)
from bikedemand.features import NUMERIC_FEATURES, build_features  # noqa: E402

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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


def plot_demand_distribution(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(frame[TARGET], bins=50, color="#4C72B0")
    axes[0].set_title("Hourly rentals (linear scale)")
    axes[0].set_xlabel("Rentals per hour")
    axes[0].set_ylabel("Hours")
    axes[1].hist(np.log1p(frame[TARGET]), bins=50, color=ACCENT)
    axes[1].set_title("After log1p transform")
    axes[1].set_xlabel("log1p(rentals)")
    axes[1].set_ylabel("Hours")
    fig.suptitle("Demand is right-skewed: model on the log scale, report on the count scale")
    return _save(fig, "01_demand_distribution.png", output_dir)


def plot_hourly_profile(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for label, mask, color in (
        ("Working day", frame["workingday"] == 1, "#4C72B0"),
        ("Weekend / holiday", frame["workingday"] == 0, ACCENT),
    ):
        subset = frame.loc[mask]
        profile = subset.groupby("hr")[TARGET].agg(["mean", "std"])
        ax.plot(profile.index, profile["mean"], marker="o", color=color, label=label)
        ax.fill_between(
            profile.index,
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            color=color,
            alpha=0.15,
        )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean rentals per hour")
    ax.set_title("Two commute peaks on working days, one afternoon hump at weekends")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    return _save(fig, "02_hourly_profile.png", output_dir)


def plot_month_hour_heatmap(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    pivot = frame.pivot_table(index="mnth", columns="hr", values=TARGET, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 4.6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pivot.index)), [MONTH_NAMES[m - 1] for m in pivot.index])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_title("Mean demand by month and hour")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Mean rentals")
    return _save(fig, "03_month_hour_heatmap.png", output_dir)


def plot_weather_effects(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    temp_bins = pd.cut(frame["temp"] * 41, bins=8)
    profile = frame.groupby(temp_bins, observed=True)[TARGET].mean()
    axes[0].plot(range(len(profile)), profile.to_numpy(), marker="o", color=ACCENT)
    axes[0].set_xticks(range(len(profile)), [f"{interval.mid:.0f}" for interval in profile.index])
    axes[0].set_xlabel("Temperature (C, bin midpoint)")
    axes[0].set_ylabel("Mean rentals")
    axes[0].set_title("Demand rises with temperature")

    weather = frame.groupby("weather_label", observed=True)[TARGET].agg(["mean", "size"])
    weather = weather.sort_values("mean")
    axes[1].barh(weather.index.astype(str), weather["mean"], color="#55A868")
    axes[1].set_xlabel("Mean rentals")
    axes[1].set_title("Weather conditions")
    for index, (_, row) in enumerate(weather.iterrows()):
        axes[1].text(row["mean"] + 2, index, f"n={int(row['size']):,}", va="center", fontsize=8)
    return _save(fig, "04_weather_effects.png", output_dir)


def plot_correlation(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    columns = [
        "hr",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "weathersit",
        "demand_lag_1h",
        "demand_lag_24h",
        "demand_lag_168h",
        TARGET,
    ]
    matrix = frame[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.045)
    ax.set_title("Lag-1 demand is strongly correlated, weather mildly so")
    return _save(fig, "05_correlation.png", output_dir)


def plot_time_series(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(frame["timestamp"], frame[TARGET], color="#9AA7B8", lw=0.5, alpha=0.6, label="Hourly")
    weekly = frame.set_index("timestamp")[TARGET].rolling(24 * 7, min_periods=24).mean()
    ax.plot(weekly.index, weekly.to_numpy(), color="#4C72B0", lw=2, label="7-day mean")
    boundary = pd.Timestamp(TEST_START)
    ax.axvline(boundary, color=PALETTE["predicted"], ls="--", lw=1.6)
    ax.annotate(
        "hold-out period",
        (boundary, frame[TARGET].max() * 0.85),
        xytext=(8, 0),
        textcoords="offset points",
        color=PALETTE["predicted"],
    )
    ax.set_ylabel("Rentals per hour")
    ax.set_title("Two years of hourly demand with a hard time split")
    ax.legend(loc="upper left")
    return _save(fig, "06_time_series.png", output_dir)


def plot_lag_relationship(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.hexbin(frame["demand_lag_24h"], frame[TARGET], gridsize=45, cmap="Blues", mincnt=1)
    limit = float(frame[TARGET].max()) * 1.02
    ax.plot([0, limit], [0, limit], color=PALETTE["predicted"], ls="--", lw=1.4)
    ax.set_xlabel("Demand 24 hours earlier")
    ax.set_ylabel("Demand now")
    ax.set_title("Same-hour-yesterday is informative but far from sufficient")
    ax.grid(False)
    return _save(fig, "07_lag_relationship.png", output_dir)


def plot_seasonal_hour_boxes(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    order = ["Winter", "Spring", "Summer", "Fall"]
    data = [frame.loc[frame["season_label"] == season, TARGET].to_numpy() for season in order]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.boxplot(
        data,
        tick_labels=order,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#B8CDE8"},
    )
    ax.set_ylabel("Rentals per hour")
    ax.set_title("Season shifts the level, not the shape, of the daily cycle")
    return _save(fig, "08_seasonal_boxes.png", output_dir)


PLOTS = (
    plot_demand_distribution,
    plot_hourly_profile,
    plot_month_hour_heatmap,
    plot_weather_effects,
    plot_correlation,
    plot_time_series,
    plot_lag_relationship,
    plot_seasonal_hour_boxes,
)


def summarise(frame: pd.DataFrame) -> dict[str, object]:
    """Compute the numbers used in the report and notebook."""
    hourly = frame.groupby("hr")[TARGET].mean()
    monthly = frame.groupby("mnth")[TARGET].mean()
    weather = frame.groupby("weather_label", observed=True)[TARGET].mean()
    correlations = (
        frame[list(NUMERIC_FEATURES) + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values()
    )
    weekday_mean = float(frame.loc[frame["workingday"] == 1, TARGET].mean())
    weekend_mean = float(frame.loc[frame["workingday"] == 0, TARGET].mean())
    return {
        "hours": int(len(frame)),
        "date_min": str(frame["timestamp"].min()),
        "date_max": str(frame["timestamp"].max()),
        "total_rentals": int(frame[TARGET].sum()),
        "mean_hourly_demand": round(float(frame[TARGET].mean()), 2),
        "median_hourly_demand": float(frame[TARGET].median()),
        "max_hourly_demand": int(frame[TARGET].max()),
        "peak_hour": int(hourly.idxmax()),
        "quietest_hour": int(hourly.idxmin()),
        "working_day_mean": round(weekday_mean, 2),
        "weekend_mean": round(weekend_mean, 2),
        "busiest_month": MONTH_NAMES[int(monthly.idxmax()) - 1],
        "quietest_month": MONTH_NAMES[int(monthly.idxmin()) - 1],
        "weather_means": {str(key): round(float(value), 2) for key, value in weather.items()},
        "train_hours": int((frame["timestamp"] < pd.Timestamp(TEST_START)).sum()),
        "test_hours": int((frame["timestamp"] >= pd.Timestamp(TEST_START)).sum()),
        "strongest_positive_correlations": {
            str(key): round(float(value), 3) for key, value in correlations.tail(5).items()
        },
        "strongest_negative_correlations": {
            str(key): round(float(value), 3) for key, value in correlations.head(5).items()
        },
    }


def generate_eda(
    frame: pd.DataFrame,
    output_dir: Path = FIGURES,
    summary_path: Path | None = REPORTS / "eda_summary.json",
) -> dict[str, object]:
    """Render every figure and write the summary JSON."""
    features = build_features(frame)
    for plot in PLOTS:
        plot(features, output_dir)
    summary = summarise(features)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

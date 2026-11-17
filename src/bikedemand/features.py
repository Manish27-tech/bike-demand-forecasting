"""Feature engineering with strict time discipline.

Every lag and rolling statistic is computed on the complete hourly grid and
shifted by at least one hour, so no feature can ever see the value it is trying
to predict. The final row selection then drops the warm-up period and the
unobserved hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bikedemand.config import (
    LAG_HOURS,
    MIN_HISTORY,
    ROLLING_WINDOWS,
    TARGET,
)

CALENDAR_FEATURES = (
    "hr",
    "weekday",
    "mnth",
    "day_of_year",
    "yr",
    "holiday",
    "workingday",
    "is_weekend",
    "season",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
)

CYCLICAL_FEATURES = (
    "hour_sin",
    "hour_cos",
    "hour_sin2",
    "hour_cos2",
    "hour_sin3",
    "hour_cos3",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
)

LAG_FEATURES = tuple(f"demand_lag_{hours}h" for hours in LAG_HOURS)
ROLLING_FEATURES = tuple(f"demand_roll_{window}h" for window in ROLLING_WINDOWS)

NUMERIC_FEATURES = CALENDAR_FEATURES + CYCLICAL_FEATURES + LAG_FEATURES + ROLLING_FEATURES


def add_cyclical(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode hour, weekday, month and day-of-year as sin/cos pairs.

    A raw hour of 23 and 0 are adjacent in reality but far apart numerically;
    the circular encoding removes that artificial discontinuity.
    """
    working = frame.copy()
    working["hour_sin"] = np.sin(2 * np.pi * working["hr"] / 24)
    working["hour_cos"] = np.cos(2 * np.pi * working["hr"] / 24)
    # Second and third harmonics let a linear model represent the twin commute
    # peaks; trees simply ignore the redundancy.
    working["hour_sin2"] = np.sin(4 * np.pi * working["hr"] / 24)
    working["hour_cos2"] = np.cos(4 * np.pi * working["hr"] / 24)
    working["hour_sin3"] = np.sin(6 * np.pi * working["hr"] / 24)
    working["hour_cos3"] = np.cos(6 * np.pi * working["hr"] / 24)
    working["weekday_sin"] = np.sin(2 * np.pi * working["weekday"] / 7)
    working["weekday_cos"] = np.cos(2 * np.pi * working["weekday"] / 7)
    working["month_sin"] = np.sin(2 * np.pi * (working["mnth"] - 1) / 12)
    working["month_cos"] = np.cos(2 * np.pi * (working["mnth"] - 1) / 12)
    working["doy_sin"] = np.sin(2 * np.pi * working["day_of_year"] / 365.25)
    working["doy_cos"] = np.cos(2 * np.pi * working["day_of_year"] / 365.25)
    return working


def add_demand_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Log-transformed lags and rolling means, all shifted to exclude the present.

    The history is stored as ``log1p`` for two reasons: demand spans 1-977 and a
    level shift of 50 rentals means something very different at 02:00 and at
    08:00, and a linear model fitted to raw lags extrapolates to physically
    impossible values (over 4,000 rentals per hour) on the hold-out quarter.
    """
    working = frame.copy()
    history = np.log1p(working[TARGET])
    for hours in LAG_HOURS:
        working[f"demand_lag_{hours}h"] = history.shift(hours)
    for window in ROLLING_WINDOWS:
        # min_periods tolerates the 165 unobserved hours without producing NaN.
        working[f"demand_roll_{window}h"] = (
            history.shift(1).rolling(window, min_periods=int(window * 0.8)).mean()
        )
    return working


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the modelling frame with calendar, cyclical and history features."""
    enriched = add_cyclical(add_demand_history(frame))

    # Rows without enough history (the first week) or without an observed
    # target (the 165 gaps) cannot be used for supervised learning.
    enriched = enriched.dropna(subset=list(NUMERIC_FEATURES) + [TARGET]).reset_index(drop=True)
    if len(enriched) < MIN_HISTORY:
        raise ValueError("not enough rows left after feature construction")
    return enriched


def feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the modelling frame into ``X`` and ``y``."""
    X = frame[list(NUMERIC_FEATURES)].astype("float32")
    y = frame[TARGET].astype("float32")
    return X, y


def split_by_time(frame: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, test)`` with a hard boundary and no shuffling."""
    boundary = pd.Timestamp(test_start)
    train = frame[frame["timestamp"] < boundary].reset_index(drop=True)
    test = frame[frame["timestamp"] >= boundary].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"time split at {test_start} produced an empty partition")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise AssertionError("time split leaked future rows into the training set")
    return train, test


def seasonal_naive(frame: pd.DataFrame) -> np.ndarray:
    """Baseline: the same hour one week earlier, back on the count scale."""
    return np.expm1(frame["demand_lag_168h"].to_numpy(dtype=float))

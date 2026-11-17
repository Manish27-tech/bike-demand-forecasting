"""Synthetic hourly fixtures: no network, no 1.1 MB CSV in the test run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_hourly_frame(hours: int = 1_200, seed: int = 13, drop: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2012-05-01", periods=hours, freq="h")
    frame = pd.DataFrame(
        {
            "instant": np.arange(1, hours + 1),
            "dteday": timestamps.strftime("%Y-%m-%d"),
            "season": [((month % 12) // 3) + 1 for month in timestamps.month],
            "yr": (timestamps.year - 2011).astype(int),
            "mnth": timestamps.month,
            "hr": timestamps.hour,
            "holiday": rng.integers(0, 2, size=hours),
            "weekday": timestamps.dayofweek,
            "workingday": ((timestamps.dayofweek < 5) & (rng.random(hours) > 0.1)).astype(int),
            "weathersit": rng.choice([1, 1, 2, 3], size=hours),
            "temp": rng.uniform(0.1, 0.9, size=hours).round(2),
            "atemp": rng.uniform(0.1, 0.9, size=hours).round(4),
            "hum": rng.uniform(0.2, 0.95, size=hours).round(2),
            "windspeed": rng.uniform(0.0, 0.5, size=hours).round(4),
        }
    )
    profile = 120 + 110 * np.sin(np.pi * (frame["hr"] - 6) / 12).clip(lower=0)
    frame["casual"] = (profile * 0.3 + rng.normal(0, 8, hours)).clip(lower=0).round()
    frame["registered"] = (profile * 0.7 + rng.normal(0, 12, hours)).clip(lower=0).round()
    frame["cnt"] = (frame["casual"] + frame["registered"]).astype(int).clip(lower=1)
    if drop:
        frame = frame.drop(index=range(10, 10 + drop)).reset_index(drop=True)
    return frame


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    return make_hourly_frame()


@pytest.fixture
def cleaned_frame(hourly_frame: pd.DataFrame):
    from bikedemand.clean import clean_hourly

    frame, report = clean_hourly(hourly_frame)
    return frame, report


@pytest.fixture
def features_frame(cleaned_frame):
    from bikedemand.features import build_features

    frame, _ = cleaned_frame
    return build_features(frame)

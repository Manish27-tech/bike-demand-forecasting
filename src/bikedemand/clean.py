"""Cleaning: put the extract on a complete hourly grid.

The published file has 17,379 rows but the calendar between 2011-01-01 and
2012-12-31 contains 17,544 hours -- 165 hours are absent. Simply shifting rows
would make a "1-hour lag" mean "the previous *observed* hour", which can be
several hours earlier. Reindexing to the full grid keeps lag semantics honest;
the 165 unobserved hours are dropped after feature construction.
"""

from __future__ import annotations

import pandas as pd

from bikedemand.config import LEAKAGE_COLUMNS, SEASON_LABELS, TARGET, WEATHER_LABELS


def add_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine ``dteday`` and ``hr`` into a single hourly timestamp."""
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["dteday"]) + pd.to_timedelta(
        working["hr"], unit="h"
    )
    return working.drop(columns=["dteday"])


def season_from_month(month: int) -> int:
    """Map a month to the dataset's season coding (1 winter .. 4 fall)."""
    if month in (12, 1, 2):
        return 1
    if month in (3, 4, 5):
        return 2
    if month in (6, 7, 8):
        return 3
    return 4


def clean_hourly(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the gridded table plus a report of every mutation."""
    working = add_timestamp(frame)
    rows_in = len(working)

    duplicates = int(working["timestamp"].duplicated().sum())
    working = working.sort_values("timestamp").drop_duplicates("timestamp")

    full_index = pd.date_range(working["timestamp"].min(), working["timestamp"].max(), freq="h")
    working = (
        working.set_index("timestamp").reindex(full_index).rename_axis("timestamp").reset_index()
    )
    missing_hours = int(working[TARGET].isna().sum())

    # Calendar fields are fully determined by the timestamp, so they can be
    # rebuilt for the gap rows as well.
    working["hr"] = working["timestamp"].dt.hour.astype("int8")
    working["mnth"] = working["timestamp"].dt.month.astype("int8")
    working["day_of_year"] = working["timestamp"].dt.dayofyear.astype("int16")
    working["weekday"] = working["timestamp"].dt.dayofweek.astype("int8")  # 0=Monday
    working["yr"] = (working["timestamp"].dt.year - 2011).astype("int8")
    working["season"] = working["mnth"].map(season_from_month).astype("int8")
    working["is_weekend"] = (working["weekday"] >= 5).astype("int8")

    # Small numbers of weather gaps: forward-fill within the grid (never
    # backwards from the future), then default the rest.
    for column in ("temp", "atemp", "hum", "windspeed", "weathersit"):
        working[column] = working[column].ffill()
    working["weathersit"] = working["weathersit"].fillna(1).astype("int8")
    working["holiday"] = working["holiday"].fillna(0).astype("int8")
    working["workingday"] = working["workingday"].fillna(0).astype("int8")

    working["weather_label"] = working["weathersit"].map(WEATHER_LABELS).fillna("Unknown")
    working["season_label"] = working["season"].map(SEASON_LABELS)
    working["observed"] = working[TARGET].notna()

    working = working.drop(columns=[column for column in LEAKAGE_COLUMNS if column in working])
    if "instant" in working.columns:
        working = working.drop(columns=["instant"])

    working[TARGET] = working[TARGET].astype("float32")

    report = {
        "rows_in": int(rows_in),
        "rows_on_grid": int(len(working)),
        "duplicate_timestamps": duplicates,
        "missing_hours": missing_hours,
        "observed_rows": int(working["observed"].sum()),
        "date_min": str(working["timestamp"].min()),
        "date_max": str(working["timestamp"].max()),
        "total_rentals": int(working[TARGET].sum()),
        "mean_hourly_demand": round(float(working[TARGET].mean()), 2),
        "max_hourly_demand": int(working[TARGET].max()),
        "share_zero_demand": round(float((working[TARGET] == 0).mean()), 4),
        "leakage_columns_dropped": list(LEAKAGE_COLUMNS),
    }
    return working, report

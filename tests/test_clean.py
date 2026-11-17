"""Cleaning tests: grid completeness, derived calendar and leakage removal."""

from __future__ import annotations

import pandas as pd

from bikedemand.clean import season_from_month


def test_grid_is_complete_and_gaps_are_flagged(cleaned_frame):
    frame, report = cleaned_frame
    expected_hours = (frame["timestamp"].max() - frame["timestamp"].min()) / pd.Timedelta("1h") + 1
    assert len(frame) == int(expected_hours)
    assert report["missing_hours"] == 4
    assert report["observed_rows"] == len(frame) - 4
    assert report["rows_on_grid"] == len(frame)


def test_timestamps_are_unique_and_sorted(cleaned_frame):
    frame, _ = cleaned_frame
    assert frame["timestamp"].is_unique
    assert frame["timestamp"].is_monotonic_increasing


def test_component_columns_are_dropped(cleaned_frame):
    frame, report = cleaned_frame
    assert "casual" not in frame.columns
    assert "registered" not in frame.columns
    assert report["leakage_columns_dropped"] == ["casual", "registered"]


def test_calendar_fields_are_rebuilt_from_timestamp(cleaned_frame):
    frame, _ = cleaned_frame
    assert (frame["hr"] == frame["timestamp"].dt.hour).all()
    assert (frame["weekday"] == frame["timestamp"].dt.dayofweek).all()
    assert (frame["yr"] == frame["timestamp"].dt.year - 2011).all()
    assert (frame["is_weekend"] == (frame["weekday"] >= 5).astype(int)).all()


def test_season_mapping_follows_the_dataset_coding():
    assert season_from_month(1) == 1
    assert season_from_month(4) == 2
    assert season_from_month(7) == 3
    assert season_from_month(10) == 4


def test_report_accounts_for_every_row(cleaned_frame):
    _, report = cleaned_frame
    assert report["rows_on_grid"] - report["missing_hours"] == report["rows_in"]


def test_gap_hours_keep_weather_but_lose_the_target(cleaned_frame):
    frame, _ = cleaned_frame
    gaps = frame[~frame["observed"]]
    assert len(gaps) == 4
    assert gaps["cnt"].isna().all()
    assert gaps["temp"].notna().all()

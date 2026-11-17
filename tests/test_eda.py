"""Tests for the EDA layer."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bikedemand import eda


def test_every_plot_writes_a_png(cleaned_frame, tmp_path):
    frame, _ = cleaned_frame
    features = eda.build_features(frame)
    paths = [plot(features, tmp_path) for plot in eda.PLOTS]
    assert len(paths) == 8
    for path in paths:
        assert path.exists() and path.stat().st_size > 5_000


def test_generate_eda_writes_summary(cleaned_frame, tmp_path):
    frame, _ = cleaned_frame
    summary_path = tmp_path / "eda.json"
    summary = eda.generate_eda(frame, output_dir=tmp_path, summary_path=summary_path)
    stored = json.loads(summary_path.read_text(encoding="utf-8"))
    assert stored["hours"] == summary["hours"]


def test_summary_totals_match_the_frame(cleaned_frame):
    frame, _ = cleaned_frame
    summary = eda.summarise(eda.build_features(frame))
    features = eda.build_features(frame)
    assert summary["hours"] == len(features)
    assert summary["total_rentals"] == int(features["cnt"].sum())
    assert summary["max_hourly_demand"] == int(features["cnt"].max())


def test_split_counts_add_up(cleaned_frame):
    frame, _ = cleaned_frame
    summary = eda.summarise(eda.build_features(frame))
    assert summary["train_hours"] + summary["test_hours"] == summary["hours"]


def test_peak_hour_is_in_range(cleaned_frame):
    frame, _ = cleaned_frame
    summary = eda.summarise(eda.build_features(frame))
    assert 0 <= summary["peak_hour"] <= 23
    assert 0 <= summary["quietest_hour"] <= 23
    assert summary["peak_hour"] != summary["quietest_hour"]


def test_plots_do_not_mutate_input(cleaned_frame, tmp_path):
    frame, _ = cleaned_frame
    before = frame.copy(deep=True)
    eda.generate_eda(frame, output_dir=tmp_path, summary_path=tmp_path / "s.json")
    pd.testing.assert_frame_equal(frame, before)

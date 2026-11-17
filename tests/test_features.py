"""Feature tests, with the emphasis on proving there is no temporal leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bikedemand.features import (
    NUMERIC_FEATURES,
    build_features,
    feature_matrix,
    seasonal_naive,
    split_by_time,
)


def test_every_feature_is_finite(features_frame):
    assert np.isfinite(features_frame[list(NUMERIC_FEATURES)].to_numpy()).all()


def test_lag_one_equals_previous_hour_log_demand(cleaned_frame):
    frame, _ = cleaned_frame
    features = build_features(frame)
    reference = np.log1p(frame["cnt"].to_numpy())
    positions = {timestamp: index for index, timestamp in enumerate(frame["timestamp"])}
    for row in features.head(20).itertuples():
        previous = positions[row.timestamp] - 1
        assert row.demand_lag_1h == pytest.approx(reference[previous], abs=1e-5)


def test_daily_lag_equals_same_hour_yesterday(cleaned_frame):
    frame, _ = cleaned_frame
    features = build_features(frame)
    reference = np.log1p(frame["cnt"].to_numpy())
    positions = {timestamp: index for index, timestamp in enumerate(frame["timestamp"])}
    for row in features.head(20).itertuples():
        earlier = positions[row.timestamp] - 24
        assert row.demand_lag_24h == pytest.approx(reference[earlier], abs=1e-5)


def test_rolling_mean_excludes_the_current_hour(cleaned_frame):
    frame, _ = cleaned_frame
    features = build_features(frame)
    reference = np.log1p(frame["cnt"].to_numpy())
    positions = {timestamp: index for index, timestamp in enumerate(frame["timestamp"])}
    row = features.iloc[500]
    earlier = positions[row["timestamp"]]
    window = reference[earlier - 24 : earlier]
    assert row["demand_roll_24h"] == pytest.approx(window.mean(), rel=1e-5)


def test_no_feature_correlates_with_future_shock(cleaned_frame):
    """A synthetic spike must not appear in any lag before it happens."""
    frame, _ = cleaned_frame
    position = 400
    frame = frame.copy()
    frame.loc[position, "cnt"] = 9000.0
    features = build_features(frame)
    spike_time = frame.loc[position, "timestamp"]
    before = features[features["timestamp"] < spike_time]
    assert before["demand_lag_1h"].max() < np.log1p(9000)
    assert before["demand_roll_24h"].max() < np.log1p(9000)


def test_cyclical_encodings_are_bounded(features_frame):
    cyclical = [column for column in NUMERIC_FEATURES if "sin" in column or "cos" in column]
    assert features_frame[cyclical].abs().max().max() <= 1.0001
    assert len(cyclical) == 12


def test_feature_matrix_types_and_shapes(features_frame):
    X, y = feature_matrix(features_frame)
    assert list(X.columns) == list(NUMERIC_FEATURES)
    assert len(X) == len(y)
    assert str(X.dtypes.iloc[0]) == "float32"


def test_split_respects_the_boundary(features_frame):
    train, test = split_by_time(features_frame, "2012-06-01")
    assert train["timestamp"].max() < pd.Timestamp("2012-06-01")
    assert test["timestamp"].min() >= pd.Timestamp("2012-06-01")
    assert len(train) + len(test) == len(features_frame)


def test_split_rejects_empty_partitions(features_frame):
    with pytest.raises(ValueError, match="empty partition"):
        split_by_time(features_frame, "1999-01-01")


def test_seasonal_naive_inverts_the_log_lag(features_frame):
    predicted = seasonal_naive(features_frame)
    expected = np.expm1(features_frame["demand_lag_168h"].to_numpy())
    assert np.allclose(predicted, expected)
    assert predicted.min() >= 0

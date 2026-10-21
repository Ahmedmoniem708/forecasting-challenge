import numpy as np
import pandas as pd
from cwfc.features import (
    extract_temporal_features,
    extract_lag_and_rolling_features,
    build_direct_forecasting_dataset,
)


def test_extract_temporal_features_shapes_and_bounds():
    timestamps = np.array([0, 300, 43200, 86400])
    feats = extract_temporal_features(timestamps)
    assert feats.shape == (4, 3)
    assert np.all(feats["sin_tod"] >= -1.0) and np.all(feats["sin_tod"] <= 1.0)
    assert np.all(feats["cos_tod"] >= -1.0) and np.all(feats["cos_tod"] <= 1.0)


def test_extract_lag_and_rolling_no_future_lookahead():
    series = np.array([10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    feats = extract_lag_and_rolling_features(series, lags=(1, 2), rolling_windows=(3,))
    # At index 2 (value 15.0), lag_1 must be 12.0, lag_2 must be 10.0
    assert feats["lag_1"].iloc[2] == 12.0
    assert feats["lag_2"].iloc[2] == 10.0
    # Rolling mean over window 3 at index 2 must be (10+12+15)/3 = 12.333
    assert round(feats["rolling_mean_3"].iloc[2], 3) == round((10 + 12 + 15) / 3, 3)


def test_build_direct_forecasting_dataset():
    n = 40
    series = np.linspace(10, 20, n)
    timestamps = np.arange(n) * 300
    train_pairs, x_latest = build_direct_forecasting_dataset(series, timestamps, horizon=5, min_history=10)
    assert len(train_pairs) == 5
    assert x_latest.shape[0] == 1

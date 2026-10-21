"""Feature engineering utilities for cloud workload time-series forecasting.

Ensures strict zero-leakage compliance: all statistical aggregations, lags,
and rolling statistics are computed exclusively on history available at or before
the forecast origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_temporal_features(timestamps: np.ndarray | pd.Series) -> pd.DataFrame:
    """Extract cyclical and normalized calendar features from relative seconds timestamps.

    Trace timestamps in Azure Public Dataset V2 represent elapsed seconds from the start.
    Assumes standard diurnal periodicity (86,400 seconds = 24 hours).
    """
    ts = np.asarray(timestamps, dtype=float)
    day_seconds = 86400.0

    time_of_day = ts % day_seconds
    sin_tod = np.sin(2.0 * np.pi * time_of_day / day_seconds)
    cos_tod = np.cos(2.0 * np.pi * time_of_day / day_seconds)
    norm_time = ts / day_seconds

    return pd.DataFrame({
        "sin_tod": sin_tod,
        "cos_tod": cos_tod,
        "norm_time": norm_time,
    })


def extract_lag_and_rolling_features(
    series: np.ndarray | pd.Series,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    rolling_windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Extract strictly historical lag and rolling statistics from an aggregate target series.

    Every feature at index `i` is computed strictly using values at index `<= i`.
    """
    s = pd.Series(series, dtype=float)
    features = {}

    # Current value at origin
    features["val_origin"] = s

    # Autoregressive lags
    for lag in lags:
        features[f"lag_{lag}"] = s.shift(lag)

    # Rolling statistics (closed='left' equivalent: includes current value at origin)
    for w in rolling_windows:
        features[f"rolling_mean_{w}"] = s.rolling(window=w, min_periods=1).mean()
        features[f"rolling_std_{w}"] = s.rolling(window=w, min_periods=1).std().fillna(0.0)
        features[f"rolling_min_{w}"] = s.rolling(window=w, min_periods=1).min()
        features[f"rolling_max_{w}"] = s.rolling(window=w, min_periods=1).max()

    # Rate of change / momentum
    features["diff_1"] = s - s.shift(1)
    features["diff_3"] = s - s.shift(3)
    features["accel_1"] = (s - s.shift(1)) - (s.shift(1) - s.shift(2))

    df_feats = pd.DataFrame(features)
    return df_feats


def build_direct_forecasting_dataset(
    series: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Series,
    horizon: int = 24,
    min_history: int = 12
) -> tuple[dict[int, tuple[pd.DataFrame, pd.Series]], pd.DataFrame]:
    """Construct multi-step direct forecasting training pairs (X_h, y_h) for h in 1..horizon.

    In direct multi-step forecasting:
    For each horizon step h in [1, H]:
      Target: y_{t+h}
      Features: X(t) strictly computed from history up to and including t.
    This eliminates compounding recursive errors and accommodates horizon-specific dynamics.

    Returns:
        train_pairs: dict mapping step_ahead h -> (X_train_h, y_train_h)
        x_latest: feature vector computed at the final historical timestamp T (origin).
    """
    s = np.asarray(series, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    n = len(s)

    lag_feats = extract_lag_and_rolling_features(s)
    time_feats = extract_temporal_features(ts)
    full_feats = pd.concat([lag_feats, time_feats], axis=1)

    train_pairs = {}
    for h in range(1, horizon + 1):
        # We need targets y_{t+h}
        # Valid origins t: from min_history to n - h - 1
        valid_origins = range(min_history, n - h)
        if len(valid_origins) > 5:
            X_h = full_feats.iloc[valid_origins].reset_index(drop=True)
            y_h = pd.Series([s[t + h] for t in valid_origins], name=f"target_h{h}")
            train_pairs[h] = (X_h, y_h)

    # Latest features at origin T = n - 1 to generate out-of-sample forecasts
    x_latest = full_feats.iloc[[-1]].reset_index(drop=True)
    return train_pairs, x_latest

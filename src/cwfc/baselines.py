from __future__ import annotations

import pandas as pd
import numpy as np


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Forecast the last observed value across all horizon timestamps."""
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def moving_average_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    window: int = 6
) -> pd.Series:
    """Forecast the trailing moving average of the last `window` observations."""
    if train.empty:
        raise ValueError("train must not be empty")
    values = train[target].astype(float).to_numpy()
    if len(values) < window:
        avg_val = float(np.mean(values))
    else:
        avg_val = float(np.mean(values[-window:]))
    return pd.Series(avg_val, index=template.index, dtype=float)


def global_mean_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Forecast the historical global sample mean of the target."""
    if train.empty:
        raise ValueError("train must not be empty")
    mean_val = float(train[target].astype(float).mean())
    return pd.Series(mean_val, index=template.index, dtype=float)


def seasonal_naive_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    season_length: int
) -> pd.Series:
    """Forecast repeating seasonal values from the most recent full cycle."""
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)

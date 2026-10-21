from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error (percentage in [0, 200])."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    """Normalized MAE relative to a baseline forecast (model MAE / baseline MAE)."""
    model_mae = mae(y_true, y_pred)
    baseline_mae = mae(y_true, baseline_pred)
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / baseline_mae)


def peak_mae(y_true, y_pred, quantile: float = 0.80) -> float:
    """MAE restricted to the highest-load timestamps in y_true.

    Operational importance: Under-forecasting or large errors during peak
    utilization events carry disproportionate risk of capacity exhaustion,
    CPU throttling, and SLA breaches.
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    threshold = np.quantile(a, quantile)
    mask = a >= threshold
    if not np.any(mask):
        return mae(a, b)
    return float(np.mean(np.abs(a[mask] - b[mask])))


def mase(y_true, y_pred, y_train, seasonality: int = 1) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scales MAE by the in-sample naive 1-step (or seasonal) forecast error.
    Values < 1.0 indicate the model outperforms the naive in-sample baseline.
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    tr = np.asarray(y_train, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(tr) <= seasonality:
        raise ValueError("y_train length must exceed seasonality")

    scale = np.mean(np.abs(tr[seasonality:] - tr[:-seasonality]))
    if scale == 0:
        return 0.0 if mae(a, b) == 0 else float("inf")
    return float(mae(a, b) / scale)


def interval_coverage(y_true, lower_bound, upper_bound) -> float:
    """Percentage of true observations captured within [lower_bound, upper_bound]."""
    a = np.asarray(y_true, dtype=float)
    l = np.asarray(lower_bound, dtype=float)
    u = np.asarray(upper_bound, dtype=float)
    if a.shape != l.shape or a.shape != u.shape:
        raise ValueError("Arrays must have identical shapes")
    contained = (a >= l) & (a <= u)
    return float(np.mean(contained) * 100.0)

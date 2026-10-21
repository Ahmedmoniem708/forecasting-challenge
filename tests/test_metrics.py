import numpy as np
from cwfc.metrics import (
    interval_coverage,
    mae,
    mase,
    normalized_mae,
    peak_mae,
    rmse,
    smape,
)


def test_metrics_basic():
    y = [1, 2, 3]
    p = [1, 3, 1]
    assert mae(y, p) == 1.0
    assert round(rmse(y, p), 8) == round((5 / 3) ** 0.5, 8)
    assert smape(y, p) > 0
    assert normalized_mae(y, p, [1, 1, 1]) > 0


def test_metrics_extended():
    y = np.array([10.0, 12.0, 20.0, 25.0])
    p = np.array([11.0, 11.0, 18.0, 24.0])
    train = np.array([8.0, 9.0, 10.0, 11.0])

    p_mae = peak_mae(y, p, quantile=0.5)
    assert p_mae > 0

    mase_val = mase(y, p, train)
    assert mase_val > 0

    cov = interval_coverage(y, p - 2.0, p + 2.0)
    assert 0 <= cov <= 100

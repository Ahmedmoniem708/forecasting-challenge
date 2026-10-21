import numpy as np
import pandas as pd
from cwfc.models import (
    PersistenceForecaster,
    MovingAverageForecaster,
    AutoRegressiveForecaster,
    DirectRidgeForecaster,
    CloudWorkloadPipeline,
)


def test_forecasters_predict_valid_shapes_and_values():
    n = 60
    train_df = pd.DataFrame({
        "timestamp": np.arange(n) * 300,
        "active_vm_count": np.full(n, 100),
        "aggregate_cpu_utilization": 10.0 + np.sin(np.arange(n) / 5.0)
    })
    template_df = pd.DataFrame({
        "timestamp": np.arange(n, n + 12) * 300
    })

    # Persistence
    pers = PersistenceForecaster().fit(train_df, "aggregate_cpu_utilization")
    p_pred = pers.predict(template_df)
    assert len(p_pred) == 12
    assert p_pred.iloc[0] == train_df["aggregate_cpu_utilization"].iloc[-1]

    # Moving Average
    ma = MovingAverageForecaster(window=6).fit(train_df, "aggregate_cpu_utilization")
    ma_pred = ma.predict(template_df)
    assert len(ma_pred) == 12

    # AutoRegressive
    ar = AutoRegressiveForecaster(lags=2).fit(train_df, "aggregate_cpu_utilization")
    ar_pred = ar.predict(template_df)
    assert len(ar_pred) == 12
    p, l, u = ar.predict_with_intervals(template_df)
    assert np.all(l <= p) and np.all(p <= u)

    # Direct Ridge
    ridge = DirectRidgeForecaster(alpha=1.0, horizon=12).fit(train_df, "aggregate_cpu_utilization")
    r_pred = ridge.predict(template_df)
    assert len(r_pred) == 12


def test_cloud_workload_pipeline():
    train_df = pd.DataFrame({
        "timestamp": np.arange(50) * 300,
        "active_vm_count": np.full(50, 500),
        "aggregate_cpu_utilization": np.random.uniform(8, 12, size=50)
    })
    template_df = pd.DataFrame({
        "timestamp": np.arange(50, 74) * 300
    })
    pipeline = CloudWorkloadPipeline(cpu_forecaster_type="autoreg")
    pipeline.fit(train_df)
    res = pipeline.predict(template_df)

    assert list(res.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(res) == 24
    assert (res["active_vm_count"] >= 0).all()
    assert (res["aggregate_cpu_utilization"] >= 0.0).all() and (res["aggregate_cpu_utilization"] <= 100.0).all()

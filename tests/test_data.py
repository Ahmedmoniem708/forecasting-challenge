import pytest
import pandas as pd
import numpy as np
from cwfc.data import construct_targets, split_train_holdout


def test_construct_targets_aggregates_correctly():
    # Synthetic smoke data
    df_cpu = pd.DataFrame({
        "timestamp": [0, 0, 0, 300, 300, 300],
        "vm_id": ["vm1", "vm2", "vm3", "vm1", "vm2", "vm3"],
        "avg_cpu": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0]
    })
    targets = construct_targets(df_cpu)

    assert len(targets) == 2
    assert list(targets.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert targets.iloc[0]["active_vm_count"] == 3
    assert targets.iloc[0]["aggregate_cpu_utilization"] == 20.0
    assert targets.iloc[1]["aggregate_cpu_utilization"] == 25.0


def test_construct_targets_deduplicates_vm_within_timestamp():
    # If a VM appears twice at same timestamp, nunique should count it once
    df_cpu = pd.DataFrame({
        "timestamp": [0, 0],
        "vm_id": ["vm1", "vm1"],
        "avg_cpu": [10.0, 20.0]
    })
    targets = construct_targets(df_cpu)
    assert targets.iloc[0]["active_vm_count"] == 1
    assert targets.iloc[0]["aggregate_cpu_utilization"] == 15.0


def test_split_train_holdout_no_leakage():
    targets = pd.DataFrame({
        "timestamp": np.arange(134) * 300,
        "active_vm_count": np.full(134, 10),
        "aggregate_cpu_utilization": np.random.uniform(5, 15, size=134)
    })
    train, holdout = split_train_holdout(targets, train_size=110, holdout_size=24)

    assert len(train) == 110
    assert len(holdout) == 24
    assert train["timestamp"].max() < holdout["timestamp"].min()
    assert holdout["timestamp"].iloc[0] == train["timestamp"].iloc[-1] + 300

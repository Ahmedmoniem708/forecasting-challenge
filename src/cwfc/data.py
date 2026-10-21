"""Data loading, validation, and target construction utilities for CWFC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class DatasetBundle:
    """Container holding raw and aggregated workload data."""
    cpu_readings: pd.DataFrame
    metadata: pd.DataFrame
    targets: pd.DataFrame
    train_targets: pd.DataFrame
    holdout_targets: pd.DataFrame


def load_raw_data(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw CPU readings and VM metadata from the data directory.

    Supports both compressed (.csv.gz) and uncompressed (.csv) formats.
    """
    data_path = Path(data_dir)
    cpu_path = data_path / "vm_cpu_readings.csv.gz"
    if not cpu_path.exists():
        cpu_path = data_path / "vm_cpu_readings.csv"

    meta_path = data_path / "vm_metadata.csv.gz"
    if not meta_path.exists():
        meta_path = data_path / "vm_metadata.csv"

    if not cpu_path.exists():
        raise FileNotFoundError(f"CPU readings not found in {data_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"VM metadata not found in {data_path}")

    df_cpu = pd.read_csv(cpu_path)
    df_meta = pd.read_csv(meta_path)

    # Basic type casting and sanitation
    df_cpu["timestamp"] = df_cpu["timestamp"].astype(int)
    df_cpu["vm_id"] = df_cpu["vm_id"].astype(str)
    for col in ["min_cpu", "max_cpu", "avg_cpu"]:
        if col in df_cpu.columns:
            df_cpu[col] = df_cpu[col].astype(float)

    return df_cpu, df_meta


def construct_targets(df_cpu: pd.DataFrame) -> pd.DataFrame:
    """Construct the challenge forecasting targets from row-level CPU readings.

    Targets required by ASSIGNMENT.md & DATASET.md:
    1. active_vm_count: number of distinct VM IDs with a valid CPU reading at timestamp t.
       Each VM is counted at most once per timestamp.
    2. aggregate_cpu_utilization: mean of VM-level avg_cpu values at timestamp t,
       expressed as a percentage in [0, 100].
    """
    # Verify required columns exist
    if "timestamp" not in df_cpu.columns or "vm_id" not in df_cpu.columns or "avg_cpu" not in df_cpu.columns:
        raise ValueError("df_cpu must contain 'timestamp', 'vm_id', and 'avg_cpu'")

    grouped = df_cpu.groupby("timestamp", as_index=False).agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean")
    )

    grouped = grouped.sort_values("timestamp").reset_index(drop=True)

    # Validation checks
    if grouped["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found after target aggregation")

    # Ensure bounds
    grouped["active_vm_count"] = grouped["active_vm_count"].clip(lower=0)
    grouped["aggregate_cpu_utilization"] = grouped["aggregate_cpu_utilization"].clip(0.0, 100.0)

    return grouped


def split_train_holdout(
    targets: pd.DataFrame,
    train_size: int = 110,
    holdout_size: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split targets into public training window and holdout horizon.

    Strictly preserves temporal ordering with zero leakage.
    """
    total_expected = train_size + holdout_size
    if len(targets) < total_expected:
        raise ValueError(
            f"Target series length ({len(targets)}) is shorter than train + holdout ({total_expected})"
        )

    train = targets.iloc[:train_size].copy().reset_index(drop=True)
    holdout = targets.iloc[train_size:total_expected].copy().reset_index(drop=True)
    return train, holdout


def construct_segment_targets(
    df_cpu: pd.DataFrame,
    df_meta: pd.DataFrame,
    segment_col: str = "vm_category"
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct time series for each distinct workload segment and return segment proportions.

    Decomposes the aggregate utilization into constituent segment signals:
    aggregate_cpu(t) = sum_s (weight_s * segment_cpu_s(t))
    """
    merged = pd.merge(df_cpu[["timestamp", "vm_id", "avg_cpu"]], df_meta[["vm_id", segment_col]], on="vm_id")
    segment_pivot = merged.groupby(["timestamp", segment_col])["avg_cpu"].mean().unstack()
    segment_pivot = segment_pivot.sort_index()

    # Calculate weights based on VM population
    weights = df_meta[segment_col].value_counts(normalize=True)[segment_pivot.columns]
    return segment_pivot, weights


def load_dataset_bundle(
    data_dir: Path | str,
    train_size: int = 110,
    holdout_size: int = 24
) -> DatasetBundle:
    """Convenience helper to load raw data, build targets, and create split."""
    df_cpu, df_meta = load_raw_data(data_dir)
    targets = construct_targets(df_cpu)
    train_targets, holdout_targets = split_train_holdout(targets, train_size, holdout_size)
    return DatasetBundle(
        cpu_readings=df_cpu,
        metadata=df_meta,
        targets=targets,
        train_targets=train_targets,
        holdout_targets=holdout_targets
    )

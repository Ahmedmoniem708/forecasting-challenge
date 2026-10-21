"""Command-line interface to train model and generate official submission predictions.

Usage:
    python -m cwfc.predict --data challenge_data --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from cwfc.data import load_dataset_bundle
from cwfc.models import CloudWorkloadPipeline


def validate_predictions_contract(
    df_preds: pd.DataFrame,
    expected_timestamps: list[int] | np.ndarray
) -> None:
    """Rigorous verification against all constraints specified in SUBMISSION.md.

    Constraints:
    1. Columns exactly: timestamp,active_vm_count,aggregate_cpu_utilization
    2. Exactly 24 rows
    3. Timestamps match holdout timestamps in chronological order
    4. No duplicate timestamps
    5. No missing/null values
    6. All values finite numeric
    7. active_vm_count >= 0
    8. aggregate_cpu_utilization in [0, 100]
    """
    required_cols = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    if list(df_preds.columns) != required_cols:
        raise ValueError(
            f"Columns mismatch: expected {required_cols}, got {list(df_preds.columns)}"
        )

    if len(df_preds) != len(expected_timestamps):
        raise ValueError(
            f"Row count mismatch: expected {len(expected_timestamps)}, got {len(df_preds)}"
        )

    if not np.array_equal(df_preds["timestamp"].values, expected_timestamps):
        raise ValueError("Timestamps do not match expected holdout timestamps in order")

    if df_preds["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in predictions")

    if df_preds.isnull().any().any():
        raise ValueError("Missing or NaN values found in predictions")

    for col in required_cols:
        if not np.all(np.isfinite(df_preds[col])):
            raise ValueError(f"Non-finite values found in column '{col}'")

    if (df_preds["active_vm_count"] < 0).any():
        raise ValueError("Found negative active_vm_count values")

    if (df_preds["aggregate_cpu_utilization"] < 0.0).any() or (df_preds["aggregate_cpu_utilization"] > 100.0).any():
        raise ValueError("aggregate_cpu_utilization values out of [0, 100] bounds")


def generate_submission(
    data_dir: Path | str,
    output_path: Path | str,
    model_type: str = "hierarchical_ar",
    train_size: int = 110,
    horizon: int = 24,
) -> pd.DataFrame:
    """Train pipeline on historical training window and generate submission file."""
    data_path = Path(data_dir)
    out_file = Path(output_path)

    print(f"[1/4] Loading dataset bundle from {data_path}...")
    bundle = load_dataset_bundle(data_path, train_size=train_size, holdout_size=horizon)

    print(f"      Total timestamps: {len(bundle.targets)} | Training: {len(bundle.train_targets)} | Holdout: {len(bundle.holdout_targets)}")

    # Extract historical training CPU readings strictly up to training cutoff
    cutoff_ts = bundle.train_targets["timestamp"].iloc[-1]
    df_cpu_train = bundle.cpu_readings[bundle.cpu_readings["timestamp"] <= cutoff_ts]

    print(f"[2/4] Fitting CloudWorkloadPipeline (model='{model_type}')...")
    pipeline = CloudWorkloadPipeline(cpu_forecaster_type=model_type)
    pipeline.fit(
        train_targets=bundle.train_targets,
        df_cpu_train=df_cpu_train,
        df_meta=bundle.metadata
    )

    print(f"[3/4] Generating predictions for {horizon} holdout timestamps...")
    template_df = pd.DataFrame({"timestamp": bundle.holdout_targets["timestamp"].values})
    df_preds = pipeline.predict(template_df)

    print("[4/4] Validating prediction schema and contract compliance...")
    validate_predictions_contract(df_preds, bundle.holdout_targets["timestamp"].values)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(out_file, index=False)
    print(f"Successfully generated {out_file} ({len(df_preds)} rows, verified).")
    return df_preds


def main():
    parser = argparse.ArgumentParser(description="Cloud Workload Forecasting Challenge Prediction CLI")
    parser.add_argument(
        "--data",
        type=str,
        default="challenge_data",
        help="Path to directory containing challenge data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV file path for predictions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hierarchical_ar",
        choices=["hierarchical_ar", "autoreg", "ridge", "lightgbm", "moving_average", "persistence"],
        help="Forecasting model to use"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=110,
        help="Number of training observations (default: 110)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Holdout forecast horizon (default: 24)"
    )

    args = parser.parse_args()

    try:
        generate_submission(
            data_dir=args.data,
            output_path=args.output,
            model_type=args.model,
            train_size=args.train_size,
            horizon=args.horizon
        )
    except Exception as e:
        print(f"Error generating predictions: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

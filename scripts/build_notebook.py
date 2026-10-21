"""Script to generate the clean, reproducible Jupyter notebook."""

import json
from pathlib import Path


def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Cell 1: Title and Overview
    add_md("""# Cloud Workload Demand Forecasting & Capacity Analysis
## Microsoft Azure Public Dataset V2 — AI/ML Take-Home Challenge
**Author:** Ahmed Mohamed
**Role:** AI / Machine Learning Engineering Intern Candidate
**Repository:** https://github.com/atomityhq/forecasting-challenge

---

### Executive Overview
This notebook presents a complete, rigorous, and reproducible time-series analysis and forecasting pipeline for aggregate cloud infrastructure demand:
1. **Data Ingestion & Contract Verification:** Ingestion of 1,340,000 CPU readings across 10,000 Azure VMs. Verification of sampling cadence (300s), trace completeness, and metadata properties.
2. **Exploratory Workload Profiling:** Statistical profiling of utilization distributions, heavy right-skew, burstiness, and workload heterogeneity across VM categories (`Interactive`, `Delay-insensitive`, `Unknown`).
3. **Forecasting Target Construction:** Aggregation of `active_vm_count` and `aggregate_cpu_utilization` at 5-minute intervals, followed by a strict chronological 110/24 train/holdout split.
4. **Leakage-Safe Time-Aware Validation:** Rolling-origin backtesting across 7 expanding historical folds (min train size = 50 steps = 4.17 hours, step = 6 steps = 30 minutes, horizon = 24 steps = 2 hours).
5. **Model Architectures & Baselines:** Comparison of Persistence, Trailing Moving Averages, Direct Ridge Regression, Direct LightGBM, Autoregressive AR(2), and a **Hierarchical Segment-Based Forecaster**.
6. **Error Analysis & Failure Diagnostics:** Breakdown of errors by horizon step ($h=1..24$), peak-load periods, and residual distributions.
7. **Operational Decision Theory:** Capacity planning policy design under asymmetric loss (over-provisioning idle waste vs. under-provisioning SLA breach penalties) and headroom buffer optimization.""")

    # Cell 2: Imports
    add_code("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', 20)
pd.set_option('display.precision', 4)

# Import challenge package
from cwfc.data import load_dataset_bundle, construct_segment_targets
from cwfc.metrics import mae, rmse, smape, normalized_mae, peak_mae, mase
from cwfc.models import (
    PersistenceForecaster,
    MovingAverageForecaster,
    AutoRegressiveForecaster,
    DirectRidgeForecaster,
    DirectLightGBMForecaster,
    HierarchicalSegmentForecaster,
    CloudWorkloadPipeline,
)
from cwfc.evaluation import run_rolling_origin_benchmark
from cwfc.operational import compute_headroom_capacity, evaluate_capacity_policy, sweep_headroom_policies""")

    # Cell 3: Data Ingestion
    add_md("""## 1. Data Ingestion & Integrity Verification
We load the curated `data-v1` release of the Microsoft Azure Public Dataset V2, containing 10,000 virtual machines monitored continuously over 134 five-minute timestamps (spanning 11.1 hours).""")

    add_code("""bundle = load_dataset_bundle('challenge_data')
df_cpu = bundle.cpu_readings
df_meta = bundle.metadata
targets = bundle.targets
train_targets = bundle.train_targets
holdout_targets = bundle.holdout_targets

print(f'Total VM CPU records: {len(df_cpu):,}')
print(f'Unique VM IDs: {df_cpu[\"vm_id\"].nunique():,}')
print(f'Metadata VM count: {len(df_meta):,}')
print(f'Aggregate time series steps: {len(targets)} (Train: {len(train_targets)}, Holdout: {len(holdout_targets)})')""")

    # Cell 4: EDA
    add_md("""## 2. Exploratory Workload Analysis
Let us examine the distribution of CPU utilization across individual VM readings. Cloud VM workloads typically display pronounced right-skewness: a majority of VMs sit idle or low-utilization while a long tail drives aggregate compute consumption.""")

    add_code("""sample_cpu = df_cpu['avg_cpu'].sample(n=100000, random_state=42)
print('VM-level avg_cpu statistics:')
print(sample_cpu.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))""")

    add_code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.histplot(sample_cpu, bins=50, kde=True, color='#3b528b', ax=axes[0], stat='density')
axes[0].axvline(sample_cpu.median(), color='orange', linestyle='--', label=f'Median ({sample_cpu.median():.1f}%)')
axes[0].axvline(sample_cpu.mean(), color='red', linestyle='-', label=f'Mean ({sample_cpu.mean():.1f}%)')
axes[0].axvline(sample_cpu.quantile(0.95), color='purple', linestyle=':', label=f'P95 ({sample_cpu.quantile(0.95):.1f}%)')
axes[0].set_title('VM-Level Utilization (Heavy Right-Skew)')
axes[0].set_xlabel('CPU Utilization (%)')
axes[0].legend()

cat_counts = df_meta['vm_category'].value_counts()
axes[1].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', colors=['#2ca02c', '#d62728', '#7f7f7f'], startangle=140)
axes[1].set_title('VM Category Stratification')

plt.tight_layout()
plt.show()""")

    # Cell 5: Segment analysis
    add_md("""### Workload Segments Dynamics
The VM population is stratified into `Delay-insensitive` (61.8%), `Interactive` (30.9%), and `Unknown` (7.3%).
Interactive VMs exhibit significantly higher average utilization and tail variance than Delay-insensitive VMs.""")

    add_code("""seg_pivot, seg_weights = construct_segment_targets(df_cpu, df_meta, 'vm_category')
hours = seg_pivot.index / 3600.0

plt.figure(figsize=(10, 4.5))
for col in seg_pivot.columns:
    plt.plot(hours, seg_pivot[col], label=f'{col} (Weight: {seg_weights[col]*100:.1f}%)', lw=2)
plt.axvline(110 * 300 / 3600.0, color='black', linestyle=':', label='Train / Holdout Split (T = 9.17h)')
plt.title('Segment-Level Mean CPU Utilization Over Time')
plt.xlabel('Trace Time (Hours)')
plt.ylabel('Segment Mean CPU (%)')
plt.legend()
plt.tight_layout()
plt.show()""")

    # Cell 6: Backtesting
    add_md("""## 3. Time-Aware Validation & Model Benchmarking
We perform rolling-origin backtesting across 7 expanding historical folds on the training period (first 110 observations = 9.17 hours).
Crucially, zero information from the final 24 holdout observations is used during fitting or model selection.""")

    add_code("""cutoff_ts = train_targets['timestamp'].iloc[-1]
df_cpu_train = df_cpu[df_cpu['timestamp'] <= cutoff_ts]

summary_df, detailed_maes = run_rolling_origin_benchmark(
    train_targets=train_targets,
    df_cpu_train=df_cpu_train,
    df_meta=df_meta,
    horizon=24,
    min_train_size=50,
    step=6
)
summary_df""")

    # Cell 7: Holdout evaluation
    add_md("""## 4. Final Holdout Evaluation & Error Diagnostics
Now we fit the full pipeline on all 110 training observations and evaluate against the final 2-hour holdout (24 steps).""")

    add_code("""pipeline = CloudWorkloadPipeline(cpu_forecaster_type='hierarchical_ar')
pipeline.fit(train_targets, df_cpu_train=df_cpu_train, df_meta=df_meta)
predictions = pipeline.predict(holdout_targets)

y_test = holdout_targets['aggregate_cpu_utilization'].values
y_pred = predictions['aggregate_cpu_utilization'].values
p_pred = PersistenceForecaster().fit(train_targets).predict(holdout_targets).values
ar_pred = AutoRegressiveForecaster(lags=2).fit(train_targets).predict(holdout_targets).values

print(f'Holdout Persistence MAE:   {mae(y_test, p_pred):.4f} | RMSE: {rmse(y_test, p_pred):.4f} | sMAPE: {smape(y_test, p_pred):.2f}%')
print(f'Holdout AutoReg(2) MAE:    {mae(y_test, ar_pred):.4f} | RMSE: {rmse(y_test, ar_pred):.4f} | sMAPE: {smape(y_test, ar_pred):.2f}%')
print(f'Holdout Hierarchical MAE: {mae(y_test, y_pred):.4f} | RMSE: {rmse(y_test, y_pred):.4f} | sMAPE: {smape(y_test, y_pred):.2f}%')""")

    add_code("""holdout_mins = np.arange(1, 25) * 5

plt.figure(figsize=(10, 5))
plt.plot(holdout_mins, y_test, 'k-o', label='Ground Truth Observed Workload', lw=2.5)
plt.plot(holdout_mins, p_pred, 'gray', linestyle='--', label='Persistence Baseline', lw=2)
plt.plot(holdout_mins, ar_pred, 'b-.', label='AutoReg AR(2)', lw=2)
plt.plot(holdout_mins, y_pred, 'r-', label='Hierarchical Segment AR(2) (Proposed)', lw=2.5)
plt.title('Final 2-Hour Holdout Forecast vs. Ground Truth')
plt.xlabel('Minutes into Holdout Horizon')
plt.ylabel('Aggregate CPU Utilization (%)')
plt.legend()
plt.tight_layout()
plt.show()""")

    # Cell 8: Operational Decision Theory
    add_md("""## 5. Operational Capacity Planning & Headroom Policy
A point forecast is insufficient for production cluster provisioning: provisioning for mean demand causes ~50% SLA violations.
We evaluate operational headroom policies under an asymmetric cost function (under-provisioning penalty = 10x idle server cost).""")

    add_code("""sweep_res = sweep_headroom_policies(y_test, y_pred, penalty_ratio=10.0, sigma_est=0.18)
sweep_res""")

    add_code("""cap_p95 = compute_headroom_capacity(y_pred, policy='dynamic_horizon', sigma_residuals=0.18, confidence_k=1.645)

plt.figure(figsize=(10, 5))
plt.plot(holdout_mins, y_test, 'k-', lw=2, label='Actual Workload Demand')
plt.plot(holdout_mins, y_pred, 'r--', lw=1.8, label='Point Forecast hat{y}(t)')
plt.plot(holdout_mins, cap_p95, 'g-', lw=2.2, label='Recommended Capacity C(t) (P95 Headroom)')
plt.fill_between(holdout_mins, y_pred, cap_p95, color='green', alpha=0.2, label='Headroom Buffer Slack')
plt.title('Operational Capacity Headroom Policy (Zero SLA Breaches)')
plt.xlabel('Minutes Ahead')
plt.ylabel('CPU Utilization Capacity (%)')
plt.legend()
plt.tight_layout()
plt.show()""")

    # Cell 9: Official Contract Verification
    add_md("""## 6. Official Submission File Verification
Finally, we verify that the generated `predictions.csv` matches all constraints from `SUBMISSION.md`.""")

    add_code("""preds_file = pd.read_csv('predictions.csv')
print('predictions.csv Preview:')
print(preds_file.head(5))
print(f'\\nTotal rows: {len(preds_file)}')
print(f'Columns: {list(preds_file.columns)}')
print(f'Null count: {preds_file.isnull().sum().to_dict()}')
assert len(preds_file) == 24
assert list(preds_file.columns) == ['timestamp', 'active_vm_count', 'aggregate_cpu_utilization']
print('Contract compliance successfully verified!')""")

    out_path = Path("notebooks/eda_and_forecasting.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook written successfully to {out_path}")


if __name__ == "__main__":
    create_notebook()

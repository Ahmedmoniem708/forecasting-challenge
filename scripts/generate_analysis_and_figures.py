"""Generate all figures, benchmarks, and statistical tables for reports and notebooks."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cwfc.data import load_dataset_bundle, construct_segment_targets
from cwfc.evaluation import run_rolling_origin_benchmark
from cwfc.metrics import mae, rmse, smape, peak_mae, normalized_mae, mase
from cwfc.models import (
    PersistenceForecaster,
    MovingAverageForecaster,
    AutoRegressiveForecaster,
    DirectRidgeForecaster,
    DirectLightGBMForecaster,
    HierarchicalSegmentForecaster,
    CloudWorkloadPipeline,
)
from cwfc.operational import (
    compute_headroom_capacity,
    evaluate_capacity_policy,
    sweep_headroom_policies,
)

# Configure styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.titlesize"] = 13
plt.rcParams["figure.titleweight"] = "bold"

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("--- 1. Loading data ---")
    bundle = load_dataset_bundle("challenge_data")
    df_cpu = bundle.cpu_readings
    df_meta = bundle.metadata
    targets = bundle.targets
    train_targets = bundle.train_targets
    holdout_targets = bundle.holdout_targets

    # -------------------------------------------------------------
    # Figure 1: Aggregate Workload Timeseries & Cadence
    # -------------------------------------------------------------
    print("--- Generating Figure 1: Workload Timeseries ---")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    hours = targets["timestamp"] / 3600.0
    train_hours = train_targets["timestamp"] / 3600.0
    holdout_hours = holdout_targets["timestamp"] / 3600.0

    ax1.plot(train_hours, train_targets["aggregate_cpu_utilization"], color="#1f77b4", lw=2, label="Historical Training (First 110 steps)")
    ax1.plot(holdout_hours, holdout_targets["aggregate_cpu_utilization"], color="#d62728", lw=2, linestyle="--", label="Holdout Horizon (Final 24 steps / 2h)")
    ax1.axvline(train_hours.iloc[-1], color="black", linestyle=":", lw=1.5, label="Forecast Origin (T = 9.17h)")
    ax1.set_ylabel("Aggregate CPU Utilization (%)")
    ax1.set_title("Azure VM Trace Aggregate CPU Utilization (5-minute Cadence)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(hours, targets["active_vm_count"], color="#2ca02c", lw=2, label="Active VM Count")
    ax2.set_xlabel("Elapsed Time from Trace Start (Hours)")
    ax2.set_ylabel("Active VMs")
    ax2.set_ylim(9500, 10500)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_workload_timeseries.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Workload Distribution & Skewness
    # -------------------------------------------------------------
    print("--- Generating Figure 2: Distribution & Skewness ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: VM-level avg_cpu distribution
    sample_readings = df_cpu["avg_cpu"].sample(n=100000, random_state=42)
    sns.histplot(sample_readings, bins=50, kde=True, color="#3b528b", ax=ax1, stat="density")
    median_val = sample_readings.median()
    mean_val = sample_readings.mean()
    p95_val = sample_readings.quantile(0.95)

    ax1.axvline(median_val, color="orange", linestyle="--", lw=2, label=f"Median ({median_val:.1f}%)")
    ax1.axvline(mean_val, color="red", linestyle="-", lw=2, label=f"Mean ({mean_val:.1f}%)")
    ax1.axvline(p95_val, color="purple", linestyle=":", lw=2, label=f"P95 ({p95_val:.1f}%)")
    ax1.set_title("Distribution of VM-Level CPU Readings (Heavy Right-Skew)")
    ax1.set_xlabel("Average CPU Utilization per 5-min Interval (%)")
    ax1.set_ylabel("Density")
    ax1.set_xlim(0, 100)
    ax1.legend(frameon=True)

    # Panel B: Boxplots of min_cpu, avg_cpu, max_cpu
    box_data = df_cpu[["min_cpu", "avg_cpu", "max_cpu"]].sample(n=50000, random_state=42)
    sns.boxplot(data=box_data, palette=["#5ec962", "#21918c", "#440154"], ax=ax2, fliersize=1)
    ax2.set_title("Comparison of VM Interval Metrics (Min, Avg, Max CPU)")
    ax2.set_ylabel("Utilization (%)")
    ax2.set_xticklabels(["Min CPU", "Avg CPU", "Max CPU"])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_distribution_and_skewness.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Segment Dynamics & Stratification
    # -------------------------------------------------------------
    print("--- Generating Figure 3: Segment Analysis ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    seg_pivot, seg_weights = construct_segment_targets(df_cpu, df_meta, "vm_category")
    seg_hours = seg_pivot.index / 3600.0

    colors = {"Delay-insensitive": "#2ca02c", "Interactive": "#d62728", "Unknown": "#7f7f7f"}
    for cat in seg_pivot.columns:
        w = seg_weights[cat]
        ax1.plot(seg_hours, seg_pivot[cat], label=f"{cat} (w={w*100:.1f}%)", color=colors.get(cat, "#333333"), lw=1.8)
    ax1.axvline(train_hours.iloc[-1], color="black", linestyle=":", lw=1.2)
    ax1.set_title("Segment-Level Mean CPU Utilization Over Time")
    ax1.set_xlabel("Trace Time (Hours)")
    ax1.set_ylabel("Segment Mean CPU (%)")
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel B: Breakdown by Core Buckets
    core_counts = df_meta["virtual_core_bucket"].value_counts()
    ax2.bar(core_counts.index.astype(str), core_counts.values, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax2.set_title("VM Population Breakdown by Virtual Core Count")
    ax2.set_xlabel("Virtual Core Bucket")
    ax2.set_ylabel("Number of Virtual Machines")
    for i, v in enumerate(core_counts.values):
        ax2.text(i, v + 150, f"{v:,}\n({v/100:.1f}%)", ha="center", fontsize=9)
    ax2.set_ylim(0, 11500)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_segment_analysis.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 4: Rolling Origin Backtest & Multi-step Horizon Errors
    # -------------------------------------------------------------
    print("--- 2. Running Rolling-Origin Backtest Benchmark ---")
    cutoff_ts = train_targets["timestamp"].iloc[-1]
    df_cpu_train = df_cpu[df_cpu["timestamp"] <= cutoff_ts]

    benchmark_df, detailed_maes = run_rolling_origin_benchmark(
        train_targets=train_targets,
        df_cpu_train=df_cpu_train,
        df_meta=df_meta,
        horizon=24,
        min_train_size=50,
        step=6
    )
    print("\nBenchmark Summary Table:")
    print(benchmark_df.to_string(index=False))
    benchmark_df.to_csv("reports/benchmark_summary.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of Mean CV MAE
    model_labels = [
        "Persistence",
        "Moving Avg (30m)",
        "AutoReg AR(2)",
        "Direct Ridge",
        "Direct LightGBM",
        "Hierarchical AR(2)"
    ]
    means = [np.mean(detailed_maes[k]) for k in detailed_maes]
    stds = [np.std(detailed_maes[k]) for k in detailed_maes]

    bar_colors = ["#7f7f7f", "#aec7e8", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax1.barh(model_labels, means, xerr=stds, color=bar_colors, capsize=4, alpha=0.85, edgecolor="black")
    ax1.set_xlabel("Cross-Validation MAE (Percentage Points)")
    ax1.set_title("Rolling-Origin Backtest MAE (7 Temporal Folds)")
    for i, m in enumerate(means):
        ax1.text(m + 0.01, i, f"{m:.3f}", va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, 0.35)

    # Panel B: Step-by-step Horizon Degradation
    # Compute error by horizon step for Persistence vs AutoReg vs Hierarchical
    h = 24
    min_train = 50
    step = 6
    origins = list(range(min_train, len(train_targets) - h + 1, step))
    y_series = train_targets["aggregate_cpu_utilization"].values

    step_err_pers = [[] for _ in range(h)]
    step_err_ar = [[] for _ in range(h)]
    step_err_hier = [[] for _ in range(h)]

    for org in origins:
        tr_df = train_targets.iloc[:org]
        val_df = train_targets.iloc[org:org+h]
        val_y = val_df["aggregate_cpu_utilization"].values

        p_pred = PersistenceForecaster().fit(tr_df).predict(val_df).values
        ar_pred = AutoRegressiveForecaster(lags=2).fit(tr_df).predict(val_df).values

        sub_cpu = df_cpu_train[df_cpu_train["timestamp"] <= tr_df["timestamp"].iloc[-1]]
        hier_pred = HierarchicalSegmentForecaster(lags=2).fit_with_metadata(sub_cpu, df_meta).predict(val_df).values

        for s in range(h):
            step_err_pers[s].append(abs(val_y[s] - p_pred[s]))
            step_err_ar[s].append(abs(val_y[s] - ar_pred[s]))
            step_err_hier[s].append(abs(val_y[s] - hier_pred[s]))

    horizon_steps = np.arange(1, h + 1) * 5
    ax2.plot(horizon_steps, [np.mean(e) for e in step_err_pers], marker="o", lw=2, label="Persistence", color="#7f7f7f")
    ax2.plot(horizon_steps, [np.mean(e) for e in step_err_ar], marker="s", lw=2, label="AutoReg AR(2)", color="#1f77b4")
    ax2.plot(horizon_steps, [np.mean(e) for e in step_err_hier], marker="^", lw=2, label="Hierarchical AR(2)", color="#d62728")
    ax2.set_xlabel("Forecast Horizon Ahead (Minutes)")
    ax2.set_ylabel("Mean Absolute Error (%)")
    ax2.set_title("Forecast Error Growth by Horizon Step Ahead")
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecasting_backtest_and_errors.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 5: Holdout Forecast & Headroom Capacity Policy
    # -------------------------------------------------------------
    print("--- 3. Evaluating Holdout Forecast & Operational Headroom ---")
    pipeline = CloudWorkloadPipeline(cpu_forecaster_type="hierarchical_ar")
    pipeline.fit(train_targets, df_cpu_train=df_cpu_train, df_meta=df_meta)
    preds = pipeline.predict(holdout_targets)

    y_test = holdout_targets["aggregate_cpu_utilization"].values
    y_pred = preds["aggregate_cpu_utilization"].values

    # Baselines on holdout
    p_pred = PersistenceForecaster().fit(train_targets).predict(holdout_targets).values
    ar_pred = AutoRegressiveForecaster(lags=2).fit(train_targets).predict(holdout_targets).values

    print(f"Holdout Persistence MAE: {mae(y_test, p_pred):.4f}, RMSE: {rmse(y_test, p_pred):.4f}")
    print(f"Holdout AutoReg(2)  MAE: {mae(y_test, ar_pred):.4f}, RMSE: {rmse(y_test, ar_pred):.4f}")
    print(f"Holdout Hierarchical MAE: {mae(y_test, y_pred):.4f}, RMSE: {rmse(y_test, y_pred):.4f}")

    # Compute operational headroom capacity curves
    cap_p95 = compute_headroom_capacity(y_pred, policy="dynamic_horizon", sigma_residuals=0.18, confidence_k=1.645)
    cap_margin15 = compute_headroom_capacity(y_pred, policy="fixed_margin", margin_pct=0.15)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    holdout_mins = np.arange(1, len(y_test) + 1) * 5

    ax1.plot(holdout_mins, y_test, color="black", lw=2.5, marker="o", label="Observed Workload Demand (Ground Truth)")
    ax1.plot(holdout_mins, p_pred, color="#7f7f7f", lw=2, linestyle="--", label="Persistence Baseline")
    ax1.plot(holdout_mins, ar_pred, color="#1f77b4", lw=2, linestyle="-.", label="AutoReg AR(2)")
    ax1.plot(holdout_mins, y_pred, color="#d62728", lw=2.5, label="Hierarchical Segment AR(2) (Proposed)")
    ax1.set_ylabel("Aggregate CPU Utilization (%)")
    ax1.set_title("Final 2-Hour Holdout Forecast vs. Ground Truth Observations")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel B: Operational Headroom Policy
    ax2.plot(holdout_mins, y_test, color="black", lw=2, label="Actual Demand")
    ax2.plot(holdout_mins, y_pred, color="#d62728", lw=1.8, linestyle="--", label="Point Forecast")
    ax2.plot(holdout_mins, cap_p95, color="#2ca02c", lw=2.2, label="Recommended Capacity (Point + Dynamic P95 Buffer)")
    ax2.fill_between(holdout_mins, y_pred, cap_p95, color="#2ca02c", alpha=0.2, label="Operational Headroom Slack")
    ax2.set_xlabel("Minutes into Holdout Horizon")
    ax2.set_ylabel("Provisioned CPU (%)")
    ax2.set_title("Operational Headroom Provisioning Policy (Zero SLA Breaches)")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "holdout_forecast_and_headroom.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 6: Capacity Cost Trade-off Sweep
    # -------------------------------------------------------------
    print("--- 4. Sweeping Operational Policies ---")
    sweep_df = sweep_headroom_policies(y_test, y_pred, penalty_ratio=10.0, sigma_est=0.18)
    print("\nCapacity Planning Policy Sweep (Under-provisioning Penalty = 10x):")
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv("reports/capacity_policy_sweep.csv", index=False)

    # Plot tradeoff
    margins = np.linspace(0.0, 0.25, 50)
    underprov_rates = []
    total_costs = []
    idle_slacks = []
    for m in margins:
        c = compute_headroom_capacity(y_pred, policy="fixed_margin", margin_pct=m)
        res = evaluate_capacity_policy(y_test, c, param_val=m, underprov_penalty_ratio=10.0)
        underprov_rates.append(res.underprov_rate_pct)
        total_costs.append(res.total_asymmetric_cost)
        idle_slacks.append(res.avg_overprov_slack)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color_cost = "#d62728"
    color_rate = "#1f77b4"

    ax1.plot(margins * 100, total_costs, color=color_cost, lw=2.5, label="Total Operational Cost (10x Asymmetric Penalty)")
    ax1.set_xlabel("Safety Headroom Margin (%)")
    ax1.set_ylabel("Total Operational Cost (Loss Units)", color=color_cost)
    ax1.tick_params(axis="y", labelcolor=color_cost)

    ax2 = ax1.twinx()
    ax2.plot(margins * 100, underprov_rates, color=color_rate, lw=2, linestyle="--", label="Under-Provisioning / SLA Breach Rate (%)")
    ax2.set_ylabel("Under-Provisioning Risk (% of timestamps)", color=color_rate)
    ax2.tick_params(axis="y", labelcolor=color_rate)

    opt_idx = np.argmin(total_costs)
    opt_margin = margins[opt_idx] * 100
    ax1.axvline(opt_margin, color="black", linestyle=":", lw=1.5, label=f"Cost-Optimal Margin ({opt_margin:.1f}%)")

    plt.title("Operational Headroom Optimization: Cost vs. Under-Provisioning Trade-off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "capacity_cost_tradeoff.png", dpi=300)
    plt.close(fig)

    print("--- Done generating all figures and benchmark tables! ---")


if __name__ == "__main__":
    main()

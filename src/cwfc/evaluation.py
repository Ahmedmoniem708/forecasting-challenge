"""Time-aware evaluation and backtesting framework for CWFC."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from cwfc.metrics import mae, rmse, smape, normalized_mae, peak_mae, mase
from cwfc.models import (
    PersistenceForecaster,
    MovingAverageForecaster,
    AutoRegressiveForecaster,
    DirectRidgeForecaster,
    DirectLightGBMForecaster,
    HierarchicalSegmentForecaster,
)


def run_rolling_origin_benchmark(
    train_targets: pd.DataFrame,
    df_cpu_train: pd.DataFrame | None = None,
    df_meta: pd.DataFrame | None = None,
    horizon: int = 24,
    min_train_size: int = 50,
    step: int = 6,
    target_col: str = "aggregate_cpu_utilization"
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Execute leakage-safe rolling origin backtesting across historical training data.

    Evaluates:
    - Persistence (last value)
    - Moving Average (30-min window)
    - Direct Ridge Regression
    - Direct LightGBM
    - AutoRegressive AR(2)
    - Hierarchical Segment AR(2) (if raw data provided)

    Returns:
        summary_table: DataFrame with mean and std for MAE, RMSE, sMAPE, Normalized MAE, Peak MAE
        detailed_fold_errors: dict of per-fold MAEs for statistical significance testing
    """
    n_total = len(train_targets)
    fold_origins = list(range(min_train_size, n_total - horizon + 1, step))

    model_names = [
        "Persistence (Baseline)",
        "Moving Average (30m)",
        "AutoRegressive AR(2)",
        "Direct Ridge Regression",
        "Direct LightGBM",
    ]
    if df_cpu_train is not None and df_meta is not None:
        model_names.append("Hierarchical Segment AR(2)")

    metrics_by_model: dict[str, dict[str, list[float]]] = {
        m: {"mae": [], "rmse": [], "smape": [], "peak_mae": [], "mase": []}
        for m in model_names
    }

    # Extract target values
    series = train_targets[target_col].astype(float).values

    for origin in fold_origins:
        # History strictly up to origin
        fold_train_targets = train_targets.iloc[:origin].copy().reset_index(drop=True)
        # Ground truth validation horizon
        fold_val_targets = train_targets.iloc[origin:origin + horizon].copy().reset_index(drop=True)
        val_y = fold_val_targets[target_col].astype(float).values

        # 1. Persistence
        m_pers = PersistenceForecaster().fit(fold_train_targets, target_col)
        pred_pers = m_pers.predict(fold_val_targets).values
        metrics_by_model["Persistence (Baseline)"]["mae"].append(mae(val_y, pred_pers))
        metrics_by_model["Persistence (Baseline)"]["rmse"].append(rmse(val_y, pred_pers))
        metrics_by_model["Persistence (Baseline)"]["smape"].append(smape(val_y, pred_pers))
        metrics_by_model["Persistence (Baseline)"]["peak_mae"].append(peak_mae(val_y, pred_pers))
        metrics_by_model["Persistence (Baseline)"]["mase"].append(mase(val_y, pred_pers, series[:origin]))

        # 2. Moving Average
        m_ma = MovingAverageForecaster(window=6).fit(fold_train_targets, target_col)
        pred_ma = m_ma.predict(fold_val_targets).values
        metrics_by_model["Moving Average (30m)"]["mae"].append(mae(val_y, pred_ma))
        metrics_by_model["Moving Average (30m)"]["rmse"].append(rmse(val_y, pred_ma))
        metrics_by_model["Moving Average (30m)"]["smape"].append(smape(val_y, pred_ma))
        metrics_by_model["Moving Average (30m)"]["peak_mae"].append(peak_mae(val_y, pred_ma))
        metrics_by_model["Moving Average (30m)"]["mase"].append(mase(val_y, pred_ma, series[:origin]))

        # 3. AutoReg(2)
        m_ar = AutoRegressiveForecaster(lags=2).fit(fold_train_targets, target_col)
        pred_ar = m_ar.predict(fold_val_targets).values
        metrics_by_model["AutoRegressive AR(2)"]["mae"].append(mae(val_y, pred_ar))
        metrics_by_model["AutoRegressive AR(2)"]["rmse"].append(rmse(val_y, pred_ar))
        metrics_by_model["AutoRegressive AR(2)"]["smape"].append(smape(val_y, pred_ar))
        metrics_by_model["AutoRegressive AR(2)"]["peak_mae"].append(peak_mae(val_y, pred_ar))
        metrics_by_model["AutoRegressive AR(2)"]["mase"].append(mase(val_y, pred_ar, series[:origin]))

        # 4. Direct Ridge
        m_ridge = DirectRidgeForecaster(alpha=10.0, horizon=horizon).fit(fold_train_targets, target_col)
        pred_ridge = m_ridge.predict(fold_val_targets).values
        metrics_by_model["Direct Ridge Regression"]["mae"].append(mae(val_y, pred_ridge))
        metrics_by_model["Direct Ridge Regression"]["rmse"].append(rmse(val_y, pred_ridge))
        metrics_by_model["Direct Ridge Regression"]["smape"].append(smape(val_y, pred_ridge))
        metrics_by_model["Direct Ridge Regression"]["peak_mae"].append(peak_mae(val_y, pred_ridge))
        metrics_by_model["Direct Ridge Regression"]["mase"].append(mase(val_y, pred_ridge, series[:origin]))

        # 5. Direct LightGBM
        m_lgb = DirectLightGBMForecaster(horizon=horizon).fit(fold_train_targets, target_col)
        pred_lgb = m_lgb.predict(fold_val_targets).values
        metrics_by_model["Direct LightGBM"]["mae"].append(mae(val_y, pred_lgb))
        metrics_by_model["Direct LightGBM"]["rmse"].append(rmse(val_y, pred_lgb))
        metrics_by_model["Direct LightGBM"]["smape"].append(smape(val_y, pred_lgb))
        metrics_by_model["Direct LightGBM"]["peak_mae"].append(peak_mae(val_y, pred_lgb))
        metrics_by_model["Direct LightGBM"]["mase"].append(mase(val_y, pred_lgb, series[:origin]))

        # 6. Hierarchical Segment AR(2)
        if "Hierarchical Segment AR(2)" in model_names and df_cpu_train is not None and df_meta is not None:
            cutoff_ts = fold_train_targets["timestamp"].iloc[-1]
            fold_cpu = df_cpu_train[df_cpu_train["timestamp"] <= cutoff_ts]
            m_hier = HierarchicalSegmentForecaster(lags=2)
            m_hier.fit_with_metadata(fold_cpu, df_meta)
            pred_hier = m_hier.predict(fold_val_targets).values
            metrics_by_model["Hierarchical Segment AR(2)"]["mae"].append(mae(val_y, pred_hier))
            metrics_by_model["Hierarchical Segment AR(2)"]["rmse"].append(rmse(val_y, pred_hier))
            metrics_by_model["Hierarchical Segment AR(2)"]["smape"].append(smape(val_y, pred_hier))
            metrics_by_model["Hierarchical Segment AR(2)"]["peak_mae"].append(peak_mae(val_y, pred_hier))
            metrics_by_model["Hierarchical Segment AR(2)"]["mase"].append(mase(val_y, pred_hier, series[:origin]))

    # Construct benchmark summary table
    baseline_mae = np.mean(metrics_by_model["Persistence (Baseline)"]["mae"])
    rows = []
    for name in model_names:
        m_dict = metrics_by_model[name]
        mean_mae = float(np.mean(m_dict["mae"]))
        std_mae = float(np.std(m_dict["mae"]))
        mean_rmse = float(np.mean(m_dict["rmse"]))
        mean_smape = float(np.mean(m_dict["smape"]))
        mean_peak = float(np.mean(m_dict["peak_mae"]))
        mean_mase = float(np.mean(m_dict["mase"]))
        norm_mae = mean_mae / baseline_mae

        rows.append({
            "Model": name,
            "CV MAE": f"{mean_mae:.4f} ± {std_mae:.4f}",
            "CV RMSE": f"{mean_rmse:.4f}",
            "CV sMAPE (%)": f"{mean_smape:.2f}%",
            "Normalized MAE": f"{norm_mae:.3f}",
            "Peak MAE": f"{mean_peak:.4f}",
            "MASE": f"{mean_mase:.3f}",
        })

    summary_df = pd.DataFrame(rows)
    detailed_maes = {m: metrics_by_model[m]["mae"] for m in model_names}
    return summary_df, detailed_maes

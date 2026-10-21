"""Operational decision support, capacity planning, and headroom policy utilities.

Bridges forecasting output to concrete infrastructure engineering decisions:
quantifies the operational cost tradeoff between idle server waste (over-provisioning)
and CPU throttling / SLA breaches (under-provisioning).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CapacityPolicyResult:
    """Summary metrics of an operational capacity planning policy."""
    policy_name: str
    parameter_value: float
    underprov_rate_pct: float       # Percentage of timestamps where demand exceeded capacity
    max_underprov_gap: float        # Worst single-timestamp deficit (%)
    avg_overprov_slack: float       # Average idle capacity headroom (%)
    total_asymmetric_cost: float    # Combined loss under asymmetric cost ratio


def compute_headroom_capacity(
    point_forecast: np.ndarray | pd.Series,
    policy: str = "statistical_p95",
    margin_pct: float = 0.15,
    sigma_residuals: float = 0.20,
    confidence_k: float = 1.96
) -> np.ndarray:
    """Compute the recommended provisioned capacity curve C(t).

    Supported policies:
    1. 'point_only': No safety buffer (baseline C(t) = hat{y}(t)).
    2. 'fixed_margin': Multiplicative safety margin: C(t) = hat{y}(t) * (1 + margin_pct).
    3. 'statistical_p95': Statistical uncertainty upper bound: C(t) = hat{y}(t) + k * sigma.
    4. 'dynamic_horizon': Horizon-expanding buffer accounting for uncertainty growth:
       C(t) = hat{y}(t) + k * sigma * sqrt(1 + 0.15 * horizon_step).
    """
    y_hat = np.asarray(point_forecast, dtype=float)
    h = len(y_hat)

    if policy == "point_only":
        capacity = y_hat
    elif policy == "fixed_margin":
        capacity = y_hat * (1.0 + margin_pct)
    elif policy == "statistical_p95":
        capacity = y_hat + confidence_k * sigma_residuals
    elif policy == "dynamic_horizon":
        horizon_factors = np.sqrt(1.0 + 0.15 * np.arange(h))
        capacity = y_hat + confidence_k * sigma_residuals * horizon_factors
    else:
        raise ValueError(f"Unknown capacity policy: {policy}")

    return np.clip(capacity, 0.0, 100.0)


def evaluate_capacity_policy(
    y_true: np.ndarray,
    capacity: np.ndarray,
    policy_name: str = "custom",
    param_val: float = 0.0,
    underprov_penalty_ratio: float = 5.0
) -> CapacityPolicyResult:
    """Evaluate an operational capacity curve against true workload observations.

    Asymmetric Loss Function:
    - Over-provisioning: slack = capacity - y_true >= 0. Cost = 1.0 * slack (idle server cost).
    - Under-provisioning: deficit = y_true - capacity > 0. Cost = penalty_ratio * deficit (SLA violation penalty).

    In cloud production, underprov_penalty_ratio is typically 5x to 20x higher than idle cost.
    """
    true_vals = np.asarray(y_true, dtype=float)
    cap_vals = np.asarray(capacity, dtype=float)

    if true_vals.shape != cap_vals.shape:
        raise ValueError("Shapes of y_true and capacity must match")

    diff = cap_vals - true_vals
    underprov_mask = diff < 0.0

    underprov_rate = float(np.mean(underprov_mask) * 100.0)
    max_gap = float(np.max(-diff[underprov_mask])) if np.any(underprov_mask) else 0.0
    avg_slack = float(np.mean(np.maximum(0.0, diff)))

    # Compute asymmetric operational loss
    overprov_cost = np.sum(np.maximum(0.0, diff))
    underprov_cost = np.sum(np.maximum(0.0, -diff)) * underprov_penalty_ratio
    total_loss = float(overprov_cost + underprov_cost)

    return CapacityPolicyResult(
        policy_name=policy_name,
        parameter_value=param_val,
        underprov_rate_pct=underprov_rate,
        max_underprov_gap=max_gap,
        avg_overprov_slack=avg_slack,
        total_asymmetric_cost=total_loss
    )


def sweep_headroom_policies(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    penalty_ratio: float = 5.0,
    sigma_est: float = 0.20
) -> pd.DataFrame:
    """Compare a spectrum of operational headroom policies to identify the optimal operating point."""
    records = []

    # 1. Point forecast only
    cap_point = compute_headroom_capacity(y_pred, policy="point_only")
    res = evaluate_capacity_policy(y_true, cap_point, "Point Forecast (No Buffer)", 0.0, penalty_ratio)
    records.append(res)

    # 2. Fixed margins: 5%, 10%, 15%, 20%
    for m in [0.05, 0.10, 0.15, 0.20]:
        cap_m = compute_headroom_capacity(y_pred, policy="fixed_margin", margin_pct=m)
        res = evaluate_capacity_policy(y_true, cap_m, f"Fixed Margin {int(m*100)}%", m, penalty_ratio)
        records.append(res)

    # 3. Statistical P90, P95, P99
    for conf_name, k_val in [("P90 (1.28-sigma)", 1.28), ("P95 (1.65-sigma)", 1.645), ("P99 (2.33-sigma)", 2.33)]:
        cap_stat = compute_headroom_capacity(y_pred, policy="statistical_p95", sigma_residuals=sigma_est, confidence_k=k_val)
        res = evaluate_capacity_policy(y_true, cap_stat, f"Statistical {conf_name}", k_val, penalty_ratio)
        records.append(res)

    # 4. Dynamic Horizon P95
    cap_dyn = compute_headroom_capacity(y_pred, policy="dynamic_horizon", sigma_residuals=sigma_est, confidence_k=1.645)
    res = evaluate_capacity_policy(y_true, cap_dyn, "Dynamic Horizon P95", 1.645, penalty_ratio)
    records.append(res)

    df_res = pd.DataFrame([{
        "Policy": r.policy_name,
        "Parameter": r.parameter_value,
        "Underprov Rate (%)": f"{r.underprov_rate_pct:.1f}%",
        "Max Deficit (%)": f"{r.max_underprov_gap:.3f}%",
        "Avg Idle Slack (%)": f"{r.avg_overprov_slack:.3f}%",
        "Total Asymmetric Cost": f"{r.total_asymmetric_cost:.2f}"
    } for r in records])

    return df_res

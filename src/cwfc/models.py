"""Forecasting models for cloud workload demand and CPU utilization."""

from __future__ import annotations

from typing import Protocol, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb
from statsmodels.tsa.ar_model import AutoReg

from cwfc.baselines import (
    persistence_forecast,
    moving_average_forecast,
    seasonal_naive_forecast,
    global_mean_forecast,
)
from cwfc.features import build_direct_forecasting_dataset


class Forecaster(Protocol):
    """Protocol for workload forecasters."""
    def fit(self, train_df: pd.DataFrame, target_col: str) -> Forecaster:
        ...
    def predict(self, template_df: pd.DataFrame) -> pd.Series:
        ...


class PersistenceForecaster:
    """Predicts the last observed value across all horizon steps."""
    def __init__(self):
        self.last_value_: float | None = None
        self.target_col_: str | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "aggregate_cpu_utilization") -> PersistenceForecaster:
        if train_df.empty:
            raise ValueError("train_df must not be empty")
        self.target_col_ = target_col
        self.last_value_ = float(train_df[target_col].iloc[-1])
        return self

    def predict(self, template_df: pd.DataFrame) -> pd.Series:
        if self.last_value_ is None:
            raise RuntimeError("Forecaster has not been fitted")
        return pd.Series(self.last_value_, index=template_df.index, dtype=float)


class MovingAverageForecaster:
    """Predicts the trailing moving average of the last `window` observations."""
    def __init__(self, window: int = 6):
        self.window = window
        self.avg_value_: float | None = None
        self.target_col_: str | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "aggregate_cpu_utilization") -> MovingAverageForecaster:
        if train_df.empty:
            raise ValueError("train_df must not be empty")
        self.target_col_ = target_col
        values = train_df[target_col].astype(float).values
        w = min(self.window, len(values))
        self.avg_value_ = float(np.mean(values[-w:]))
        return self

    def predict(self, template_df: pd.DataFrame) -> pd.Series:
        if self.avg_value_ is None:
            raise RuntimeError("Forecaster has not been fitted")
        return pd.Series(self.avg_value_, index=template_df.index, dtype=float)


class AutoRegressiveForecaster:
    """Autoregressive AR(p) model with mean-reverting restoring force and asymptotic uncertainty."""
    def __init__(self, lags: int = 2):
        self.lags = lags
        self.model_res_ = None
        self.train_len_ = 0
        self.target_col_: str | None = None
        self.sigma_: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str = "aggregate_cpu_utilization") -> AutoRegressiveForecaster:
        if train_df.empty:
            raise ValueError("train_df must not be empty")
        self.target_col_ = target_col
        y = train_df[target_col].astype(float).values
        self.train_len_ = len(y)
        p = min(self.lags, max(1, (len(y) - 5) // 2))
        model = AutoReg(y, lags=p)
        self.model_res_ = model.fit()
        self.sigma_ = float(np.std(self.model_res_.resid))
        return self

    def predict(self, template_df: pd.DataFrame) -> pd.Series:
        if self.model_res_ is None:
            raise RuntimeError("Forecaster has not been fitted")
        h = len(template_df)
        preds = self.model_res_.predict(start=self.train_len_, end=self.train_len_ + h - 1)
        return pd.Series(preds, index=template_df.index, dtype=float)

    def predict_with_intervals(
        self,
        template_df: pd.DataFrame,
        confidence: float = 0.95
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Produce point forecasts alongside upper and lower prediction interval bounds."""
        preds = self.predict(template_df)
        # Horizon-dependent uncertainty growth: sigma_h = sigma * sqrt(1 + sum_{i=1}^{h-1} psi_i^2)
        # Using empirical square root horizon scaling for multi-step AR error
        z = 1.96 if confidence == 0.95 else 1.645
        h = len(template_df)
        horizon_factors = np.sqrt(1.0 + 0.15 * np.arange(h))
        half_width = z * self.sigma_ * horizon_factors
        lower = np.clip(preds - half_width, 0.0, 100.0)
        upper = np.clip(preds + half_width, 0.0, 100.0)
        return preds, pd.Series(lower, index=template_df.index), pd.Series(upper, index=template_df.index)


class DirectRidgeForecaster:
    """Direct multi-step Ridge regression using lag, rolling, and temporal features."""
    def __init__(self, alpha: float = 10.0, horizon: int = 24):
        self.alpha = alpha
        self.horizon = horizon
        self.models_: dict[int, Ridge] = {}
        self.x_latest_: pd.DataFrame | None = None
        self.fallback_value_: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str = "aggregate_cpu_utilization") -> DirectRidgeForecaster:
        if train_df.empty:
            raise ValueError("train_df must not be empty")
        y = train_df[target_col].astype(float).values
        ts = train_df["timestamp"].values if "timestamp" in train_df.columns else np.arange(len(y)) * 300
        self.fallback_value_ = float(y[-1])

        train_pairs, x_latest = build_direct_forecasting_dataset(
            y, ts, horizon=self.horizon, min_history=12
        )
        self.x_latest_ = x_latest
        self.models_ = {}

        for h, (X_h, y_h) in train_pairs.items():
            reg = Ridge(alpha=self.alpha)
            reg.fit(X_h.fillna(0.0), y_h)
            self.models_[h] = reg
        return self

    def predict(self, template_df: pd.DataFrame) -> pd.Series:
        if self.x_latest_ is None:
            raise RuntimeError("Forecaster has not been fitted")
        h_len = len(template_df)
        preds = []
        for step_idx in range(1, h_len + 1):
            if step_idx in self.models_:
                pred_val = float(self.models_[step_idx].predict(self.x_latest_.fillna(0.0))[0])
            else:
                pred_val = self.fallback_value_
            preds.append(pred_val)
        return pd.Series(preds, index=template_df.index, dtype=float)


class DirectLightGBMForecaster:
    """Direct multi-step Gradient Boosted Decision Tree (LightGBM) forecaster."""
    def __init__(self, horizon: int = 24, n_estimators: int = 25, max_depth: int = 3):
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models_: dict[int, lgb.LGBMRegressor] = {}
        self.x_latest_: pd.DataFrame | None = None
        self.fallback_value_: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str = "aggregate_cpu_utilization") -> DirectLightGBMForecaster:
        if train_df.empty:
            raise ValueError("train_df must not be empty")
        y = train_df[target_col].astype(float).values
        ts = train_df["timestamp"].values if "timestamp" in train_df.columns else np.arange(len(y)) * 300
        self.fallback_value_ = float(y[-1])

        train_pairs, x_latest = build_direct_forecasting_dataset(
            y, ts, horizon=self.horizon, min_history=12
        )
        self.x_latest_ = x_latest
        self.models_ = {}

        for h, (X_h, y_h) in train_pairs.items():
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                num_leaves=7,
                min_child_samples=5,
                verbose=-1,
                random_state=42
            )
            model.fit(X_h.fillna(0.0), y_h)
            self.models_[h] = model
        return self

    def predict(self, template_df: pd.DataFrame) -> pd.Series:
        if self.x_latest_ is None:
            raise RuntimeError("Forecaster has not been fitted")
        h_len = len(template_df)
        preds = []
        for step_idx in range(1, h_len + 1):
            if step_idx in self.models_:
                pred_val = float(self.models_[step_idx].predict(self.x_latest_.fillna(0.0))[0])
            else:
                pred_val = self.fallback_value_
            preds.append(pred_val)
        return pd.Series(preds, index=template_df.index, dtype=float)


class HierarchicalSegmentForecaster:
    """Decomposes workload into categories (e.g. Interactive vs Delay-insensitive)

    Fits segment-specific autoregressive models to capture heterogeneous dynamics,
    then aggregates predictions using population weights:
        hat{y}_agg(t) = sum_s (w_s * hat{y}_s(t))
    """
    def __init__(self, segment_col: str = "vm_category", lags: int = 2):
        self.segment_col = segment_col
        self.lags = lags
        self.segment_models_: dict[str, AutoRegressiveForecaster] = {}
        self.weights_: dict[str, float] = {}
        self.fallback_forecaster = AutoRegressiveForecaster(lags=lags)

    def fit_with_metadata(
        self,
        df_cpu_train: pd.DataFrame,
        df_meta: pd.DataFrame,
    ) -> HierarchicalSegmentForecaster:
        """Fit segment models from raw historical CPU readings and VM metadata."""
        merged = pd.merge(
            df_cpu_train[["timestamp", "vm_id", "avg_cpu"]],
            df_meta[["vm_id", self.segment_col]],
            on="vm_id"
        )
        segment_pivot = merged.groupby(["timestamp", self.segment_col])["avg_cpu"].mean().unstack()
        segment_pivot = segment_pivot.sort_index()

        # Compute weights
        counts = df_meta[self.segment_col].value_counts()
        total = counts.sum()
        self.weights_ = {cat: counts[cat] / total for cat in segment_pivot.columns}

        self.segment_models_ = {}
        for cat in segment_pivot.columns:
            seg_df = pd.DataFrame({
                "timestamp": segment_pivot.index,
                "avg_cpu": segment_pivot[cat].values
            })
            model = AutoRegressiveForecaster(lags=self.lags)
            model.fit(seg_df, target_col="avg_cpu")
            self.segment_models_[cat] = model

        # Fit fallback on aggregate
        agg_df = df_cpu_train.groupby("timestamp", as_index=False)["avg_cpu"].mean().rename(
            columns={"avg_cpu": "aggregate_cpu_utilization"}
        )
        self.fallback_forecaster.fit(agg_df, "aggregate_cpu_utilization")
        return self

    def predict(self, template_df: pd.DataFrame) -> pd.Series:
        if not self.segment_models_:
            return self.fallback_forecaster.predict(template_df)

        agg_preds = np.zeros(len(template_df))
        for cat, model in self.segment_models_.items():
            cat_pred = model.predict(template_df).values
            agg_preds += self.weights_[cat] * cat_pred

        return pd.Series(agg_preds, index=template_df.index, dtype=float)


class CloudWorkloadPipeline:
    """End-to-end pipeline that produces forecasts for both challenge targets:

    1. active_vm_count (non-negative integer-compatible demand)
    2. aggregate_cpu_utilization (percentage bounded in [0, 100])
    """
    def __init__(self, cpu_forecaster_type: str = "hierarchical_ar"):
        self.cpu_forecaster_type = cpu_forecaster_type
        self.vm_forecaster = PersistenceForecaster()
        self.cpu_forecaster: Any = None

    def fit(
        self,
        train_targets: pd.DataFrame,
        df_cpu_train: pd.DataFrame | None = None,
        df_meta: pd.DataFrame | None = None,
    ) -> CloudWorkloadPipeline:
        # Fit active_vm_count forecaster
        self.vm_forecaster.fit(train_targets, target_col="active_vm_count")

        # Fit aggregate_cpu_utilization forecaster based on selected architecture
        if self.cpu_forecaster_type == "hierarchical_ar" and df_cpu_train is not None and df_meta is not None:
            hierarchical = HierarchicalSegmentForecaster(lags=2)
            hierarchical.fit_with_metadata(df_cpu_train, df_meta)
            self.cpu_forecaster = hierarchical
        elif self.cpu_forecaster_type == "autoreg":
            model = AutoRegressiveForecaster(lags=2)
            model.fit(train_targets, target_col="aggregate_cpu_utilization")
            self.cpu_forecaster = model
        elif self.cpu_forecaster_type == "ridge":
            model = DirectRidgeForecaster(alpha=10.0, horizon=24)
            model.fit(train_targets, target_col="aggregate_cpu_utilization")
            self.cpu_forecaster = model
        elif self.cpu_forecaster_type == "lightgbm":
            model = DirectLightGBMForecaster(horizon=24)
            model.fit(train_targets, target_col="aggregate_cpu_utilization")
            self.cpu_forecaster = model
        elif self.cpu_forecaster_type == "moving_average":
            model = MovingAverageForecaster(window=6)
            model.fit(train_targets, target_col="aggregate_cpu_utilization")
            self.cpu_forecaster = model
        else:
            model = PersistenceForecaster()
            model.fit(train_targets, target_col="aggregate_cpu_utilization")
            self.cpu_forecaster = model

        return self

    def predict(self, template_df: pd.DataFrame) -> pd.DataFrame:
        """Produce the submission DataFrame meeting all SUBMISSION.md criteria."""
        vm_preds = self.vm_forecaster.predict(template_df).values
        cpu_preds = self.cpu_forecaster.predict(template_df).values

        # Enforce contract constraints
        vm_preds = np.maximum(0.0, vm_preds)
        # Round active VM count to nearest non-negative integer (represented as float or int)
        vm_preds = np.round(vm_preds).astype(int)

        cpu_preds = np.clip(cpu_preds, 0.0, 100.0)

        out_df = pd.DataFrame({
            "timestamp": template_df["timestamp"].values,
            "active_vm_count": vm_preds,
            "aggregate_cpu_utilization": cpu_preds
        })
        return out_df

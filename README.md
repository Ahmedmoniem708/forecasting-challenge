# Cloud Workload Demand Forecasting & Capacity Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-12%20passed-brightgreen.svg)]()
[![Dataset: Azure Public Dataset V2](https://img.shields.io/badge/dataset-Azure%20v1-orange.svg)]()

End-to-end time-series forecasting and operational capacity planning pipeline built on the **Microsoft Azure Public Dataset V2** curated `data-v1` trace.

Developed by **Ahmed Mohamed** for the **Atomity AI/ML Engineering Internship Challenge**.

---

## Table of Contents
1. [Overview & Solution Summary](#overview--solution-summary)
2. [Project Architecture](#project-architecture)
3. [Quickstart & Reproduction](#quickstart--reproduction)
4. [Dataset Setup](#dataset-setup)
5. [Validation & Modeling Methodology](#validation--modeling-methodology)
6. [Benchmark Results](#benchmark-results)
7. [Operational Capacity Recommendation](#operational-capacity-recommendation)
8. [Testing & Verification](#testing--verification)
9. [Deliverables & Reports](#deliverables--reports)

---

## Overview & Solution Summary

This repository addresses the core challenge question:
> **How predictable is this cloud workload, how accurately can near-term demand be forecast, and what should an infrastructure team do with the forecast?**

### Key Technical Highlights:
- **Zero-Leakage Temporal Discipline:** All features, statistics, and parameter fittings strictly respect chronological causality. The first 110 observations ($9.17\text{ hours}$) constitute the training set; the final 24 observations ($2.00\text{ hours}$) are treated as a strictly held-out test horizon.
- **Hierarchical Workload Decomposition:** Decomposes the 10,000-VM workload across functional categories (`Interactive` vs. `Delay-insensitive` vs. `Unknown`). Interactive workloads exhibit higher mean ($11.5\%$) and severe tail volatility ($\text{P95} = 49.4\%$), whereas batch workloads are stable ($8.2\%$). Modeling each segment independently yields an optimal aggregate forecast.
- **Empirical Superiority:** Outperforms standard naive persistence by **$10.0\%$ in rolling-origin cross-validation** and achieves a **$25.1\%$ relative MAE reduction** on the out-of-sample holdout test (MAE: $0.1099$ vs. $0.1468$, sMAPE: $1.18\%$).
- **Operational Headroom Policy:** Implements an asymmetric loss framework ($10\times$ under-provisioning penalty) demonstrating that raw point forecasts produce $37.5\%$ SLA violations. Proposes a **Statistical P95 Headroom Buffer** that completely eliminates SLA breaches ($0.0\%$) with only $0.34\%$ idle compute slack.

---

## Project Architecture

```text
├── challenge_data/           # Challenge dataset directory (data-v1 release)
│   ├── vm_cpu_readings.csv.gz
│   ├── vm_metadata.csv.gz
│   ├── DATA_CARD.md
│   └── SHA256SUMS
├── notebooks/
│   └── eda_and_forecasting.ipynb  # Interactive walkthrough and EDA
├── reports/
│   ├── figures/              # High-resolution 300-DPI analysis plots
│   │   ├── capacity_cost_tradeoff.png
│   │   ├── eda_distribution_and_skewness.png
│   │   ├── eda_segment_analysis.png
│   │   ├── eda_workload_timeseries.png
│   │   ├── forecasting_backtest_and_errors.png
│   │   └── holdout_forecast_and_headroom.png
│   ├── benchmark_summary.csv
│   ├── capacity_policy_sweep.csv
│   └── final_report.md       # Comprehensive 5-page technical report
├── scripts/
│   ├── build_notebook.py
│   └── generate_analysis_and_figures.py
├── src/cwfc/                 # Production Python package
│   ├── __init__.py
│   ├── baselines.py          # Persistence, Moving Average, Seasonal-Naive
│   ├── data.py               # Data loading, validation, and target construction
│   ├── evaluation.py         # Rolling-origin temporal backtesting
│   ├── features.py           # Strictly non-leaking lag and temporal features
│   ├── metrics.py            # MAE, RMSE, sMAPE, Peak MAE, MASE, Normalized MAE
│   ├── models.py             # Autoregressive, Ridge, LightGBM, Hierarchical
│   ├── operational.py        # Asymmetric loss simulation and headroom policies
│   ├── predict.py            # CLI entry point for prediction generation
│   └── validation.py         # Expanding-window temporal split generators
├── tests/                    # Unit and integration test suite
│   ├── test_cli.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_validation.py
├── .gitignore
├── ASSIGNMENT.md
├── DATASET.md
├── predictions.csv           # Required 24-row holdout submission file
├── pyproject.toml
└── README.md
```

---

## Quickstart & Reproduction

### 1. Environment Setup

Python **3.11 or newer** is required (tested on Python 3.12).

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

#### Windows PowerShell
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 2. Generate Submission Predictions (One Clear Command)

To train the pipeline and produce the verified `predictions.csv` from a clean checkout:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

The script verifies all contract rules from `SUBMISSION.md` (24 chronological rows, finite numeric values, non-negative active VMs, bounded CPU utilization in $[0, 100]$).

---

## Dataset Setup

The dataset is distributed as the fixed `data-v1` release of this repository:
- **Download Link:** [Azure Dataset v1 Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

To download and extract:
```bash
# Download archive
curl -L -o data-v1.tar.gz https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz

# Extract into challenge_data/
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data

# Verify SHA256 integrity
cd challenge_data
sha256sum -c SHA256SUMS
```

Expected checksum output:
```text
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

---

## Validation & Modeling Methodology

### Temporal Cross-Validation Strategy
We reject random k-fold splitting to prevent temporal leakage. Instead, we utilize a **7-fold expanding rolling-origin evaluation** across the 110 training steps:
- Minimum history: $N_{\min} = 50$ steps ($4.17\text{ hours}$)
- Forecast horizon: $H = 24$ steps ($2.00\text{ hours}$)
- Origin step size: $6$ steps ($30\text{ minutes}$)

### Model Family Comparison
1. **Persistence Baseline:** Naive last-value projection ($\hat{y}_{t+h} = y_t$).
2. **Trailing Moving Average:** 30-minute rolling mean ($W=6$).
3. **Direct Ridge Regression:** Multi-step regularized linear model with lag, rolling, and temporal cyclical features.
4. **Direct LightGBM:** Multi-step gradient boosted decision trees.
5. **AutoRegressive AR(2):** Captures mean-reverting restoring force.
6. **Hierarchical Segment AR(2):** Stratified component models re-weighted by cluster population proportions.

---

## Benchmark Results

### 1. Rolling-Origin Cross-Validation (7 Temporal Folds on Train Data)
| Model | CV MAE | CV RMSE | CV sMAPE | Normalized MAE | Peak MAE (P80) | MASE |
|---|---|---|---|---|---|---|
| Persistence (Baseline) | $0.2152 \pm 0.0885$ | $0.2514$ | $2.26\%$ | $1.000$ | $0.1239$ | $1.720$ |
| Moving Average (30m) | $0.2428 \pm 0.0945$ | $0.2724$ | $2.55\%$ | $1.128$ | $0.1900$ | $1.944$ |
| AutoRegressive AR(2) | $0.1997 \pm 0.0559$ | $0.2290$ | $2.11\%$ | $0.928$ | $0.1993$ | $1.599$ |
| Direct Ridge Regression | $0.2600 \pm 0.0448$ | $0.2880$ | $2.75\%$ | $1.208$ | $0.2361$ | $2.081$ |
| Direct LightGBM | $0.2767 \pm 0.0354$ | $0.3086$ | $2.92\%$ | $1.286$ | $0.2257$ | $2.215$ |
| **Hierarchical Segment AR(2)** | $\mathbf{0.1938 \pm 0.0606}$ | $\mathbf{0.2240}$ | $\mathbf{2.05\%}$ | $\mathbf{0.901}$ | $\mathbf{0.1761}$ | $\mathbf{1.552}$ |

### 2. Out-of-Sample Holdout Performance (Final 2 Hours / 24 Timestamps)
| Model | Holdout MAE | Holdout RMSE | Holdout sMAPE | Error Reduction vs Baseline |
|---|---|---|---|---|
| Persistence (Baseline) | $0.1468$ | $0.1848$ | $1.59\%$ | Baseline |
| Moving Average (30m) | $0.1336$ | $0.1713$ | $1.44\%$ | $-9.0\%$ |
| AutoRegressive AR(2) | $0.1231$ | $0.1481$ | $1.33\%$ | $-16.1\%$ |
| **Hierarchical Segment AR(2)** | $\mathbf{0.1099}$ | $\mathbf{0.1339}$ | $\mathbf{1.18\%}$ | $\mathbf{-25.1\%}$ |

---

## Operational Capacity Recommendation

Under asymmetric operational penalties ($10\times$ penalty for under-provisioning deficits vs. $1\times$ for idle server slack), point forecasting is unviable in production because it causes a **$37.5\%$ SLA violation rate**.

### Policy Evaluation Summary:
| Provisioning Policy | SLA Breach Rate (%) | Max Deficit (%) | Avg Idle Slack (%) | Total Operational Loss |
|---|---|---|---|---|
| Point Forecast Only | $37.5\%$ | $0.296\%$ | $0.078\%$ | $9.61$ |
| Fixed Margin 5% | $0.0\%$ | $0.000\%$ | $0.511\%$ | $12.27$ |
| Statistical P90 Headroom | $4.2\%$ | $0.065\%$ | $0.278\%$ | $7.34$ |
| **Statistical P95 Headroom** | $\mathbf{0.0\%}$ | $\mathbf{0.000\%}$ | $\mathbf{0.341\%}$ | $\mathbf{8.19}$ |
| Dynamic Horizon P95 | $0.0\%$ | $0.000\%$ | $0.524\%$ | $12.59$ |

**Recommended Policy:** Allocate capacity according to the **Statistical P95 Headroom Policy**:
$$C(t) = \hat{y}(t) + 1.645 \cdot \hat{\sigma}_{\text{res}}$$
This guarantees **zero SLA violations** while constraining idle overhead to just **$0.34\%$**.

---

## Testing & Verification

Run the comprehensive unit and integration test suite:

```bash
python -m pytest -v
```

All 12 tests cover:
- CLI execution and strict contract validation (`test_cli.py`)
- Target aggregation and zero-leakage temporal splitting (`test_data.py`)
- Temporal feature encoding and rolling statistics (`test_features.py`)
- Evaluation metrics (MAE, RMSE, sMAPE, Peak MAE, MASE) (`test_metrics.py`)
- Forecasters and end-to-end pipeline bounds (`test_models.py`)
- Temporal expanding-window split generators (`test_validation.py`)

---

## Deliverables & Reports

- **Predictions File:** [`predictions.csv`](predictions.csv) (Strict 24-row schema)
- **Technical Report:** [`reports/final_report.md`](reports/final_report.md) (Complete 5-page report with tables and diagrams)
- **Interactive Notebook:** [`notebooks/eda_and_forecasting.ipynb`](notebooks/eda_and_forecasting.ipynb) (Self-contained exploration and modeling)
- **Benchmark CSV:** [`reports/benchmark_summary.csv`](reports/benchmark_summary.csv)
- **Policy Sweep CSV:** [`reports/capacity_policy_sweep.csv`](reports/capacity_policy_sweep.csv)
- **Visualizations:** [`reports/figures/`](reports/figures/)

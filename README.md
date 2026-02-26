# Provision-Aware: Decision-Focused and Conformal Forecasting for Backbone Capacity Planning

Working repo for the *Provision-Aware* paper (CNSM 2026). Goal: show that asymmetric/quantile-loss forecasters wrapped in conformal prediction beat MSE-trained baselines on **operational cost** at equal RMSE on real backbone traces (Abilene, GÉANT, CESNET-TimeSeries24).

The previous synthetic-only SARIMA-vs-LSTM comparison is preserved as an ablation only; see `plan.md` for the roadmap and `STEPS.md` for the working notebook.

## Overview

This project implements a complete simulation pipeline to:
1. Generate synthetic network traffic on a 12-node backbone topology
2. Train SARIMA (per-link) and LSTM (joint) forecasting models
3. Evaluate forecasting accuracy (RMSE, MAE, MAPE)
4. Analyze capacity planning implications (utilization, overload rates)

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (single seed)
python scripts/main.py

# Or run multi-seed experiments for robust results
python scripts/run_experiments.py
```

## Project Structure

```
├── src/                # Core modules
│   ├── config.py       # Global configuration parameters
│   ├── utils.py        # Shared utility functions
│   ├── simulate_data.py    # Synthetic data generation
│   ├── train_arima.py      # SARIMA model training
│   ├── train_lstm.py       # LSTM model training
│   └── eval_capacity.py    # Evaluation and capacity planning
├── scripts/
│   ├── main.py             # Pipeline orchestrator
│   └── run_experiments.py  # Multi-seed experiment runner
├── tests/              # Test suite
├── requirements.txt    # Python dependencies
├── data/               # Generated data files
├── models/             # Trained model artifacts
├── results/            # Metrics and predictions
├── plots/              # Visualization outputs
└── report/             # Academic paper (LaTeX)
```

## Configuration

Key parameters in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_nodes` | 12 | Network nodes |
| `days` | 14 | Simulation duration |
| `time_step_minutes` | 5 | Sampling interval |
| `window_size` | 72 | LSTM input window (6 hours) |
| `arima_order` | (2,1,2) | ARIMA order |
| `seasonal_order` | (1,0,1,72) | SARIMA seasonal (6-hour period) |
| `capacity_margin` | 1.1 | 10% safety margin |
| Random seeds | 42, 123, 456, 789, 1024 | For multi-seed experiments |

## Results

Results from multi-seed experiments (n=5 seeds: 42, 123, 456, 789, 1024) showing mean ± std.

### Forecasting Metrics

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| SARIMA | 25.32 ± 1.70 | 20.45 ± 1.59 | 47.3% ± 3.0% |
| LSTM | **16.50 ± 0.21** | **12.17 ± 0.13** | **26.0% ± 0.7%** |

| RMSE (mean ± std) | MAE (mean ± std) | MAPE (mean ± std) |
|:-----------------:|:----------------:|:-----------------:|
| ![RMSE](plots/aggregated_forecast_rmse.png) | ![MAE](plots/aggregated_forecast_mae.png) | ![MAPE](plots/aggregated_forecast_mape.png) |

### Capacity Planning

| Model | Mean U_max | Overload Rate |
|-------|------------|---------------|
| SARIMA | 2.72 ± 0.20 | 34.6% ± 3.2% |
| LSTM | **1.94 ± 0.04** | **10.6% ± 0.3%** |

| Max Utilization (mean ± std) | Overload Rate (mean ± std) |
|:----------------------------:|:--------------------------:|
| ![U_max](plots/aggregated_capacity_u_max.png) | ![Overload](plots/aggregated_capacity_f_over.png) |

**Key Findings**:
- LSTM achieves **35% lower RMSE** than SARIMA (16.5 vs 25.3)
- LSTM shows **45% lower MAPE** (26% vs 47%)
- LSTM reduces overload rate by **69%** (10.6% vs 34.6%)
- LSTM exhibits much lower variance across seeds, indicating more stable training

### Per-Link RMSE Distribution

![RMSE Histogram](plots/aggregated_rmse_histogram.png)

### Time Series Examples

| LSTM Wins | SARIMA Wins |
|:---------:|:-----------:|
| ![LSTM Better](plots/timeseries_link_8_lstm_better.png) | ![SARIMA Better](plots/timeseries_link_4_sarima_better.png) |

## Usage Options

```bash
# Full pipeline (single seed)
python scripts/main.py

# Skip data generation (reuse existing)
python scripts/main.py --skip-data

# Only run evaluation
python scripts/main.py --eval-only

# Multi-seed experiments (recommended)
python scripts/run_experiments.py --seeds 42 123 456 789 1024

# Run individual modules
python -m src.simulate_data
python -m src.train_arima
python -m src.train_lstm
python -m src.eval_capacity
```

## Generated Outputs

- **Data**: `data/topology.npz`, `data/traffic_data.npz`
- **Model**: `models/lstm_forecaster.pt`
- **Metrics**: `results/combined_results.json`, `results/aggregated_results.json`
- **Per-seed**: `results/seed_*/` (individual seed results)
- **Plots**: `plots/*.png` (histograms, comparisons, time series)

## Report

A detailed academic report is available: **[paper.pdf](report/paper.pdf)** (IEEE format)

It includes:

- Literature survey of ML approaches for traffic prediction
- Detailed methodology and mathematical formulation
- Extended discussion of results and limitations

LaTeX source: [`report/paper.tex`](report/paper.tex)

## Cloud sweep (paper-grade runs)

The laptop runs setup + tests only; full sweeps live on a cloud box.

```bash
# 1. provision a Python 3.13 box, clone the repo, then:
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-cloud.txt

# 2. download datasets (one time per box):
#    Abilene — 24 weekly files from UT Austin, ~170 MB total:
mkdir -p data/raw/abilene && cd data/raw/abilene
for w in $(seq -f "%02g" 1 24); do
  curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/X${w}.gz"
done
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/links"
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/demands"
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/A"
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/topo-2003-04-10.txt"
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/readme.txt"
cd -
python -m src.data.abilene_loader

#    GÉANT — preprocessed CSV (46 MB):
mkdir -p data/raw/geant && cd data/raw/geant
curl -L -O "https://raw.githubusercontent.com/duchuyle108/SDN-TMprediction/master/dataset/geant-flat-tms.csv"
cd -
python -m src.data.geant_loader

#    CESNET-TimeSeries24 — Zenodo (CC-BY, several GB):
mkdir -p data/raw/cesnet
# fetch the 10-min institution-aggregation parquet bundle from
#   https://zenodo.org/records/13382427  (DOI 10.5281/zenodo.13382427)
# unpack so .parquet files land directly under data/raw/cesnet/.
python -m src.data.cesnet_loader  # defaults to top-20 institutions.

# 3. run the v0 gate (Abilene, 5 seeds, mse + asym + pinball):
python scripts/run_v0.py --seeds 42 123 456 789 1024
# pass criterion: see CLAUDE.md / plan.md.

# 4. one-off cells (any dataset × loss):
python scripts/run_experiments.py --dataset abilene --loss asym \
    --alpha 5 --beta 1 --seeds 42 123 456 789 1024
python scripts/run_experiments.py --dataset geant   --loss pinball \
    --alpha 5 --beta 1 --seeds 42 123 456 789 1024
```

Per-cell artefacts land under `results/<dataset>_<loss>/seed_<N>/`; the aggregated JSON at `results/<dataset>_<loss>/aggregated_results.json` is what `run_v0.py --from-cache` reads back.

## Requirements

- Python 3.9+
- numpy, pandas, networkx
- matplotlib, statsmodels
- torch, scikit-learn, joblib
- (cloud) pyarrow, mapie — see `requirements-cloud.txt` for pinned versions

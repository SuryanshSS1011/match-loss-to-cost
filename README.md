# Provision-Aware: Decision-Focused and Conformal Forecasting for Backbone Capacity Planning

Research code for the *Provision-Aware* paper (target: **CNSM 2026**). The thesis: backbone
operators minimise the **joint cost of SLA violations + over-provisioning**, not RMSE. Training a
forecaster on an **asymmetric capacity-cost surrogate** and wrapping it in **adaptive conformal
prediction** reduces operational cost and SLA-violation rate *at equal RMSE* versus an MSE-trained
baseline — demonstrated on three real backbone traces (**Abilene, GÉANT, CESNET-TimeSeries24**).

The architecture is held fixed across loss variants — the contribution is the **loss + calibration**,
not a new model. The earlier synthetic 12-node SARIMA-vs-LSTM comparison is retained only as a
topology-size ablation.

---

## Three findings (real Abilene/GÉANT/CESNET, LSTM backbone, 20 seeds)

1. **An operator-cost frontier exists, and asymmetric training beats MSE on it.** Sweeping the
   training cost ratio α:β and evaluating against an operator's own α:β, a matched asymmetric model
   beats the MSE baseline on operator-relevant cost at every realistic operator ratio (1:1 → 100:1),
   with significant paired-bootstrap confidence intervals.

2. **"Match your loss to your cost."** The best training α tracks the operator's α (the operator-eval
   heatmap is diagonal-dominant) — a simple, deployable rule.

3. **Cusp-linear (L1) asymmetric loss is safer than squared.** On heavy-tailed traffic (GÉANT,
   link loads spanning 5 orders of magnitude) the **squared** asymmetric loss collapses — MSE wins —
   while **L1 wins 62–75 % at every operator**. On tamer traces (Abilene, CESNET) both work. So the
   *loss formulation* matters, not just the α:β knob. Corroborates Eramo et al. 2020.

**Conformal calibration (companion contribution).** Split-CQR under-covers on temporally-correlated
traffic; **Adaptive Conformal Inference (ACI) tracks the target coverage** online:

| Target coverage | Split-CQR | ACI |
|---|---|---|
| 0.95 | 0.886 (under) | **0.947** |
| 0.90 | 0.844 (under) | **0.895** |
| 0.80 | 0.768 (under) | **0.789** |

---

## Pipeline

1. **Loaders** map each dataset to a uniform `(time × link)` tensor + routing matrix
   (`src/data/{abilene,geant,cesnet}_loader.py`).
2. **Forecasters** share one training loop (`src/train_neural.py`): LSTM (primary), plus DLinear,
   PatchTST, iTransformer, DCRNN, Chronos-Bolt (zero-shot), and classical SARIMA / seasonal-naive /
   Holt-Winters baselines.
3. **Losses** (`src/losses/`): MSE, asymmetric-squared `α·max(y−ŷ,0)² + β·max(ŷ−y,0)²`,
   cusp-linear/L1, and pinball at τ=α/(α+β).
4. **Calibration** (`src/calibration/`): CQR (Romano 2019) and ACI (Gibbs–Candès 2021).
5. **Evaluation** (`src/evaluation/`): operational cost / overload rate / over-provisioning, plus the
   significance protocol (paired Wilcoxon + Holm, Demšar critical-difference diagrams).

## Project structure

```
src/
├── data/         # abilene/geant/cesnet loaders → uniform (T × link) tensors + routing
├── models/       # patchtst, itransformer, dlinear, dcrnn model classes
├── losses/       # asymmetric (squared), cusp_linear (L1), pinball, + factory
├── calibration/  # cqr.py, aci.py
├── evaluation/   # operational.py (cost metrics), significance.py (Wilcoxon/Holm/CD)
├── train_neural.py            # shared neural training loop
└── train_{arima,chronos,...}.py
scripts/
├── run_pareto.py              # operator α:β sweep → Pareto frontier plot
├── run_operator_eval.py       # post-hoc operator-cost heatmap (per-seed values + bootstrap CI)
├── run_pareto_calibration.py  # CQR/ACI coverage-vs-width sweep
├── build_significance.py      # CD diagrams + Holm-Wilcoxon from operator_eval.json
├── build_headline_table.py    # LaTeX/Markdown headline tables
└── run_experiments.py         # generic per-(dataset, loss, model) runner
report/                        # generated headline tables (LaTeX + Markdown)
```

## Quick start

This repo runs locally (e.g. Apple Silicon with the MPS backend) or on a CPU/GPU box.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # loose spec (numpy, torch, statsmodels, mapie, chronos, ...)
# — or, for the exact environment that produced the committed results:
# pip install -r requirements.lock.txt
pytest -q                              # ~318 tests
```

## Datasets (one-time download)

```bash
# Abilene — 24 weekly traffic-matrix files from UT Austin (~170 MB)
mkdir -p data/raw/abilene && cd data/raw/abilene
for w in $(seq -f "%02g" 1 24); do
  curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/X${w}.gz"; done
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/links"
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/demands"
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/A"
cd - && python -m src.data.abilene_loader            # → data/abilene_traffic.npz

# GÉANT — preprocessed flat CSV (48 MB). The published file has a leading
# timestamp column; the loader detects and drops it.
mkdir -p data/raw/geant && curl -L -o data/raw/geant/geant-flat-tms.csv \
  "https://raw.githubusercontent.com/duchuyle108/SDN-TMprediction/master/dataset/geant-flat-tms.csv"
python -m src.data.geant_loader                       # → data/geant_traffic.npz (23 nodes, 58 links)

# CESNET-TimeSeries24 — Zenodo DOI 10.5281/zenodo.13382427 (CC-BY).
# The release ships CSV files inside tar.gz archives (NOT loose parquet):
mkdir -p data/raw/cesnet
curl -L -o data/raw/cesnet/institutions.tar.gz \
  "https://zenodo.org/api/records/13382427/files/institutions.tar.gz/content"   # 479 MB
tar xzf data/raw/cesnet/institutions.tar.gz -C data/raw/cesnet/ \
  institutions/agg_10_minutes institutions/identifiers.csv
python -m src.data.cesnet_loader                      # → data/cesnet_traffic.npz (top-20 institutions)
```

The loaders accept either format and document any subsetting (CESNET drops empty/truncated
institutions and keeps the top-20 by total bytes — see the loader docstring).

## Reproducing the results

All headline runs use **20 seeds** (`42 123 456 789 1024 1 2 3 7 13 17 99 256 512 2048 31337 8191 65521 100003 271828`).
The runner caches per-seed predictions, so re-runs resume cheaply; `--from-cache` re-aggregates and
re-plots without retraining.

```bash
SEEDS="42 123 456 789 1024 1 2 3 7 13 17 99 256 512 2048 31337 8191 65521 100003 271828"

# 1) Operator α:β sweep → Pareto frontier (both loss forms, per dataset)
for d in abilene geant cesnet; do for form in asym asym_l1; do
  python scripts/run_pareto.py --dataset $d --models lstm \
    --ratios 1:1 2:1 5:1 10:1 20:1 100:1 --loss-form $form --include-mse-baseline \
    --seeds $SEEDS --plot-path plots/pareto_${d}_${form}.png
done; done

# 2) Operator-cost heatmaps (post-hoc, per-seed bootstrap CIs)
for d in abilene geant cesnet; do for form in asym asym_l1; do
  python scripts/run_operator_eval.py --dataset $d --loss-form $form
done; done

# 3) Conformal calibration sweep (Abilene: CQR + ACI at target coverage 0.95/0.90/0.80)
python scripts/run_pareto_calibration.py --dataset abilene --models lstm \
  --calibration both --target-alphas 0.05 0.10 0.20 --seeds $SEEDS --alpha 5 --beta 1 \
  --plot-path plots/pareto_calibration_abilene.png

# 4) Significance (CD diagrams + Holm-Wilcoxon) and headline tables
for d in abilene geant cesnet; do for form in asym asym_l1; do
  python scripts/build_significance.py --dataset $d --loss-form $form
done; done
for d in abilene geant cesnet; do
  python scripts/build_headline_table.py \
    --inputs results/${d}_pareto_asym/ratio_5_1/aggregated_results.json \
    --output report/table_${d}.tex --markdown
done
```

### Statistics

- Per-cell significance: **paired bootstrap on win % vs MSE** (valid at any seed count) — the headline.
- Ranking across operators: **Demšar critical-difference diagrams** (Nemenyi).
- **Holm-corrected paired Wilcoxon** per (dataset, operator). Note: the Wilcoxon signed-rank test
  cannot reach significance under Holm correction with fewer than ~8 seeds (min p ≈ 0.06 at n=5),
  which is why the headline cells use 20 seeds.

## Key outputs

| Artifact | Path |
|---|---|
| Pareto frontiers | `plots/pareto_<dataset>_<form>.png` |
| Operator-cost heatmaps | `plots/operator_eval_<dataset>_<form>.png` |
| Critical-difference diagrams | `plots/cd_<dataset>_<form>.png` |
| Calibration coverage-vs-width | `plots/pareto_calibration_abilene.png` |
| Per-(cell × operator) costs + CIs | `results/<dataset>_pareto_<form>/operator_eval.json` |
| Significance (CD ranks + Wilcoxon) | `results/<dataset>_pareto_<form>/significance.json` |
| Headline tables | `report/table_<dataset>.{tex,md}` |

## Datasets & baselines reference

| Dataset | Scope | Source |
|---|---|---|
| Abilene TM | 12 nodes / 144 OD-pairs, 5-min, 24 wk | UT Austin (Y. Zhang) |
| GÉANT TM | 23 nodes / 58 links, 15-min, 4 mo | TOTEM / duchuyle108 CSV mirror |
| CESNET-TimeSeries24 | per-institution, 10-min, ~40 wk | Zenodo 10.5281/zenodo.13382427 (CC-BY) |

Baselines implemented: seasonal-naive, Holt-Winters, historical-average, SARIMA, LSTM, DLinear,
PatchTST, iTransformer, DCRNN, Chronos-Bolt (zero-shot). Cross-architecture loss ablation
(PatchTST/iTransformer) and the full 9-baseline matrix are planned as a journal/INFOCOM extension.

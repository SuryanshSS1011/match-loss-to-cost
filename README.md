# Provisioning-Aware Traffic Forecasting: Asymmetric Losses and Conformal Capacity Bands

Research code for the *Provisioning-Aware Traffic Forecasting* paper (target: **CNSM 2026**). The
thesis is that backbone operators minimise the **joint cost of SLA violations and over-provisioning**
rather than RMSE, so training a forecaster on an **asymmetric capacity-cost surrogate** and wrapping
it in **adaptive conformal prediction** reduces operational cost and SLA-violation rate at equal RMSE
versus an MSE-trained baseline. The result is demonstrated on three real backbone traces
(**Abilene, GÉANT, CESNET-TimeSeries24**) across six modern forecaster architectures.

The architecture is held fixed across loss variants, so the contribution is the **loss + calibration**
rather than a new model. The earlier synthetic 12-node SARIMA vs. LSTM comparison is retained only as
a topology-size ablation in the appendix.

---

## Headline findings (real Abilene/GÉANT/CESNET, LSTM backbone, 20 seeds)

1. **An operator-cost frontier exists, and asymmetric training beats MSE on it.** Sweeping the
   training cost ratio α:β and evaluating against an operator's own α:β, a matched asymmetric model
   beats the MSE baseline on operator-relevant cost at every realistic operator ratio (1:1 through
   100:1) with significant paired-bootstrap confidence intervals.

2. **"Match your loss to your cost."** The best training α tracks the operator's α (the operator-eval
   heatmap is diagonal-dominant), which gives a simple and deployable rule.

3. **Cusp-linear (L1) asymmetric loss is safer than squared.** On heavy-tailed traffic such as GÉANT,
   where link loads span 5 orders of magnitude, the **squared** asymmetric loss collapses and MSE
   wins, while the **L1** form still wins 62–75% at every operator. On tamer traces (Abilene,
   CESNET) both forms work, so the *loss formulation* matters in addition to the α:β knob.
   Corroborates Eramo et al. 2020.

4. **The cost-aware loss generalises across architectures.** The same operational-cost wins reproduce
   on LSTM, DLinear, iTransformer, PatchTST, and DCRNN backbones. This rules out an
   "architecture-driven not loss-driven" alternative hypothesis.

5. **Conformal calibration of the capacity band (companion contribution).** Split-CQR under-covers on
   temporally-correlated traffic, while **Adaptive Conformal Inference (ACI) tracks the target
   coverage** online and reduces overload 3 to 155× across the three datasets:

   | Dataset | Target | CQR coverage | ACI coverage |
   |---|---|---|---|
   | Abilene | 0.95 | 0.886 | **0.947** |
   | Abilene | 0.90 | 0.844 | **0.895** |
   | Abilene | 0.80 | 0.768 | **0.789** |
   | GÉANT   | 0.95 | 0.951 | 0.938 |
   | GÉANT   | 0.90 | 0.911 | 0.878 |
   | GÉANT   | 0.80 | 0.820 | 0.755 |
   | CESNET  | 0.95 | 0.949 | 0.947 |
   | CESNET  | 0.90 | 0.871 | **0.893** |
   | CESNET  | 0.80 | 0.735 | **0.787** |

   ACI has 30 to 200× lower across-seed coverage variance than CQR at every cell. The cost winner
   is dataset-dependent at the strict 0.95 target, but the overload winner is uniformly ACI, which
   makes ACI the SLA-conservative choice and CQR the cost-aggressive one.

---

## Pipeline

1. **Loaders** map each dataset to a uniform `(time × link)` tensor and routing matrix
   (`src/data/{abilene,geant,cesnet}_loader.py`).
2. **Forecasters** share one training loop (`src/train_neural.py`): LSTM (primary), plus DLinear,
   PatchTST, iTransformer, DCRNN, and Chronos-Bolt (zero-shot), as well as classical SARIMA,
   seasonal-naive, and Holt-Winters baselines.
3. **Losses** (`src/losses/`): MSE, asymmetric-squared `α·max(y−ŷ,0)² + β·max(ŷ−y,0)²`, cusp-linear
   (L1), and pinball at τ=α/(α+β) for the conformal band training.
4. **Calibration** (`src/calibration/`): split CQR (Romano 2019) and ACI (Gibbs–Candès 2021).
5. **Evaluation** (`src/evaluation/`): operational cost, overload rate, over-provisioning cost, plus
   the significance protocol (paired Wilcoxon with Holm correction and Demšar critical-difference
   diagrams).

## Project structure

```
src/
├── data/         # abilene/geant/cesnet loaders → uniform (T × link) tensors + routing
├── models/       # patchtst, itransformer, dlinear, dcrnn model classes
├── losses/       # asymmetric (squared), cusp_linear (L1), pinball, plus a factory
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
├── build_calibration_table.py # multi-dataset calibration table for §5.4
├── aggregate_from_seed_npz.py # rebuild aggregated_results.json from per-seed predictions
└── run_experiments.py         # generic per-(dataset, loss, model) runner
report/                        # generated headline tables (LaTeX + Markdown)
```

## Quick start

This repo runs locally on Apple Silicon with the MPS backend or on any CPU or CUDA box.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # loose spec (numpy, torch, statsmodels, mapie, chronos, ...)
# Or, for the exact environment that produced the committed results:
# pip install -r requirements.lock.txt
pytest -q                              # 327 passed, 4 skipped
```

## Datasets (one-time download)

```bash
# Abilene: 24 weekly traffic-matrix files from UT Austin (~170 MB)
mkdir -p data/raw/abilene && cd data/raw/abilene
for w in $(seq -f "%02g" 1 24); do
  curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/X${w}.gz"; done
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/links"
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/demands"
curl -O "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/A"
cd - && python -m src.data.abilene_loader            # → data/abilene_traffic.npz

# GÉANT: preprocessed flat CSV (48 MB). The published file has a leading
# timestamp column, and the loader detects and drops it automatically.
mkdir -p data/raw/geant && curl -L -o data/raw/geant/geant-flat-tms.csv \
  "https://raw.githubusercontent.com/duchuyle108/SDN-TMprediction/master/dataset/geant-flat-tms.csv"
python -m src.data.geant_loader                       # → data/geant_traffic.npz (23 nodes, 58 links)

# CESNET-TimeSeries24: Zenodo DOI 10.5281/zenodo.13382427 (CC-BY).
# The release ships CSV files inside tar.gz archives, not loose parquet:
mkdir -p data/raw/cesnet
curl -L -o data/raw/cesnet/institutions.tar.gz \
  "https://zenodo.org/api/records/13382427/files/institutions.tar.gz/content"   # 479 MB
tar xzf data/raw/cesnet/institutions.tar.gz -C data/raw/cesnet/ \
  institutions/agg_10_minutes institutions/identifiers.csv
python -m src.data.cesnet_loader                      # → data/cesnet_traffic.npz (top-20 institutions)
```

The loaders accept either format and document any subsetting (CESNET drops empty or truncated
institutions and keeps the top-20 by total bytes; see the loader docstring).

## Reproducing the results

All headline runs use **20 seeds**
(`42 123 456 789 1024 1 2 3 7 13 17 99 256 512 2048 31337 8191 65521 100003 271828`).
The runner caches per-seed predictions, so re-runs resume cheaply, and `--from-cache` re-aggregates
and re-plots without retraining.

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

# 3) Conformal calibration sweep (CQR + ACI at target coverages 0.95/0.90/0.80, all 3 datasets)
for d in abilene geant cesnet; do
  python scripts/run_pareto_calibration.py --dataset $d --models lstm \
    --calibration both --target-alphas 0.05 0.10 0.20 --seeds $SEEDS --alpha 5 --beta 1 \
    --plot-path plots/pareto_calibration_${d}.png
done

# 4) Significance (CD diagrams + Holm-Wilcoxon) and headline tables
for d in abilene geant cesnet; do for form in asym asym_l1; do
  python scripts/build_significance.py --dataset $d --loss-form $form
done; done
for d in abilene geant cesnet; do
  python scripts/build_headline_table.py \
    --inputs results/${d}_pareto_asym/ratio_5_1/aggregated_results.json \
    --output report/table_${d}.tex --markdown
done
python scripts/build_calibration_table.py  # multi-dataset coverage table for §5.4
```

### Statistics

- Per-cell significance: **paired bootstrap on win % vs. MSE** (valid at any seed count) is the
  headline.
- Ranking across operators: **Demšar critical-difference diagrams** (Nemenyi).
- **Holm-corrected paired Wilcoxon** per (dataset, operator). The Wilcoxon signed-rank test cannot
  reach significance under Holm correction with fewer than ~8 seeds (min p ≈ 0.06 at n=5), so the
  headline cells use 20 seeds.

## Key outputs

| Artifact | Path |
|---|---|
| Pareto frontiers | `plots/pareto_<dataset>_<form>.png` |
| Operator-cost heatmaps | `plots/operator_eval_<dataset>_<form>.png` |
| Critical-difference diagrams | `plots/cd_<dataset>_<form>.png` |
| Calibration coverage-vs-width | `plots/pareto_calibration_<dataset>.png` |
| Per-(cell × operator) costs + CIs | `results/<dataset>_pareto_<form>/operator_eval.json` |
| Significance (CD ranks + Wilcoxon) | `results/<dataset>_pareto_<form>/significance.json` |
| Headline tables | `report/table_<dataset>.{tex,md}` |
| Calibration table (multi-dataset) | `report/table_calibration_main.{tex,md}` |

## Datasets and baselines reference

| Dataset | Scope | Source |
|---|---|---|
| Abilene TM | 12 nodes / 144 OD-pairs, 5-min, 24 wk | UT Austin (Y. Zhang) |
| GÉANT TM | 23 nodes / 58 links, 15-min, 4 mo | TOTEM / duchuyle108 CSV mirror |
| CESNET-TimeSeries24 | per-institution, 10-min, ~40 wk | Zenodo 10.5281/zenodo.13382427 (CC-BY) |

Baselines implemented: seasonal-naive, Holt-Winters, historical-average, SARIMA, LSTM, DLinear,
PatchTST, iTransformer, DCRNN, and Chronos-Bolt (zero-shot). The 9-baseline full matrix and a
cross-topology generalization study are planned as a journal or INFOCOM extension.

## Artifacts on HuggingFace

- **Predictions (reproduction path):**
  [`SuryanshSS1011/provisioning-aware-predictions`](https://huggingface.co/datasets/SuryanshSS1011/provisioning-aware-predictions)
  carries per-(dataset, sweep, cell, seed) forecast `.npz` files and regenerates every table and
  figure without retraining.
- **Reference checkpoints (illustrative, not the reproduction path):**
  [`SuryanshSS1011/provisioning-aware-checkpoints`](https://huggingface.co/SuryanshSS1011/provisioning-aware-checkpoints)
  carries one example end-of-run weight file per architecture. These are illustrative only, so for
  canonical models, retrain from the seed list above.

Raw traffic data is not rehosted, so use the dataset download instructions above.

#!/bin/bash
# Pareto α/β sweep with cusp-linear (L1) asymmetric loss + MSE baseline reference.
# Companion to run_sweep_squared.sh; cusp-linear is less knob-sensitive
# (Eramo 2020 style). Wall-clock estimate on Codespaces 4c/16GB: ~2.5 hrs.
python scripts/run_pareto.py --dataset abilene --models lstm --ratios 1:1 2:1 5:1 10:1 20:1 100:1 --loss-form asym_l1 --include-mse-baseline --seeds 42 123 456 789 1024 --plot-path plots/pareto_abilene_l1.png

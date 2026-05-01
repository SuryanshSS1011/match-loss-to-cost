#!/bin/bash
# Pareto α/β sweep with squared asymmetric loss + MSE baseline reference.
# Headline figure for the v0 frontier-vs-MSE comparison.
# Wall-clock estimate on Codespaces 4c/16GB: ~2.5 hrs.
python scripts/run_pareto.py --dataset abilene --models lstm --ratios 1:1 2:1 5:1 10:1 20:1 100:1 --loss-form asym --include-mse-baseline --seeds 42 123 456 789 1024 --plot-path plots/pareto_abilene_squared.png

#!/usr/bin/env bash
# Cascade the CQR/ACI calibration sweep across CESNET → Abilene.
# Invoked manually after GÉANT finishes (the launcher of this script
# waits for results/geant_calib_pareto/alpha_0.20/aggregated_results.json
# as the completion gate).
#
# Each dataset: 3 target_alphas × 20 seeds, LSTM pinball + CQR + ACI.
#
# Logs land in logs/{cesnet,abilene}_calib_<HHMM>.log and the script
# refuses to start if GÉANT's last cell hasn't completed.
set -e
cd "$(dirname "$0")/.."

SEEDS="42 123 456 789 1024 1 2 3 7 13 17 99 256 512 2048 31337 8191 65521 100003 271828"
GEANT_DONE="results/geant_calib_pareto/alpha_0.20/aggregated_results.json"

if [[ ! -f "$GEANT_DONE" ]]; then
  echo "ERR: GÉANT calibration not complete (missing $GEANT_DONE)."
  echo "Wait for it, or override by touching the file if you really mean it."
  exit 1
fi

run_one() {
  local ds="$1"
  local log="logs/${ds}_calib_$(date +%H%M).log"
  echo "=== $(echo "$ds" | tr '[:lower:]' '[:upper:]') calibration sweep → $log ==="
  {
    for ta in 0.05 0.10 0.20; do
      echo "=== $(echo "$ds" | tr '[:lower:]' '[:upper:]') target_alpha=$ta ==="
      PYTHONPATH=. .venv/bin/python scripts/run_experiments.py \
        --dataset "$ds" --loss asym --alpha 5 --beta 1 \
        --seeds $SEEDS \
        --models lstm \
        --calibration both \
        --target-alpha "$ta" \
        --output-dir "results/${ds}_calib_pareto/alpha_${ta}"
    done
    echo "=== DONE $(echo "$ds" | tr '[:lower:]' '[:upper:]') $(date) ==="
  } > "$log" 2>&1
}

run_one cesnet
# Skip Abilene: results/abilene_calib_pareto/alpha_{0.05,0.1,0.2}/aggregated_results.json
# already exist at 20 seeds from an earlier run. Saves ~3h wall-clock.
echo "=== CASCADE COMPLETE $(date) ==="

#!/usr/bin/env bash
# PatchTST on Abilene, on the M5 Pro CPU. MPS hits INT_MAX on Abilene's
# 144-link × 48k-step series so the GPU path is unavailable; the CPU
# path takes longer but stays on the same consumer hardware, which is
# the resource-constrained framing the paper sells.
#
# Strategy: 4-way parallelism across 20 seeds. Each subprocess pins
# OMP_NUM_THREADS=4 so the 4 in-flight seeds share the 18-core M5 Pro
# without oversubscribing (16 of 18 cores used; 2 left for the OS).
#
# Output dir: results/abilene_patchtst_cpu/seed_<S>/patchtst_predictions.npz
# Logs: logs/patchtst_abilene_cpu_seed_<S>.log
#
# Wait for CESNET calibration to finish before running this — otherwise
# the laptop is double-loaded. Refuse to start if the cascade is alive.
set -euo pipefail
cd "$(dirname "$0")/.."

CASCADE_PID_FILE=/tmp/cascade_pid
if [[ -f "$CASCADE_PID_FILE" ]] && ps -p "$(cat $CASCADE_PID_FILE)" > /dev/null 2>&1; then
  echo "ERR: CESNET cascade still running (PID $(cat $CASCADE_PID_FILE)). Wait for it to finish before starting CPU sweep."
  exit 1
fi

mkdir -p logs

SEEDS="42 123 456 789 1024 1 2 3 7 13 17 99 256 512 2048 31337 8191 65521 100003 271828"
PARALLEL=4

run_one() {
  local seed=$1
  local log="logs/patchtst_abilene_cpu_seed_${seed}.log"
  echo "[$(date +%H:%M:%S)] launching seed=$seed → $log"
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  PROVISION_AWARE_DEVICE=cpu \
  PYTHONPATH=. \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/run_experiments.py \
    --dataset abilene --loss asym --alpha 5 --beta 1 \
    --seeds "$seed" \
    --models patchtst \
    --output-dir "results/abilene_patchtst_cpu" \
    > "$log" 2>&1 &
}

# Launch in batches of PARALLEL. Wait for each batch to finish before
# starting the next so we don't oversubscribe.
batch=()
for seed in $SEEDS; do
  run_one "$seed"
  batch+=("$!")
  if [[ ${#batch[@]} -ge $PARALLEL ]]; then
    for pid in "${batch[@]}"; do wait "$pid"; done
    batch=()
    echo "[$(date +%H:%M:%S)] batch of $PARALLEL complete"
  fi
done
# Drain any remaining
for pid in "${batch[@]}"; do wait "$pid"; done
echo "[$(date +%H:%M:%S)] === ALL 20 SEEDS DONE ==="

#!/usr/bin/env bash
#SBATCH --job-name=pa-patchtst-abilene
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/work/%u/pa-cnsm-work/logs/patchtst_abilene_%j.out
#SBATCH --error=/storage/work/%u/pa-cnsm-work/logs/patchtst_abilene_%j.err
#
# PatchTST on Abilene — the local M5 Pro hits the MPS INT_MAX tensor-dim
# limit on this combination (Abilene's 144-link-wide patches + 50-epoch
# training blow the MPSGraph constructor). Move the one missing
# multi-architecture cell to CUDA.
#
# Output: results/abilene_pareto_asym/ratio_5_1/seed_*/patchtst_predictions.npz
# for 20 seeds, then re-aggregate locally with
# scripts/run_experiments.py --from-cache.
#
# Wall budget: ~1.5h at 20 seeds × ~4 min/seed on P100. Time cap 2h.

set -euo pipefail

USER_ID="$(whoami)"
CODE_DIR="/storage/home/${USER_ID}/pa-cnsm"
ENV_NAME="pa-cnsm"

cd "${CODE_DIR}"

module purge
module load anaconda/2023.09 cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[job] node=$(hostname)  user=${USER_ID}  jobid=${SLURM_JOB_ID}"
echo "[job] git commit: $(git rev-parse HEAD)"
nvidia-smi || echo "  (no nvidia-smi — odd, GPU should be allocated)"

export PYTHONUNBUFFERED=1
SEEDS="42 123 456 789 1024 1 2 3 7 13 17 99 256 512 2048 31337 8191 65521 100003 271828"

# Write to a dedicated dir so we don't clobber the existing
# DLinear+iTransformer aggregated_results.json. Laptop-side merge handles
# combining the rows post-hoc.
python scripts/run_experiments.py \
    --dataset abilene --loss asym --alpha 5 --beta 1 \
    --seeds $SEEDS \
    --models patchtst \
    --output-dir results/abilene_patchtst_only

echo "[job] PatchTST/Abilene sweep done. Pushing artefacts..."
BRANCH="roar-patchtst-abilene-${SLURM_JOB_ID}"
git checkout -B "${BRANCH}"
git add 'results/abilene_patchtst_only/' 2>/dev/null || true
if ! git diff --cached --quiet; then
    git -c user.email="${USER_ID}@psu.edu" -c user.name="${USER_ID}" \
        commit -m "PatchTST/Abilene results (Roar job ${SLURM_JOB_ID})"
    git push --set-upstream origin "${BRANCH}" \
        || echo "[job] git push failed — copy artefacts manually"
else
    echo "[job] nothing new to commit"
fi

echo "[job] done."

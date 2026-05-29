# Roar Collab cloud jobs

SLURM scripts for the one experiment that won't run on Apple MPS:
**PatchTST/Abilene at 20 seeds**, which hits the MPS INT_MAX
tensor-dim ceiling locally.

## Prereqs

Set up once per Roar account:

```bash
# On Roar (after `ssh roar`):
USER_ID=$(whoami)
mkdir -p "/storage/home/${USER_ID}" "/storage/work/${USER_ID}/pa-cnsm-work/logs"
cd "/storage/home/${USER_ID}"
git clone https://github.com/SuryanshSS1011/ML-for-Network-Traffic-Prediction-and-Capacity-Planning pa-cnsm
cd pa-cnsm

module load anaconda/2023.09 cuda/12.6.2
conda create -y -n pa-cnsm python=3.11
conda activate pa-cnsm
pip install -r requirements.txt
# pip install -r requirements.lock.txt  # if you want exact pins

# Get the Abilene data (uses our existing loader):
python -c "from src.data_loaders import abilene; abilene.load()"
```

## Submit the PatchTST/Abilene job

```bash
cd /storage/home/$(whoami)/pa-cnsm
sbatch scripts/roar/patchtst_abilene.sbatch
```

Watch:

```bash
squeue --me
tail -f /storage/work/$(whoami)/pa-cnsm-work/logs/patchtst_abilene_<jobid>.out
```

## Pull artefacts back to the laptop

The job pushes results to a branch named `roar-patchtst-abilene-<jobid>`.

```bash
# On the laptop:
git fetch origin
# Pull just the patchtst aggregated json (the .npz files are large; pick what you need)
git checkout roar-patchtst-abilene-<jobid> -- results/abilene_patchtst_only/
git checkout main
# The cloud job wrote a standalone aggregated_results.json containing only PatchTST.
# Merge it into the Abilene headline table by passing it as another --inputs:
.venv/bin/python scripts/build_headline_table.py \
  --inputs \
    results/abilene_pareto_asym/ratio_5_1/aggregated_results_lstm.json \
    results/abilene_pareto_asym/ratio_5_1/aggregated_results.json \
    results/abilene_baselines/chronos_mse/aggregated_results.json \
    results/abilene_baselines/dcrnn_mse/aggregated_results.json \
    results/abilene_patchtst_only/aggregated_results.json \
  --output report/table_abilene.tex --markdown \
  --caption "Provisioning-Aware vs baselines on Abilene (asym 5:1, 20 seeds)." \
  --label "tab:abilene_main"
```

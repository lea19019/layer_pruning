#!/bin/bash
#SBATCH --job-name=jobA_fast
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/jobA_fast_%j.out
#SBATCH --error=logs/jobA_fast_%j.err
set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1
echo "=== Job A: Fast experiments ==="
# BnB eval
python scripts/run_bnb_eval.py --all-missing || echo "BnB eval had failures"
# I5 prune-only + MP_16_enes
for exp in I5_t03 I5_t05 I5_t07 I5_t03_enes I5_t05_enes I5_t07_enes MP_16_enes; do
    [ -f "experiments/results/${exp}/results.json" ] && continue
    echo "=== Running ${exp} at $(date) ==="
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || echo "FAILED ${exp}"
done
echo "=== Done at $(date) ==="

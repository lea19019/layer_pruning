#!/bin/bash
#SBATCH --job-name=jC2_eskd
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
#SBATCH --output=logs/jC2_eskd_%j.out
#SBATCH --error=logs/jC2_eskd_%j.err
set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1
for exp in I5_t03_kd_enes I5_t05_kd_enes I5_t07_kd_enes; do
    [ -f "experiments/results/${exp}/results.json" ] && { echo "SKIP ${exp}"; continue; }
    echo "=== Running ${exp} at $(date) ==="
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || echo "FAILED ${exp}"
done

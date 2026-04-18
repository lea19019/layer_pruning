#!/bin/bash
#SBATCH --job-name=jB2_esft
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
#SBATCH --output=logs/jB2_esft_%j.out
#SBATCH --error=logs/jB2_esft_%j.err
set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1
for exp in I5_t03_ft_enes I5_t05_ft_enes I5_t07_ft_enes; do
    [ -f "experiments/results/${exp}/results.json" ] && { echo "SKIP ${exp}"; continue; }
    echo "=== Running ${exp} at $(date) ==="
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || echo "FAILED ${exp}"
done

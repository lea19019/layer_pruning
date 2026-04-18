#!/bin/bash
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/enes_%x_%j.out
#SBATCH --error=logs/enes_%x_%j.err

set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

EXP=$1
echo "=== Running ${EXP} at $(date) ==="
python -m src.run_experiment "experiments/configs/${EXP}.yaml"
echo "=== Done at $(date) ==="

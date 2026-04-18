#!/bin/bash
#SBATCH --job-name=m_enes_fix
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/m_enes_fix_%j.out
#SBATCH --error=logs/m_enes_fix_%j.err

# Rerun all M-series en-es experiments with fixed heuristic pruning.
# Prunes ONCE from 32→16 layers, saves at 24/20/16, then fine-tunes all 6.

set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== M-series en-es rerun with fixed pruning ==="
echo "=== Started at $(date) ==="

python scripts/rerun_m_enes.py

echo "=== Completed at $(date) ==="

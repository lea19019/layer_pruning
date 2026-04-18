#!/bin/bash
#SBATCH --job-name=ablation-surgical
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/ablation_surgical_%j.out
#SBATCH --error=logs/ablation_surgical_%j.err

set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

APPROACH=${1:-all}

echo "Surgical fix approach: ${APPROACH}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -m ablation.scripts.surgical_fix --approach ${APPROACH}

echo "Done: $(date)"

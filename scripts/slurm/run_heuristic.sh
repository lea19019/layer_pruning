#!/bin/bash
#SBATCH --job-name=heuristic
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/heuristic_%j.out
#SBATCH --error=logs/heuristic_%j.err

# Heuristic pruning (Moslem et al.): for each pruning step, evaluates the
# model N times (once per remaining layer). With 32 layers and target 16,
# this means (32+31+...+17) = ~392 evaluations. Each evaluation translates
# 200 sentences. On H200, pruning takes ~2-3h + ~3h fine-tuning.
#
# Usage: sbatch scripts/slurm/run_heuristic.sh experiments/configs/M1_8.yaml

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch $0 <config.yaml>"
    exit 1
fi

CONFIG="$1"

source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Running heuristic pruning experiment: ${CONFIG} ==="
python -m src.run_experiment "${CONFIG}"

echo "=== Done ==="

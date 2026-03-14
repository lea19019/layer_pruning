#!/bin/bash
#SBATCH --job-name=heuristic
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/heuristic_%j.out
#SBATCH --error=logs/heuristic_%j.err

# Heuristic pruning is very expensive: for each pruning step, it evaluates
# the model N times (once per remaining layer). With 32 layers and target 16,
# this means (32+31+...+17) = ~392 evaluations. Each evaluation translates
# 50 sentences. Hence the long walltime.
#
# Usage: sbatch scripts/slurm/run_heuristic.sh experiments/configs/M1_8.yaml

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch $0 <config.yaml>"
    exit 1
fi

CONFIG="$1"

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs

source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Running heuristic pruning experiment: ${CONFIG} ==="
python -m src.run_experiment "${CONFIG}"

echo "=== Done ==="

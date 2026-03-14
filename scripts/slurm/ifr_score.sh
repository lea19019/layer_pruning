#!/bin/bash
#SBATCH --job-name=ifr_score
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/ifr_score_%j.out
#SBATCH --error=logs/ifr_score_%j.err

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs

source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== IFR Layer Importance Scoring ==="
python -m src.attribution.score_layers \
    --model CohereForAI/aya-expanse-8b \
    --n-samples 200 \
    --max-length 256 \
    --output experiments/results/ifr_scores.json

echo "=== Done ==="

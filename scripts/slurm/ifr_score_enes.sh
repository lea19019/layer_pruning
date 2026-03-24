#!/bin/bash
#SBATCH --job-name=ifr_enes
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/ifr_enes_%j.out
#SBATCH --error=logs/ifr_enes_%j.err

set -euo pipefail

source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== IFR Layer Importance Scoring (English-Spanish) ==="
python -m src.attribution.score_layers \
    --model CohereForAI/aya-expanse-8b \
    --src data/filtered_en_es/test.en \
    --tgt data/filtered_en_es/test.es \
    --src-lang-name English \
    --tgt-lang-name Spanish \
    --n-samples 200 \
    --max-length 256 \
    --output experiments/results/ifr_scores_enes.json

echo "=== Done ==="

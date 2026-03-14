#!/bin/bash
#SBATCH --job-name=generate_kd
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/generate_kd_%j.out
#SBATCH --error=logs/generate_kd_%j.err

set -euo pipefail

source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Generating KD data with Aya 32B teacher ==="
python -m src.distillation.generate_kd \
    --src data/filtered/train.cs \
    --ref data/filtered/train.de \
    --output-dir data/kd \
    --teacher CohereForAI/aya-expanse-32b \
    --tp-size 4

echo "=== Done ==="

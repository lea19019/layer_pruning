#!/bin/bash
#SBATCH --job-name=kd_enes
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/kd_enes_%j.out
#SBATCH --error=logs/kd_enes_%j.err

set -euo pipefail

source .venv/bin/activate
export HF_HUB_OFFLINE=1
export C_INCLUDE_PATH=/home/vacl2/python3.11-headers2/usr/include/python3.11

echo "=== Generating KD data (English-Spanish) with Aya 32B teacher ==="
python -m src.distillation.generate_kd \
    --src data/filtered_en_es/train.en \
    --ref data/filtered_en_es/train.es \
    --output-dir data/kd_en_es \
    --teacher CohereForAI/aya-expanse-32b \
    --tp-size 1 \
    --src-ext en \
    --tgt-ext es

echo "=== Done ==="

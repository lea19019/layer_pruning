#!/bin/bash
#SBATCH --job-name=prep_enes
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
#SBATCH --output=logs/prep_enes_%j.out
#SBATCH --error=logs/prep_enes_%j.err

set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Filtering English-Spanish data ==="
python scripts/prep_enes_data.py

echo "=== Done ==="

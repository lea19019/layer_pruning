#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.err

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs

source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Step 1: Download corpus ==="
python -m src.data_prep.download

echo "=== Step 2: Filter ==="
python -m src.data_prep.filter

echo "=== Step 3: Split ==="
python -m src.data_prep.split

echo "=== Done ==="

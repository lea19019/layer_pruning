#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err

# Usage: sbatch scripts/slurm/run_experiment.sh experiments/configs/I1_8.yaml
# Or for heuristic pruning (needs more time):
#   sbatch --time=3-00:00:00 scripts/slurm/run_experiment.sh experiments/configs/M1_8.yaml

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

echo "=== Running experiment: ${CONFIG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi

python -m src.run_experiment "${CONFIG}"

echo "=== Done ==="

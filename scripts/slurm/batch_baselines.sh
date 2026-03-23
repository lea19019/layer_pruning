#!/bin/bash
#SBATCH --job-name=baselines
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
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

# Re-run baselines with the fixed translation code (stop strings).

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

for exp in B0 B1; do
    echo ""
    echo "============================================================"
    echo "=== Running ${exp} at $(date) ==="
    echo "============================================================"
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || {
        echo "=== FAILED ${exp} (exit $?) — continuing to next ==="
    }
done

echo ""
echo "=== Baselines complete at $(date) ==="

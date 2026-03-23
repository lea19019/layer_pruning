#!/bin/bash
#SBATCH --job-name=batch_fft
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=72:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/batch_fft_%j.out
#SBATCH --error=logs/batch_fft_%j.err

# Full fine-tuning variants (Moslem exact replication).

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

CONFIGS=(M5_12 M5_16)

for exp in "${CONFIGS[@]}"; do
    RESULTS="experiments/results/${exp}/results.json"
    if [ -f "$RESULTS" ]; then
        echo "=== SKIP ${exp}: results already exist ==="
        continue
    fi
    echo ""
    echo "============================================================"
    echo "=== Running ${exp} at $(date) ==="
    echo "============================================================"
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || {
        echo "=== FAILED ${exp} (exit $?) — continuing to next ==="
    }
done

echo ""
echo "=== Batch complete at $(date) ==="

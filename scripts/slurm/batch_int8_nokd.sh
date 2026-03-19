#!/bin/bash
#SBATCH --job-name=batch_int8
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
#SBATCH --output=logs/batch_int8_%j.out
#SBATCH --error=logs/batch_int8_%j.err

# INT8 quantization variants (no KD dependency).
# Skips experiments that already have results.

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

CONFIGS=(
    B1_int8
    I3_8_int8 I3_12_int8 I3_16_int8
    I5_threshold_int8
    L3_8_int8 L3_12_int8 L3_16_int8
    M3_8_int8 M3_12_int8 M3_16_int8
)

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

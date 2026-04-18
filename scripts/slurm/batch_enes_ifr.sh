#!/bin/bash
#SBATCH --job-name=enes_ifr
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
#SBATCH --output=logs/enes_ifr_%j.out
#SBATCH --error=logs/enes_ifr_%j.err

# En-es IFR experiments (fast). Skips anything already done.

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

CONFIGS=(
    I1_8_enes I1_12_enes I1_16_enes
    I3_8_int8_enes I3_12_int8_enes I3_16_int8_enes
    I5_threshold_int8_enes
    I2_8_enes I2_12_enes I2_16_enes
    I4_8_int8_enes I4_12_int8_enes I4_16_int8_enes
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

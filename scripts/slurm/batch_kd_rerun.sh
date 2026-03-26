#!/bin/bash
#SBATCH --job-name=kd_rerun
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
#SBATCH --output=logs/kd_rerun_%j.out
#SBATCH --error=logs/kd_rerun_%j.err

# Re-run all cs-de KD experiments with fixed (aligned) data.

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

CONFIGS=(
    I2_8 I2_12 I2_16
    I4_8_int8 I4_12_int8 I4_16_int8
    M2_8 M2_12 M2_16
    M4_8_int8 M4_12_int8 M4_16_int8
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
echo "=== KD re-run complete at $(date) ==="

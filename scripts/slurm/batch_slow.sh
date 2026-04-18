#!/bin/bash
#SBATCH --job-name=slow
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
#SBATCH --output=logs/slow_%j.out
#SBATCH --error=logs/slow_%j.err

# All slow experiments: iterative pruning (both lang pairs, all variants).
# Iterative pruning takes 5-12h per experiment.
# Ordered by layers removed (8 first = fastest).

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

CONFIGS=(
    # cs-de iterative + KD (8 layers first = fastest)
    M2_8 M2_12 M2_16
    M4_8_int8 M4_12_int8 M4_16_int8
    # en-es iterative (core)
    M1_8_enes M1_12_enes M1_16_enes
    # en-es iterative + INT8
    M3_8_int8_enes M3_12_int8_enes M3_16_int8_enes
    # en-es iterative + KD
    M2_8_enes M2_12_enes M2_16_enes
    M4_8_int8_enes M4_12_int8_enes M4_16_int8_enes
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
echo "=== Slow batch complete at $(date) ==="

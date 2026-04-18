#!/bin/bash
#SBATCH --job-name=fast
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
#SBATCH --output=logs/fast_%j.out
#SBATCH --error=logs/fast_%j.err

# All fast experiments: baselines + IFR (both lang pairs, including KD and INT8).
# IFR pruning is instant (score lookup), so these are just FT + eval.
# ~3-4h each, ~20 experiments.

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

CONFIGS=(
    # cs-de KD (IFR only — fast pruning)
    I2_8 I2_12 I2_16
    I4_8_int8 I4_12_int8 I4_16_int8
    # en-es baselines + IFR
    B2_enes B3_enes
    I1_8_enes I1_12_enes I1_16_enes
    I3_8_int8_enes I3_12_int8_enes I3_16_int8_enes
    I5_threshold_int8_enes
    # en-es KD (IFR only — fast pruning)
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
echo "=== Fast batch complete at $(date) ==="

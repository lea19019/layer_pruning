#!/bin/bash
#SBATCH --job-name=batch_heur
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
#SBATCH --output=logs/batch_heur_%j.out
#SBATCH --error=logs/batch_heur_%j.err

# Runs all heuristic (Moslem et al.) experiments that don't need KD data.
# Heuristic pruning is slow (~5-6h each), so this job needs a long walltime.
# Skips any experiment that already has results.

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

CONFIGS=(
    M1_8 M1_12 M1_16
    M3_8 M3_12 M3_16
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

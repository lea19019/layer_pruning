#!/bin/bash
#SBATCH --job-name=job2_i5
#SBATCH --partition=m13h
#SBATCH --qos=gpu

#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/job2_i5_%j.out
#SBATCH --error=logs/job2_i5_%j.err

# Job 2: I5 threshold exploration
# 3 thresholds (0.3, 0.5, 0.7) × 3 configs (prune-only, +FT, +FT+KD) × 2 languages
# Prune-only are fast (~15 min), FT ~3h, FT+KD ~4h
# Run prune-only first (fast), then FT, then FT+KD

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Job 2: I5 threshold exploration ==="
echo "=== Started at $(date) ==="

# Prune-only first (fast)
for exp in I5_t03 I5_t05 I5_t07 I5_t03_enes I5_t05_enes I5_t07_enes; do
    RESULTS="experiments/results/${exp}/results.json"
    if [ -f "$RESULTS" ]; then
        echo "=== SKIP ${exp}: already done ==="
        continue
    fi
    echo ""
    echo "============================================================"
    echo "=== Running ${exp} at $(date) ==="
    echo "============================================================"
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || {
        echo "=== FAILED ${exp} (exit $?) — continuing ==="
    }
done

# Then +FT
for exp in I5_t03_ft I5_t05_ft I5_t07_ft I5_t03_ft_enes I5_t05_ft_enes I5_t07_ft_enes; do
    RESULTS="experiments/results/${exp}/results.json"
    if [ -f "$RESULTS" ]; then
        echo "=== SKIP ${exp}: already done ==="
        continue
    fi
    echo ""
    echo "============================================================"
    echo "=== Running ${exp} at $(date) ==="
    echo "============================================================"
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || {
        echo "=== FAILED ${exp} (exit $?) — continuing ==="
    }
done

# Then +FT+KD
for exp in I5_t03_kd I5_t05_kd I5_t07_kd I5_t03_kd_enes I5_t05_kd_enes I5_t07_kd_enes; do
    RESULTS="experiments/results/${exp}/results.json"
    if [ -f "$RESULTS" ]; then
        echo "=== SKIP ${exp}: already done ==="
        continue
    fi
    echo ""
    echo "============================================================"
    echo "=== Running ${exp} at $(date) ==="
    echo "============================================================"
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || {
        echo "=== FAILED ${exp} (exit $?) — continuing ==="
    }
done

echo ""
echo "=== Job 2 complete at $(date) ==="

#!/bin/bash
#SBATCH --job-name=remaining
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/remaining_%j.out
#SBATCH --error=logs/remaining_%j.err

# Run all remaining experiments:
# 1. B4, B4_enes (no pruning + FT + KD baselines)
# 2. BnB INT8 eval for M3/M4/I4 en-es (reuses existing finetuned models)
# 3. Speed benchmark for I2_12_enes and any new experiments

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Remaining experiments ==="
echo "=== Started at $(date) ==="

# 1. B4 baselines (FT + KD, no pruning)
for exp in B4 B4_enes; do
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

# 2. BnB INT8 eval for en-es (reuses existing models, ~20 min each)
echo ""
echo "============================================================"
echo "=== BnB INT8 evaluations at $(date) ==="
echo "============================================================"
python scripts/run_bnb_eval.py --all-missing || {
    echo "=== BnB eval had some failures ==="
}

# 3. Speed benchmark for anything new
echo ""
echo "============================================================"
echo "=== Speed benchmarks at $(date) ==="
echo "============================================================"
python scripts/benchmark_speed.py

echo ""
echo "=== All done at $(date) ==="

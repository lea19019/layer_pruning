#!/bin/bash
#SBATCH --job-name=j2c_kd
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
#SBATCH --output=logs/j2c_kd_%j.out
#SBATCH --error=logs/j2c_kd_%j.err

# I5 prune + FT + KD — 6 experiments, ~4-5h each

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== I5 + FT + KD ==="
for exp in I5_t03_kd I5_t05_kd I5_t07_kd I5_t03_kd_enes I5_t05_kd_enes I5_t07_kd_enes; do
    RESULTS="experiments/results/${exp}/results.json"
    if [ -f "$RESULTS" ]; then
        echo "=== SKIP ${exp} ==="
        continue
    fi
    echo ""
    echo "=== Running ${exp} at $(date) ==="
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || {
        echo "=== FAILED ${exp} (exit $?) ==="
    }
done
echo "=== Done at $(date) ==="

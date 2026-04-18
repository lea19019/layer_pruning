#!/bin/bash
#SBATCH --job-name=j2a_prune
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/j2a_prune_%j.out
#SBATCH --error=logs/j2a_prune_%j.err

# I5 prune-only (no FT) — all 6 experiments, fast (~15 min each)

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== I5 prune-only ==="
for exp in I5_t03 I5_t05 I5_t07 I5_t03_enes I5_t05_enes I5_t07_enes; do
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

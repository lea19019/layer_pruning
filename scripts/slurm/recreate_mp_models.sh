#!/bin/bash
#SBATCH --job-name=recreate_mp
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
#SBATCH --output=logs/recreate_mp_%j.out
#SBATCH --error=logs/recreate_mp_%j.err

# Recreate MP pruning-only models (just prune and save, skip eval if model exists)

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Recreating MP pruning-only models ==="

for exp in MP_8 MP_12 MP_16 MP_8_enes MP_12_enes; do
    PRUNED="experiments/results/${exp}/pruned_model/config.json"
    if [ -f "$PRUNED" ]; then
        echo "SKIP ${exp}: model exists"
        continue
    fi
    echo "=== Recreating ${exp} at $(date) ==="
    python -m src.run_experiment "experiments/configs/${exp}.yaml" || echo "FAILED ${exp} — continuing"
done

echo "=== Done at $(date) ==="

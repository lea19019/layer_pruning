#!/bin/bash
#SBATCH --job-name=job1_prio
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
#SBATCH --output=logs/job1_prio_%j.out
#SBATCH --error=logs/job1_prio_%j.err

# Job 1: Priority experiments
# 1. B4, B4_enes (FT + KD baselines)
# 2. Pruning-only baselines (IP and MP, both languages)
# 3. BnB INT8 eval for en-es (M3/M4/I4)
# 4. Speed benchmarks for all new experiments

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== Job 1: Priority experiments ==="
echo "=== Started at $(date) ==="

# 1. B4 baselines
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

# 2. Pruning-only baselines (fast - no training)
for exp in IP_8 IP_12 IP_16 IP_8_enes IP_12_enes IP_16_enes \
           MP_8 MP_12 MP_16 MP_8_enes MP_12_enes MP_16_enes; do
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

# 3. BnB INT8 eval for en-es
echo ""
echo "============================================================"
echo "=== BnB INT8 evaluations at $(date) ==="
echo "============================================================"
python scripts/run_bnb_eval.py --all-missing || {
    echo "=== BnB eval had some failures ==="
}

echo ""
echo "=== Job 1 complete at $(date) ==="

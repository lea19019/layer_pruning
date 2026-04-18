#!/bin/bash
#SBATCH --job-name=bench_pruneonly
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/bench_pruneonly_%j.out
#SBATCH --error=logs/bench_pruneonly_%j.err

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export C_INCLUDE_PATH=/home/vacl2/python3.11-headers2/usr/include/python3.11

EXPERIMENTS=(
    IP_8 IP_12 IP_16
    IP_8_enes IP_12_enes IP_16_enes
    MP_8 MP_12 MP_16
    MP_8_enes MP_12_enes MP_16_enes
    I5_t03 I5_t05 I5_t07
    I5_t03_enes I5_t05_enes I5_t07_enes
)

echo "=== Step 1: Recreate missing pruned_model dirs ==="
python scripts/recreate_pruned.py "${EXPERIMENTS[@]}"

echo ""
echo "=== Step 2: Run GPTQ benchmark with --force to overwrite stale results ==="
python scripts/benchmark_speed.py --experiment "${EXPERIMENTS[@]}" --force

echo "=== Done at $(date) ==="

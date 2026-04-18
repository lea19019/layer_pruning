#!/bin/bash
#SBATCH --job-name=bench_all
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
#SBATCH --output=logs/bench_all_%j.out
#SBATCH --error=logs/bench_all_%j.err

# GPTQ 4-bit export + vLLM speed benchmark only (no model recreation).
# Skips experiments that already have speed_benchmark results.
# Safe to rerun — picks up where it left off.

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== GPTQ + vLLM Speed Benchmark ==="
echo "=== Started at $(date) ==="

python scripts/benchmark_speed.py

echo ""
echo "=== Completed at $(date) ==="

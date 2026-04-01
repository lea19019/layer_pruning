#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export C_INCLUDE_PATH=/home/vacl2/python3.11-headers2/usr/include/python3.11

echo "=== Speed Benchmark (vLLM fp16 + GPTQ) ==="
python scripts/benchmark_speed.py --experiment I1_8

echo "=== Done ==="

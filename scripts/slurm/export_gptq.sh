#!/bin/bash
#SBATCH --job-name=gptq_exp
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
#SBATCH --output=logs/gptq_exp_%j.out
#SBATCH --error=logs/gptq_exp_%j.err

set -uo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== GPTQ 4-bit export ==="

python3 -c "
from scripts.benchmark_speed import export_gptq
from pathlib import Path

models = [
    ('I2_8_enes', 'experiments/results/I2_8_enes/finetuned/merged'),
    ('I2_12_enes', 'experiments/results/I2_12_enes/finetuned/merged'),
    ('I2_16_enes', 'experiments/results/I2_16_enes/finetuned/merged'),
    ('M2_8_enes', 'experiments/results/M2_8_enes/finetuned/merged'),
    ('M2_12_enes', 'experiments/results/M2_12_enes/finetuned/merged'),
    ('I5_t05_kd', 'experiments/results/I5_t05_kd/finetuned/merged'),
    ('I5_t05_kd_enes', 'experiments/results/I5_t05_kd_enes/finetuned/merged'),
]

for name, fp16_path in models:
    exp_dir = Path('experiments/results') / name
    gptq_dir = exp_dir / 'gptq_4bit'
    if gptq_dir.exists() and (gptq_dir / 'config.json').exists():
        print(f'SKIP {name}: GPTQ already exists')
        continue
    print(f'Exporting {name}...')
    try:
        export_gptq(fp16_path, exp_dir, bits=4)
    except Exception as e:
        print(f'FAILED {name}: {e}')

print('Done.')
"

echo "=== Completed at $(date) ==="

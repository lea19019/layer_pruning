#!/bin/bash
#SBATCH --job-name=ablation-categorize
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/ablation_categorize_%j.out
#SBATCH --error=logs/ablation_categorize_%j.err

# This is CPU-only — no GPU needed.
# Can also run directly on the login node:
#   source .venv/bin/activate && python -m ablation.scripts.output_categorization

set -euo pipefail
source .venv/bin/activate

echo "Running output categorization"
echo "Start: $(date)"

python -m ablation.scripts.output_categorization

echo "Done: $(date)"

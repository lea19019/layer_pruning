#!/bin/bash
#SBATCH --job-name=ablation-analysis
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/ablation_analysis_%j.out
#SBATCH --error=logs/ablation_analysis_%j.err

set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

ANALYSIS=${1:-hidden_state}
N_SAMPLES=${2:-100}
BATCH_SIZE=${3:-4}

echo "Running ablation analysis: ${ANALYSIS}"
echo "Samples: ${N_SAMPLES}, Batch size: ${BATCH_SIZE}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

case ${ANALYSIS} in
    hidden_state)
        python -m ablation.scripts.hidden_state_divergence \
            --n-samples ${N_SAMPLES} --batch-size ${BATCH_SIZE}
        ;;
    redundancy)
        python -m ablation.scripts.redundancy_analysis \
            --n-samples ${N_SAMPLES} --batch-size ${BATCH_SIZE}
        ;;
    logit_lens)
        python -m ablation.scripts.logit_lens \
            --n-samples ${N_SAMPLES}
        ;;
    attention)
        python -m ablation.scripts.attention_comparison \
            --n-samples ${N_SAMPLES}
        ;;
    *)
        echo "Unknown analysis: ${ANALYSIS}"
        exit 1
        ;;
esac

echo "Done: $(date)"

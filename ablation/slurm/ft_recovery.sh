#!/bin/bash
#SBATCH --job-name=ablation-ft-recovery
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/ablation_ft_recovery_%j.out
#SBATCH --error=logs/ablation_ft_recovery_%j.err

set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# Experiment variant: full_kd, no_kd, frac_25, frac_50, frac_75
VARIANT=${1:-full_kd}
EVAL_STEPS=${2:-200}

echo "FT recovery experiment: ${VARIANT}"
echo "Eval every ${EVAL_STEPS} steps"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

case ${VARIANT} in
    full_kd)
        python -m ablation.scripts.ft_recovery_curve \
            --eval-steps ${EVAL_STEPS}
        ;;
    no_kd)
        python -m ablation.scripts.ft_recovery_curve \
            --no-kd --eval-steps ${EVAL_STEPS}
        ;;
    frac_25)
        python -m ablation.scripts.ft_recovery_curve \
            --data-fraction 0.25 --eval-steps ${EVAL_STEPS}
        ;;
    frac_50)
        python -m ablation.scripts.ft_recovery_curve \
            --data-fraction 0.5 --eval-steps ${EVAL_STEPS}
        ;;
    frac_75)
        python -m ablation.scripts.ft_recovery_curve \
            --data-fraction 0.75 --eval-steps ${EVAL_STEPS}
        ;;
    eval_only)
        # Just evaluate existing checkpoints
        RUN_DIR=${3:-ablation/results/ft_recovery/full_kd}
        python -m ablation.scripts.ft_recovery_curve \
            --eval-only ${RUN_DIR}
        ;;
    *)
        echo "Unknown variant: ${VARIANT}"
        echo "Usage: sbatch ft_recovery.sh [full_kd|no_kd|frac_25|frac_50|frac_75|eval_only] [eval_steps]"
        exit 1
        ;;
esac

echo "Done: $(date)"

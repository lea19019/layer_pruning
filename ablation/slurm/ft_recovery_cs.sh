#!/bin/bash
#SBATCH --job-name=ablation-ft-recovery
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/vacl2/attention_lp
#SBATCH --output=logs/ablation_ft_recovery_%j.out
#SBATCH --error=logs/ablation_ft_recovery_%j.err

set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1

VARIANT=${1:-full_kd}
SAVE_STEPS=${2:-1000}

echo "FT recovery experiment: ${VARIANT}"
echo "Save adapter every ${SAVE_STEPS} steps"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

case ${VARIANT} in
    full_kd)
        python -m ablation.scripts.ft_recovery_curve \
            --save-steps ${SAVE_STEPS}
        ;;
    no_kd)
        python -m ablation.scripts.ft_recovery_curve \
            --no-kd --save-steps ${SAVE_STEPS}
        ;;
    frac_25)
        python -m ablation.scripts.ft_recovery_curve \
            --data-fraction 0.25 --save-steps ${SAVE_STEPS}
        ;;
    frac_50)
        python -m ablation.scripts.ft_recovery_curve \
            --data-fraction 0.5 --save-steps ${SAVE_STEPS}
        ;;
    frac_75)
        python -m ablation.scripts.ft_recovery_curve \
            --data-fraction 0.75 --save-steps ${SAVE_STEPS}
        ;;
    eval_only)
        RUN_DIR=${3:-ablation/results/ft_recovery/full_kd}
        python -m ablation.scripts.ft_recovery_curve \
            --eval-only ${RUN_DIR}
        ;;
    *)
        echo "Unknown variant: ${VARIANT}"
        exit 1
        ;;
esac

echo "Done: $(date)"

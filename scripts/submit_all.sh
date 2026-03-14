#!/bin/bash
# Submit all experiments in the correct order.
# Dependencies are handled via SLURM --dependency flags.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs

echo "=== Phase 0: Data preparation ==="
DATA_JOB=$(sbatch --parsable scripts/slurm/data_prep.sh)
echo "  Data prep: job ${DATA_JOB}"

echo ""
echo "=== Phase 1: IFR scoring (depends on data) ==="
IFR_JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB} scripts/slurm/ifr_score.sh)
echo "  IFR scoring: job ${IFR_JOB}"

echo ""
echo "=== Phase 1b: KD data generation (depends on data) ==="
KD_JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB} scripts/slurm/generate_kd.sh)
echo "  KD generation: job ${KD_JOB}"

echo ""
echo "=== Phase 2: Baselines (depend on data) ==="
for cfg in B0 B1; do
    JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB} \
        scripts/slurm/run_experiment.sh "experiments/configs/${cfg}.yaml")
    echo "  ${cfg}: job ${JOB}"
done

echo ""
echo "=== Phase 3: Heuristic pruning (depends on data) ==="
for n in 8 12 16; do
    # M1: heuristic + FT (no KD dependency)
    JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB} \
        scripts/slurm/run_heuristic.sh "experiments/configs/M1_${n}.yaml")
    echo "  M1_${n}: job ${JOB}"

    # M3: heuristic + FT + quant (no KD dependency)
    JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB} \
        scripts/slurm/run_heuristic.sh "experiments/configs/M3_${n}.yaml")
    echo "  M3_${n}: job ${JOB}"

    # M2, M4: need KD data
    JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB}:${KD_JOB} \
        scripts/slurm/run_heuristic.sh "experiments/configs/M2_${n}.yaml")
    echo "  M2_${n}: job ${JOB}"

    JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB}:${KD_JOB} \
        scripts/slurm/run_heuristic.sh "experiments/configs/M4_${n}.yaml")
    echo "  M4_${n}: job ${JOB}"
done

echo ""
echo "=== Phase 4: IFR-guided pruning (depends on IFR scores) ==="
for n in 8 12 16; do
    # I1: IFR + FT
    JOB=$(sbatch --parsable --dependency=afterok:${IFR_JOB} \
        scripts/slurm/run_experiment.sh "experiments/configs/I1_${n}.yaml")
    echo "  I1_${n}: job ${JOB}"

    # I3: IFR + FT + quant
    JOB=$(sbatch --parsable --dependency=afterok:${IFR_JOB} \
        scripts/slurm/run_experiment.sh "experiments/configs/I3_${n}.yaml")
    echo "  I3_${n}: job ${JOB}"

    # I2, I4: need KD data + IFR scores
    JOB=$(sbatch --parsable --dependency=afterok:${IFR_JOB}:${KD_JOB} \
        scripts/slurm/run_experiment.sh "experiments/configs/I2_${n}.yaml")
    echo "  I2_${n}: job ${JOB}"

    JOB=$(sbatch --parsable --dependency=afterok:${IFR_JOB}:${KD_JOB} \
        scripts/slurm/run_experiment.sh "experiments/configs/I4_${n}.yaml")
    echo "  I4_${n}: job ${JOB}"
done

# I5: threshold-based
JOB=$(sbatch --parsable --dependency=afterok:${IFR_JOB} \
    scripts/slurm/run_experiment.sh "experiments/configs/I5_threshold.yaml")
echo "  I5_threshold: job ${JOB}"

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: squeue -u \$USER"

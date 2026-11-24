#!/bin/bash
#SBATCH --job-name=r1_rebuttal
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=reports/rebuttal/reviewer1/logs/%x-%j.out
#SBATCH --error=reports/rebuttal/reviewer1/logs/%x-%j.err
# Reviewer-1 rebuttal experiments (scripts/rebuttal/), same venv and thread settings as the paper runs.
set -uo pipefail
cd "$HOME/projects/AIPS/scripts/rebuttal"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export R1_WORKERS=62
PY="$HOME/projects/AIPS/.venv/bin/python3"
L="$HOME/projects/AIPS/reports/rebuttal/reviewer1/logs"
hostname; date
"$PY" r1_hurdle_glm.py > "$L/r1_hurdle_glm.log" 2>&1 &
GLM_PID=$!
"$PY" r1_poisson_baselines.py > "$L/r1_poisson_baselines.log" 2>&1; echo "EXP1 exit=$? $(date)"
"$PY" r1_neural_hurdle.py --stage all > "$L/r1_neural_hurdle.log" 2>&1; echo "EXP3 exit=$? $(date)"
wait $GLM_PID; echo "EXP2 exit=$? $(date)"
echo "ALL DONE"

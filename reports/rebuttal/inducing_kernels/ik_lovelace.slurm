#!/bin/bash
#SBATCH --job-name=ik_ablation
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=reports/rebuttal/inducing_kernels/logs/%x-%j.out
#SBATCH --error=reports/rebuttal/inducing_kernels/logs/%x-%j.err
# Inducing-point and kernel ablation of Red-LGCP for the rebuttal (scripts/rebuttal/ik_*.py,
# config/rebuttal/inducing_kernels.yaml). Same machine type as the paper's runs (lovelace).
# 0. cost against M on one core, before anything else runs in the allocation;
# 1. per-run configs of every variant from the definitive per-run configs and seeds (copy in
#    reports/rebuttal/svgp_residual/definitive, since Demetra's reports/seeded_monthly_windows_hurdle lacks seed_1);
# 2. training, largest M first; 3. scoring of every variant and of the paper's runs.
# SKIP_TRAIN=1 (sbatch --export=ALL,SKIP_TRAIN=1 ...) reruns only step 3.
set -uo pipefail
cd "$HOME/projects/AIPS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PY="$HOME/projects/AIPS/.venv/bin/python3"
B=reports/rebuttal/inducing_kernels
D=reports/rebuttal/svgp_residual/definitive
N=${SLURM_CPUS_PER_TASK:-120}
hostname; lscpu | grep -m1 "Model name"; date

if [ "${SKIP_TRAIN:-0}" != 1 ]; then
"$PY" scripts/rebuttal/ik_timing.py --config $D/seed_1/configs/2025_may_01_07.yaml --M 50 100 200 300 500 1000 \
  --steps 20 --out $B/timing.csv > $B/logs/timing.log 2>&1; echo "TIMING exit=$? $(date)"
(cd scripts/rebuttal && "$PY" ik_make_configs.py --definitive "../../$D" --out "../../$B")
cat $B/jobs.txt | xargs -P $N -I{} bash -c \
  '"$PY" scripts/rebuttal/ik_train.py --config "{}" > "{}.log" 2>&1; echo "trained {} exit=$? $(date +%H:%M)"'
echo "TRAINING DONE $(date)"
fi

cd scripts/rebuttal
"$PY" ik_score.py --workers $N > "../../$B/logs/score.log" 2>&1; echo "SCORING exit=$? $(date)"
echo "ALL DONE"

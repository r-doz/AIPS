#!/bin/bash
#SBATCH --job-name=twostage
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=reports/rebuttal/twostage/logs/%x-%j.out
#SBATCH --error=reports/rebuttal/twostage/logs/%x-%j.err
# Two-stage ablation ladder for the rebuttal (scripts/rebuttal/twostage_*.py).
# 1-2. Train Red-LGCP in two separate stages (classifier, then zero-truncated LGCP) from the definitive per-run
#      configs and seeds of the joint model (copy in reports/rebuttal/svgp_residual/definitive, since Demetra's
#      reports/seeded_monthly_windows_hurdle lacks seed_1).
# 3.   Score steps S1-S3 for every seed and test week.
# SKIP_TRAIN=1 (sbatch --export=ALL,SKIP_TRAIN=1 ...) reruns only step 3.
set -uo pipefail
cd "$HOME/projects/AIPS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PY="$HOME/projects/AIPS/.venv/bin/python3"
B=reports/rebuttal/twostage
hostname; lscpu | grep -m1 "Model name"; date

if [ "${SKIP_TRAIN:-0}" != 1 ]; then
"$PY" - <<'PYEOF'
import glob, os, yaml
B, D = "reports/rebuttal/twostage", "reports/rebuttal/svgp_residual/definitive"
for f in sorted(glob.glob(f"{D}/seed_*/configs/*.yaml")):
    seed, win = f.split("/")[-3], os.path.basename(f)[:-5]
    c = yaml.safe_load(open(f))
    c["report_root"] = os.path.abspath(f"{B}/runs/{seed}/runs")
    os.makedirs(f"{B}/configs/{seed}", exist_ok=True)
    yaml.safe_dump(c, open(f"{B}/configs/{seed}/{win}.yaml", "w"), sort_keys=False)
print("configs:", len(glob.glob(f"{B}/configs/seed_*/*.yaml")))
PYEOF
ls $B/configs/seed_*/*.yaml | xargs -P 80 -I{} bash -c \
  '"$PY" scripts/rebuttal/twostage_hurdle.py --config "{}" > "{}.log" 2>&1; echo "trained {} exit=$? $(date +%H:%M)"'
echo "TRAINING DONE $(date)"
fi

cd scripts/rebuttal
"$PY" twostage_score.py --workers 80 > "../../$B/logs/score.log" 2>&1; echo "SCORING exit=$? $(date)"
echo "ALL DONE"

#!/bin/bash
#SBATCH --job-name=svgp_residual
#SBATCH --partition=turing-wide
#SBATCH --nodelist=turing-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=150G
#SBATCH --time=10:00:00
#SBATCH --output=reports/rebuttal/svgp_residual/logs/%x-%j.out
#SBATCH --error=reports/rebuttal/svgp_residual/logs/%x-%j.err
# SVGP residual-variance experiment for the rebuttal (scripts/rebuttal/svgp_*.py).
# Retrains the paper's Red-LGCP (SoR) and the SVGP variant from the definitive per-run configs and seeds
# (copied to definitive/), scores all runs, and decomposes the latent field of both retrained sets.
set -uo pipefail
cd "$HOME/projects/AIPS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PY="$HOME/projects/AIPS/.venv/bin/python3"
B=reports/rebuttal/svgp_residual
hostname; lscpu | grep -m1 "Model name"; date

# SKIP_TRAIN=1 (sbatch --export=ALL,SKIP_TRAIN=1 ...) reruns only steps 3-4 on the trained models
if [ "${SKIP_TRAIN:-0}" != 1 ]; then
# 1. per-run configs: the definitive ones with a new report_root
"$PY" - <<'PYEOF'
import glob, os, yaml
B = "reports/rebuttal/svgp_residual"
for f in sorted(glob.glob(f"{B}/definitive/seed_*/configs/*.yaml")):
    seed, win = f.split("/")[-3], os.path.basename(f)[:-5]
    c = yaml.safe_load(open(f))
    for v in ("svgp", "sor"):
        c["report_root"] = os.path.abspath(f"{B}/runs_{v}/{seed}/runs")
        os.makedirs(f"{B}/configs_{v}/{seed}", exist_ok=True)
        yaml.safe_dump(c, open(f"{B}/configs_{v}/{seed}/{win}.yaml", "w"), sort_keys=False)
print("configs:", len(glob.glob(f"{B}/configs_svgp/seed_*/*.yaml")), len(glob.glob(f"{B}/configs_sor/seed_*/*.yaml")))
PYEOF

# 2. train both variants for every (seed, window), 40 at a time
ls $B/configs_svgp/seed_*/*.yaml $B/configs_sor/seed_*/*.yaml | xargs -P 40 -I{} bash -c \
  'v=$(echo "{}" | sed -E "s|.*/configs_([a-z]+)/.*|\1|"); "$PY" scripts/rebuttal/svgp_residual.py --config "{}" --variant $v > "{}.log" 2>&1; echo "trained {} exit=$? $(date +%H:%M)"'
echo "TRAINING DONE $(date)"
fi

# 3. score the paper's runs and both retrained sets
cd scripts/rebuttal
"$PY" svgp_score_all.py --workers 40 > "../../$B/logs/score_all.log" 2>&1; echo "SCORING exit=$? $(date)"

# 4. latent-field decomposition and covariate effects (Table 6 analysis) of both retrained sets
"$PY" svgp_interpret.py "../../$B/runs_svgp" "../../$B/interpretability_svgp" > "../../$B/logs/interpret_svgp.log" 2>&1 &
"$PY" svgp_interpret.py "../../$B/runs_sor" "../../$B/interpretability_sor" > "../../$B/logs/interpret_sor.log" 2>&1 &
wait
echo "INTERPRET done $(date)"
echo "ALL DONE"

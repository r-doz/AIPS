#!/bin/bash
# Train Red-LGCP on the new datasets (configs from scripts/rebuttal/newdata_make_configs.py), locally,
# N runs in parallel at low priority, one thread each. Run from the repository root.
N=${1:-5}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p reports/rebuttal/new_data/logs
cat reports/rebuttal/new_data/jobs.txt | xargs -P "$N" -I{} bash -c \
  'nice -n 15 python scripts/rebuttal/ik_train.py --config "{}" > "{}.log" 2>&1; echo "trained {} exit=$? $(date +%H:%M)"'
echo "TRAINING DONE $(date)"

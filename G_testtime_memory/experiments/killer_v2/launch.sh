#!/bin/bash
set -euo pipefail
cd /home/liuyibo/auto-llm-driving/TheoryScout/G_testtime_memory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
  prepare) script=prepare.py; limit=900 ;;
  smoke) script=run.py; limit=120 ;;
  formal) script=run.py; limit=3200 ;;
  analyze) script=analyze.py; limit=240 ;;
  *) exit 2 ;;
esac
args=()
if [[ "$script" == run.py ]]; then args=("$1"); fi
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  timeout --signal=TERM "${limit}s" /home/liuyibo/venvs/theory/bin/python "experiments/killer_v2/$script" "${args[@]}"

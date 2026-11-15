#!/bin/bash
set -euo pipefail
cd /home/liuyibo/auto-llm-driving/TheoryScout/G_testtime_memory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
  smoke|formal)
    systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
      timeout --signal=TERM 3500s /home/liuyibo/venvs/theory/bin/python experiments/killer/run_v2.py --mode "$1"
    ;;
  analyze)
    systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
      timeout --signal=TERM 180s /home/liuyibo/venvs/theory/bin/python experiments/killer/analyze_v2.py
    ;;
  *) exit 2 ;;
esac

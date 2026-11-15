#!/usr/bin/env bash
set -euo pipefail
cd /home/liuyibo/auto-llm-driving/TheoryScout/D_optimizer_speedup_phase
exec systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python v6_review_checks/extra_checks.py

#!/bin/bash
set -euo pipefail
cd /home/liuyibo/auto-llm-driving/TheoryScout/G_testtime_memory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  timeout --signal=TERM 60s /home/liuyibo/venvs/theory/bin/python experiments/killer_v3/describe.py

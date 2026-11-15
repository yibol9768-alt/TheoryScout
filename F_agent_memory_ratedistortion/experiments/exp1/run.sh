#!/bin/bash
set -euo pipefail
cd /home/liuyibo/auto-llm-driving/TheoryScout/F_agent_memory_ratedistortion
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 timeout --signal=TERM --kill-after=10s 3540s /home/liuyibo/venvs/theory/bin/python -u experiments/exp1/run.py "$@"

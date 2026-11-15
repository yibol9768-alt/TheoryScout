#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/theory-week1-mpl
mkdir -p experiments/week1_results
timeout --signal=TERM --kill-after=5s 3300s /home/liuyibo/venvs/theory/bin/python -u experiments/week1.py > experiments/week1_results/run.log 2>&1

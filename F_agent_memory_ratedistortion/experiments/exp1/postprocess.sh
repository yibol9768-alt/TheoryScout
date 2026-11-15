#!/bin/bash
set -euo pipefail
cd /home/liuyibo/auto-llm-driving/TheoryScout/F_agent_memory_ratedistortion
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for script in summarize.py calibration_diagnostics.py examples.py plot.py export_dp.py verify_artifacts.py make_report.py; do
  systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python "experiments/exp1/$script"
done
sha256sum -c PREREG_EXP1.sha256
sha256sum -c experiments/exp1/CODE.sha256

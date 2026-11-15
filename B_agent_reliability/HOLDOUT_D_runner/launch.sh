#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
umask 077
if [[ " $* " == *" --live "* ]]; then
  echo '本次交付禁止真实调用；未来真实运行需另获明确授权。' >&2
  exit 2
fi
stamp=$(date -u +%Y%m%dT%H%M%S%N)
parent_sid=$(ps -o sid= -p $$ | tr -d ' ')
log="HOLDOUT_D_runner/HOLDOUT_D_LAUNCH_${stamp}.log"
setsid nohup systemd-run --user --scope --unit="holdout-d-${stamp}" \
  -p MemoryMax=2G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  HOLDOUT_D_PARENT_SID="$parent_sid" PYTHONDONTWRITEBYTECODE=1 python3 -B HOLDOUT_D_runner/run.py "$@" \
  </dev/null >"$log" 2>&1 &
printf '%s\n' "启动包装PID=$! 日志=$log"

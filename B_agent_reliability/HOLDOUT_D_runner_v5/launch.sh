#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
umask 077
# 和直接 Python 入口共用解析器；校验在创建日志/会话/客户端之前完成。
PYTHONDONTWRITEBYTECODE=1 python3 -B HOLDOUT_D_runner_v5/run.py --entry-check "$@"
stamp=$(date -u +%Y%m%dT%H%M%S%N)
parent_sid=$(ps -o sid= -p $$ | tr -d ' ')
log="HOLDOUT_D_runner_v5/HOLDOUT_D_LAUNCH_${stamp}.log"
setsid nohup systemd-run --user --scope --unit="holdout-d-v5-${stamp}" \
  -p MemoryMax=2G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  HOLDOUT_D_PARENT_SID="$parent_sid" PYTHONDONTWRITEBYTECODE=1 python3 -B HOLDOUT_D_runner_v5/run.py "$@" \
  </dev/null >"$log" 2>&1 &
printf '%s\n' "启动包装PID=$! 日志=$log"

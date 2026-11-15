#!/usr/bin/env bash
set -euo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${W4_PYTHON:-/home/liuyibo/venvs/theory/bin/python}"
out_dir="${1:-$base/output}"
cd "$base"
sha256sum --check SHA256SUMS
exec systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  timeout 540 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  "$python_bin" -B "$base/W4_replay.py" --out "$out_dir"

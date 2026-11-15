#!/usr/bin/env bash
set -eu
set +x
export SOLKEY="$(sed -n '13s/^[[:space:]]*-[[:space:]]*//p' /home/liuyibo/.local/share/codex-workflow/llm-proxy-private/config-plus.yaml | tr -d "\"'")"
test -n "$SOLKEY"
python3 -B -u experiments/real/restart2_holdout_c.py formal
unset SOLKEY
exec python3 -B -u experiments/real/deliver_holdout_c.py

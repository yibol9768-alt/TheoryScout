#!/usr/bin/env bash
set -eu
for dim in 64 128 256 512
do
  /home/liuyibo/venvs/theory/bin/python review18_checks/independent_mc.py --d "$dim" --outer 8192 --inner 64 > "review18_checks/strict/mc_d${dim}.log" 2>&1
done
/home/liuyibo/venvs/theory/bin/python review18_checks/independent_rank.py > review18_checks/strict/rank_checks.log 2>&1
/home/liuyibo/venvs/theory/bin/python review18_checks/analyze_review.py > review18_checks/strict/analysis.log 2>&1

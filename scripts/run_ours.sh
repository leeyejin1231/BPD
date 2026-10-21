#!/bin/bash
# Usage: scripts/run_ours.sh <gpu> <llada|dream> <task> [<task> ...]      task: gsm8k | humaneval | mbpp | math500 | ifeval
# BPD (ours) through the eval_*_hf.py harnesses with the paper setting: one pivot per interval, threshold TAU (default 0.9),
# gen 256 / block 32, one job after another on one GPU. Overrides: see scripts/common.sh (MODEL, GEN, BLOCK, LIMIT, ...).
# Outputs: results/${PREFIX:-ours}_<label>_<task>_g<gen>_b<block>_t<tau>, logs/<same>.out
source "$(dirname "$0")/common.sh"
gpu=$1; resolve_model $2; shift 2
tau=${TAU:-0.9}; n=${N:-1}
set_prefix ours
for task in "$@"; do
  out=results/${prefix}_${label}_${task}_g${gen}_b${block}_t${tau/./}
  run_job $gpu "$task ours tau=$tau n=$n" $task $out "" --methods ours --n $n --confidence-threshold $tau
done
finish_queue $gpu

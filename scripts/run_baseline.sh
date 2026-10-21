#!/bin/bash
# Usage: scripts/run_baseline.sh <gpu> <llada|dream> <task:method[:shard]> [<task:method[:shard]> ...]
# Baselines through the eval_*_hf.py harnesses, gen 256 / block 32, one job after another on one GPU.
# task:   gsm8k | humaneval | mbpp | math500 | ifeval
# method: vanilla (low-confidence remasking) | hier (Hierarchy-dLLM) | wino (per-task thresholds) | fastdllm (threshold 0.9, prefix cache)
#         WINO and Fast-dLLM load their own model implementation, so every job runs one method.
# shard k/n (mbpp/math500/ifeval only) writes results/..._shard<k>-<n>. Overrides: see scripts/common.sh.
# Outputs: results/${PREFIX:-baseline}_<label>_<task>_g<gen>_b<block>_<method>, logs/<same>.out
source "$(dirname "$0")/common.sh"
gpu=$1; resolve_model $2; shift 2
set_prefix baseline
for job in "$@"; do
  IFS=: read task method shard <<< "$job"
  case $method in
    vanilla)  cfg=configs/vanilla_variant.json;;
    hier)     cfg=configs/hierarchy_r03_variant.json;;
    wino)     cfg=configs/wino_${task}.json;;
    fastdllm) cfg=configs/fastdllm_prefix.json;;
    *) echo "unknown method '$method' (vanilla|hier|wino|fastdllm)"; exit 2;;
  esac
  [ -f "$cfg" ] || { echo "missing config $cfg"; exit 2; }
  out=results/${prefix}_${label}_${task}_g${gen}_b${block}_${method}${shard:+_shard${shard/\//-}}
  run_job $gpu "$job" $task $out $cfg ${shard:+--shard $shard}
done
finish_queue $gpu

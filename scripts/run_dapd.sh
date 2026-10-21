#!/bin/bash
# Usage: scripts/run_dapd.sh <gpu> <llada|dream> <task> [<task> ...]      task: gsm8k | humaneval | mbpp | math500 | ifeval
# DAPD (official package in DAPD/, dapd_decode.py) through the eval_*_hf.py harnesses, gen 256 / block 32, one job after
# another on one GPU. ALG=dapd_direct|dapd_staged (default dapd_direct); TAU_MIN/TAU_MAX override the per-task defaults
# of the official evaluation scripts. For LLaDA-1.5: MODEL=GSAI-ML/LLaDA-1.5 MODEL_LABEL=llada15. Overrides: see scripts/common.sh.
# Outputs: results/${PREFIX:-dapd}_<label>_<task>_g<gen>_b<block>_<alg>, logs/<same>.out
source "$(dirname "$0")/common.sh"
gpu=$1; resolve_model $2; shift 2
alg=${ALG:-dapd_direct}
set_prefix dapd
for task in "$@"; do
  case $family:$alg:$task in
    llada:dapd_direct:gsm8k|llada:dapd_direct:math500) tmin=0.005; tmax=0.05;;
    llada:dapd_direct:mbpp)                            tmin=0.01;  tmax=0.02;;
    llada:dapd_direct:*)                               tmin=0.01;  tmax=0.05;;
    llada:dapd_staged:gsm8k|llada:dapd_staged:math500) tmin=0.01;  tmax=0.05;;
    llada:dapd_staged:*)                               tmin=0.01;  tmax=0.15;;
    dream:*:ifeval)                                    tmin=0.1;   tmax=0.5;;
    dream:dapd_direct:*)                               tmin=0.005; tmax=0.01;;
    dream:dapd_staged:*)                               tmin=0.005; tmax=0.05;;
    *) echo "unknown ALG '$alg' (dapd_direct|dapd_staged)"; exit 2;;
  esac
  tmin=${TAU_MIN:-$tmin}; tmax=${TAU_MAX:-$tmax}
  cfg=configs/${alg}_${family}_${task}_tau${tmin}_${tmax}.json
  echo "{\"$alg\": {\"decoder\": \"dapd\", \"alg\": \"$alg\", \"tau_min\": $tmin, \"tau_max\": $tmax}}" > $cfg
  out=results/${prefix}_${label}_${task}_g${gen}_b${block}_${alg}
  run_job $gpu "$task $alg tau=[$tmin,$tmax]" $task $out $cfg
done
finish_queue $gpu

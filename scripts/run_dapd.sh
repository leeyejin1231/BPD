#!/bin/bash
# Usage: scripts/run_dapd.sh <gpu> <task> [<task> ...]      task: gsm8k | humaneval | mbpp | math500 | ifeval
# DAPD (official dapd package in DAPD/, dapd_decode.py) on LLaDA-1.5 through the eval_*_hf.py harnesses, gen 256 / block 32,
# one job after another on one GPU. ALG=dapd_direct|dapd_staged (default dapd_direct); TAU_MIN/TAU_MAX override the official
# per-task defaults of DAPD/evaluation/lm-evaluation-harness/exp/dapd/_run_llada_task.sh; GEN/BLOCK override lengths;
# FAMILY=llada (default, LLaDA-1.5) | dream (Dream-v0-Instruct-7B, official Dream tau defaults from _run_dream_task.sh);
# MODEL=<snapshot> MODEL_LABEL=<tag> pick another checkpoint; LIMIT=<n> evaluates n problems (smoke test, prefix smoke_dapd).
# Outputs: results/${PREFIX:-dapd}_<model_label>_<task>_g<gen>_b<block>_<alg>, logs/<same>.out, queue log logs/<prefix>_<model_label>_gpu<gpu>.log
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
gpu=$1; shift 1
if [ -d /scratch/ssgyejin/hub ]; then
  PY=${PY:-/home/ssgyejin/miniconda3/envs/diffuguard/bin/python}; HUB=/scratch/ssgyejin/hub; D=/scratch/ssgyejin/datasets
else
  PY=${PY:-$PWD/.venv/bin/python}; HUB=/mnt/shared/huggingface-cache/hub; D=/mnt/shared/huggingface-cache/datasets
fi
family=${FAMILY:-llada}
case $family in
  llada) MODEL=${MODEL:-$HUB/models--GSAI-ML--LLaDA-1.5/snapshots/84346fd91ba60252d260022201ad6fc5a3468fb2}; label=${MODEL_LABEL:-llada15};;
  dream) MODEL=${MODEL:-$HUB/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae}; label=${MODEL_LABEL:-dream};;
  *) echo "unknown FAMILY $family (llada|dream)"; exit 2;;
esac
alg=${ALG:-dapd_direct}
gen=${GEN:-256}; block=${BLOCK:-32}
limit=${LIMIT:-}; prefix=${PREFIX:-$([ -z "$limit" ] && echo dapd || echo smoke_dapd)}
mkdir -p logs results configs
first_existing() { for f in "$@"; do for root in $D $HUB/datasets; do [ -f "$root/$f" ] && { echo "$root/$f"; return; }; done; done; echo "$D/$1"; }
for task in "$@"; do
  case $task in
    gsm8k)     data=$(first_existing openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow); script=eval_gsm8k_hf.py; extra="--num-samples ${limit:-0}";;
    humaneval) data=$(first_existing openai___openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow); script=eval_humaneval_hf.py; extra=${limit:+--limit $limit};;
    mbpp)      data=$(first_existing google-research-datasets___mbpp/full/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c/mbpp-test.arrow); script=eval_bench_hf.py; extra="--task mbpp ${limit:+--limit $limit}";;
    math500)   data=$(first_existing HuggingFaceH4___math-500/default/0.0.0/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow); script=eval_bench_hf.py; extra="--task math500 ${limit:+--limit $limit}";;
    ifeval)    data=$(first_existing google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84/if_eval-train.arrow); script=eval_bench_hf.py; extra="--task ifeval ${limit:+--limit $limit}";;
    *) echo "unknown task $task"; exit 2;;
  esac
  [ -f "$data" ] || { echo "dataset file missing: $data (falling back to the harness default / HF id)"; data=; }
  case $family:$alg:$task in
    llada:dapd_direct:gsm8k|llada:dapd_direct:math500) tmin=0.005; tmax=0.05;;
    llada:dapd_direct:mbpp)                            tmin=0.01;  tmax=0.02;;
    llada:dapd_direct:*)                               tmin=0.01;  tmax=0.05;;
    llada:dapd_staged:gsm8k|llada:dapd_staged:math500) tmin=0.01;  tmax=0.05;;
    llada:dapd_staged:*)                               tmin=0.01;  tmax=0.15;;
    dream:*:ifeval)                                    tmin=0.1;   tmax=0.5;;
    dream:dapd_direct:*)                               tmin=0.005; tmax=0.01;;
    dream:dapd_staged:*)                               tmin=0.005; tmax=0.05;;
    *) echo "unknown ALG $alg (dapd_direct|dapd_staged)"; exit 2;;
  esac
  tmin=${TAU_MIN:-$tmin}; tmax=${TAU_MAX:-$tmax}
  cfg=configs/${alg}_${family}_${task}_tau${tmin}_${tmax}.json
  echo "{\"$alg\": {\"decoder\": \"dapd\", \"alg\": \"$alg\", \"tau_min\": $tmin, \"tau_max\": $tmax}}" > $cfg
  out=results/${prefix}_${label}_${task}_g${gen}_b${block}_${alg}
  log=logs/${prefix}_${label}_${task}_g${gen}_b${block}_${alg}.out
  echo "=== $(date) START $task $alg tau=[$tmin,$tmax] -> $out" >> logs/${prefix}_${label}_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $script --model-family $family --model $MODEL --local-files-only ${data:+--dataset $data} \
    --output $out --gen-length $gen --block-length $block --steps $gen $extra --variant-config $cfg > $log 2>&1
  rc=$?
  echo "=== $(date) END $task exit=$rc" >> logs/${prefix}_${label}_gpu${gpu}.log
  [ $rc -ne 0 ] && { echo "=== $(date) QUEUE STOPPED gpu$gpu after $task failed" >> logs/${prefix}_${label}_gpu${gpu}.log; exit $rc; }
done
echo "=== $(date) QUEUE DONE gpu$gpu" >> logs/${prefix}_${label}_gpu${gpu}.log

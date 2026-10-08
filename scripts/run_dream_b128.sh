#!/bin/bash
# Usage: run_dream_b128.sh <gpu> <task:method> [<task:method> ...]
# Dream-v0-Instruct-7B baselines, gen 256 (ifeval: 512) / block 128, one job after another on one GPU.
# task: gsm8k | humaneval | mbpp | math500 (gen 256) | ifeval (gen 512). method: vanilla | hier (paper rule, hierarchy_decode.py) | wino | fastdllm (t0.9, prefix cache).
# LIMIT=<n> evaluates only n problems into results/smoke_dream_* (smoke test).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The user slice allows 1024 tasks in total; keep CPU thread pools small.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
PY=/home/yejin/anaconda3/bin/python
MODEL=/mnt/shared/huggingface-cache/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae
D=/mnt/shared/huggingface-cache/hub
gpu=$1; shift
limit=${LIMIT:-}; prefix=$([ -z "$limit" ] && echo dream || echo smoke_dream)
mkdir -p logs results
for job in "$@"; do
  IFS=: read task method <<< "$job"
  case $task in
    gsm8k)     data=$D/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow; script=eval_gsm8k.py; extra="--num-samples ${limit:-1319}";;
    humaneval) data=$D/datasets/openai___openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow; script=eval_humaneval.py; extra=${limit:+--limit $limit};;
    mbpp)      data=$D/google-research-datasets___mbpp/full/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c/mbpp-test.arrow; script=eval_bench.py; extra="--task mbpp ${limit:+--limit $limit}";;
    math500)   data=/mnt/shared/huggingface-cache/datasets/HuggingFaceH4___math-500/default/0.0.0/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow; script=eval_bench.py; extra="--task math500 ${limit:+--limit $limit}";;
    ifeval)    data=$D/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84/if_eval-train.arrow; script=eval_bench.py; extra="--task ifeval ${limit:+--limit $limit}";;
    *) echo "unknown task $task"; exit 2;;
  esac
  case $method in
    vanilla)  cfg=configs/vanilla_variant.json;;
    hier)     cfg=configs/hierarchy_r03_variant.json;;
    wino)     cfg=configs/wino_${task}.json;;
    fastdllm) cfg=configs/fastdllm_prefix.json;;
    *) echo "unknown method $method"; exit 2;;
  esac
  gen=$([ $task = ifeval ] && echo 512 || echo 256)
  out=results/${prefix}_${task}_g${gen}_b128_${method}
  echo "=== $(date) START $job -> $out" >> logs/${prefix}_b128_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $script --model-family dream --model $MODEL --dataset $data \
    --output $out --gen-length $gen --block-length 128 --steps $gen $extra --variant-config $cfg \
    > logs/${prefix}_${task}_${method}_b128.out 2>&1
  rc=$?
  echo "=== $(date) END $job exit=$rc" >> logs/${prefix}_b128_gpu${gpu}.log
  [ $rc -ne 0 ] && { echo "=== $(date) QUEUE STOPPED gpu$gpu after $job failed" >> logs/${prefix}_b128_gpu${gpu}.log; exit $rc; }
done
echo "=== $(date) QUEUE DONE gpu$gpu" >> logs/${prefix}_b128_gpu${gpu}.log

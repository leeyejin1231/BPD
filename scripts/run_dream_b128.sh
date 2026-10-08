#!/bin/bash
# Usage: run_dream_b128.sh <gpu> <task:method[:shard]> [<task:method[:shard]> ...]
# Dream-v0-Instruct-7B baselines, gen 256 / block 128 (GEN=<n> overrides), one job after another on one GPU.
# task: gsm8k | humaneval | mbpp | math500 | ifeval. method: vanilla | hier (hierarchy_decode.py) | wino | fastdllm (t0.9, prefix cache) | ours (t0.75 p32 s2) | ours_prefix | ours_s0_prefix.
# shard k/n (eval_bench tasks only) writes results/..._shard<k>-<n>; report_bench.py merges shards.
# LIMIT=<n> evaluates only n problems into results/smoke_dream_* (smoke test).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
SNAP=models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae
if [ -d /scratch/ssgyejin/hub ]; then   # SLURM cluster (node2x)
  PY=/home/ssgyejin/miniconda3/envs/diffuguard/bin/python; MODEL=/scratch/ssgyejin/hub/$SNAP; D=/scratch/ssgyejin/datasets
else                                     # node57-style host
  PY=/home/yejin/anaconda3/bin/python; MODEL=/mnt/shared/huggingface-cache/hub/$SNAP; D=/mnt/shared/huggingface-cache/datasets
fi
gpu=$1; shift
limit=${LIMIT:-}; prefix=$([ -z "$limit" ] && echo dream || echo smoke_dream)
mkdir -p logs results
for job in "$@"; do
  IFS=: read task method shard <<< "$job"
  case $task in
    gsm8k)     data=$D/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow; script=eval_gsm8k.py; extra="--num-samples ${limit:-1319}";;
    humaneval) data=$D/openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow; script=eval_humaneval.py; extra=${limit:+--limit $limit};;
    mbpp)      data=$D/google-research-datasets___mbpp/full/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c/mbpp-test.arrow; script=eval_bench.py; extra="--task mbpp ${limit:+--limit $limit}";;
    math500)   data=$D/HuggingFaceH4___math-500/default/0.0.0/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow; script=eval_bench.py; extra="--task math500 ${limit:+--limit $limit}";;
    ifeval)    data=$D/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84/if_eval-train.arrow; script=eval_bench.py; extra="--task ifeval ${limit:+--limit $limit}";;
    *) echo "unknown task $task"; exit 2;;
  esac
  case $method in
    vanilla)  cfg=configs/vanilla_variant.json;;
    hier)     cfg=configs/hierarchy_r03_variant.json;;
    wino)     cfg=configs/wino_${task}.json;;
    fastdllm) cfg=configs/fastdllm_prefix.json;;
    ours)     cfg=configs/confpivot_v2_t075_p32_s2.json;;
    ours_prefix)    cfg=configs/confpivot_v2_t075_p32_s2_prefix.json;;
    ours_s0_prefix) cfg=configs/confpivot_v2_t075_p32_s0_prefix.json;;
    ours_t06)        cfg=configs/confpivot_v2_t06_p32_s2.json;;
    ours_t06_prefix) cfg=configs/confpivot_v2_t06_p32_s2_prefix.json;;
    *) echo "unknown method $method"; exit 2;;
  esac
  gen=${GEN:-256}
  out=results/${prefix}_${task}_g${gen}_b128_${method}
  [ -n "$shard" ] && { out=${out}_shard${shard/\//-}; extra="$extra --shard $shard"; }
  log=logs/${prefix}_${task}_${method}_b128${shard:+_shard${shard/\//-}}.out
  echo "=== $(date) START $job -> $out" >> logs/${prefix}_b128_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $script --model-family dream --model $MODEL --dataset $data \
    --output $out --gen-length $gen --block-length 128 --steps $gen $extra --variant-config $cfg > $log 2>&1
  rc=$?
  echo "=== $(date) END $job exit=$rc" >> logs/${prefix}_b128_gpu${gpu}.log
  [ $rc -ne 0 ] && { echo "=== $(date) QUEUE STOPPED gpu$gpu after $job failed" >> logs/${prefix}_b128_gpu${gpu}.log; exit $rc; }
done
echo "=== $(date) QUEUE DONE gpu$gpu" >> logs/${prefix}_b128_gpu${gpu}.log

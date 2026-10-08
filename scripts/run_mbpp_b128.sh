#!/bin/bash
# Usage: run_mbpp_b128.sh <gpu> <cfg> [<cfg> ...]
# MBPP full test (500 problems), gen 256 / block 128, one config after another on one GPU.
# cfg: vanilla_s256 | hier_r03 | wino | fastdllm_* | tXX_pYY_sZ_prefix (ours, decode_confidence_v2).
# LIMIT=<n> evaluates the first n problems into results/smoke_mbpp_* (smoke test).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The user slice allows 1024 tasks in total; keep CPU thread pools small.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=2
# Fast-dLLM's modeling uses torch.compile; inductor would otherwise fork 32 compile workers.
export TORCHINDUCTOR_COMPILE_THREADS=1
PY=/home/yejin/anaconda3/bin/python
MODEL=/mnt/shared/huggingface-cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
DATA=/mnt/shared/huggingface-cache/hub/google-research-datasets___mbpp/full/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c/mbpp-test.arrow
gpu=$1; shift
limit=${LIMIT:-}; prefix=$([ -z "$limit" ] && echo mbpp || echo smoke_mbpp); extra=${limit:+--limit $limit}
mkdir -p logs results
variant_config() {
  case $1 in
    wino) echo wino_mbpp.json;;
    vanilla_*) echo vanilla_variant.json;;
    hier_r03) echo hierarchy_r03_variant.json;;
    fastdllm_*) echo $1.json;;
    *) echo confpivot_v2_$1.json;;
  esac
}
for cfg in "$@"; do
  echo "=== $(date) START $cfg" >> logs/${prefix}_b128_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES=$gpu $PY -u eval_bench.py --task mbpp --model $MODEL --dataset $DATA \
    --output results/${prefix}_g256_b128_$cfg --gen-length 256 --block-length 128 --steps 256 $extra \
    --variant-config configs/$(variant_config $cfg) > logs/${prefix}_g256_${cfg}_b128.out 2>&1
  rc=$?
  echo "=== $(date) END $cfg exit=$rc" >> logs/${prefix}_b128_gpu${gpu}.log
  [ $rc -ne 0 ] && { echo "=== $(date) QUEUE STOPPED gpu$gpu after $cfg failed" >> logs/${prefix}_b128_gpu${gpu}.log; exit $rc; }
done
echo "=== $(date) QUEUE DONE gpu$gpu" >> logs/${prefix}_b128_gpu${gpu}.log

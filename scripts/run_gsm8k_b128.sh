#!/bin/bash
# Usage: run_gsm8k_b128.sh <gpu> <cfg> [<cfg> ...]
# Full GSM8K (1319 questions), gen 256 / block 128, one config after another on one GPU.
# cfg: vanilla_s256 | hier_r03 | wino | fastdllm_* | tXX_pYY_sZ_prefix (ours, decode_confidence_v2).
# NUM=<n> evaluates a seed-42 subset into results/smoke_gsm8k_* (smoke test).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The user slice allows 1024 tasks in total; keep CPU thread pools small.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=2
# Fast-dLLM's modeling uses torch.compile; inductor would otherwise fork 32 compile workers.
export TORCHINDUCTOR_COMPILE_THREADS=1
PY=/home/yejin/anaconda3/bin/python
MODEL=/mnt/shared/huggingface-cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
DATA=/mnt/shared/huggingface-cache/hub/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow
gpu=$1; shift
num=${NUM:-1319}; prefix=$([ $num = 1319 ] && echo gsm8k_full || echo smoke_gsm8k)
mkdir -p logs results
variant_config() {
  case $1 in
    wino) echo wino_gsm8k.json;;
    vanilla_*) echo vanilla_variant.json;;
    hier_r03) echo hierarchy_r03_variant.json;;
    fastdllm_*) echo $1.json;;
    *) echo confpivot_v2_$1.json;;
  esac
}
for cfg in "$@"; do
  echo "=== $(date) START $cfg" >> logs/${prefix}_b128_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES=$gpu $PY -u eval_gsm8k.py --model $MODEL --dataset $DATA \
    --output results/${prefix}_g256_b128_$cfg --num-samples $num --gen-length 256 --block-length 128 --steps 256 \
    --variant-config configs/$(variant_config $cfg) > logs/${prefix}_g256_${cfg}_b128.out 2>&1
  rc=$?
  echo "=== $(date) END $cfg exit=$rc" >> logs/${prefix}_b128_gpu${gpu}.log
  [ $rc -ne 0 ] && { echo "=== $(date) QUEUE STOPPED gpu$gpu after $cfg failed" >> logs/${prefix}_b128_gpu${gpu}.log; exit $rc; }
done
echo "=== $(date) QUEUE DONE gpu$gpu" >> logs/${prefix}_b128_gpu${gpu}.log

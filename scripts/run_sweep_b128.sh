#!/bin/bash
# Usage: run_sweep_b128.sh <gpu> <jobs_file>
# HumanEval gen 256 / block 128 hyperparameter sweep of decode_confidence_v2 + prefix cache.
# Each line of jobs_file is a config name tXX_pYY_sZ_prefix (configs/confpivot_v2_<name>.json).
# Pops jobs one at a time under flock, so each GPU always runs exactly one job.
# A finished run returns before loading the model, so the queue can be restarted from 0.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The user slice allows 1024 tasks in total; keep CPU thread pools small.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=2
# Fast-dLLM's modeling uses torch.compile; inductor would otherwise fork 32 compile workers.
export TORCHINDUCTOR_COMPILE_THREADS=1
PY=/home/yejin/anaconda3/bin/python
MODEL=/mnt/shared/huggingface-cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
DATA=/mnt/shared/huggingface-cache/hub/datasets/openai___openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow
gpu=$1; jobs=$2; lock=$jobs.lock; cursor=$jobs.cursor
mkdir -p logs results
touch "$lock"; [ -f "$cursor" ] || echo 0 > "$cursor"
while true; do
  cfg=$(flock "$lock" bash -c "i=\$(cat '$cursor'); n=\$(wc -l < '$jobs'); if [ \$i -lt \$n ]; then echo \$((i+1)) > '$cursor'; sed -n \"\$((i+1))p\" '$jobs'; fi")
  [ -z "$cfg" ] && break
  echo "=== $(date) START $cfg" >> logs/sweep_b128_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES=$gpu $PY -u eval_humaneval.py --model $MODEL --dataset $DATA \
    --output results/humaneval_g256_b128_$cfg --gen-length 256 --block-length 128 \
    --variant-config configs/confpivot_v2_$cfg.json > logs/humaneval_g256_${cfg}_b128.out 2>&1
  rc=$?
  echo "=== $(date) END $cfg exit=$rc" >> logs/sweep_b128_gpu${gpu}.log
  # A failed job is not retried by the cursor: stop instead of burning through the queue.
  [ $rc -ne 0 ] && { echo "=== $(date) POOL STOPPED gpu$gpu after $cfg failed" >> logs/sweep_b128_gpu${gpu}.log; exit $rc; }
done
echo "=== $(date) POOL DONE gpu$gpu" >> logs/sweep_b128_gpu${gpu}.log

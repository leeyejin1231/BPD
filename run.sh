#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python eval_gsm8k_hf.py --model GSAI-ML/LLaDA-8B-Instruct --output ./results/gsm8k/M_I2_n1b128_0.7/42 --n 1 --remask-k 0 --confidence-threshold 0.7 --gen-length 256 --block-length 128 --num-samples 0

CUDA_VISIBLE_DEVICES=0 python eval_bench_hf.py --task math500 --model GSAI-ML/LLaDA-8B-Instruct --output ./results/math500/M_I2_n1b128_0.7/42 --n 1 --remask-k 0 --confidence-threshold 0.7 --gen-length 256 --block-length 128

CUDA_VISIBLE_DEVICES=0 python eval_bench_hf.py --task mbpp --model GSAI-ML/LLaDA-8B-Instruct --output ./results/mbpp/M_I2_n1b128_0.7/42 --n 1 --remask-k 0 --confidence-threshold 0.7 --gen-length 256 --block-length 128

CUDA_VISIBLE_DEVICES=0 python eval_bench_hf.py --task ifeval --model GSAI-ML/LLaDA-8B-Instruct --output ./results/ifeval/M_I2_n1b128_0.7/42 --n 1 --remask-k 0 --confidence-threshold 0.7 --gen-length 256 --block-length 128
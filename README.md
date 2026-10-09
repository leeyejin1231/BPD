## 준비
### Fast-dLLM
```bash
git clone https://github.com/NVlabs/Fast-dLLM.git
```
### WINO
```bash
git clone https://github.com/Feng-Hong/WINO-DLLM.git
```
### Hierarchy-dLLM  
(구현 달라서 논문 구현으로 해야함)
```bash
git clone https://github.com/inclusionAI/dInfer.git
```

## Dream baselines (eval_*_hf.py)
`eval_gsm8k_hf.py`, `eval_bench_hf.py`, `eval_humaneval_hf.py` take `--model-family dream`
(Dream-v0-Instruct-7B) for the baselines; `eval_models.py` picks the model implementation:

| decoder (`remasking`) | LLaDA | Dream |
|---|---|---|
| `low_confidence` / `random` / `random_pivot` (vanilla, `decode.py`) | HF model | `dream_adapter` (shifted logits, full attention) |
| `hierarchy` (`hierarchy_decode.py`) | HF model | `dream_adapter` |
| `wino` (`wino_decode.py`) | WINO `modeling_llada`, alone | raw Dream model, `dream=True`, alone |
| `fastdllm` | Fast-dLLM `modeling_llada`, alone | `fastdllm_dream_decode.py` (Fast-dLLM `modeling_dream`, prefix cache), alone |
| `ours` (`decode_confidence_v2.py`) | HF model + prefix KV cache | not ported |

```bash
# gen 256 / block 128 (GEN=, BLOCK= override), one method per job
scripts/run_dream_hf.sh <gpu> math500:vanilla math500:hier math500:wino math500:fastdllm
```

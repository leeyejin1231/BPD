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
(구현 달라서 논문 구현으로 해야함, hierarch_decode.py 확인)
```bash
git clone https://github.com/inclusionAI/dInfer.git
```

### Python version
python==3.12.14


### Execute
Use `decode_confidence_v2.py`
```sh
"confidence_threshold": 0.75, 
"small_interval_size": 2, 
"pivots_per_interval": 32, 
"pivot_schedule": [[16, 32], [8, 16], [4, 4], [2, 2]], 
"prefix_cache": true
```
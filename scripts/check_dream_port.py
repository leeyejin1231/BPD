import sys, torch
sys.path.insert(0, '/home/ssgyejin/contents/DLM_random_pivot')
from transformers import AutoTokenizer
import dream_adapter, fastdllm_dream_decode
from decode_confidence_v2 import generate as g_llada_style
from decode_confidence_v2_dream import generate as g_dream
M = '/scratch/ssgyejin/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae'
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True, local_files_only=True)
adapter = dream_adapter.load_model(M)
fast = fastdllm_dream_decode.load_model(M)
qs = ['What is the sum of the first 20 positive odd integers? Please reason step by step, and put your final answer within \\boxed{}.',
      'Write a haiku about autumn. Your entire response must be in lowercase letters.',
      'Solve for x: 3x + 7 = 25. Please reason step by step, and put your final answer within \\boxed{}.']
kw = dict(gen_length=256, block_length=128, confidence_threshold=0.75, small_interval_size=2, pivots_per_interval=32, return_stats=True, mask_id=151666, temperature=0, cfg_scale=0)
for q in qs:
    ids = tok(tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True), add_special_tokens=False, return_tensors='pt').input_ids.cuda()
    outs = {}
    for name, fn, model, extra in [('adapter_v2', g_llada_style, adapter, {}), ('dream_port', g_dream, adapter, {}), ('dream_port_prefix', g_dream, fast, {'prefix_cache': True})]:
        torch.manual_seed(42)
        x, st = fn(model, ids, **kw, **extra)
        outs[name] = (x[0, ids.shape[1]:].cpu(), st['forward_steps'])
    a, b, c = outs['adapter_v2'], outs['dream_port'], outs['dream_port_prefix']
    print(f'adapter_v2 vs dream_port: identical={torch.equal(a[0], b[0])} steps={a[1]}/{b[1]} | '
          f'dream_port vs prefix: agree={(b[0]==c[0]).float().mean():.3f} steps={b[1]}/{c[1]}', flush=True)
    print('   ', repr(tok.decode(c[0], skip_special_tokens=True)[:100]))

"""DAPD (Kim, Jeon, Jeon, No, arXiv:2603.12996) as a baseline decoder: the official dapd package cloned in DAPD/, block-wise selection on the stock HF LLaDA model (generate_dapd_with_blocks) or, with dream=True, on the raw Dream model (dream_generate_dapd_with_blocks hooks Dream's attention layers and shifts the logits itself), stats converted to the harness format."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().with_name('DAPD')))
from dapd.dream_generation import dream_generate_dapd_with_blocks
from dapd.generation import generate_dapd_with_blocks

ALGS = ('dapd_direct', 'dapd_staged')


@torch.no_grad()
def generate(model, prompt, attention_mask=None, gen_length=128, block_length=128, temperature=0., mask_id=126336, alg='dapd_direct', tau_min=0.01, tau_max=0.05, layer_ratio=0.3, dream=False, return_stats=False, **unused):
    if unused.get('cfg_scale', 0):
        raise NotImplementedError('CFG is not part of the DAPD baseline.')
    if temperature != 0:
        raise ValueError('DAPD is greedy (argmax of the marginals); temperature must be 0.')
    if alg not in ALGS:
        raise ValueError(f'alg must be one of {ALGS}.')
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError('dapd supports a single nonempty prompt of shape (1, prompt_length).')
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError('gen_length must be positive and divisible by block_length.')
    device = model.device
    prompt = prompt.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if dream:
        x, stats = dream_generate_dapd_with_blocks(model=model, tokenizer=SimpleNamespace(mask_token_id=mask_id), input_ids=prompt, gen_length=gen_length, block_length=block_length, attention_scope='global', layer_ratio=layer_ratio, tau_min=tau_min, tau_max=tau_max, dapd_alg=alg, verbose=False)
    else:
        x, stats = generate_dapd_with_blocks(model=model, prompt=prompt, attention_mask=attention_mask, gen_length=gen_length, block_length=block_length, attention_scope='global', layer_ratio=layer_ratio, tau_min=tau_min, tau_max=tau_max, mask_id=mask_id, verbose=False, alg=alg, collect_step_history=False)
    if not return_stats:
        return x
    conf = stats.get('direct_conf_unmasked_per_step', stats.get('staged_conf_unmasked_per_step', []))
    dependency = stats.get('avg_dependency', [])
    return x, {'forward_steps': stats['total_forward_passes'], 'unmasked_per_step': [[int(n)] for n in stats['tokens_per_step']], 'conf_unmasked_per_step': [int(n) for n in conf], 'steps_per_block': [int(b['steps']) for b in stats.get('per_block_stats', [])], 'mean_dependency': sum(dependency) / len(dependency) if dependency else 0.0, 'alg': alg, 'tau_min': tau_min, 'tau_max': tau_max, 'layer_ratio': layer_ratio, 'capped_incomplete': bool((x[:, prompt.shape[1]:] == mask_id).any())}

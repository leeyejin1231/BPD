import importlib.util
import sys
from pathlib import Path

import torch

FAST_DIR = Path(__file__).resolve().parent / 'Fast-dLLM' / 'v1' / 'llada'


def _original():
    if 'fastdllm_generate' not in sys.modules:
        if str(FAST_DIR) not in sys.path:
            sys.path.insert(0, str(FAST_DIR))
        spec = importlib.util.spec_from_file_location('fastdllm_generate', FAST_DIR / 'generate.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules['fastdllm_generate'] = module
        spec.loader.exec_module(module)
    return sys.modules['fastdllm_generate']


def load_model(path, device='cuda'):
    return _original().LLaDAModelLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()


@torch.no_grad()
def generate(model, prompt, attention_mask=None, gen_length=128, block_length=32, temperature=0.,
             mask_id=126336, threshold=0.9, cache=None, return_stats=False, **unused):
    if unused.get('cfg_scale', 0):
        raise NotImplementedError('CFG is not part of the Fast-dLLM baseline.')
    if prompt.shape[0] != 1:
        raise ValueError('The Fast-dLLM port supports batch size 1 only.')
    if cache not in (None, 'prefix', 'dual'):
        raise ValueError("cache must be None, 'prefix' or 'dual'.")
    assert gen_length % block_length == 0
    get_transfer_index = _original().get_transfer_index
    P = prompt.shape[1]
    x = torch.full((1, P + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :P] = prompt.clone()
    unmasked_per_step, full_forwards = [], 0

    def select(logits, mask_index, tokens):
        x0, transfer_index = get_transfer_index(logits, temperature, 'low_confidence', mask_index,
                                                tokens, None, threshold)
        unmasked_per_step.append([int(transfer_index.sum())])
        return x0, transfer_index

    for num_block in range(gen_length // block_length):
        s, e = P + num_block * block_length, P + (num_block + 1) * block_length
        if cache is None:
            while (x[:, s:e] == mask_id).any():
                mask_index = (x == mask_id)
                logits = model(x).logits
                full_forwards += 1
                mask_index[:, e:] = 0
                x0, transfer_index = select(logits, mask_index, x)
                x[transfer_index] = x0[transfer_index]
            continue

        # First step of the block: full forward, which also fills the KV cache.
        output = model(x, use_cache=True)
        full_forwards += 1
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, e:] = 0
        x0, transfer_index = select(output.logits, mask_index, x)
        x[transfer_index] = x0[transfer_index]

        if cache == 'prefix':
            past_key_values = [tuple(t[:, :, :s] for t in layer) for layer in past_key_values]
            while (x[:, s:e] == mask_id).any():
                mask_index = (x[:, s:] == mask_id)
                mask_index[:, block_length:] = 0
                logits = model(x[:, s:], past_key_values=past_key_values, use_cache=True).logits
                x0, transfer_index = select(logits, mask_index, x[:, s:])
                x[:, s:][transfer_index] = x0[transfer_index]
        else:
            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, s:e] = True
            while (x[:, s:e] == mask_id).any():
                logits = model(x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits
                mask_index = (x[:, s:e] == mask_id)
                x0, transfer_index = select(logits, mask_index, x[:, s:e])
                x[:, s:e] = torch.where(transfer_index, x0, x[:, s:e])

    if not return_stats:
        return x
    return x, {
        'forward_steps': len(unmasked_per_step),
        'full_forward_steps': full_forwards,
        'unmasked_per_step': unmasked_per_step,
        'threshold': threshold, 'cache': cache,
        'capped_incomplete': False,
    }

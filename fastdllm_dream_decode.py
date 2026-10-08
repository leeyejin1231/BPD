"""Fast-dLLM baseline on Dream-v0-Instruct-7B, ported from Fast-dLLM/v1/dream/model/generation_utils_block.py.

The loop is DreamGenerationMixin._sample with alg='confidence_threshold' and the prefix KV
cache (use_cache=true, dual_cache=false), the setting Fast-dLLM's Dream scripts call
"prefix cache+parallel", written out with per-step accounting:
  * every block starts with a full forward whose keys/values before the block are kept;
    Fast-dLLM also commits the first position of the block from that forward (Dream
    predicts position i from the logits at i-1, so the shifted logits are used throughout);
  * later steps recompute the block and the suffix with the cached prefix and commit every
    masked position of the block with confidence >= threshold, plus the single most
    confident one.
sample_tokens is imported unchanged (temperature 0: argmax and its probability). The
cache needs past_key_values, which the Hugging Face Dream modeling does not accept, so
load_model() builds the model with Fast-dLLM's modeling_dream.py (same weights). The
model is used directly, not through dream_adapter. Batch size 1; a cached forward counts
as one forward step. Token ids follow dream_adapter (mask 151666).
"""
import importlib.util
import sys
from pathlib import Path

import torch

FAST_DIR = Path(__file__).resolve().parent / 'Fast-dLLM' / 'v1' / 'dream' / 'model'


def _load(name, path, package=False):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(path.parent)] if package else None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _model_package():
    """Fast-dLLM's dream/model package under a private name (its modules use relative imports)."""
    return _load('fastdllm_dream_model', FAST_DIR / '__init__.py', package=True)


def _block_generation():
    return _load('fastdllm_dream_generation_block', FAST_DIR / 'generation_utils_block.py')


def load_model(path, device='cuda'):
    return _model_package().DreamModel.from_pretrained(
        path, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()


def _shift(logits):
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@torch.no_grad()
def generate(model, prompt, attention_mask=None, gen_length=128, block_length=32, temperature=0.,
             mask_id=151666, threshold=0.9, cache='prefix', return_stats=False, **unused):
    """Fast-dLLM confidence-threshold decoding on Dream with per-step accounting; unused absorbs steps/cfg_scale."""
    if unused.get('cfg_scale', 0):
        raise NotImplementedError('CFG is not part of the Fast-dLLM baseline.')
    if attention_mask is not None:
        raise ValueError('The Dream port expects attention_mask=None (unpadded batch of 1).')
    if prompt.shape[0] != 1:
        raise ValueError('The Fast-dLLM port supports batch size 1 only.')
    if cache != 'prefix':
        raise ValueError("Only cache='prefix' is ported for Dream.")
    assert gen_length % block_length == 0
    sample_tokens = _block_generation().sample_tokens
    P = prompt.shape[1]
    x = torch.full((1, P + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :P] = prompt.clone()
    unmasked_per_step, full_forwards = [], 0

    for num_block in range(gen_length // block_length):
        s, e = P + num_block * block_length, P + (num_block + 1) * block_length
        # First step of the block: full forward, which also fills the KV cache.
        output = model(x, attention_mask='full', position_ids=None, use_cache=True)
        full_forwards += 1
        _, x0 = sample_tokens(_shift(output.logits), temperature=temperature)
        x[:, s] = x0[:, s]
        unmasked_per_step.append([1])
        past_key_values = [tuple(t[:, :s, :] for t in layer) for layer in output.past_key_values]

        while (x[:, s:e] == mask_id).any():
            mask_index = x[:, s:] == mask_id
            logits = _shift(model(x[:, s:], attention_mask='full', position_ids=None,
                                  past_key_values=past_key_values, use_cache=True).logits)
            confidence, x0 = sample_tokens(logits[mask_index], temperature=temperature)
            x_ = torch.full_like(x[:, s:], mask_id)
            full_confidence = torch.full_like(x[:, s:], -torch.inf, dtype=logits.dtype)
            x_[mask_index] = x0
            full_confidence[mask_index] = confidence
            full_confidence[:, block_length:] = -torch.inf
            current_transfer_tokens = int((x[:, s:e] == mask_id).sum())
            selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
            transfer_index = torch.zeros_like(x_, dtype=torch.bool)
            transfer_index[0, select_index[0]] = True
            for k in range(1, current_transfer_tokens):
                if selected_confidence[0, k] < threshold:
                    transfer_index[0, select_index[0, k]] = False
            x[:, s:][transfer_index] = x_[transfer_index]
            unmasked_per_step.append([int(transfer_index.sum())])

    if not return_stats:
        return x
    return x, {
        'forward_steps': len(unmasked_per_step),
        'full_forward_steps': full_forwards,
        'unmasked_per_step': unmasked_per_step,
        'threshold': threshold, 'cache': cache,
        'capped_incomplete': False,
    }

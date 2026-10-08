"""Confidence-filtered random pivots with bulk completion on Dream-v0-Instruct-7B.

Same rule as decode_confidence_v2.generate (select_pivots is imported unchanged), written
against Dream's forward instead of LLaDA's:
  * Dream wants attention_mask="full" (fully bidirectional, batch of 1, no padding);
  * Dream predicts position i from the logits at i-1, so every logits tensor is shifted
    right by one before scoring, as generation_utils._sample does;
  * prefix_cache uses Fast-dLLM's Dream modeling (fastdllm_dream_decode.load_model): the
    first forward of a block is a full forward whose keys/values for positions < block_start-1
    are kept; later forwards feed x[:, block_start-1:] so the logits at block_start-1, which
    predict the first block position, are recomputed each step. The pivot rule is therefore
    identical with and without the cache (no forced first-position commit as in Fast-dLLM).
Accepts the raw Dream model (Hugging Face or Fast-dLLM modeling) or a
dream_adapter.DreamForwardAdapter. Token ids: mask 151666, eos 151643.

Example (model and prompt already loaded):
    from decode_confidence_v2_dream import generate
    output, stats = generate(model, prompt, gen_length=256, block_length=128,
                             confidence_threshold=0.75, small_interval_size=2,
                             pivots_per_interval=32, prefix_cache=True, return_stats=True)

This module does not load models or start experiments when imported or executed.
"""
import torch

from decode import add_gumbel_noise
from decode_confidence_v2 import _DIAGNOSTICS, select_pivots
from dream_adapter import DreamForwardAdapter, MASK_ID


def _shift(logits):
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def _validate(confidence_threshold, confidence_threshold_max, small_interval_size,
              pivots_per_interval, pivot_schedule, gen_length, block_length, prompt,
              attention_mask, temperature, cfg_scale):
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('confidence_threshold must be between 0 and 1.')
    if confidence_threshold_max is not None and not confidence_threshold <= confidence_threshold_max <= 1:
        raise ValueError('confidence_threshold_max must be between confidence_threshold and 1.')
    if (isinstance(small_interval_size, bool)
            or not isinstance(small_interval_size, int) or small_interval_size < 0):
        raise ValueError('small_interval_size must be a non-negative integer.')
    if (isinstance(pivots_per_interval, bool)
            or not isinstance(pivots_per_interval, int) or pivots_per_interval < 1):
        raise ValueError('pivots_per_interval must be a positive integer.')
    if pivot_schedule is not None:
        pivot_schedule = [[int(min_size), int(budget)] for min_size, budget in pivot_schedule]
        if (not pivot_schedule or any(b < 1 for _, b in pivot_schedule)
                or any(a[0] <= b[0] for a, b in zip(pivot_schedule, pivot_schedule[1:]))):
            raise ValueError('pivot_schedule needs [min_size, p>=1] pairs in descending min_size order.')
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError('Positive gen_length must be divisible by positive block_length.')
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError('The Dream port expects a non-empty prompt of shape (1, prompt_length).')
    if attention_mask is not None:
        raise ValueError('The Dream port expects attention_mask=None (unpadded batch of 1).')
    if temperature < 0:
        raise ValueError('temperature must be non-negative.')
    if cfg_scale:
        raise NotImplementedError('CFG is not ported to Dream.')
    return pivot_schedule


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=None, gen_length=128,
             block_length=32, temperature=0., cfg_scale=0., mask_id=MASK_ID,
             confidence_threshold=0.8, logits_eos_inf=False, eos_token_id=151643,
             return_stats=False, small_interval_size=2, pivots_per_interval=1,
             pivot_schedule=None, confidence_threshold_max=None, prefix_cache=False):
    """decode_confidence_v2.generate on Dream; see that docstring for the rule and the stats.

    steps is accepted and ignored. With prefix_cache the model must accept past_key_values
    (Fast-dLLM's modeling_dream, loaded by fastdllm_dream_decode.load_model); a cached
    forward counts as one forward step, as in the LLaDA version.
    """
    pivot_schedule = _validate(confidence_threshold, confidence_threshold_max, small_interval_size,
                               pivots_per_interval, pivot_schedule, gen_length, block_length,
                               prompt, attention_mask, temperature, cfg_scale)
    raw = model.model if isinstance(model, DreamForwardAdapter) else model
    device = raw.device
    prompt_length = prompt.shape[1]
    x = torch.full((1, prompt_length + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt.to(device)
    unmasked_per_step = []
    diagnostics_per_step = {key: [] for key in _DIAGNOSTICS}
    steps_per_block = []
    forward_steps = 0

    for block_start in range(prompt_length, prompt_length + gen_length, block_length):
        block_end = block_start + block_length
        intervals = [[(0, block_length)]]
        block_steps = 0
        past_key_values = None
        while any(intervals):
            if prefix_cache and past_key_values is None:
                output = raw(x, attention_mask='full', position_ids=None, use_cache=True)
                logits = _shift(output.logits)[:, block_start:block_end]
                # Keep keys/values before block_start-1 (seq axis 1); block_start-1 is recomputed
                # so that its logits, which predict block_start, stay available.
                past_key_values = [tuple(t[:, :block_start - 1] for t in layer)
                                   for layer in output.past_key_values]
                del output
            elif prefix_cache:
                # Unshifted logits at block_start-1 .. block_end-2 predict block_start .. block_end-1.
                logits = raw(x[:, block_start - 1:], attention_mask='full', position_ids=None,
                             past_key_values=past_key_values, use_cache=True).logits[:, :block_length]
            else:
                logits = _shift(raw(x, attention_mask='full', position_ids=None).logits)[:, block_start:block_end]

            logits[..., mask_id] = -torch.inf
            if logits_eos_inf:
                logits[..., eos_token_id] = -torch.inf
            predictions = add_gumbel_noise(logits, temperature).argmax(dim=-1)
            float_logits = logits.float()
            confidence = torch.exp(
                float_logits.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
                - torch.logsumexp(float_logits, dim=-1)
            )
            selected, intervals, diagnostics = select_pivots(
                confidence, intervals, confidence_threshold, small_interval_size,
                pivots_per_interval, pivot_schedule, confidence_threshold_max, block_length)
            block = x[:, block_start:block_end]
            block[selected] = predictions[selected]
            forward_steps += 1
            block_steps += 1
            if return_stats:
                unmasked_per_step.append(selected.sum(dim=1))
                for key, values in diagnostics.items():
                    diagnostics_per_step[key].append(values)
            del logits, float_logits, confidence, predictions
        steps_per_block.append(block_steps)

    if not return_stats:
        return x
    stats = {
        'forward_steps': forward_steps,
        'steps_per_block': steps_per_block,
        'unmasked_per_step': torch.stack(unmasked_per_step).cpu().tolist(),
        'confidence_threshold': confidence_threshold,
        'confidence_threshold_max': confidence_threshold_max,
        'small_interval_size': small_interval_size,
        'pivots_per_interval': pivots_per_interval,
        'pivot_schedule': pivot_schedule,
        'prefix_cache': prefix_cache,
        'model_family': 'dream',
        'capped_incomplete': False,
    }
    stats.update({f'{key}_per_step': values for key, values in diagnostics_per_step.items()})
    return x, stats

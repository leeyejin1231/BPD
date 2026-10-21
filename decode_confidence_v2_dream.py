import math

import torch


def load_model(path, device='cuda'):
    from fastdllm_dream_decode import load_model as load
    return load(path, device)


def prepare_prefix_cache(model):
    if type(model).__name__ != 'DreamModel' or type(model).__module__ != 'fastdllm_dream_model':
        raise RuntimeError('ours on Dream needs Fast-dLLM\'s modeling_dream (decode_confidence_v2_dream.load_model); '
                           'the stock HF Dream model and DreamForwardAdapter do not accept past_key_values.')
    return 'native'


def _sample_tokens():
    from fastdllm_dream_decode import _block_generation
    return _block_generation().sample_tokens


def _shift(logits):
    # Fast-dLLM: logits at i-1 predict token i.
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def _prefix_only(cache, length):
    if not isinstance(cache, (tuple, list)) or not cache:
        raise RuntimeError('ours requires a model returning legacy (key, value) KV tuples; cache is missing or unsupported.')
    result = []
    for layer in cache:
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise RuntimeError('Expected one (key, value) pair per layer.')
        for tensor in layer:
            if tensor.ndim != 3 or tensor.shape[1] < length:
                raise RuntimeError('Expected Dream KV shape [batch, sequence, hidden].')
        result.append(tuple(t[:, :length, :] for t in layer))
    return tuple(result)


@torch.no_grad()
def generate(
    model,
    prompt,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    mask_id=151666,
    n=2,
    confidence_threshold=0.9,
    logits_eos_inf=False,
    eos_token_id=151643,
    *,
    attention_mask=None,
    return_stats=False,
    remask_k=0,
    remask_threshold=0.5,
    n_switch_threshold=128,
    force_confidence=None,
):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError(
            'prefix-cache generation requires nonempty prompt shape '
            '(1, prompt_length).'
        )
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError(
            'gen_length must be positive and divisible by block_length.'
        )
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer.')
    if (
        isinstance(remask_k, bool)
        or not isinstance(remask_k, int)
        or remask_k < 0
    ):
        raise ValueError('remask_k must be a non-negative integer.')
    if not 0.0 <= remask_threshold <= 1.0:
        raise ValueError('remask_threshold must be between 0 and 1.')
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError('confidence_threshold must be between 0 and 1.')
    if force_confidence is not None and not 0.0 <= force_confidence <= 1.0:
        raise ValueError('force_confidence must be None or between 0 and 1.')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and non-negative.')

    if attention_mask is not None:
        if (
            attention_mask.shape != prompt.shape
            or not bool(attention_mask.eq(1).all())
        ):
            raise ValueError('ours supports an unpadded single prompt only.')

    device = model.device
    prompt = prompt.to(device)
    if bool(prompt.eq(mask_id).any()):
        raise ValueError('The prompt must not contain MASK tokens.')
    sample_tokens = _sample_tokens()
    prompt_length = prompt.shape[1]
    x = torch.full((1, prompt_length + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt
    index = torch.arange(block_length, device=device)
    counts = []
    forward_steps = prefill_steps = cached_steps = 0

    for block_start in range(prompt_length, prompt_length + gen_length, block_length):
        block_end = block_start + block_length
        block = x[0, block_start:block_end]

        # Fast-dLLM block start: one full forward fills the cache and commits block
        # position 0 from the shifted logits (raw row block_start-1), unconditionally.
        output = model(x, attention_mask='full', position_ids=None, use_cache=True)
        first_logits = output.logits[0, block_start - 1:block_start].clone()
        first_logits[..., mask_id] = -torch.inf
        if logits_eos_inf:
            first_logits[..., eos_token_id] = -torch.inf
        _, first = sample_tokens(first_logits, temperature=temperature)
        block[0] = first[0]
        past_key_values = _prefix_only(getattr(output, 'past_key_values', None), block_start)
        del output
        forward_steps += 1
        prefill_steps += 1
        if return_stats:
            counts.append(torch.ones(1, dtype=torch.long, device=device))

        # Policy steps on the cached prefix. Each step fills >= 1 token, so
        # block_length steps suffice for the block_length-1 remaining masks.
        for step_idx in range(block_length):
            output = model(x[:, block_start:], attention_mask='full', position_ids=None, past_key_values=past_key_values, use_cache=True)
            # Fast-dLLM shift on the suffix logits (bf16, model dtype); keep the block rows.
            logits = _shift(output.logits)[0, :block_length]
            del output
            forward_steps += 1
            cached_steps += 1

            logits = logits.clone()
            logits[..., mask_id] = -torch.inf
            if logits_eos_inf:
                logits[..., eos_token_id] = -torch.inf
            confidence, predictions = sample_tokens(logits, temperature=temperature)
            confidence = confidence.float()
            if force_confidence is not None:
                # Same bf16 softmax confidence as Fast-dLLM's threshold rule.
                force_hit = confidence.ge(force_confidence)
            del logits
            masked = block.eq(mask_id)
            starts = masked.clone()
            starts[1:] &= ~masked[:-1]
            segment = starts.cumsum(dim=0)
            same_interval = masked[:, None] & masked[None, :] & segment[:, None].eq(segment[None, :])
            interval_size = same_interval.sum(dim=1)
            eligible = masked & confidence.ge(confidence_threshold)
            eligible_count = (same_interval & eligible[None, :]).sum(dim=1)

            ##############################################################################
            interval_left = torch.where(
                starts, index, 0
            ).cummax(dim=0).values

            ends = masked.clone()
            ends[:-1] &= ~masked[1:]

            interval_right = torch.where(
                ends, index, block_length
            ).flip(0).cummin(dim=0).values.flip(0)

            center_score = -(
                2 * index - interval_left - interval_right
            ).abs().to(confidence.dtype)

            enough = eligible_count.ge(n)
            priority = torch.where(
                enough,
                torch.where(eligible, center_score, -torch.inf),
                torch.where(eligible, 2.0, confidence),
            )
            ##############################################################################

            higher_priority = same_interval & (
                priority[None, :].gt(priority[:, None])
                | (priority[None, :].eq(priority[:, None]) & index[None, :].lt(index[:, None])))
            rank = higher_priority.sum(dim=1)
            selected = masked & (interval_size.le(n) | rank.lt(n))
            ##############################################################################
            if step_idx <= 2:
                first_selected = masked & confidence.ge(confidence_threshold)

                selected = torch.where(
                    first_selected.any(),
                    first_selected,
                    selected,
                )
            ##############################################################################
            if force_confidence is not None:
                # Always commit tokens whose bf16 confidence reaches
                # force_confidence, regardless of n, interval rank, or step index.
                selected = selected | (masked & force_hit)
            ##############################################################################
            block[selected] = predictions[selected]

            if return_stats:
                num_unmasked = (
                    selected & ~block.eq(mask_id)
                ).sum()
                counts.append(num_unmasked.reshape(1))

            if not block.eq(mask_id).any():
                break
        else:
            raise RuntimeError('Block did not finish; check model logits for NaN/Inf or MASK-only predictions.')
        # Refresh from the full updated sequence at the next block boundary.
        del past_key_values

    if return_stats:
        stats = {
            'forward_steps': forward_steps,
            'unmasked_per_step': torch.stack(counts).cpu().tolist(),
            'prefill_steps': prefill_steps, 'cached_steps': cached_steps,
            'prefix_cache': True, 'capped_incomplete': False,
        }
        return x, stats
    return x

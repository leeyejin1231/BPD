"""Confidence pivots with a fixed prefix KV cache, refreshed at every block."""
import ast
import inspect
import math
import textwrap
import types

import torch


def prepare_prefix_cache(model):
    """Enable the existing cache implementation in stock HF LLaDA, if needed.

    Only the known cache prohibition is removed, on this model instance.
    Weights, attention, RoPE, and use_cache=False behavior are unchanged.
    Models already supporting caching are left alone. No disk files are edited.
    """
    core = getattr(model, 'model', None)
    if core is None or type(core).__name__ != 'LLaDAModel':
        return 'native'
    if getattr(core, '_prefix_cache_prepared', False):
        return 'stock_llada_guard_removed'
    method = core.forward
    try:
        source = textwrap.dedent(inspect.getsource(method))
    except (OSError, TypeError):
        return 'native'
    tree = ast.parse(source)
    function = tree.body[0]
    expected = ast.dump(ast.parse('past_key_values is None and not use_cache', mode='eval').body)
    guards = [node for node in function.body if isinstance(node, ast.Assert)
              and ast.dump(node.test) == expected]
    if not guards:
        return 'native'
    if (len(guards) != 1 or core.config.alibi or not core.config.rope
            or 'self.get_bidirectional_attention_bias(' not in source
            or getattr(method.__func__, '__closure__', None)):
        raise RuntimeError('Unrecognized LLaDA cache implementation; use a cache-capable model.')
    function.body.remove(guards[0])
    function.decorator_list = []
    # Compile inside a class to preserve Python's private-name mangling (__cache).
    wrapper = ast.ClassDef(name='LLaDAModel', bases=[], keywords=[], body=[function], decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              wrapper], type_ignores=[])
    namespace = dict(method.__func__.__globals__)
    exec(compile(ast.fix_missing_locations(module), '<llada-prefix-cache>', 'exec'), namespace)
    core.forward = types.MethodType(namespace['LLaDAModel'].forward, core)
    core._prefix_cache_prepared = True
    return 'stock_llada_guard_removed'


def _prefix_only(cache, length):
    if not isinstance(cache, (tuple, list)) or not cache:
        raise RuntimeError('ours requires a model returning legacy (key, value) KV tuples; cache is missing or unsupported.')
    result = []
    for layer in cache:
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise RuntimeError('Expected one (key, value) pair per layer.')
        for tensor in layer:
            if tensor.ndim != 4 or tensor.shape[-2] < length:
                raise RuntimeError('Expected KV shape [batch, heads, sequence, head_dim].')
        # Clone: a view would retain the entire full-sequence cache allocation.
        result.append(tuple(t[:, :, :length, :] for t in layer))
    return tuple(result)


def _add_gumbel_noise(logits, temperature):
    # Same sampling rule as decode.py, without importing the optional baseline.
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


@torch.no_grad()
def generate(
    model,
    prompt,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    mask_id=126336,
    n=2,
    confidence_threshold=0.9,
    logits_eos_inf=False,
    eos_token_id=126081,
    *,
    attention_mask=None,
    return_stats=False,
    remask_k=0,
    remask_threshold=0.5,
    n_switch_threshold=128,
):
    """Confidence pivots with an initial n=1 phase.

    At the start of each step:
      - use n=1 when net unmasked generated positions < n_switch_threshold
      - otherwise use the configured n

    The count excludes the prompt and spans completed blocks plus the
    current block. Remasked positions do not count as unmasked.

    Remasking selects globally within the current block, capped at
    min(remask_k, newly_unmasked - 1). Completed blocks exit immediately.
    """
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
        isinstance(n_switch_threshold, bool)
        or not isinstance(n_switch_threshold, int)
        or n_switch_threshold < 0
    ):
        raise ValueError(
            'n_switch_threshold must be a non-negative integer.'
        )
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

    prompt_length = prompt.shape[1]
    x = torch.full(
        (1, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt

    index = torch.arange(block_length, device=device)
    counts = []
    remasked_counts = []
    n_per_step = []
    forward_steps = prefill_steps = cached_steps = 0

    for block_start in range(
        prompt_length,
        prompt_length + gen_length,
        block_length,
    ):
        block_end = block_start + block_length
        block = x[0, block_start:block_end]
        past_key_values = None

        # Each valid incomplete step makes at least one net unmask:
        # remasking is capped at newly_unmasked - 1.
        for step_idx in range(block_length):
            if past_key_values is None:
                output = model(x, use_cache=True)
                logits = output.logits[
                    0, block_start:block_end
                ].float()
                past_key_values = _prefix_only(
                    getattr(output, 'past_key_values', None),
                    block_start,
                )
                prefill_steps += 1
            else:
                output = model(
                    x[:, block_start:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = output.logits[0, :block_length].float()
                cached_steps += 1

            del output
            forward_steps += 1

            logits[..., mask_id] = -torch.inf
            if logits_eos_inf:
                logits[..., eos_token_id] = -torch.inf

            if temperature == 0.0:
                predicted_logits, predictions = logits.max(dim=-1)
            else:
                predictions = _add_gumbel_noise(
                    logits, temperature
                ).argmax(dim=-1)
                predicted_logits = logits.gather(
                    1, predictions[:, None]
                ).squeeze(1)

            log_normalizer = torch.logsumexp(logits, dim=-1)
            confidence = (
                predicted_logits - log_normalizer
            ).exp()

            if remask_k:
                # Probability of the token currently stored at each position.
                committed_confidence = (
                    logits.gather(1, block[:, None]).squeeze(1)
                    - log_normalizer
                ).exp()

            del logits

            masked = block.eq(mask_id)

            # Completed generated blocks + current block's unmasked positions.
            # Excludes the prompt and does not double-count refilled positions.
            # generated_before_step = (
            #     (block_start - prompt_length)
            #     + (~masked).sum()
            # )
            generated_before_step = (~masked).sum()

            # Device-side scalar: no .item() or extra CPU synchronization.
            step_n = (
                1
                + (n - 1)
                * generated_before_step.le(
                    n_switch_threshold
                ).to(torch.long)
            )

            starts = masked.clone()
            starts[1:] &= ~masked[:-1]
            segment = starts.cumsum(dim=0)

            same_interval = (
                masked[:, None]
                & masked[None, :]
                & segment[:, None].eq(segment[None, :])
            )
            interval_size = same_interval.sum(dim=1)

            eligible = masked & confidence.ge(confidence_threshold)
            eligible_count = (
                same_interval & eligible[None, :]
            ).sum(dim=1)

            random_score = torch.rand_like(confidence)
            enough = eligible_count.ge(step_n)

            priority = torch.where(
                enough,
                torch.where(eligible, random_score, -torch.inf),
                torch.where(eligible, 2.0, confidence),
            )

            higher_priority = same_interval & (
                priority[None, :].gt(priority[:, None])
                | (
                    priority[None, :].eq(priority[:, None])
                    & index[None, :].lt(index[:, None])
                )
            )
            rank = higher_priority.sum(dim=1)

            selected = masked & (
                interval_size.le(step_n)
                | rank.lt(step_n)
            )
            ##############################################################################
            if step_idx <= 7:
                first_selected = masked & confidence.ge(confidence_threshold)

                selected = torch.where(
                    first_selected.any(),
                    first_selected,
                    selected,
                )
            ##############################################################################
            block[selected] = predictions[selected]

            if return_stats or remask_k:
                num_unmasked = (
                    selected & ~block.eq(mask_id)
                ).sum()

            if return_stats:
                counts.append(num_unmasked.reshape(1))
                n_per_step.append(step_n.reshape(1))
                if remask_k:
                    remasked_counts.append(
                        torch.zeros_like(counts[-1])
                    )

            # Never remask a completed block.
            if not block.eq(mask_id).any():
                break

            if remask_k:
                token_confidence = torch.where(
                    masked,
                    confidence,
                    committed_confidence,
                )
                candidates = (
                    ~block.eq(mask_id)
                    & token_confidence.lt(remask_threshold)
                )
                scores = torch.where(
                    candidates,
                    token_confidence,
                    torch.inf,
                )

                # Global remasking within the current block.
                limit = min(remask_k, block_length)
                values, positions = torch.topk(
                    scores,
                    limit,
                    largest=False,
                    sorted=True,
                )

                # At most newly_unmasked - 1 positions may be remasked.
                keep = (
                    values.lt(remask_threshold)
                    & index[:limit].lt(num_unmasked - 1)
                )
                remask_positions = positions[keep]
                block[remask_positions] = mask_id

                if return_stats:
                    remasked_counts[-1] = keep.sum().reshape(1)
        else:
            raise RuntimeError(
                'Block did not finish; check model logits for '
                'NaN/Inf or MASK-only predictions.'
            )

        del past_key_values

    if return_stats:
        stats = {
            'forward_steps': forward_steps,
            'unmasked_per_step': torch.stack(counts).cpu().tolist(),
            'n_per_step': torch.stack(n_per_step).cpu().tolist(),
            'prefill_steps': prefill_steps,
            'cached_steps': cached_steps,
            'prefix_cache': True,
            'capped_incomplete': False,
        }
        if remask_k:
            stats['remasked_per_step'] = (
                torch.stack(remasked_counts).cpu().tolist()
            )
            stats['net_unmasked_per_step'] = [
                [filled[0] - remasked[0]]
                for filled, remasked in zip(
                    stats['unmasked_per_step'],
                    stats['remasked_per_step'],
                )
            ]
        return x, stats

    return x

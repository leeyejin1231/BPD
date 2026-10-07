import torch
from decode import add_gumbel_noise

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
):
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("prefix-cache generation requires prompt shape (1, prompt_length).")
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError("gen_length must be positive and divisible by block_length.")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")

    device = model.device
    prompt = prompt.to(device)
    prompt_length = prompt.shape[1]

    x = torch.full(
        (1, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt
    index = torch.arange(block_length, device=device)

    for block_start in range(prompt_length, prompt_length + gen_length, block_length):
        block_end = block_start + block_length
        block = x[0, block_start:block_end]
        past_key_values = None

        while True:
            if past_key_values is None:
                output = model(x, use_cache=True)
                logits = output.logits[0, block_start:block_end]
                past_key_values = [
                    tuple(t[:, :, :block_start] for t in layer)
                    for layer in output.past_key_values
                ]
                del output
            else:
                logits = model(
                    x[:, block_start:],
                    past_key_values=past_key_values,
                    use_cache=True,
                ).logits[0, :block_length]

            logits[..., mask_id] = -torch.inf
            if logits_eos_inf:
                logits[..., eos_token_id] = -torch.inf

            if temperature == 0.0:
                predicted_logits, predictions = logits.max(dim=-1)
            else:
                predictions = add_gumbel_noise(logits, temperature).argmax(dim=-1)
                predicted_logits = logits.gather(1, predictions[:, None]).squeeze(1)

            confidence = (predicted_logits - torch.logsumexp(logits, dim=-1)).exp()
            masked = block.eq(mask_id)

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
            eligible_count = (same_interval & eligible[None, :]).sum(dim=1)

            random_score = torch.rand_like(confidence)
            enough = eligible_count.ge(n)
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

            selected = masked & (interval_size.le(n) | rank.lt(n))
            block[selected] = predictions[selected]

            if not block.eq(mask_id).any():
                break

    return x

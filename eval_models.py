import inspect
from pathlib import Path

import torch
from transformers import AutoModel

from dream_adapter import END_IDS as DREAM_END_IDS, MASK_ID as DREAM_MASK_ID

MODEL_FAMILIES = ('llada', 'dream')
VANILLA = ('low_confidence', 'random', 'random_pivot')
BASELINES = VANILLA + ('hierarchy', 'bisection', 'dapd', 'wino', 'fastdllm')
DECODERS = ('ours',) + BASELINES
EXCLUSIVE = ('wino', 'fastdllm')  # own modeling files: one of them per process, nothing else
TOKEN_IDS = {
    'llada': {'mask_id': 126336, 'end_ids': frozenset({126081, 126348})},
    'dream': {'mask_id': DREAM_MASK_ID, 'end_ids': frozenset(DREAM_END_IDS)},
}
# generate() parameters the harness sets itself; a variant config may not override them.
RESERVED = {'model', 'prompt', 'attention_mask', 'steps', 'gen_length', 'block_length',
            'temperature', 'cfg_scale', 'remasking', 'mask_id', 'return_stats', 'dream'}


def baseline_generate(decoder, model_family):
    if decoder in VANILLA:
        from decode import generate
    elif decoder == 'hierarchy':
        from hierarchy_decode import generate_hierarchy as generate
    elif decoder == 'bisection':
        from bisection.decode_bisection import generate
    elif decoder == 'dapd':
        from dapd_decode import generate
    elif decoder == 'wino':
        from wino_decode import generate
    elif decoder == 'fastdllm':
        if model_family == 'dream':
            from fastdllm_dream_decode import generate
        else:
            from fastdllm_decode import generate
    else:
        raise ValueError(f'unsupported baseline decoder {decoder!r}; choose from {BASELINES}')
    return generate


def baseline_options(decoder, model_family):
    parameters = inspect.signature(baseline_generate(decoder, model_family)).parameters
    return {name for name, p in parameters.items()
            if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)} - RESERVED


def resolve_loader(variants, model_family):
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f'unknown model family {model_family!r}; choose from {MODEL_FAMILIES}')
    decoders = {v['decoder'] for v in variants.values()}
    exclusive = decoders & set(EXCLUSIVE)
    if exclusive and len(decoders) > 1:
        raise ValueError(f'{sorted(exclusive)} need their own modeling files: run each of them alone in a separate process from any other decoder.')
    if model_family == 'dream':
        if 'ours' in decoders:
            if len(decoders) > 1:
                raise ValueError('ours on Dream uses the Fast-dLLM Dream modeling (prefix KV cache); run it alone, in a separate process from the other decoders.')
            return 'fastdllm_dream'
        return 'fastdllm_dream' if 'fastdllm' in decoders else 'dream_adapter'
    if 'wino' in decoders:
        return 'wino'
    if 'fastdllm' in decoders:
        return 'fastdllm'
    return 'hf'


def load_model(path, loader, pretrained):
    if loader == 'hf':
        return AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, **pretrained).to('cuda').eval()
    if pretrained.get('revision') is not None or pretrained.get('cache_dir') is not None:
        raise ValueError(f'loader {loader} reads a cached checkpoint with local_files_only=True; pass its local snapshot path as --model instead of --model-revision/--cache-dir.')
    if loader == 'wino':
        from wino_decode import load_model as load
    elif loader == 'fastdllm':
        from fastdllm_decode import load_model as load
    elif loader == 'dream_adapter':
        from dream_adapter import load_model as load
    elif loader == 'fastdllm_dream':
        from fastdllm_dream_decode import load_model as load
    else:
        raise ValueError(f'unknown loader {loader!r}')
    return load(path)


def source_files(variants, model_family):
    files = []
    for decoder in dict.fromkeys(v['decoder'] for v in variants.values()):
        if decoder != 'ours':
            generate = inspect.unwrap(baseline_generate(decoder, model_family))  # past torch.no_grad
            files.append(Path(inspect.getfile(generate)).resolve())
    if model_family == 'dream':
        import dream_adapter
        files.append(Path(dream_adapter.__file__).resolve())
    if files:
        files.append(Path(__file__).resolve())
    return list(dict.fromkeys(files))


def run_baseline(model, model_family, decoder, options, input_ids, attention_mask, *, mask_id, gen_length, block_length, steps):
    generate = baseline_generate(decoder, model_family)
    target, options = model, dict(options)
    if model_family == 'dream':
        attention_mask = None  # unpadded batch of 1; the adapter and the ports insist on it
        if decoder in ('wino', 'dapd'):
            target, options['dream'] = model.model, True  # raw Dream model; shadow-slot shift inside
    kwargs = dict(gen_length=gen_length, block_length=block_length, temperature=0.0, mask_id=mask_id, return_stats=True, attention_mask=attention_mask)
    if decoder in VANILLA:
        kwargs.update(steps=steps, cfg_scale=0.0, remasking=decoder)
    return generate(target, input_ids, **kwargs, **options)

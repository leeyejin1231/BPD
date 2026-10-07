"""Resumable HumanEval for LLaDA: ours by default, optional low_confidence.

Downloads openai/openai_humaneval from Hugging Face on first use.
Candidate programs are executed with a timeout, as in the supplied evaluator.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import transformers
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoTokenizer

# Baseline and optional custom model loader are imported only when selected.
from decode_confidence_v2 import generate as generate_ours, prepare_prefix_cache


MODES = ('low_confidence', 'ours')
STOP_SEQUENCES = ('\nif __name__', '\nprint(', '\nassert ')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def extract_code(text):
    """Return (code, source). Prefer the first fenced python block, then any fence, else raw text."""
    fenced = re.findall(r'```(?:python|py|Python)?[ \t]*\n(.*?)(?:```|$)', text, flags=re.S)
    if fenced:
        return fenced[0], 'fence'
    return text, 'raw'


def build_program(prompt, code, entry_point):
    """Combine prompt and model code the way lm-eval/Fast-dLLM instruct HumanEval evals do."""
    code = code.rstrip()
    defines = re.search(rf'^\s*def\s+{re.escape(entry_point)}\s*\(', code, flags=re.M) is not None
    if defines:
        # The prompt's function body is docstring-only, so re-defining it below is valid.
        return prompt.rstrip() + '\n\n\n' + code + '\n', 'full_function'
    # Continuation: cut at typical trailing test/print blocks and indent if needed.
    for stop in STOP_SEQUENCES:
        cut = code.find(stop)
        if cut != -1:
            code = code[:cut]
    lines = code.splitlines()
    first = next((line for line in lines if line.strip()), '')
    if first and not first.startswith((' ', '\t')):
        lines = ['    ' + line if line.strip() else line for line in lines]
    return prompt + '\n'.join(lines) + '\n', 'continuation'


def run_tests(program, test, entry_point, timeout, workdir):
    """Execute program + test + check(entry_point) in a fresh interpreter. Returns (passed, status, stderr_tail)."""
    source = program + '\n\n' + test + '\n\n' + f'check({entry_point})\n'
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        path = Path(tmp) / 'candidate.py'
        path.write_text(source, encoding='utf-8')
        try:
            proc = subprocess.run(
                [sys.executable, '-I', str(path)], cwd=tmp, capture_output=True, text=True,
                timeout=timeout, env={'PATH': os.environ.get('PATH', ''), 'PYTHONHASHSEED': '0'})
        except subprocess.TimeoutExpired:
            return False, 'timeout', ''
    if proc.returncode == 0:
        return True, 'passed', ''
    tail = proc.stderr.strip().splitlines()
    status = 'failed'
    if tail and 'SyntaxError' in tail[-1] or any('SyntaxError' in line for line in tail[-3:]):
        status = 'syntax_error'
    elif tail and 'AssertionError' in tail[-1]:
        status = 'assertion_failed'
    return False, status, '\n'.join(tail[-5:])


def summarize(rows):
    summary = {}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        subset = [r for r in rows if r['mode'] == mode]
        seconds = sum(r['decode_seconds'] for r in subset)
        steps = sum(r['forward_steps'] for r in subset)
        unmasked = sum(r['unmasked_tokens'] for r in subset)
        statuses = {}
        for r in subset:
            statuses[r['status']] = statuses.get(r['status'], 0) + 1
        summary[mode] = {
            'n': len(subset),
            'pass_at_1_percent': 100 * sum(r['passed'] for r in subset) / len(subset),
            'passed': sum(r['passed'] for r in subset),
            'status_counts': statuses,
            'decode_seconds': seconds,
            'mean_seconds_per_problem': seconds / len(subset),
            'unmasked_tokens_per_second': unmasked / seconds,
            'answer_tokens_per_second': sum(r['answer_tokens'] for r in subset) / seconds,
            'unmasked_tokens_per_step': unmasked / steps,
            'mean_forward_steps': steps / len(subset),
            'mean_answer_tokens': sum(r['answer_tokens'] for r in subset) / len(subset),
            'code_source_counts': {k: sum(r['code_source'] == k for r in subset) for k in ('fence', 'raw')},
            'program_kind_counts': {k: sum(r['program_kind'] == k for r in subset) for k in ('full_function', 'continuation')},
            'no_end_token_count': sum(not r['has_end_token'] for r in subset),
            'residual_mask_tokens': sum(r['residual_mask_tokens'] for r in subset),
            'peak_allocated_gib': max(r['peak_allocated_gib'] for r in subset),
        }
        if all('decoder_stats' in r for r in subset):
            summary[mode]['capped_incomplete_count'] = sum(
                r['decoder_stats'].get('capped_incomplete', False) for r in subset)
            for name in ('remasked', 'edited'):
                if all(f'{name}_per_step' in r['decoder_stats'] for r in subset):
                    summary[mode][f'{name}_tokens_total'] = sum(
                        sum(c[0] for c in r['decoder_stats'][f'{name}_per_step']) for r in subset)
        net = sum(r['final_unmasked_tokens'] for r in subset)
        summary[mode].update({
            'answer_tokens_per_step': sum(r['answer_tokens'] for r in subset) / steps,
            'final_tokens_per_second': net / seconds,
            'final_tokens_per_step': net / steps,
        })
    if all(mode in summary for mode in MODES):
        paired = {}
        for row in rows:
            paired.setdefault(row['test_index'], {})[row['mode']] = row
        pairs = [p for p in paired.values() if all(m in p for m in MODES)]
        if pairs:
            summary['paired'] = {
                'n': len(pairs),
                'decode_speedup': sum(p['low_confidence']['decode_seconds'] for p in pairs) / sum(p['ours']['decode_seconds'] for p in pairs),
                'vanilla_only_passed': sum(p['low_confidence']['passed'] and not p['ours']['passed'] for p in pairs),
                'ours_only_passed': sum(p['ours']['passed'] and not p['low_confidence']['passed'] for p in pairs),
                'both_passed': sum(p['ours']['passed'] and p['low_confidence']['passed'] for p in pairs),
            }
    return summary


def load_humaneval(args):
    path = Path(args.dataset).expanduser()
    if path.is_file():
        dataset = Dataset.from_file(str(path))
    else:
        dataset = load_dataset(args.dataset, split=args.dataset_split,
                               revision=args.dataset_revision, cache_dir=args.cache_dir)
    required = {'task_id', 'prompt', 'test', 'entry_point'}
    if not required.issubset(dataset.column_names):
        raise ValueError(f'Dataset is missing columns: {sorted(required - set(dataset.column_names))}')
    return dataset


def get_variants(args):
    if args.variant_config:
        raw = json.loads(Path(args.variant_config).read_text())
    else:
        raw = {mode: {'decoder': mode} for mode in args.methods}
    if not isinstance(raw, dict) or not raw:
        raise ValueError('variant-config must be a nonempty JSON object.')
    result = {}
    for label, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f'{label}: expected a dictionary of decoder options.')
        kwargs = dict(value)
        decoder = kwargs.pop('decoder', kwargs.pop('remasking', label))
        if decoder == 'confidence_pivot_v2':
            decoder = 'ours'
        if decoder not in ('ours', 'low_confidence', 'random_pivot', 'random'):
            raise ValueError(f'{label}: unsupported decoder {decoder!r}.')
        # Model selection must never be inferred from a baseline's options.
        cache = kwargs.pop('prefix_cache', None)
        if cache is not None and (decoder != 'ours' or cache is not True):
            raise ValueError(f'{label}: prefix_cache is only supported as true for ours. '
                             'Select a model implementation with --model-loader instead.')
        if decoder == 'ours':
            kwargs.setdefault('n', args.n)
            kwargs.setdefault('confidence_threshold', args.confidence_threshold)
            kwargs.setdefault('remask_k', args.remask_k)
            kwargs.setdefault('remask_threshold', args.remask_threshold)
            allowed = {'n', 'confidence_threshold', 'logits_eos_inf', 'eos_token_id',
                       'remask_k', 'remask_threshold'}
            n, k = kwargs['n'], kwargs['remask_k']
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError(f'{label}: n must be a positive integer.')
            # if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n:
            #     raise ValueError(f'{label}: remask_k must satisfy 0 <= remask_k < n.')
            if not 0 <= kwargs['remask_threshold'] <= 1:
                raise ValueError(f'{label}: remask_threshold must be between 0 and 1.')
        else:
            from decode import generate as baseline
            reserved = {'model', 'prompt', 'attention_mask', 'steps', 'gen_length',
                        'block_length', 'temperature', 'cfg_scale', 'remasking',
                        'mask_id', 'return_stats'}
            allowed = set(inspect.signature(baseline).parameters) - reserved
        if set(kwargs) - allowed:
            raise ValueError(f'{label}: unsupported options: {sorted(set(kwargs) - allowed)}')
        result[label] = {'decoder': decoder, **kwargs}
    return result


def run_decoder(model, encoded, variant, args, mask_id):
    kwargs = dict(variant)
    decoder = kwargs.pop('decoder')
    common = dict(gen_length=args.gen_length, block_length=args.block_length,
                  mask_id=mask_id, temperature=0.0, return_stats=True)
    if decoder == 'ours':
        return generate_ours(model, encoded['input_ids'],
                             attention_mask=encoded.get('attention_mask'), **common, **kwargs)
    from decode import generate as baseline
    return baseline(model, encoded['input_ids'],
                    attention_mask=encoded.get('attention_mask'),
                    steps=args.steps, cfg_scale=0.0, remasking=decoder, **common, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default='openai/openai_humaneval', help='HF dataset ID or local Arrow file')
    parser.add_argument('--dataset-split', default='test')
    parser.add_argument('--dataset-revision', default='main')
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--model-revision', default=None)
    parser.add_argument('--local-files-only', action='store_true', help='Use cached model/tokenizer files only')
    parser.add_argument('--model-loader', default='hf', help='hf or module:function (for example fastdllm_decode:load_model)')
    parser.add_argument('--native-cache-only', action='store_true', help='Disable the stock LLaDA cache compatibility patch')
    parser.add_argument('--methods', nargs='+', choices=('ours', 'low_confidence', 'random_pivot'), default=['ours'])
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--confidence-threshold', type=float, default=0.9)
    parser.add_argument('--remask-k', type=int, default=0, help='Remask at most k tokens per incomplete block step; 0 disables, k < n')
    parser.add_argument('--remask-threshold', type=float, default=0.3, help='Only remask tokens with current-token probability strictly below this value')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen-length', type=int, default=256)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--exec-timeout', type=float, default=10.0)
    parser.add_argument('--limit', type=int, default=None, help='Evaluate only the first N tasks (debug)')
    parser.add_argument('--variant-config', help='JSON mapping experiment labels to generate keyword arguments')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this benchmark requires a GPU.')
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_humaneval(args)
    indices = list(range(len(dataset)))[:args.limit]
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if args.gen_length <= 0 or args.block_length <= 0 or args.gen_length % args.block_length:
        parser.error('--gen-length must be positive and divisible by --block-length.')
    if args.exec_timeout <= 0:
        parser.error('--exec-timeout must be positive.')
    variants = get_variants(args)
    if any(v['decoder'] in ('low_confidence', 'random') for v in variants.values()):
        blocks = args.gen_length // args.block_length
        if args.steps < blocks or args.steps % blocks:
            parser.error('--steps must be positive and divisible by the number of blocks for low_confidence/random.')
    uses_ours = any(v['decoder'] == 'ours' for v in variants.values())
    loader = args.model_loader
    source_paths = [Path(__file__).resolve(), Path(__file__).with_name('decode_confidence_v2.py').resolve()]
    if any(v['decoder'] != 'ours' for v in variants.values()):
        source_paths.append(Path(__file__).with_name('decode.py').resolve())
    custom_load = None
    if loader != 'hf':
        module_name, sep, function_name = loader.partition(':')
        if not sep:
            parser.error('--model-loader must be hf or module:function.')
        module = importlib.import_module(module_name)
        custom_load = getattr(module, function_name)
        if getattr(module, '__file__', None):
            source_paths.append(Path(module.__file__).resolve())
    modes_to_run = tuple(variants)
    prompt_template = ('Complete the following Python function. Reply with the complete '
                       'function definition in a ```python code block.\n\n```python\n{prompt}```')
    config = vars(args).copy()
    config.update({
        'resolved_model_loader': loader,
        'dataset_fingerprint': getattr(dataset, '_fingerprint', None),
        'num_tasks': len(indices), 'batch_size': 1, 'num_fewshot': 0,
        'variants': variants,
        'temperature': 0, 'cfg_scale': 0, 'dtype': 'bfloat16',
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'prompt_template': prompt_template,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'metrics': {
            'pass_at_1': 'One greedy sample per task; prompt + extracted code + test + check(entry_point) executed in a subprocess with timeout.',
            'timing': 'CUDA-synchronized wall time around generate; excludes tokenization, loading, warmup, code extraction, and test execution.',
            'tokens_per_second': 'Gross unmask events, including refills after remasking, special tokens and post-EOS positions / decode seconds. See final_tokens_per_second for final output throughput.',
            'answer_tokens_per_second': 'Non-special tokens before first EOS/EOT / decode seconds.',
            'tokens_per_step': 'Actual unmasked positions / forward calls, batch size 1.',
        },
    })
    config_path = out_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing output has a different configuration; use a new output directory.')
    write_json(config_path, config)
    source_dir = out_dir / 'source'
    source_dir.mkdir(exist_ok=True)
    for path in source_paths:
        (source_dir / path.name).write_bytes(path.read_bytes())
    workdir = out_dir / 'exec_tmp'
    workdir.mkdir(exist_ok=True)
    records_path = out_dir / 'results.jsonl'
    rows = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    done = {(r['test_index'], r['mode']) for r in rows}
    if len(done) == len(indices) * len(modes_to_run):
        write_json(out_dir / 'summary.json', summarize(rows))
        print(json.dumps(summarize(rows), indent=2), flush=True)
        return
    print(f'Loading {args.model} on {config["gpu"]}', flush=True)
    pretrained = dict(trust_remote_code=True, local_files_only=args.local_files_only)
    if args.model_revision is not None:
        pretrained['revision'] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **pretrained)
    if custom_load is not None:
        if args.model_revision is not None:
            raise ValueError('For a custom loader, pass a pinned local checkpoint as --model instead of --model-revision.')
        model = custom_load(args.model)
    else:
        model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16,
                                          **pretrained).to('cuda').eval()
    model.eval()
    # No global cache toggle: each decoder explicitly chooses use_cache per call.
    if uses_ours and loader == 'hf' and not args.native_cache_only:
        print('Cache preparation: ' + prepare_prefix_cache(model), flush=True)
    print('Decoders: ' + json.dumps(variants, ensure_ascii=False), flush=True)
    print(f'Model loader: {loader}; ours uses prefix KV cache independently of baseline selection.', flush=True)
    mask_id, end_ids = 126336, {126081, 126348}
    eos = tokenizer.eos_token_id
    if eos is not None:
        end_ids.add(eos)
    special_ids = set(tokenizer.all_special_ids) | end_ids | {mask_id}

    def encode(prompt):
        messages = [{'role': 'user', 'content': prompt_template.format(prompt=prompt)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')

    def run(encoded, mode):
        return run_decoder(model, encoded, variants[mode], args, mask_id)

    warmup = encode('def add(a: int, b: int) -> int:\n    """Return the sum of a and b.\n    >>> add(1, 2)\n    3\n    """\n')
    for mode in modes_to_run:
        torch.manual_seed(args.seed)
        warmup_tokens, warmup_stats = run(warmup, mode)
        # if variants[mode]['decoder'] == 'ours' and warmup_stats['prefill_steps'] != args.gen_length // args.block_length:
        #     raise RuntimeError('Unexpected cache prefill count.')
        ################################################################################
        if variants[mode]['decoder'] == 'ours':
            generated_length = (
                warmup_tokens.shape[1]
                - warmup['input_ids'].shape[1]
            )
            expected_prefills = generated_length // args.block_length

            if (
                generated_length <= 0
                or generated_length % args.block_length != 0
                or warmup_stats['prefill_steps'] != expected_prefills
            ):
                raise RuntimeError('Unexpected cache prefill count.')
        ################################################################################
        del warmup_tokens, warmup_stats
        torch.cuda.synchronize()
        print(f'Warmup complete: {mode}', flush=True)
    del warmup

    with records_path.open('a', buffering=1) as output:
        for ordinal, index in enumerate(indices):
            sample = dataset[index]
            encoded = encode(sample['prompt'])
            offset = ordinal % len(modes_to_run)
            modes = modes_to_run[offset:] + modes_to_run[:offset]
            for mode in modes:
                if (index, mode) in done:
                    continue
                sample_seed = args.seed + index
                torch.manual_seed(sample_seed)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                tokens, stats = run(encoded, mode)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                generated = tokens[0, encoded['input_ids'].shape[1]:].tolist()
                end = next((i for i, token in enumerate(generated) if token in end_ids), len(generated))
                text = tokenizer.decode(generated[:end], skip_special_tokens=True)
                code, code_source = extract_code(text)
                program, program_kind = build_program(sample['prompt'], code, sample['entry_point'])
                passed, status, error_tail = run_tests(program, sample['test'], sample['entry_point'],
                                                       args.exec_timeout, workdir)
                counts = [count[0] for count in stats['unmasked_per_step']]
                row = {
                    'test_index': index, 'task_id': sample['task_id'], 'mode': mode, 'seed': sample_seed,
                    'entry_point': sample['entry_point'],
                    'passed': passed, 'status': status, 'error_tail': error_tail,
                    'code_source': code_source, 'program_kind': program_kind,
                    'has_end_token': end < len(generated),
                    'text': text, 'code': code, 'program': program,
                    'raw_text': tokenizer.decode(generated, skip_special_tokens=False),
                    'generated_ids': generated, 'prompt_tokens': encoded['input_ids'].shape[1],
                    'answer_tokens': sum(token not in special_ids for token in generated[:end]),
                    'unmasked_tokens': sum(counts), 'unmasked_per_step': counts,
                    'final_unmasked_tokens': sum(token != mask_id for token in generated),
                    'decoder_stats': stats,
                    'forward_steps': stats['forward_steps'], 'decode_seconds': elapsed,
                    'residual_mask_tokens': generated.count(mask_id),
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                }
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
                done.add((index, mode))
                write_json(out_dir / 'summary.json', summarize(rows))
                print(f'[{len(done)}/{len(indices) * len(modes_to_run)}] {sample["task_id"]} {mode}: '
                      f'passed={passed} ({status}) steps={row["forward_steps"]} '
                      f'time={elapsed:.2f}s tokens/s={row["unmasked_tokens"]/elapsed:.2f} '
                      f'tokens/step={row["unmasked_tokens"]/row["forward_steps"]:.2f}', flush=True)
                del tokens
    print('COMPLETE\n' + json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Resumable HF MATH-500 / MBPP / IFEval evaluation for LLaDA.

Default: ours only; decode_confidence_v2.py must be beside this file.
Optional baselines are imported only when requested.
HF defaults: MATH-500 test; MBPP full/test; IFEval train.
Local Arrow files remain supported via --dataset.

Scoring preserves the supplied evaluator's protocol: normalized boxed-answer
matching for MATH (not symbolic equivalence), visible-test MBPP, and the local
ifeval_lib strict/loose instruction checkers. IFEval requires ifeval_lib/ beside
this script, including its dependencies. MBPP runs generated Python in a
subprocess with a timeout; that subprocess is not a security sandbox.

--shard k/n selects indices with index % n == k; use separate output directories.
--limit applies after sharding. Each example uses seed + its dataset index.
"""
import argparse
import hashlib
import inspect
import json
import math
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
from decode_confidence_v2 import generate as generate_ours, prepare_prefix_cache

def last_boxed(text):
    """Return the content of the last \\boxed{...} / \\fbox{...} with balanced braces, else None."""
    start = max(text.rfind('\\boxed'), text.rfind('\\fbox'))
    if start < 0:
        return None
    i = text.find('{', start)
    if i < 0:
        tail = text[start:].split('$')[0]
        tail = tail.replace('\\boxed', '').replace('\\fbox', '').strip()
        return tail or None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += '\\frac'
            if substr and substr[0] == '{':
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != '{':
                    new_str += '{' + a + '}{' + b + '}' + substr[2:]
                else:
                    new_str += '{' + a + '}' + b + substr[2:]
    return new_str


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a, b = string.split('/')
    try:
        a, b = int(a), int(b)
        assert string == f'{a}/{b}'
        return '\\frac{' + str(a) + '}{' + str(b) + '}'
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string):
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        return splits[0]
    return string


def _fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != '{':
            new_string += '\\sqrt{' + split[0] + '}' + split[1:]
        else:
            new_string += '\\sqrt' + split
    return new_string


def strip_string(string):
    """Hendrycks MATH normalization (math_equivalence.strip_string) with a few extras."""
    string = str(string).replace('\n', '').replace('\\!', '').replace('\\\\', '\\')
    string = string.replace('tfrac', 'frac').replace('dfrac', 'frac')
    string = string.replace('\\left', '').replace('\\right', '')
    string = string.replace('^{\\circ}', '').replace('^\\circ', '')
    string = string.replace('\\$', '').replace('$', '')
    string = re.sub(r'\\text\{\s*([^}]*)\}', r'\1', string) if '\\text{ ' not in string else _remove_right_units(string)
    string = string.replace('\\%', '').replace('%', '')
    string = string.replace(' .', ' 0.').replace('{.', '{0.')
    if string and string[0] == '.':
        string = '0' + string
    if len(string.split('=')) == 2 and len(string.split('=')[0]) <= 2:
        string = string.split('=')[1]
    string = _fix_sqrt(string)
    string = string.replace(' ', '')
    string = _fix_fracs(string)
    if string == '0.5':
        string = '\\frac{1}{2}'
    string = _fix_a_slash_b(string)
    string = string.replace(',\\!', '').rstrip('.')
    return string


def math_equal(prediction, gold):
    if prediction is None:
        return False
    try:
        a, b = strip_string(prediction), strip_string(gold)
    except (ValueError, AssertionError, IndexError):
        return False
    if a == b:
        return True
    try:
        return abs(float(a.replace(',', '')) - float(b.replace(',', ''))) < 1e-6
    except (ValueError, OverflowError):
        return False


# ----------------------------------------------------------------------------- code

def extract_code(text):
    fenced = re.findall(r'```(?:python|py|Python)?[ \t]*\n(.*?)(?:```|$)', text, flags=re.S)
    return (fenced[0], 'fence') if fenced else (text, 'raw')


def run_program(source, timeout, workdir):
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        path = Path(tmp) / 'candidate.py'
        path.write_text(source)
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
    if any(name in line for line in tail[-3:] for name in ('SyntaxError', 'IndentationError', 'TabError')):
        status = 'syntax_error'
    elif tail and 'AssertionError' in tail[-1]:
        status = 'assertion_failed'
    return False, status, '\n'.join(tail[-5:])


# ----------------------------------------------------------------------------- ifeval

def ifeval_score(sample, response):
    from ifeval_lib import instructions_registry
    ids = sample['instruction_id_list']
    strict, loose = [], []
    variants = [response]
    stripped = response.replace('*', '')
    lines = response.split('\n')
    no_first = '\n'.join(lines[1:]).strip()
    no_last = '\n'.join(lines[:-1]).strip()
    no_both = '\n'.join(lines[1:-1]).strip()
    for v in (no_first, no_last, no_both, stripped, no_first.replace('*', ''),
              no_last.replace('*', ''), no_both.replace('*', '')):
        variants.append(v)
    for index, instruction_id in enumerate(ids):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        kwargs = {k: v for k, v in sample['kwargs'][index].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and 'prompt' in args:
            instruction.build_description(prompt=sample['prompt'])
        strict.append(bool(response.strip()) and instruction.check_following(response))
        loose.append(any(v.strip() and instruction.check_following(v) for v in variants))
    return {'strict_follow': strict, 'loose_follow': loose,
            'prompt_strict': all(strict), 'prompt_loose': all(loose)}


# ----------------------------------------------------------------------------- shared

TASKS = {
    'math500': {
        'question': lambda s: s['problem'],
        'suffix': '\n\nSolve this problem step by step. Put your final answer within \\boxed{}.',
        'warmup': 'What is $2 + 3$?',
    },
    'mbpp': {
        'question': lambda s: ('You are an expert Python programmer, and here is your task: '
                               + s['text'].strip() + ' Your code should pass these tests:\n\n'
                               + '\n'.join(s['test_list'])),
        'suffix': '\n\nReply with the complete solution in a ```python code block.',
        'warmup': ('You are an expert Python programmer, and here is your task: Write a function '
                   'to add two numbers. Your code should pass these tests:\n\nassert add(1, 2) == 3'),
    },
    'ifeval': {
        'question': lambda s: s['prompt'],
        'suffix': '',
        'warmup': 'Write a short poem about the sea. Do not use any commas.',
    },
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def summarize(rows, task):
    summary = {}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        subset = [r for r in rows if r['mode'] == mode]
        seconds = sum(r['decode_seconds'] for r in subset)
        steps = sum(r['forward_steps'] for r in subset)
        unmasked = sum(r['unmasked_tokens'] for r in subset)
        entry = {
            'n': len(subset),
            'decode_seconds': seconds,
            'mean_seconds_per_problem': seconds / len(subset),
            'unmasked_tokens_per_second': unmasked / seconds,
            'answer_tokens_per_second': sum(r['answer_tokens'] for r in subset) / seconds,
            'unmasked_tokens_per_step': unmasked / steps,
            'mean_forward_steps': steps / len(subset),
            'mean_answer_tokens': sum(r['answer_tokens'] for r in subset) / len(subset),
            'final_tokens_per_second': sum(r['final_unmasked_tokens'] for r in subset) / seconds,
            'final_tokens_per_step': sum(r['final_unmasked_tokens'] for r in subset) / steps,
            'answer_tokens_per_step': sum(r['answer_tokens'] for r in subset) / steps,
            'remasked_tokens_total': sum(
                sum(sum(c) for c in r['decoder_stats'].get('remasked_per_step', []))
                for r in subset),
            'no_end_token_count': sum(not r['has_end_token'] for r in subset),
            'residual_mask_tokens': sum(r['residual_mask_tokens'] for r in subset),
            'peak_allocated_gib': max(r['peak_allocated_gib'] for r in subset),
        }
        if task == 'math500':
            entry['correct'] = sum(r['correct'] for r in subset)
            entry['accuracy_percent'] = 100 * sum(r['correct'] for r in subset) / len(subset)
            entry['missing_boxed_count'] = sum(r['prediction'] is None for r in subset)
        elif task == 'mbpp':
            entry['passed'] = sum(r['passed'] for r in subset)
            entry['pass_at_1_percent'] = 100 * sum(r['passed'] for r in subset) / len(subset)
            statuses = {}
            for r in subset:
                statuses[r['status']] = statuses.get(r['status'], 0) + 1
            entry['status_counts'] = statuses
        else:
            n_inst = sum(len(r['ifeval']['strict_follow']) for r in subset)
            entry.update({
                'prompt_strict_accuracy_percent': 100 * sum(r['ifeval']['prompt_strict'] for r in subset) / len(subset),
                'prompt_loose_accuracy_percent': 100 * sum(r['ifeval']['prompt_loose'] for r in subset) / len(subset),
                'instruction_strict_accuracy_percent': 100 * sum(sum(r['ifeval']['strict_follow']) for r in subset) / n_inst if n_inst else 0.0,
                'instruction_loose_accuracy_percent': 100 * sum(sum(r['ifeval']['loose_follow']) for r in subset) / n_inst if n_inst else 0.0,
            })
        if all('decoder_stats' in r for r in subset):
            entry['capped_incomplete_count'] = sum(r['decoder_stats'].get('capped_incomplete', False) for r in subset)
            for diagnostic in ('deferred', 'fallback', 'bulk_tokens'):
                entry[f'{diagnostic}_total'] = sum(
                    sum(sum(c) for c in r['decoder_stats'].get(f'{diagnostic}_per_step', [])) for r in subset)
        summary[mode] = entry
    return summary




def get_variants(args):
    raw = (
        json.loads(Path(args.variant_config).read_text(encoding='utf-8'))
        if args.variant_config
        else {name: {'decoder': name} for name in args.methods}
    )
    if not isinstance(raw, dict) or not raw:
        raise ValueError('variant-config must be a nonempty JSON object.')

    variants = {}
    reserved = {
        'model', 'prompt', 'attention_mask', 'steps', 'gen_length',
        'block_length', 'temperature', 'cfg_scale', 'remasking',
        'mask_id', 'return_stats',
    }

    for label, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f'{label}: expected decoder options.')

        options = dict(value)
        decoder = options.pop('decoder', options.pop('remasking', label))
        if decoder == 'confidence_pivot_v2':
            decoder = 'ours'

        if decoder == 'ours':
            for key in (
                'n', 'confidence_threshold', 'remask_k', 'remask_threshold'
            ):
                options.setdefault(key, getattr(args, key))

            if args.n_switch_threshold is not None:
                options.setdefault('n_switch_threshold', args.n_switch_threshold)

            allowed = {
                'n', 'confidence_threshold', 'remask_k', 'remask_threshold',
                'logits_eos_inf', 'eos_token_id', 'n_switch_threshold',
            }

            integer_options = [('n', 1), ('remask_k', 0)]
            if 'n_switch_threshold' in options:
                integer_options.append(('n_switch_threshold', 0))
            for key, minimum in integer_options:
                value = options[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < minimum
                ):
                    raise ValueError(
                        f'{label}: {key} must be an integer >= {minimum}.'
                    )

            for key in ('confidence_threshold', 'remask_threshold'):
                if not 0 <= options[key] <= 1:
                    raise ValueError(
                        f'{label}: {key} must be between 0 and 1.'
                    )

        elif decoder in ('low_confidence', 'random_pivot', 'random'):
            from decode import generate as baseline
            allowed = set(inspect.signature(baseline).parameters) - reserved

        else:
            raise ValueError(f'{label}: unsupported decoder {decoder!r}.')

        if set(options) - allowed:
            raise ValueError(
                f'{label}: unsupported options {sorted(set(options) - allowed)}'
            )

        if decoder == 'ours':
            parameters = inspect.signature(generate_ours).parameters
            unsupported = set(options) - set(parameters)
            if unsupported:
                raise ValueError(f'{label}: current decoder does not accept {sorted(unsupported)}')
        variants[label] = {'decoder': decoder, **options}

    return variants

def run_decoder(model, encoded, variant, args):
    options = dict(variant)
    decoder = options.pop('decoder')
    common = dict(
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        mask_id=126336,
        return_stats=True,
        attention_mask=encoded.get('attention_mask'),
    )

    if decoder == 'ours':
        return generate_ours(
            model, encoded['input_ids'], **common, **options
        )

    from decode import generate as baseline
    return baseline(
        model,
        encoded['input_ids'],
        steps=args.steps,
        cfg_scale=0.0,
        remasking=decoder,
        **common,
        **options,
    )


HF_TASKS = {
    'math500': ('HuggingFaceH4/MATH-500', None, 'test', {'problem', 'answer', 'unique_id'}),
    'mbpp': ('google-research-datasets/mbpp', 'full', 'test',
             {'text', 'test_list', 'test_setup_code', 'task_id'}),
    'ifeval': ('google/IFEval', None, 'train',
               {'prompt', 'instruction_id_list', 'kwargs', 'key'}),
}


def load_data(args):
    path = Path(args.dataset).expanduser()
    if path.is_file():
        data = Dataset.from_file(str(path))
    else:
        data = load_dataset(
            args.dataset, args.dataset_config, split=args.dataset_split,
            revision=args.dataset_revision, cache_dir=args.cache_dir,
        )
    missing = HF_TASKS[args.task][3] - set(data.column_names)
    if missing:
        raise ValueError(f'{args.task}: missing dataset columns {sorted(missing)}')
    if not len(data):
        raise ValueError('Dataset is empty.')
    return data


def select_indices(size, shard=None, limit=None):
    indices = list(range(size))
    if shard is not None:
        try:
            k, n = map(int, shard.split('/'))
        except ValueError as exc:
            raise ValueError('--shard must be k/n, for example 0/2.') from exc
        if n <= 0 or not 0 <= k < n:
            raise ValueError('--shard requires n > 0 and 0 <= k < n.')
        indices = [i for i in indices if i % n == k]
    if limit is not None:
        if limit <= 0:
            raise ValueError('--limit must be positive.')
        indices = indices[:limit]
    if not indices:
        raise ValueError('No examples selected by this shard/limit.')
    return indices


def load_records(path, expected):
    if not path.exists():
        return [], set()
    rows, done = [], set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid results.jsonl line {line_number}; inspect an interrupted write.') from exc
        key = (row['test_index'], row['mode'])
        if key not in expected or key in done:
            raise ValueError(f'Unexpected or duplicate result: {key}')
        rows.append(row)
        done.add(key)
    return rows, done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=sorted(TASKS))
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', help='HF dataset ID or local Arrow file; defaults by task')
    parser.add_argument('--dataset-config')
    parser.add_argument('--dataset-split')
    parser.add_argument('--dataset-revision', default='main')
    parser.add_argument('--cache-dir')
    parser.add_argument('--model-revision')
    parser.add_argument('--local-files-only', action='store_true',
                        help='Use cached model/tokenizer files only; does not control datasets')
    parser.add_argument('--native-cache-only', action='store_true',
                        help='Skip the stock LLaDA cache-guard compatibility patch')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen-length', type=int, default=256)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument('--steps', type=int, default=256, help='Baseline only; ignored by ours')
    parser.add_argument('--exec-timeout', type=float, default=10.0)
    parser.add_argument('--limit', type=int, help='First N selected examples, after sharding')
    parser.add_argument('--shard', help='k/n; use a separate output directory per shard')
    parser.add_argument('--methods', nargs='+', default=['ours'],
                        choices=('ours', 'low_confidence', 'random_pivot', 'random'))
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--confidence-threshold', type=float, default=0.9)
    parser.add_argument('--remask-k', type=int, default=0)
    parser.add_argument('--remask-threshold', type=float, default=0.5)
    parser.add_argument('--n-switch-threshold', type=int, default=None,
                        help='Omit to use the current decoder default, if it has this schedule')
    parser.add_argument('--variant-config', help='Optional JSON; decoder/remasking aliases supported')
    args = parser.parse_args()

    if args.gen_length <= 0 or args.block_length <= 0 or args.gen_length % args.block_length:
        parser.error('--gen-length must be positive and divisible by --block-length.')
    if not math.isfinite(args.exec_timeout) or args.exec_timeout <= 0:
        parser.error('--exec-timeout must be finite and positive.')
    dataset_id, dataset_config, dataset_split, _ = HF_TASKS[args.task]
    if args.dataset is None:
        args.dataset = dataset_id
    if args.dataset_config is None and args.dataset == dataset_id:
        args.dataset_config = dataset_config
    if args.dataset_split is None:
        args.dataset_split = dataset_split
    variants = get_variants(args)
    blocks = args.gen_length // args.block_length
    if any(v['decoder'] != 'ours' for v in variants.values()):
        if args.steps < blocks or args.steps % blocks:
            parser.error('--steps must be positive and divisible by the number of blocks.')

    dataset = load_data(args)
    indices = select_indices(len(dataset), args.shard, args.limit)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this benchmark requires a GPU.')
    task = TASKS[args.task]
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__).resolve()]
    if any(v['decoder'] == 'ours' for v in variants.values()):
        sources.append(Path(inspect.getfile(inspect.unwrap(generate_ours))).resolve())
    if any(v['decoder'] != 'ours' for v in variants.values()):
        from decode import generate as baseline
        sources.append(Path(inspect.getfile(inspect.unwrap(baseline))).resolve())
    ifeval_sources = []
    if args.task == 'ifeval':
        try:
            from ifeval_lib import instructions_registry
        except ImportError as exc:
            raise RuntimeError(
                'IFEval requires the original ifeval_lib/ package and its dependencies '
                'beside this script. HF downloads the dataset, not the checker. '
                f'Import error: {exc}'
            ) from exc
        checker_dir = Path(instructions_registry.__file__).resolve().parent
        ifeval_sources = sorted(checker_dir.rglob('*.py'))
        checker_hashes = {
            str(p.relative_to(checker_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ifeval_sources
        }
    modes_to_run = tuple(variants)
    config = vars(args).copy()
    config['cli_decoder_defaults'] = {
        key: config.pop(key) for key in (
            'n', 'confidence_threshold', 'remask_k', 'remask_threshold', 'n_switch_threshold')
    }
    config.update({
        'model_family': 'llada', 'model_loader': 'hf',
        'num_tasks': len(indices), 'test_indices': indices,
        'dataset_fingerprint': getattr(dataset, '_fingerprint', None),
        'batch_size': 1, 'num_fewshot': 0, 'variants': variants,
        'decoder_signature': str(inspect.signature(generate_ours)),
        'temperature': 0, 'cfg_scale': 0, 'dtype': 'bfloat16',
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'prompt_suffix': task['suffix'],
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'metrics': {
            'timing': 'CUDA-synchronized generate time; excludes loading, tokenization, warmup and scoring.',
            'tokens_per_second': 'Gross unmask events, including refills and special/post-EOS tokens / decode seconds.',
            'final_tokens_per_second': 'Final non-MASK generated positions, including special/post-EOS tokens / decode seconds.',
            'answer_tokens_per_second': 'Non-special tokens before first EOS/EOT / decode seconds.',
            'math500': 'Normalized last boxed answer plus numeric tolerance 1e-6; not symbolic equivalence.',
            'mbpp': 'Generated code plus test_setup_code and visible test_list, subprocess timeout.',
            'ifeval': 'Original ifeval_lib checkers; prompt/instruction strict and loose scores.',
        },
    })
    if args.task == 'ifeval':
        config['ifeval_checker_sha256'] = checker_hashes
    config_path = out_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing output has a different configuration; use a new output directory.')
    write_json(config_path, config)
    source_dir = out_dir / 'source'
    source_dir.mkdir(exist_ok=True)
    for path in sources:
        (source_dir / path.name).write_bytes(path.read_bytes())
    for path in ifeval_sources:
        dest = source_dir / 'ifeval_lib' / path.relative_to(checker_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    write_json(out_dir / 'samples.json', [{'test_index': i, **dataset[i]} for i in indices])
    workdir = out_dir / 'exec_tmp'
    workdir.mkdir(exist_ok=True)
    records_path = out_dir / 'results.jsonl'
    expected = {(i, mode) for i in indices for mode in variants}
    rows, done = load_records(records_path, expected)
    if done == expected:
        summary = summarize(rows, args.task)
        write_json(out_dir / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        return

    pretrained = dict(trust_remote_code=True, local_files_only=args.local_files_only)
    if args.cache_dir is not None:
        pretrained['cache_dir'] = args.cache_dir
    if args.model_revision is not None:
        pretrained['revision'] = args.model_revision
    print(f'Loading {args.model}; variants={json.dumps(variants)}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **pretrained)
    model = AutoModel.from_pretrained(
        args.model, dtype=torch.bfloat16, **pretrained,
    ).to('cuda').eval()
    if any(v['decoder'] == 'ours' for v in variants.values()) and not args.native_cache_only:
        print('Cache preparation: ' + prepare_prefix_cache(model), flush=True)
    mask_id, end_ids = 126336, {126081, 126348}
    if tokenizer.eos_token_id is not None:
        end_ids.add(tokenizer.eos_token_id)
    special_ids = set(tokenizer.all_special_ids) | end_ids | {mask_id}

    def encode(question):
        messages = [{'role': 'user', 'content': question + task['suffix']}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')

    def run(encoded, mode):
        return run_decoder(model, encoded, variants[mode], args)

    warmup = encode(task['warmup'])
    for mode in modes_to_run:
        torch.manual_seed(args.seed)
        run(warmup, mode)
        torch.cuda.synchronize()
        print(f'Warmup complete: {mode}', flush=True)
    del warmup

    with records_path.open('a', buffering=1) as output:
        for ordinal, index in enumerate(indices):
            sample = dataset[index]
            encoded = encode(task['question'](sample))
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
                counts = [count[0] for count in stats['unmasked_per_step']]
                row = {
                    'test_index': index, 'mode': mode, 'seed': sample_seed,
                    'text': text, 'raw_text': tokenizer.decode(generated, skip_special_tokens=False),
                    'has_end_token': end < len(generated),
                    'reached_generation_limit_without_end': end == len(generated) and len(generated) >= args.gen_length,
                    'generated_ids': generated, 'prompt_tokens': encoded['input_ids'].shape[1],
                    'answer_tokens': sum(token not in special_ids for token in generated[:end]),
                    'unmasked_tokens': sum(counts), 'unmasked_per_step': counts,
                    'final_unmasked_tokens': sum(token != mask_id for token in generated),
                    'decoder_stats': stats,
                    'forward_steps': stats['forward_steps'], 'decode_seconds': elapsed,
                    'residual_mask_tokens': generated.count(mask_id),
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                }
                if args.task == 'math500':
                    prediction = last_boxed(text)
                    row.update({'unique_id': sample['unique_id'], 'gold': sample['answer'],
                                'prediction': prediction, 'correct': math_equal(prediction, sample['answer'])})
                    score = f'correct={row["correct"]}'
                elif args.task == 'mbpp':
                    code, code_source = extract_code(text)
                    for stop in ('\nif __name__', '\nprint(', '\nassert '):
                        cut = code.find(stop)
                        if cut != -1:
                            code = code[:cut]
                    program = (sample['test_setup_code'] or '') + '\n' + code.rstrip() + '\n\n' + '\n'.join(sample['test_list']) + '\n'
                    passed, status, error_tail = run_program(program, args.exec_timeout, workdir)
                    row.update({'task_id': sample['task_id'], 'code': code, 'code_source': code_source,
                                'program': program, 'passed': passed, 'status': status, 'error_tail': error_tail})
                    score = f'passed={passed} ({status})'
                else:
                    result = ifeval_score(sample, text)
                    row.update({'key': sample['key'], 'ifeval': result})
                    score = f'strict={result["prompt_strict"]} loose={result["prompt_loose"]}'
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
                done.add((index, mode))
                write_json(out_dir / 'summary.json', summarize(rows, args.task))
                print(f'[{len(done)}/{len(indices) * len(modes_to_run)}] index={index} {mode}: {score} '
                      f'steps={row["forward_steps"]} time={elapsed:.2f}s '
                      f'answer_tokens/s={row["answer_tokens"]/elapsed:.2f} '
                      f'final_tokens/s={row["final_unmasked_tokens"]/elapsed:.2f}', flush=True)
                del tokens
    print('COMPLETE\n' + json.dumps(summarize(rows, args.task), indent=2), flush=True)


if __name__ == '__main__':
    main()

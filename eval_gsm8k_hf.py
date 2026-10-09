"""Resumable GSM8K evaluation for LLaDA or Dream using Hugging Face datasets.

Decoders: ours (decode_confidence_v2.py, LLaDA only) and the baselines low_confidence /
random / random_pivot (decode.py), hierarchy, wino and fastdllm; see eval_models.py for the
model implementation each one needs and for --model-family dream.
"""
import argparse
import hashlib
import json
import random
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import transformers
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from decode_confidence_v2 import (
    generate as generate_ours,
    prepare_prefix_cache,
)
from eval_models import (
    BASELINES, DECODERS, MODEL_FAMILIES, TOKEN_IDS,
    baseline_options, load_model, resolve_loader, run_baseline, source_files,
)


NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
PROMPT_SUFFIX = (
    '\n\nSolve this problem step by step. End your response with '
    '"#### <answer>", where <answer> is the final numerical answer.'
)


def normalize_number(value):
    try:
        return str(Decimal(value.replace(',', '').strip()).normalize())
    except InvalidOperation:
        return None


def extract_answer(text):
    marked = re.findall(r'####\s*\$?\s*(' + NUMBER + r')', text)
    if marked:
        return normalize_number(marked[-1]), True
    numbers = re.findall(NUMBER, text)
    return (normalize_number(numbers[-1]) if numbers else None), False


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def load_data(args):
    path = Path(args.dataset).expanduser()
    if path.is_file():
        data = Dataset.from_file(str(path))
    else:
        data = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.dataset_split,
            revision=args.dataset_revision,
            cache_dir=args.cache_dir,
        )

    if not {'question', 'answer'}.issubset(data.column_names):
        raise ValueError('Dataset must have question and answer columns.')
    if not len(data):
        raise ValueError('Dataset is empty.')
    return data


def get_variants(args):
    raw = (
        json.loads(Path(args.variant_config).read_text(encoding='utf-8'))
        if args.variant_config
        else {name: {'decoder': name} for name in args.methods}
    )
    if not isinstance(raw, dict) or not raw:
        raise ValueError('variant-config must be a nonempty JSON object.')

    variants = {}

    for label, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f'{label}: expected decoder options.')

        options = dict(value)
        decoder = options.pop('decoder', options.pop('remasking', label))
        if decoder == 'confidence_pivot_v2':
            decoder = 'ours'

        if decoder == 'ours':
            for key in (
                'n', 'confidence_threshold', 'remask_k', 'remask_threshold',
                'force_confidence',
            ):
                options.setdefault(key, getattr(args, key))

            allowed = {
                'n', 'confidence_threshold', 'remask_k', 'remask_threshold',
                'logits_eos_inf', 'eos_token_id', 'force_confidence',
            }

            for key, minimum in (('n', 1), ('remask_k', 0)):
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

        elif decoder in BASELINES:
            allowed = baseline_options(decoder, args.model_family)

        else:
            raise ValueError(f'{label}: unsupported decoder {decoder!r}.')

        if set(options) - allowed:
            raise ValueError(
                f'{label}: unsupported options {sorted(set(options) - allowed)}'
            )

        variants[label] = {'decoder': decoder, **options}

    return variants


def run_decoder(model, encoded, variant, args):
    options = dict(variant)
    decoder = options.pop('decoder')
    mask_id = TOKEN_IDS[args.model_family]['mask_id']

    if decoder == 'ours':
        return generate_ours(
            model, encoded['input_ids'],
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=0.0,
            mask_id=mask_id,
            return_stats=True,
            attention_mask=encoded.get('attention_mask'),
            dream=args.model_family == 'dream',
            **options,
        )

    return run_baseline(
        model, args.model_family, decoder, options,
        encoded['input_ids'], encoded.get('attention_mask'),
        mask_id=mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
    )


def summarize(rows):
    summary = {}

    for mode in dict.fromkeys(r['mode'] for r in rows):
        subset = [r for r in rows if r['mode'] == mode]
        count = len(subset)
        seconds = sum(r['decode_seconds'] for r in subset)
        steps = sum(r['forward_steps'] for r in subset)
        gross = sum(r['unmasked_tokens'] for r in subset)
        final = sum(r['final_unmasked_tokens'] for r in subset)
        answers = sum(r['answer_tokens'] for r in subset)

        summary[mode] = {
            'n': count,
            'correct': sum(r['correct'] for r in subset),
            'accuracy_percent':
                100 * sum(r['correct'] for r in subset) / count,
            'strict_marker_accuracy_percent':
                100 * sum(
                    r['correct'] and r['has_answer_marker']
                    for r in subset
                ) / count,
            'decode_seconds': seconds,
            'mean_seconds_per_problem': seconds / count,
            'mean_forward_steps': steps / count,
            'mean_answer_tokens': answers / count,
            'unmasked_tokens_per_second': gross / seconds,
            'unmasked_tokens_per_step': gross / steps,
            'final_tokens_per_second': final / seconds,
            'final_tokens_per_step': final / steps,
            'answer_tokens_per_second': answers / seconds,
            'answer_tokens_per_step': answers / steps,
            'missing_answer_marker_count':
                sum(not r['has_answer_marker'] for r in subset),
            'no_end_token_count':
                sum(not r['has_end_token'] for r in subset),
            'possible_truncation_count':
                sum(r['possible_truncation'] for r in subset),
            'residual_mask_tokens':
                sum(r['residual_mask_tokens'] for r in subset),
            'remasked_tokens_total': sum(
                sum(
                    sum(c)
                    for c in r['decoder_stats'].get('remasked_per_step', [])
                )
                for r in subset
            ),
            'capped_incomplete_count': sum(
                r['decoder_stats'].get('capped_incomplete', False)
                for r in subset
            ),
            'peak_allocated_gib':
                max(r['peak_allocated_gib'] for r in subset),
        }

    if 'ours' in summary and 'low_confidence' in summary:
        by_index = {}
        for row in rows:
            by_index.setdefault(row['test_index'], {})[row['mode']] = row
        pairs = [
            p for p in by_index.values()
            if 'ours' in p and 'low_confidence' in p
        ]
        if pairs:
            summary['paired'] = {
                'n': len(pairs),
                'decode_speedup':
                    sum(p['low_confidence']['decode_seconds'] for p in pairs)
                    / sum(p['ours']['decode_seconds'] for p in pairs),
                'vanilla_only_correct': sum(
                    p['low_confidence']['correct'] and not p['ours']['correct']
                    for p in pairs
                ),
                'ours_only_correct': sum(
                    p['ours']['correct'] and not p['low_confidence']['correct']
                    for p in pairs
                ),
            }

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument(
        '--model-family', choices=MODEL_FAMILIES, default='llada',
        help='dream: Dream-v0-Instruct-7B for the baselines (vanilla/hierarchy via '
             'dream_adapter, wino on the raw model, fastdllm via fastdllm_dream_decode); '
             'ours is LLaDA-only',
    )
    parser.add_argument('--dataset', default='openai/gsm8k')
    parser.add_argument('--dataset-config', default='main')
    parser.add_argument('--dataset-split', default='test')
    parser.add_argument('--dataset-revision', default='main')
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--model-revision', default=None)
    parser.add_argument(
        '--local-files-only', action='store_true',
        help='Use only cached model/tokenizer files',
    )
    parser.add_argument('--native-cache-only', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--num-samples', type=int, default=0,
        help='Random subset size; 0 means the full split',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen-length', type=int, default=256)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument(
        '--steps', type=int, default=256,
        help='Baseline schedule; ignored by ours',
    )
    parser.add_argument(
        '--methods', nargs='+',
        choices=DECODERS,
        default=['ours'],
    )
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--confidence-threshold', type=float, default=0.9)
    parser.add_argument('--remask-k', type=int, default=0)
    parser.add_argument('--remask-threshold', type=float, default=0.5)
    parser.add_argument(
        '--force-confidence', type=float, default=None,
        help='Always unmask tokens whose confidence >= this value (None disables)',
    )
    parser.add_argument('--variant-config')
    args = parser.parse_args()

    if (
        args.gen_length <= 0
        or args.block_length <= 0
        or args.gen_length % args.block_length
    ):
        parser.error(
            '--gen-length must be positive and divisible by --block-length.'
        )

    variants = get_variants(args)
    loader = resolve_loader(variants, args.model_family)
    blocks = args.gen_length // args.block_length
    if any(
        v['decoder'] in ('low_confidence', 'random')
        for v in variants.values()
    ):
        if args.steps < blocks or args.steps % blocks:
            parser.error(
                '--steps must be positive and divisible by the number of blocks.'
            )

    dataset = load_data(args)
    if not 0 <= args.num_samples <= len(dataset):
        parser.error(
            f'--num-samples must be between 0 and {len(dataset)}.'
        )
    sample_count = args.num_samples or len(dataset)
    indices = sorted(
        random.Random(args.seed).sample(range(len(dataset)), sample_count)
    )

    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA.')

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name('decode_confidence_v2.py').resolve(),
    ]
    source_paths.extend(source_files(variants, args.model_family))

    config = vars(args).copy()
    # JSON에서 덮어쓴 실제 설정과 CLI 기본값을 구분합니다.
    config['cli_decoder_defaults'] = {
        key: config.pop(key)
        for key in (
            'n', 'confidence_threshold', 'remask_k', 'remask_threshold'
        )
    }
    config.update({
        'model_loader': loader,
        'variants': variants,
        'num_tasks': sample_count,
        'test_indices': indices,
        'dataset_fingerprint': getattr(dataset, '_fingerprint', None),
        'batch_size': 1,
        'num_fewshot': 0,
        'temperature': 0,
        'cfg_scale': 0,
        'dtype': 'bfloat16',
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'prompt_suffix': PROMPT_SUFFIX,
        'source_sha256': {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        'metrics': {
            'accuracy':
                'Last #### number, falling back to last number; '
                'normalized numeric equality.',
            'strict_accuracy':
                'Requires a #### number and numeric equality.',
            'timing':
                'CUDA-synchronized generation time; '
                'excludes loading, warmup and parsing.',
            'unmasked_tokens':
                'Gross unmask events, including refills after remasking.',
            'final_tokens':
                'Final non-MASK output positions, '
                'including special and post-EOS tokens.',
        },
    })

    config_path = out_dir / 'config.json'
    if (
        config_path.exists()
        and json.loads(config_path.read_text()) != config
    ):
        raise ValueError(
            'Output configuration differs; use a new output directory.'
        )
    write_json(config_path, config)

    source_dir = out_dir / 'source'
    source_dir.mkdir(exist_ok=True)
    for path in source_paths:
        (source_dir / path.name).write_bytes(path.read_bytes())

    write_json(
        out_dir / 'samples.json',
        [{'test_index': i, **dataset[i]} for i in indices],
    )

    records_path = out_dir / 'results.jsonl'
    rows = (
        [
            json.loads(line)
            for line in records_path.read_text().splitlines()
            if line.strip()
        ]
        if records_path.exists() else []
    )
    done = {(r['test_index'], r['mode']) for r in rows}
    expected = {(i, mode) for i in indices for mode in variants}

    if done == expected:
        write_json(out_dir / 'summary.json', summarize(rows))
        print(json.dumps(summarize(rows), indent=2), flush=True)
        return

    pretrained = dict(
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if args.model_revision is not None:
        pretrained['revision'] = args.model_revision

    print(
        f'Loading {args.model}; variants={json.dumps(variants)}',
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, **pretrained)
    model = load_model(args.model, loader, pretrained)
    print(f'Model family: {args.model_family}; loader: {loader}', flush=True)

    if (
        loader == 'hf'
        and any(v['decoder'] == 'ours' for v in variants.values())
        and not args.native_cache_only
    ):
        print(
            'Cache preparation: ' + prepare_prefix_cache(model),
            flush=True,
        )

    ids = TOKEN_IDS[args.model_family]
    mask_id, end_ids = ids['mask_id'], set(ids['end_ids'])
    if tokenizer.eos_token_id is not None:
        end_ids.add(tokenizer.eos_token_id)
    special_ids = set(tokenizer.all_special_ids) | end_ids | {mask_id}

    def encode(question):
        messages = [
            {'role': 'user', 'content': question + PROMPT_SUFFIX}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(
            text,
            add_special_tokens=False,
            return_tensors='pt',
        ).to('cuda')

    modes = tuple(variants)
    warmup = encode(
        'A box contains 3 red balls and 4 blue balls. '
        'How many balls are in the box?'
    )
    for mode in modes:
        torch.manual_seed(args.seed)
        run_decoder(model, warmup, variants[mode], args)
        torch.cuda.synchronize()
        print(f'Warmup complete: {mode}', flush=True)
    del warmup

    with records_path.open(
        'a', encoding='utf-8', buffering=1
    ) as output:
        for ordinal, index in enumerate(indices):
            sample = dataset[index]
            gold, _ = extract_answer(sample['answer'])
            if gold is None:
                raise ValueError(
                    f'No numeric reference answer at index {index}.'
                )

            encoded = encode(sample['question'])
            offset = ordinal % len(modes)

            for mode in modes[offset:] + modes[:offset]:
                if (index, mode) in done:
                    continue

                sample_seed = args.seed + index
                torch.manual_seed(sample_seed)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()

                tokens, stats = run_decoder(
                    model, encoded, variants[mode], args
                )

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                generated = tokens[
                    0, encoded['input_ids'].shape[1]:
                ].tolist()
                end = next(
                    (
                        i for i, token in enumerate(generated)
                        if token in end_ids
                    ),
                    len(generated),
                )
                text = tokenizer.decode(
                    generated[:end], skip_special_tokens=True
                )
                prediction, marked = extract_answer(text)
                counts = [c[0] for c in stats['unmasked_per_step']]

                row = {
                    'test_index': index,
                    'mode': mode,
                    'seed': sample_seed,
                    'question': sample['question'],
                    'reference': sample['answer'],
                    'gold': gold,
                    'prediction': prediction,
                    'correct':
                        prediction is not None and prediction == gold,
                    'has_answer_marker': marked,
                    'has_end_token': end < len(generated),
                    'possible_truncation':
                        end == len(generated) and not marked,
                    'text': text,
                    'raw_text': tokenizer.decode(
                        generated, skip_special_tokens=False
                    ),
                    'generated_ids': generated,
                    'prompt_tokens': encoded['input_ids'].shape[1],
                    'answer_tokens': sum(
                        t not in special_ids for t in generated[:end]
                    ),
                    'unmasked_tokens': sum(counts),
                    'unmasked_per_step': counts,
                    'final_unmasked_tokens': sum(
                        t != mask_id for t in generated
                    ),
                    'decoder_stats': stats,
                    'forward_steps': stats['forward_steps'],
                    'decode_seconds': elapsed,
                    'residual_mask_tokens': generated.count(mask_id),
                    'peak_allocated_gib':
                        torch.cuda.max_memory_allocated() / 2**30,
                }

                output.write(
                    json.dumps(row, ensure_ascii=False) + '\n'
                )
                rows.append(row)
                done.add((index, mode))
                write_json(out_dir / 'summary.json', summarize(rows))

                print(
                    f'[{len(done)}/{len(expected)}] index={index} {mode}: '
                    f'correct={row["correct"]} '
                    f'steps={row["forward_steps"]} '
                    f'time={elapsed:.2f}s',
                    flush=True,
                )
                del tokens

    print(
        'COMPLETE\n' + json.dumps(summarize(rows), indent=2),
        flush=True,
    )


if __name__ == '__main__':
    main()
"""Summarise results/dapd_* runs: accuracy (pass@1 / exact match / IFEval prompt-level strict, loose in parentheses) and final tokens per second, two decimals; * = partial run."""
import argparse
import json
import re
from pathlib import Path

TASKS = ('humaneval', 'mbpp', 'math500', 'gsm8k', 'ifeval')
METRIC = {'humaneval': 'pass_at_1_percent', 'mbpp': 'pass_at_1_percent', 'math500': 'accuracy_percent', 'ifeval': 'prompt_strict_accuracy_percent', 'gsm8k': 'accuracy_percent'}
SIZES = {'humaneval': 164, 'mbpp': 500, 'math500': 500, 'ifeval': 541, 'gsm8k': 1319}
PATTERN = re.compile(r'^(?P<prefix>.+?)_(?P<model>[a-z0-9]+)_(?P<task>[a-z0-9]+)_g(?P<gen>\d+)_b(?P<block>\d+)_(?P<alg>dapd_direct|dapd_staged)$')


def load(results_dir, prefix, gen, block):
    rows = []
    for path in sorted(Path(results_dir).glob(f'{prefix}_*')):
        match = PATTERN.match(path.name)
        if not match or int(match['gen']) != gen or int(match['block']) != block or not (path / 'summary.json').exists():
            continue
        summary = json.loads((path / 'summary.json').read_text())
        variants = json.loads((path / 'config.json').read_text()).get('variants', {})
        for label, entry in summary.items():
            if not isinstance(entry, dict) or 'n' not in entry:
                continue
            options = variants.get(label, {})
            rows.append({**match.groupdict(), 'label': label, 'n': entry['n'], 'accuracy': entry.get(METRIC[match['task']]), 'tokens_per_second': entry.get('final_tokens_per_second'), 'steps': entry.get('mean_forward_steps'), 'loose': entry.get('prompt_loose_accuracy_percent'), 'tau': (options.get('tau_min'), options.get('tau_max'))})
    return rows


def fmt(value):
    return '-' if value is None else f'{value:.2f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default=str(Path(__file__).resolve().parent / 'results'))
    parser.add_argument('--prefix', default='dapd')
    parser.add_argument('--gen', type=int, default=256)
    parser.add_argument('--block', type=int, default=32)
    parser.add_argument('--steps', action='store_true', help='also print mean forward steps (NFE) per cell')
    args = parser.parse_args()
    rows = load(args.results, args.prefix, args.gen, args.block)
    if not rows:
        print('no finished runs found'); return
    for model in dict.fromkeys(r['model'] for r in rows):
        subset = [r for r in rows if r['model'] == model]
        print(f'\n## {model} (gen {args.gen}, block {args.block}); cells are accuracy % / tokens per second{" (NFE)" if args.steps else ""}, * = partial run\n')
        print('| method | ' + ' | '.join(TASKS) + ' |')
        print('|---|' + '---|' * len(TASKS))
        for alg in dict.fromkeys(r['alg'] for r in subset):
            cells = []
            for task in TASKS:
                hits = [r for r in subset if r['alg'] == alg and r['task'] == task]
                if not hits:
                    cells.append('-'); continue
                r = hits[0]
                partial = '*' if r['n'] < SIZES[task] else ''
                acc = f"{fmt(r['accuracy'])} (loose {fmt(r['loose'])})" if task == 'ifeval' and r['loose'] is not None else fmt(r['accuracy'])
                nfe = f" ({fmt(r['steps'])})" if args.steps else ''
                cells.append(f"{acc} / {fmt(r['tokens_per_second'])}{nfe}{partial}")
            print(f'| {alg} | ' + ' | '.join(cells) + ' |')
        taus = sorted({(r['alg'], r['task'], r['tau']) for r in subset})
        print('\ntau per run: ' + ', '.join(f"{alg} {task} {tau[0]}->{tau[1]}" for alg, task, tau in taus))


if __name__ == '__main__':
    main()

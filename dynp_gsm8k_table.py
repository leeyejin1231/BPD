"""LLaDA GSM8K full (1,319, g128/b32): dynamic-p schedules vs fixed p=32, vanilla and hier."""
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def load(d):
    p = Path('results') / d / 'results.jsonl'
    return {json.loads(l)['test_index']: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else {}


def mcnemar(a, b):
    c = set(a) & set(b)
    x = sum(a[i]['correct'] and not b[i]['correct'] for i in c)
    y = sum(b[i]['correct'] and not a[i]['correct'] for i in c)
    n = x + y
    return x, y, (min(1, sum(comb(n, k) for k in range(min(x, y) + 1)) * 2 / 2 ** n) if n else 1)


runs = {'Vanilla': 'gsm8k_full_g128_b32_vanilla_s128', 'Hierarchy-dLLM': 'gsm8k_full_g128_b32_hier_r03',
        'Ours p=32': 'gsm8k_full_g128_b32_t075_p32_s2',
        'Ours dynamic p (32/16/4/2)': 'gsm8k_full_g128_b32_t075_dyn32_16_4_2_s2'}
data = {k: load(v) for k, v in runs.items()}
print('| method | accuracy | tok/step | tok/s | steps | cap activation rate | vs vanilla | vs p=32 |')
print('|---|---|---|---|---|---|---|---|')
buckets = {}
for name, R in data.items():
    L = list(R.values())
    steps = sum(r['forward_steps'] for r in L)
    unm = sum(r['unmasked_tokens'] for r in L)
    rate = '—'
    if 'intervals_per_step' in L[0]['decoder_stats']:
        per = defaultdict(lambda: [0, 0])
        for r in L:
            for step in r['decoder_stats']['intervals_per_step']:
                for log in step[0]:
                    if log['eligible_count'] > 0:
                        per[log['pivot_budget']][1] += 1
                        per[log['pivot_budget']][0] += log['eligible_count'] > log['pivot_budget']
        c, a = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
        rate = f'{100 * c / a:.1f}%'
        buckets[name] = per
    pv = '—' if name == 'Vanilla' else '{2:.3f} ({0}:{1})'.format(*mcnemar(data['Vanilla'], R))
    pp = '—' if name in ('Vanilla', 'Ours p=32') else '{2:.3f} ({0}:{1})'.format(*mcnemar(data['Ours p=32'], R))
    print(f"| {name} | {100 * sum(r['correct'] for r in L) / len(L):.2f} (n={len(L)}) | {unm / steps:.2f} | "
          f"{unm / sum(r['decode_seconds'] for r in L):.1f} | {steps / len(L):.1f} | {rate} | {pv} | {pp} |")
for name, per in buckets.items():
    print(f'\n{name} — by p(A): ' + ', '.join(
        f'p={b}: {100 * per[b][0] / per[b][1]:.1f}%' for b in sorted(per, reverse=True)))

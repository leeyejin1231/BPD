"""LLaDA HumanEval g256 dynamic-p runs: metrics plus cap activation rate from the interval log.

cap activation rate = #{A : |E_A| > p(A)} / #{A : |E_A| > 0}, over every interval that
reached the pivot rule (size > s) in every forward step; also broken down by p(A).
"""
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def load(d):
    p = Path('results') / d / 'results.jsonl'
    return {json.loads(l)['test_index']: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else {}


def mcnemar(a, b):
    c = set(a) & set(b)
    x = sum(a[i]['passed'] and not b[i]['passed'] for i in c)
    y = sum(b[i]['passed'] and not a[i]['passed'] for i in c)
    n = x + y
    return x, y, (min(1, sum(comb(n, k) for k in range(min(x, y) + 1)) * 2 / 2 ** n) if n else 1)


runs = {'Vanilla': 'humaneval_g256_b32_vanilla_s256', 'Hierarchy-dLLM': 'humaneval_g256_b32_hier_r03',
        'Ours p=4': 'humaneval_g256_b32_t075_p4_s2', 'Ours p=32': 'humaneval_g256_b32_t075_p32_s2',
        'Ours dynamic p (32/16/4/2)': 'humaneval_g256_b32_t075_dyn32_16_4_2_s2'}
data = {k: load(v) for k, v in runs.items()}
van = data['Vanilla']
print('| method | pass@1 | tok/step | tok/s | steps | syntax err | cap activation rate | McNemar vs vanilla |')
print('|---|---|---|---|---|---|---|---|')
buckets = {}
for name, R in data.items():
    if not R:
        print(f'| {name} | (missing) |')
        continue
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
        capped, active = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
        rate = f'{100 * capped / active:.1f}% ({capped}/{active})'
        buckets[name] = per
    pv = '—' if name == 'Vanilla' else '{2:.3f} ({0}:{1})'.format(*mcnemar(van, R))
    print(f"| {name} | {100 * sum(r['passed'] for r in L) / len(L):.2f} (n={len(L)}) | {unm / steps:.2f} | "
          f"{unm / sum(r['decode_seconds'] for r in L):.1f} | {steps / len(L):.1f} | "
          f"{sum(r['status'] == 'syntax_error' for r in L)} | {rate} | {pv} |")
for name, per in buckets.items():
    print(f'\n{name} — by pivot budget p(A):')
    for budget in sorted(per, reverse=True):
        c, a = per[budget]
        print(f'  p={budget:2d}: {100 * c / a:5.1f}% capped ({c}/{a} intervals with |E|>0)')

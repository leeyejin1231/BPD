"""HumanEval (LLaDA, g256/b32) p-sweep at fixed tau/s, with diagnostics on how often p binds.

`cand>p %` is the share of forward steps whose above-threshold candidates (summed over all
intervals) exceed p: an upper bound on how often the random choice actually dropped a
candidate, since p applies per interval. With block length 32, p >= 32 can never bind.
"""
import json
import sys
from math import comb
from pathlib import Path

TAU, S = sys.argv[1] if len(sys.argv) > 1 else '075', sys.argv[2] if len(sys.argv) > 2 else '2'


def load(d):
    p = Path('results') / d / 'results.jsonl'
    return {json.loads(l)['test_index']: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else {}


def mcnemar(a, b):
    c = set(a) & set(b)
    x = sum(a[i]['passed'] and not b[i]['passed'] for i in c)
    y = sum(b[i]['passed'] and not a[i]['passed'] for i in c)
    n = x + y
    return x, y, (min(1, sum(comb(n, k) for k in range(min(x, y) + 1)) * 2 / 2 ** n) if n else 1)


van = load('humaneval_g256_b32_vanilla_s256')
print(f"| method | pass@1 | tok/step | tok/s | steps | syntax err | cand>p % | fallback % | p vs vanilla (v:m) |")
print('|---|---|---|---|---|---|---|---|---|')
for label, d, p in ([('Vanilla', 'humaneval_g256_b32_vanilla_s256', None),
                     ('Hierarchy-dLLM', 'humaneval_g256_b32_hier_r03', None)]
                    + [(f'Ours p={p}', f'humaneval_g256_b32_t{TAU}_p{p}_s{S}', p) for p in (2, 4, 8, 16, 32)]):
    R = load(d)
    if not R:
        print(f'| {label} | (missing) |')
        continue
    L = list(R.values())
    n = len(L)
    steps = sum(r['forward_steps'] for r in L)
    perf = 100 * sum(r['passed'] for r in L) / n
    tps = sum(r['unmasked_tokens'] for r in L) / steps
    toks = sum(r['unmasked_tokens'] for r in L) / sum(r['decode_seconds'] for r in L)
    syn = sum(r['status'] == 'syntax_error' for r in L)
    bind = fb = '—'
    if p is not None:
        cands = [c[0] for r in L for c in r['decoder_stats']['candidates_per_step']]
        fbs = [f[0] for r in L for f in r['decoder_stats']['fallback_per_step']]
        bind = f'{100 * sum(c > p for c in cands) / len(cands):.1f}'
        fb = f'{100 * sum(fbs) / len(fbs):.1f}'
    pv = '—' if label == 'Vanilla' else '{2:.3f} ({0}:{1})'.format(*mcnemar(van, R))
    part = '' if n >= 164 else f' (n={n})'
    print(f'| {label} | {perf:.2f}{part} | {tps:.2f} | {toks:.1f} | {steps / n:.1f} | {syn} | {bind} | {fb} | {pv} |')

"""LLaDA: size-dependent threshold tau(A) = 0.75 + 0.10 * |A| / 32 vs fixed thresholds (p=32, s=2).

Reports performance, tokens/step and throughput per benchmark, McNemar vs vanilla and
vs fixed tau=0.75, and for the dynamic run the share of pivot-rule intervals with an
eligible token, by interval size.
"""
from collections import defaultdict

from final_table import PERF, SUCCESS, TOTAL, load, mcnemar, metrics

DYN = 'dyntau075_085_p32_s2'
# task -> (directory pattern, {label: cfg})
RUNS = {
    'gsm8k': ('results/gsm8k_full_g128_b32_{cfg}', 'vanilla_s128', 't075_p32_s2'),
    'humaneval': ('results/humaneval_g256_b32_{cfg}', 'vanilla_s256', 't075_p32_s2'),
    'math500': ('results/math500_g256_b32_{cfg}', 'vanilla', 'ours'),
    'mbpp': ('results/mbpp_g256_b32_{cfg}', 'vanilla', 'ours'),
    'ifeval': ('results/ifeval_g512_b32_{cfg}', 'vanilla', 'ours'),
}

for task, (pattern, vanilla, fixed075) in RUNS.items():
    rows = {'Vanilla': load(pattern, vanilla), 'Fixed tau=0.75': load(pattern, fixed075),
            'Fixed tau=0.80': load(pattern, 't08_p32_s2'), 'Fixed tau=0.85': load(pattern, 't085_p32_s2'),
            'Dynamic tau 0.75-0.85': load(pattern, DYN)}
    print(f'\n### {task} ({PERF[task]}, n={TOTAL[task]})')
    print('| method | perf | tok/step | tok/s | fallback steps | n | vs vanilla | vs fixed 0.75 |')
    print('|---|---|---|---|---|---|---|---|')
    for name, r in rows.items():
        if not r:
            print(f'| {name} | (no results) | | | | | | |')
            continue
        m = metrics(task, r)
        fb = '—'
        if 'fallback_per_step' in next(iter(r.values())).get('decoder_stats', {}):
            fired = sum(s[0] for x in r.values() for s in x['decoder_stats']['fallback_per_step'])
            fb = f"{100 * fired / sum(x['forward_steps'] for x in r.values()):.1f}%"
        pv = '—' if name == 'Vanilla' else '{2:.3f} ({0}:{1})'.format(*mcnemar(task, rows['Vanilla'], r))
        pf = ('—' if name in ('Vanilla', 'Fixed tau=0.75')
              else '{2:.3f} ({0}:{1})'.format(*mcnemar(task, rows['Fixed tau=0.75'], r)))
        partial = '' if m['n'] >= TOTAL[task] else ' (partial)'
        print(f"| {name} | {m['perf']:.2f} | {m['tps']:.2f} | {m['tok_s']:.1f} | {fb} | {m['n']}{partial} | {pv} | {pf} |")
    dyn = rows['Dynamic tau 0.75-0.85']
    if dyn:
        per = defaultdict(lambda: [0, 0, 0])  # size bucket -> [intervals, with eligible, eligible tokens]
        for r in dyn.values():
            for step in r['decoder_stats']['intervals_per_step']:
                for log in step[0]:
                    size = log['interval_size']
                    bucket = next(b for b in (32, 24, 16, 8, 4, 3) if size >= b)
                    per[bucket][0] += 1
                    per[bucket][1] += log['eligible_count'] > 0
                    per[bucket][2] += log['eligible_count']
        print('dynamic tau, intervals with >=1 eligible token by size: ' + ', '.join(
            f'|A|>={b}: {100 * per[b][1] / per[b][0]:.0f}% ({per[b][2] / per[b][0]:.1f} tok)'
            for b in sorted(per, reverse=True)))

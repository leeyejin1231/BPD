"""LLaDA HumanEval (gen 256): every method at block length 32 / 64 / 128 / 256 (one block).

Prints one table per metric (performance, tokens/step, throughput) with a column per
block length, then the per-run detail with McNemar vs vanilla and vs Ours p32/s2 at
the same block length. Partial runs are flagged with *.
"""
from final_table import TOTAL, load, mcnemar, metrics

TASK = 'humaneval'
BLOCKS = (32, 64, 128, 256)
PATTERN = 'results/humaneval_g256_b{block}_{{cfg}}'
METHODS = [
    ('Vanilla', 'vanilla_s256'),
    ('Hierarchy-dLLM', 'hier_r03'),
    ('WINO', 'wino'),
    ('Fast-dLLM parallel t0.9', 'fastdllm_parallel'),
    ('Fast-dLLM parallel t0.75', 'fastdllm_parallel_t075'),
    ('Fast-dLLM parallel + dual cache t0.9', 'fastdllm_dual'),
    ('Fast-dLLM parallel + prefix cache t0.75', 'fastdllm_prefix_t075'),
    ('Ours t0.75 p32 s2', 't075_p32_s2'),
    ('Ours t0.75 p32 s0', 't075_p32_s0'),
    ('Ours t0.75 p dyn(32/16/4/2) s2', 't075_dyn32_16_4_2_s2'),
    ('Ours t0.75 p dyn(32/16/4/2) s0', 't075_dyn32_16_4_2_s0'),
    ('Ours t0.75 p32 s2 + prefix cache', 't075_p32_s2_prefix'),
    ('Ours t0.75 p32 s0 + prefix cache', 't075_p32_s0_prefix'),
    ('Ours t0.75 p dyn(32/16/4/2) s2 + prefix cache', 't075_dyn32_16_4_2_s2_prefix'),
    ('Ours t0.75 p dyn(32/16/4/2) s0 + prefix cache', 't075_dyn32_16_4_2_s0_prefix'),
]
# Runs stored under another config name (same decoder settings).
ALIASES = {('t075_p32_s0_prefix', 32): 'v3_t075_p32_s0_prefix'}

runs = {(cfg, b): load(PATTERN.format(block=b), ALIASES.get((cfg, b), cfg)) for _, cfg in METHODS for b in BLOCKS}
stats = {k: metrics(TASK, r) for k, r in runs.items() if r}

for title, key, fmt in (('performance (pass@1)', 'perf', '{:.2f}'), ('tokens/step', 'tps', '{:.2f}'),
                        ('tokens/second', 'tok_s', '{:.1f}')):
    print(f'\n### {title}')
    print('| method | ' + ' | '.join(f'block {b}' for b in BLOCKS) + ' |')
    print('|---|' + '---|' * len(BLOCKS))
    for name, cfg in METHODS:
        cells = []
        for b in BLOCKS:
            m = stats.get((cfg, b))
            cells.append('—' if m is None else fmt.format(m[key]) + ('' if m['n'] >= TOTAL[TASK] else '*'))
        print(f'| {name} | ' + ' | '.join(cells) + ' |')

print('\n### detail')
print('| method | block | perf | tok/step | tok/s | fwd/problem | ms/fwd | n | vs vanilla | vs Ours p32 s2 |')
print('|---|---|---|---|---|---|---|---|---|---|')
for name, cfg in METHODS:
    for b in BLOCKS:
        r = runs[(cfg, b)]
        if not r:
            continue
        m = stats[(cfg, b)]
        forwards = sum(x['forward_steps'] for x in r.values())
        seconds = sum(x['decode_seconds'] for x in r.values())
        test = lambda a: '{2:.3f} ({0}:{1})'.format(*mcnemar(TASK, a, r)) if a and a is not r else '—'
        print(f"| {name} | {b} | {m['perf']:.2f} | {m['tps']:.2f} | {m['tok_s']:.1f} | {forwards / len(r):.1f} | "
              f"{1000 * seconds / forwards:.0f} | {m['n']} | {test(runs[('vanilla_s256', b)])} | "
              f"{test(runs[('t075_p32_s2', b)])} |")

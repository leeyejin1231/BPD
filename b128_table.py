"""LLaDA block-128 benchmark table (gen 256): every method with step counts.

Usage: python b128_table.py [gsm8k|mbpp|humaneval] [llada|dream]

Speed is reported two ways: gross (every mask-to-token transition, including tokens that
were remasked and committed again) and net (final non-MASK positions only). The net
numbers are the ones to use for Hierarchy-dLLM and WINO, whose remasking inflates the
gross count. McNemar exact test vs vanilla and vs Ours t0.75/s0. Partial runs are flagged *.
"""
import sys

from final_table import PERF, TOTAL, load, mcnemar

TASK = sys.argv[1] if len(sys.argv) > 1 else 'gsm8k'
FAMILY = sys.argv[2] if len(sys.argv) > 2 else 'llada'
SUCCESS = {'gsm8k': 'correct', 'mbpp': 'passed', 'humaneval': 'passed'}[TASK]
WINO_TAU = {'gsm8k': 0.6, 'mbpp': 0.8, 'humaneval': 0.7}[TASK]
if FAMILY == 'llada':
    PATTERN = {'gsm8k': 'results/gsm8k_full_g256_b128_{cfg}', 'mbpp': 'results/mbpp_g256_b128_{cfg}',
               'humaneval': 'results/humaneval_g256_b128_{cfg}'}[TASK]
    METHODS = [
        ('Vanilla (256 steps)', 'vanilla_s256'),
        ('Hierarchy-dLLM (0.82/0.3/r0.3)', 'hier_r03'),
        (f'WINO ({WINO_TAU}/back 0.9)', 'wino'),
        ('Fast-dLLM t0.9 + prefix cache', 'fastdllm_prefix'),
        ('Ours t0.85 s2 p32 + prefix cache', 't085_p32_s2_prefix'),
        ('Ours t0.75 s0 p32 + prefix cache', 't075_p32_s0_prefix'),
    ]
    BASE = ('vanilla_s256', 't075_p32_s0_prefix')
else:
    PATTERN = f'results/dream_{TASK}_g256_b128_{{cfg}}'
    METHODS = [
        ('Vanilla (256 steps)', 'vanilla'),
        ('Hierarchy-dLLM (0.82/0.3/r0.3)', 'hier'),
        (f'WINO ({WINO_TAU}/back 0.9)', 'wino'),
        ('Fast-dLLM t0.9 + prefix cache', 'fastdllm'),
    ]
    BASE = ('vanilla', 'fastdllm')

METHODS2 = [(cfg, name) for name, cfg in METHODS]
runs = {cfg: load(PATTERN, cfg) for _, cfg in METHODS}
vanilla, ours = runs[BASE[0]], runs[BASE[1]]

print(f'| method | n | {PERF[TASK]} | gross tok/step | gross tok/s | net tok/step | net tok/s | total steps | steps/q | ms/fwd | vs vanilla | vs {dict(METHODS2)[BASE[1]]} |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for name, cfg in METHODS:
    R = list(runs[cfg].values())
    if not R:
        print(f'| {name} | — |')
        continue
    steps = sum(r['forward_steps'] for r in R)
    seconds = sum(r['decode_seconds'] for r in R)
    gross = sum(r['unmasked_tokens'] for r in R)
    net = sum(r['final_unmasked_tokens'] for r in R)
    acc = 100 * sum(r[SUCCESS] for r in R) / len(R)
    flag = '' if len(R) >= TOTAL[TASK] else '*'
    test = lambda base: ('{2:.3f} ({0}:{1})'.format(*mcnemar(TASK, base, runs[cfg]))
                         if base and base is not runs[cfg] else '—')
    print(f'| {name} | {len(R)}{flag} | {acc:.2f} | {gross / steps:.2f} | {gross / seconds:.1f} | '
          f'{net / steps:.2f} | {net / seconds:.1f} | {steps} | {steps / len(R):.1f} | '
          f'{1000 * seconds / steps:.0f} | {test(vanilla)} | {test(ours)} |')

"""Dream-v0-Instruct-7B block-128 baseline table (MATH-500, IFEval): perf, net tok/step, net tok/s, steps/problem, ms/forward.

Usage: python dream_b128_table.py [--tasks math500 ifeval] [--methods wino hier fastdllm]
tok/step and tok/s use final_unmasked_tokens (remasked tokens counted once); dup/prob = re-unmasked tokens per problem.
"""
import argparse, glob, json
from pathlib import Path

TOTAL = {'math500': 500, 'ifeval': 541, 'mbpp': 500, 'humaneval': 164, 'gsm8k': 1319}
NAMES = {'wino': 'WINO (t0.7/0.6, back 0.9)', 'hier': 'Hierarchy-dLLM (r0.3)', 'fastdllm': 'Fast-dLLM prefix t0.9', 'vanilla': 'Vanilla', 'ours': 'Ours t0.75 p32 s2', 'ours_prefix': 'Ours t0.75 p32 s2 + prefix', 'ours_s0_prefix': 'Ours t0.75 p32 s0 + prefix', 'ours_t06_prefix': 'Ours t0.6 p32 s2 + prefix'}

def load(task, method, block=128):
    rows = {}
    for d in sorted(glob.glob(f'results/dream_{task}_g*_b{block}_{method}') + glob.glob(f'results/dream_{task}_g*_b{block}_{method}_shard*')):
        p = Path(d) / 'results.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line); rows[r['test_index']] = r
    return rows

def perf(task, R):
    if task == 'math500': return 100 * sum(r['correct'] for r in R) / len(R), ''
    if task in ('mbpp', 'humaneval'): return 100 * sum(r['passed'] for r in R) / len(R), ''
    if task == 'gsm8k': return 100 * sum(r['correct'] for r in R) / len(R), ''
    n_inst = sum(len(r['ifeval']['strict_follow']) for r in R)
    return (100 * sum(r['ifeval']['prompt_strict'] for r in R) / len(R),
            f" loose={100*sum(r['ifeval']['prompt_loose'] for r in R)/len(R):.2f} inst_strict={100*sum(sum(r['ifeval']['strict_follow']) for r in R)/n_inst:.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['math500', 'ifeval'])
    ap.add_argument('--methods', nargs='+', default=['wino', 'hier', 'fastdllm'])
    ap.add_argument('--block', type=int, default=128)
    a = ap.parse_args()
    for task in a.tasks:
        print(f'\n=== Dream {task} gen256 block{a.block} ===')
        print(f"{'method':28s} {'n':>8s} {'perf':>6s} {'tok/step':>8s} {'tok/s':>6s} {'steps/prob':>10s} {'ms/fwd':>7s} {'dup/prob':>8s} {'no_end':>6s}")
        for m in a.methods:
            rows = load(task, m, a.block)
            if not rows: print(f'{NAMES.get(m, m):28s} (no results)'); continue
            R = list(rows.values())
            steps = sum(r['forward_steps'] for r in R); secs = sum(r['decode_seconds'] for r in R)
            net = sum(r['final_unmasked_tokens'] for r in R); gross = sum(r['unmasked_tokens'] for r in R)
            p, extra = perf(task, R)
            print(f"{NAMES.get(m, m):28s} {len(R):3d}/{TOTAL[task]:<4d} {p:6.2f} {net/steps:8.2f} {net/secs:6.1f} {steps/len(R):10.1f} {1000*secs/steps:7.1f} {(gross-net)/len(R):8.1f} {sum(not r['has_end_token'] for r in R):6d}{extra}")

if __name__ == '__main__':
    main()

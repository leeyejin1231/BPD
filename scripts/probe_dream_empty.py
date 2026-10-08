"""Why does Dream at block 128 emit an empty answer? First-forward confidence profile + vanilla check."""
import sys, json, torch
sys.path.insert(0, '/home/ssgyejin/contents/DLM_random_pivot')
from datasets import Dataset
from transformers import AutoTokenizer
import dream_adapter
from decode import generate as vanilla
M = '/scratch/ssgyejin/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae'
DATA = '/scratch/ssgyejin/datasets/HuggingFaceH4___math-500/default/0.0.0/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow'
SUFFIX = '\n\nSolve this problem step by step. Put your final answer within \\boxed{}.'
MASK, EOS = 151666, 151643
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True, local_files_only=True)
model = dream_adapter.load_model(M)
ds = Dataset.from_file(DATA)
vcfg = next(iter(json.load(open('configs/vanilla_variant.json')).values()))
for idx in [10, 14, 17, 0, 1]:
    text = tok.apply_chat_template([{'role': 'user', 'content': ds[idx]['problem'] + SUFFIX}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False, return_tensors='pt').input_ids.cuda()
    P = ids.shape[1]
    x = torch.cat([ids, torch.full((1, 256), MASK, device='cuda')], 1)
    with torch.no_grad():
        logits = model(x).logits[0, P:].float()
    logits[:, MASK] = -torch.inf
    prob = torch.softmax(logits, -1)
    conf, pred = prob.max(-1)
    conf, pred = conf.cpu(), pred.cpu()
    line = f'idx {idx:3d}:'
    for B in (32, 128):
        c, p = conf[:B], pred[:B]
        top = int(c.argmax())
        elig = (c >= 0.75).nonzero().flatten().tolist()
        line += (f' | block{B}: top pos={top} tok={tok.decode([int(p[top])])!r} conf={float(c[top]):.3f};'
                 f' eligible(>=0.75)={len(elig)} of which eos={sum(int(p[i])==EOS for i in elig)}, first eligible pos={elig[:3]}')
    line += f' | pos0 tok={tok.decode([int(pred[0])])!r} conf={float(conf[0]):.3f}'
    print(line, flush=True)
    for B in (128, 32):
        kw = dict(vcfg); kw.pop('remasking', None)
        torch.manual_seed(42)
        out = vanilla(model, ids, attention_mask=None, mask_id=MASK, gen_length=256, block_length=B, steps=256,
                      temperature=0, cfg_scale=0, remasking=vcfg.get('remasking', 'low_confidence'), **{k: v for k, v in kw.items() if k in ()})
        gen = out[0, P:].tolist()
        n_ans = next((i for i, t in enumerate(gen) if t in (151643, 151645)), len(gen))
        print(f'     vanilla block{B}: answer_tokens={n_ans} text={tok.decode(gen[:12], skip_special_tokens=True)!r}', flush=True)

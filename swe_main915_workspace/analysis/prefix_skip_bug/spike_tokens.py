"""List the individual generated tokens whose engine/trainer logprobs disagree by > SPIKE nats."""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(
    0,
    "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools",
)
import pt_numpy  # noqa: E402

RUN = sys.argv[1]
SPIKE = float(os.environ.get("SPIKE", "3.0"))
MAX_PRINT = int(os.environ.get("MAX_PRINT", "60"))

msgs_by_sid = {}
for jl in sorted(glob.glob(os.path.join(RUN, "dumps/rollouts/*.jsonl"))):
    with open(jl) as f:
        for line in f:
            rec = json.loads(line)
            msgs_by_sid[rec["sample_id"]] = [(m["role"], int(m["n_tokens"])) for m in rec["messages"]]

spikes = []  # dicts
n_gen_total = 0
n_gen_corrupt = 0
first_tok_total = 0
for pt in sorted(glob.glob(os.path.join(RUN, "dumps/token_level/*.pt"))):
    d = pt_numpy.load(pt)
    sids = d["sample_ids"]
    lens = d["input_lengths"].astype(np.int64)
    mask = d["token_mask"].astype(bool)
    ids = d["input_ids"].astype(np.int64)
    gen_lp = d["generation_logprobs"].astype(np.float64)
    prev_lp = d["prev_logprobs"].astype(np.float64)
    err = d["seq_mult_prob_error"].astype(np.float64)
    off = int(d["logprob_offset"])
    tok_start = lp_start = 0
    for i, sid in enumerate(sids):
        L = int(lens[i])
        nlp = L - off
        m = mask[tok_start : tok_start + L]
        tid = ids[tok_start : tok_start + L]
        g = gen_lp[lp_start : lp_start + nlp]
        p = prev_lp[lp_start : lp_start + nlp]
        gm = m[off:]
        n_gen_total += int(gm.sum())
        if err[i] > 2.0:
            n_gen_corrupt += int(gm.sum())
        msgs = msgs_by_sid[sid]
        # message boundaries -> per-token (msg idx, in-msg offset, role)
        bounds = np.cumsum([0] + [n for _, n in msgs])
        roles = [r for r, _ in msgs]
        # first generated token of each assistant message
        for mi, (role, n) in enumerate(msgs):
            if role == "assistant":
                first_tok_total += 1
        diff = g - p
        big = np.nonzero((np.abs(diff) > SPIKE) & gm)[0]
        for t in big:
            pos = int(t) + off  # token position in the sample
            mi = int(np.searchsorted(bounds, pos, side="right") - 1)
            in_msg = pos - int(bounds[mi])
            spikes.append(
                dict(
                    sid=sid,
                    err=float(err[i]),
                    step=int(d["step"]),
                    pos=pos,
                    L=L,
                    mi=mi,
                    role=roles[mi],
                    in_msg=in_msg,
                    msg_len=int(msgs[mi][1]),
                    eng=float(g[t]),
                    trn=float(p[t]),
                    tok=int(tid[pos]),
                    prev_toks=[int(x) for x in tid[max(0, pos - 4) : pos]],
                    next_toks=[int(x) for x in tid[pos + 1 : pos + 4]],
                    prompt_len=int(bounds[mi]) if roles[mi] == "assistant" else None,
                )
            )
        tok_start += L
        lp_start += nlp

print(f"generated tokens total: {n_gen_total}; in err>2 sequences: {n_gen_corrupt}; assistant messages: {first_tok_total}")
print(f"spikes (|eng-trn| > {SPIKE}): {len(spikes)}; in err>2 sequences: {sum(s['err']>2 for s in spikes)}")
print(f"  at first generated token of an assistant message (in_msg==0): {sum(s['in_msg']==0 for s in spikes)}")
print(f"  at last token of its message (in_msg==msg_len-1): {sum(s['in_msg']==s['msg_len']-1 for s in spikes)}")
print(f"  engine > trainer (engine more confident): {sum(s['eng']>s['trn'] for s in spikes)}; trainer > engine: {sum(s['eng']<s['trn'] for s in spikes)}")
print(f"  step 1: {sum(s['step']==1 for s in spikes)}, step 2: {sum(s['step']==2 for s in spikes)}")
pl = np.array([s['prompt_len'] for s in spikes if s['prompt_len'] is not None])
if pl.size:
    print(f"  prompt_len at spike: min {pl.min()} p25 {np.percentile(pl,25):.0f} median {np.median(pl):.0f} p75 {np.percentile(pl,75):.0f} max {pl.max()}")
    print(f"  fraction with prompt_len >= 131072: {(pl>=131072).mean():.2f}; >= 98304: {(pl>=98304).mean():.2f}; >= 65536: {(pl>=65536).mean():.2f}")
print(f"  most common spike token ids: {Counter(s['tok'] for s in spikes).most_common(12)}")
print(f"  most common prev token (immediately before spike): {Counter(s['prev_toks'][-1] if s['prev_toks'] else -1 for s in spikes).most_common(12)}")
print(f"  in_msg offset histogram (0,1,2,3-9,10-99,100+): "
      f"{sum(s['in_msg']==0 for s in spikes)}, {sum(s['in_msg']==1 for s in spikes)}, {sum(s['in_msg']==2 for s in spikes)}, "
      f"{sum(3<=s['in_msg']<10 for s in spikes)}, {sum(10<=s['in_msg']<100 for s in spikes)}, {sum(s['in_msg']>=100 for s in spikes)}")

print()
print("largest spikes:")
hdr = "  |diff|   eng       trn       pos      msg#  in_msg/len   prompt_len  P%256 P%16384 tok     prev4                      next3            sid"
print(hdr)
for s in sorted(spikes, key=lambda s: -abs(s['eng'] - s['trn']))[:MAX_PRINT]:
    P = s['prompt_len'] if s['prompt_len'] is not None else -1
    print(f"  {abs(s['eng']-s['trn']):<7.2f} {s['eng']:<9.3f} {s['trn']:<9.3f} {s['pos']:<8} {s['mi']:<5} {s['in_msg']:>5}/{s['msg_len']:<6} {P:<11} {P%256 if P>=0 else '':<5} {P%16384 if P>=0 else '':<7} {s['tok']:<7} {str(s['prev_toks']):<26} {str(s['next_toks']):<16} {s['sid'][:8]}")

# healthy-sequence spikes for reference
h = [s for s in spikes if s['err'] <= 2.0]
print()
print(f"spikes in healthy (err<=2) sequences: {len(h)}; in_msg==0: {sum(s['in_msg']==0 for s in h)}; max |diff|: {max((abs(s['eng']-s['trn']) for s in h), default=0):.2f}")

"""One-shot validation summary for a smoke run: err>2 by turn bucket, spikes by position, masks, cached tokens."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(
    0,
    "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools",
)
import pt_numpy  # noqa: E402

RUN = sys.argv[1]
SPIKE = 3.0

turns_by_sid, reward_by_sid, cached_turns_by_sid = {}, {}, {}
for jl in sorted(glob.glob(os.path.join(RUN, "dumps/rollouts/*.jsonl"))):
    with open(jl) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["sample_id"]
            turns_by_sid[sid] = int(rec["num_assistant_turns"])
            reward_by_sid[sid] = rec.get("reward")
            usages = ((rec.get("full_result") or {}).get("per_turn_metrics") or {}).get("token_usages") or []
            cached_turns_by_sid[sid] = (len(usages), sum(1 for u in usages if int(u.get("cache_read_tokens") or 0) > 0))

rows = []
pos_t = np.zeros(13, dtype=np.int64)
pos_s = np.zeros(13, dtype=np.int64)
per_step = {}
for pt in sorted(glob.glob(os.path.join(RUN, "dumps/token_level/*.pt"))):
    d = pt_numpy.load(pt)
    step = int(d["step"])
    sids = d["sample_ids"]
    lens = d["input_lengths"].astype(np.int64)
    mask = d["token_mask"].astype(bool)
    diff = np.abs(d["generation_logprobs"].astype(np.float64) - d["prev_logprobs"].astype(np.float64))
    err = d["seq_mult_prob_error"].astype(np.float64)
    smb = d["sample_mask_before"].astype(np.float64)
    sma = d["sample_mask_after"].astype(np.float64)
    ps = per_step.setdefault(step, [0, 0.0, 0.0])
    ps[0] += len(sids); ps[1] += float(smb.sum()); ps[2] += float(sma.sum())
    off = int(d["logprob_offset"])
    tok_start = lp_start = 0
    for i, sid in enumerate(sids):
        L = int(lens[i]); nlp = L - off
        gm = mask[tok_start : tok_start + L][off:]
        gen_idx = np.nonzero(gm)[0]
        pos = gen_idx + off
        sp = diff[lp_start : lp_start + nlp][gen_idx] > SPIKE
        cb = np.minimum(pos // 16384, 12)
        pos_t += np.bincount(cb, minlength=13); pos_s += np.bincount(cb[sp], minlength=13)
        rows.append((sid, step, L, float(err[i]), turns_by_sid.get(sid), int(sp.sum())))
        tok_start += L; lp_start += nlp

print(f"{os.path.basename(RUN)}: {len(rows)} sequences")
for step in sorted(per_step):
    n, b, a = per_step[step]
    print(f"  step {step}: {n} sequences, sample_mask before {b:.0f} / after logprob-error masking {a:.0f}")
errs = np.array([r[3] for r in rows])
print(f"  seq_mult_prob_error: p50 {np.percentile(errs,50):.4f} p90 {np.percentile(errs,90):.4f} p99 {np.percentile(errs,99):.4f} max {errs.max():.4g}; err>2: {(errs>2).sum()} ({100*(errs>2).mean():.1f}%)")
print("  by num_assistant_turns bucket: n, err>2 %, median err, spike tokens")
for a, b in [(1, 24), (25, 49), (50, 74), (75, 99), (100, 149), (150, 200)]:
    sel = [r for r in rows if r[4] is not None and a <= r[4] <= b]
    if sel:
        e = np.array([r[3] for r in sel])
        print(f"    {a:>3}-{b:<3}: n {len(sel):>4}  err>2 {100*(e>2).mean():5.1f}%  median {np.median(e):.4f}  spikes {sum(r[5] for r in sel)}")
print("  spike rate (|engine-trainer|>3 nats) per 1e4 generated tokens by position:")
for k in range(13):
    if pos_t[k]:
        lo, hi = k * 16384, (k + 1) * 16384
        print(f"    [{lo:>6},{hi:>6}): tokens {pos_t[k]:>8} spikes {pos_s[k]:>4} rate {1e4*pos_s[k]/pos_t[k]:6.2f}")
tot_turns = sum(v[0] for v in cached_turns_by_sid.values()); cached = sum(v[1] for v in cached_turns_by_sid.values())
print(f"  turns with engine cache_read_tokens>0: {cached} / {tot_turns}")
rw = [reward_by_sid[r[0]] for r in rows if reward_by_sid.get(r[0]) is not None]
print(f"  mean reward {np.mean(rw):.3f}; turns median {np.median([r[4] for r in rows if r[4] is not None]):.0f}, max L {max(r[2] for r in rows)}")

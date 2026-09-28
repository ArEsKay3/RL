"""Per-turn engine-reported cached tokens (OpenHands token_usages.cache_read_tokens) vs spikes.

In memory-only hybrid prefix caching the engine must report 0 cached (skipped) tokens on every turn;
any non-zero value is a chunk that skipped prompt tokens the Mamba layers never saw.
"""
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

# ---- spikes per (sid, assistant-turn index) from the .pt dumps ----
msgs_by_sid = {}
usages_by_sid = {}
for jl in sorted(glob.glob(os.path.join(RUN, "dumps/rollouts/*.jsonl"))):
    with open(jl) as f:
        for line in f:
            rec = json.loads(line)
            msgs_by_sid[rec["sample_id"]] = [(m["role"], int(m["n_tokens"])) for m in rec["messages"]]
            ptm = (rec.get("full_result") or {}).get("per_turn_metrics") or {}
            usages_by_sid[rec["sample_id"]] = ptm.get("token_usages") or []

spikes_by_turn = {}  # (sid, turn_idx) -> count
err_by_sid = {}
for pt in sorted(glob.glob(os.path.join(RUN, "dumps/token_level/*.pt"))):
    d = pt_numpy.load(pt)
    sids = d["sample_ids"]
    lens = d["input_lengths"].astype(np.int64)
    mask = d["token_mask"].astype(bool)
    diff = np.abs(d["generation_logprobs"].astype(np.float64) - d["prev_logprobs"].astype(np.float64))
    err = d["seq_mult_prob_error"].astype(np.float64)
    off = int(d["logprob_offset"])
    tok_start = lp_start = 0
    for i, sid in enumerate(sids):
        L = int(lens[i]); nlp = L - off
        err_by_sid[sid] = float(err[i])
        gm = mask[tok_start : tok_start + L][off:]
        gen_idx = np.nonzero(gm)[0]
        pos = gen_idx + off
        sp = diff[lp_start : lp_start + nlp][gen_idx] > SPIKE
        msgs = msgs_by_sid[sid]
        bounds = np.cumsum([0] + [n for _, n in msgs])
        mi = np.searchsorted(bounds, pos[sp], side="right") - 1
        # assistant turn index of message mi
        asst_idx = np.cumsum([1 if r == "assistant" else 0 for r, _ in msgs]) - 1
        for m_ in mi:
            t = int(asst_idx[int(m_)])
            spikes_by_turn[(sid, t)] = spikes_by_turn.get((sid, t), 0) + 1
        tok_start += L; lp_start += nlp

# ---- per-turn cached tokens ----
n_turns = n_cached = 0
cached_vals = []
rows = []  # (sid, turn, prompt_tokens, cache_read, spikes)
for sid, usages in usages_by_sid.items():
    if sid not in err_by_sid:
        continue
    for t, u in enumerate(usages):
        cr = int(u.get("cache_read_tokens") or 0)
        ptk = int(u.get("prompt_tokens") or 0)
        n_turns += 1
        if cr > 0:
            n_cached += 1
            cached_vals.append(cr)
        rows.append((sid, t, ptk, cr, spikes_by_turn.get((sid, t), 0)))

print(f"turns: {n_turns}; turns with cache_read_tokens>0: {n_cached}")
if cached_vals:
    cv = np.array(cached_vals)
    print(f"  cache_read_tokens: min {cv.min()} median {np.median(cv):.0f} max {cv.max()}")
sp_turns = [r for r in rows if r[4] > 0]
print(f"turns with spikes: {len(sp_turns)}; of which cache_read>0: {sum(1 for r in sp_turns if r[3] > 0)}")
nosp = [r for r in rows if r[4] == 0]
print(f"turns without spikes: {len(nosp)}; of which cache_read>0: {sum(1 for r in nosp if r[3] > 0)}")
# by prompt length bucket
print("prompt_tokens bucket -> turns, cache_read>0 fraction, spiked fraction")
edges = [0, 32768, 65536, 98304, 131072, 163840, 262144]
for a, b in zip(edges[:-1], edges[1:]):
    sel = [r for r in rows if a <= r[2] < b]
    if not sel:
        continue
    print(f"  [{a:>6},{b:>6}): turns {len(sel):>6}  cache_read>0 {sum(1 for r in sel if r[3]>0)/len(sel):6.3f}  spiked {sum(1 for r in sel if r[4]>0)/len(sel):6.3f}")
print()
print("examples of turns with cache_read>0 (sid, turn, prompt_tokens, cache_read_tokens, spikes):")
for r in sorted([r for r in rows if r[3] > 0], key=lambda r: -r[3])[:25]:
    print("  ", r[0][:8], r[1], r[2], r[3], r[4])

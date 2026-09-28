"""Spike rate by absolute position using only the token-level .pt dumps (no rollout jsonl)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(
    0,
    "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools",
)
import pt_numpy  # noqa: E402

RUN = sys.argv[1]
SPIKE = float(os.environ.get("SPIKE", "3.0"))
MAX_PT = int(os.environ.get("MAX_PT", "9999"))
files = sorted(glob.glob(os.path.join(RUN, "dumps/token_level/*.pt")))[:MAX_PT]
STEP = os.environ.get("STEP")
if STEP:
    files = [f for f in files if os.path.basename(f).startswith(f"step_{int(STEP):05d}_")]
coarse_t = np.zeros(13, dtype=np.int64)
coarse_s = np.zeros(13, dtype=np.int64)
fine_t = {}
fine_s = {}
n_seq = n_bad = 0
maxL = 0
for pt in files:
    d = pt_numpy.load(pt)
    lens = d["input_lengths"].astype(np.int64)
    mask = d["token_mask"].astype(bool)
    diff = np.abs(d["generation_logprobs"].astype(np.float64) - d["prev_logprobs"].astype(np.float64))
    err = d["seq_mult_prob_error"].astype(np.float64)
    off = int(d["logprob_offset"])
    tok_start = lp_start = 0
    for i, L in enumerate(lens):
        L = int(L)
        nlp = L - off
        gm = mask[tok_start : tok_start + L][off:]
        gen_idx = np.nonzero(gm)[0]
        pos = gen_idx + off
        sp = diff[lp_start : lp_start + nlp][gen_idx] > SPIKE
        cb = np.minimum(pos // 16384, 12)
        coarse_t += np.bincount(cb, minlength=13)
        coarse_s += np.bincount(cb[sp], minlength=13)
        sel = pos >= 98304
        for k, c in zip(*np.unique(pos[sel] // 4096, return_counts=True)):
            fine_t[int(k)] = fine_t.get(int(k), 0) + int(c)
        ks = pos[sel & sp] // 4096
        for k, c in zip(*np.unique(ks, return_counts=True)):
            fine_s[int(k)] = fine_s.get(int(k), 0) + int(c)
        n_seq += 1
        n_bad += int(err[i] > 2.0)
        maxL = max(maxL, L)
        tok_start += L
        lp_start += nlp

print(f"{os.path.basename(RUN)}: files {len(files)}, sequences {n_seq}, err>2: {n_bad} ({100*n_bad/max(n_seq,1):.1f}%), max L {maxL}")
print("spike rate by ABSOLUTE POSITION (bin=16384); rate = spikes per 1e4 generated tokens")
for b in range(13):
    if coarse_t[b]:
        print(f"  [{b*16384:>6},{(b+1)*16384:>6}): tokens {coarse_t[b]:>9}  spikes {coarse_s[b]:>5}  rate {1e4*coarse_s[b]/coarse_t[b]:8.2f}")
print("fine bins (4096) for position >= 98304")
for k in sorted(fine_t):
    t, s = fine_t[k], fine_s.get(k, 0)
    print(f"  [{k*4096:>6},{(k+1)*4096:>6}): tokens {t:>8}  spikes {s:>5}  rate {1e4*s/t:8.2f}")

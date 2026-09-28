"""Is the engine/trainer disagreement dense or sparse past the threshold?

For each absolute-position bucket, fraction of generated tokens with |diff| above several thresholds.
"""
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
MAX_PT = int(os.environ.get("MAX_PT", "9999"))
ONLY_BAD = os.environ.get("ONLY_BAD") == "1"  # restrict to err>2 sequences
THRESH = [0.01, 0.05, 0.2, 1.0, 3.0]
EDGES = [0, 32768, 65536, 98304, 114688, 131072, 147456, 163840, 196608]
tot = np.zeros(len(EDGES) - 1, dtype=np.int64)
cnt = np.zeros((len(THRESH), len(EDGES) - 1), dtype=np.int64)
files = sorted(glob.glob(os.path.join(RUN, "dumps/token_level/*.pt")))[:MAX_PT]
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
        if not ONLY_BAD or err[i] > 2.0:
            gm = mask[tok_start : tok_start + L][off:]
            gen_idx = np.nonzero(gm)[0]
            pos = gen_idx + off
            dg = diff[lp_start : lp_start + nlp][gen_idx]
            b = np.searchsorted(EDGES, pos, side="right") - 1
            tot += np.bincount(b, minlength=len(tot))[: len(tot)]
            for ti, t in enumerate(THRESH):
                cnt[ti] += np.bincount(b[dg > t], minlength=len(tot))[: len(tot)]
        tok_start += L
        lp_start += nlp

label = os.path.basename(RUN) + (" [err>2 sequences only]" if ONLY_BAD else " [all sequences]")
print(label)
print("position bucket        tokens      " + "  ".join(f">{t:<5}" for t in THRESH) + "   (fraction of generated tokens)")
for k in range(len(tot)):
    if tot[k] == 0:
        continue
    fr = "  ".join(f"{cnt[ti][k]/tot[k]:7.4f}" for ti in range(len(THRESH)))
    print(f"[{EDGES[k]:>6},{EDGES[k+1]:>6})  {tot[k]:>9}   {fr}")

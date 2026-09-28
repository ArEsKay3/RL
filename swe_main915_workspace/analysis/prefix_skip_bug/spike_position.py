"""Spike rate as a function of absolute context position, prompt length, offset in message; per-message clustering."""
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
SPIKE = float(os.environ.get("SPIKE", "3.0"))
BIN = 16384

msgs_by_sid = {}
for jl in sorted(glob.glob(os.path.join(RUN, "dumps/rollouts/*.jsonl"))):
    with open(jl) as f:
        for line in f:
            rec = json.loads(line)
            msgs_by_sid[rec["sample_id"]] = [(m["role"], int(m["n_tokens"])) for m in rec["messages"]]

pos_tokens = {}   # bin -> generated tokens
pos_spikes = {}   # bin -> spikes
pl_tokens = {}    # prompt-length bin -> generated tokens
pl_spikes = {}
inmsg_tokens = {}  # in-message offset bucket -> tokens
inmsg_spikes = {}
per_seq = []      # (sid, err, n_asst_msgs, n_msgs_with_spike, first_spike_msg, last_spike_msg, n_spikes, max_pos)
fine_bins = {}    # 4096-wide bins between 98304 and 196608 for a sharper threshold look
fine_spikes = {}


def bucket_inmsg(k):
    if k == 0:
        return "0"
    if k < 4:
        return "1-3"
    if k < 16:
        return "4-15"
    if k < 64:
        return "16-63"
    if k < 256:
        return "64-255"
    if k < 1024:
        return "256-1023"
    return "1024+"


for pt in sorted(glob.glob(os.path.join(RUN, "dumps/token_level/*.pt")))[: int(os.environ.get("MAX_PT", "9999"))]:
    d = pt_numpy.load(pt)
    sids = d["sample_ids"]
    lens = d["input_lengths"].astype(np.int64)
    mask = d["token_mask"].astype(bool)
    gen_lp = d["generation_logprobs"].astype(np.float64)
    prev_lp = d["prev_logprobs"].astype(np.float64)
    err = d["seq_mult_prob_error"].astype(np.float64)
    off = int(d["logprob_offset"])
    tok_start = lp_start = 0
    for i, sid in enumerate(sids):
        L = int(lens[i])
        nlp = L - off
        m = mask[tok_start : tok_start + L]
        g = gen_lp[lp_start : lp_start + nlp]
        p = prev_lp[lp_start : lp_start + nlp]
        gm = m[off:]
        diff = np.abs(g - p)
        msgs = msgs_by_sid[sid]
        bounds = np.cumsum([0] + [n for _, n in msgs])
        roles = [r for r, _ in msgs]
        gen_idx = np.nonzero(gm)[0]
        pos = gen_idx + off
        is_spike = diff[gen_idx] > SPIKE
        mi = np.searchsorted(bounds, pos, side="right") - 1
        in_msg = pos - bounds[mi]
        prompt_len = bounds[mi]
        def acc(tok_d, sp_d, keys, spk):
            if keys.size == 0:
                return
            for k, c in zip(*np.unique(keys, return_counts=True)):
                tok_d[int(k)] = tok_d.get(int(k), 0) + int(c)
            ks = keys[spk]
            if ks.size:
                for k, c in zip(*np.unique(ks, return_counts=True)):
                    sp_d[int(k)] = sp_d.get(int(k), 0) + int(c)
        acc(pos_tokens, pos_spikes, pos // BIN, is_spike)
        acc(pl_tokens, pl_spikes, prompt_len // BIN, is_spike)
        edges = np.array([0, 1, 4, 16, 64, 256, 1024, 1 << 40])
        labels = ["0", "1-3", "4-15", "16-63", "64-255", "256-1023", "1024+"]
        kb = np.searchsorted(edges, in_msg, side="right") - 1
        for k, c in zip(*np.unique(kb, return_counts=True)):
            inmsg_tokens[labels[int(k)]] = inmsg_tokens.get(labels[int(k)], 0) + int(c)
        kbs = kb[is_spike]
        if kbs.size:
            for k, c in zip(*np.unique(kbs, return_counts=True)):
                inmsg_spikes[labels[int(k)]] = inmsg_spikes.get(labels[int(k)], 0) + int(c)
        sel = pos >= 98304
        acc(fine_bins, fine_spikes, pos[sel] // 4096, is_spike[sel])
        n_asst = sum(1 for r in roles if r == "assistant")
        spike_msgs = sorted(set(int(x) for x in mi[is_spike]))
        per_seq.append((sid, float(err[i]), n_asst, len(spike_msgs), spike_msgs[0] if spike_msgs else None, spike_msgs[-1] if spike_msgs else None, int(is_spike.sum()), L, spike_msgs))
        tok_start += L
        lp_start += nlp

print(f"spike rate by ABSOLUTE POSITION (bin={BIN}); rate = spikes per 1e4 generated tokens")
for b in sorted(pos_tokens):
    n, s = pos_tokens[b], pos_spikes[b]
    print(f"  [{b*BIN:>6},{(b+1)*BIN:>6}): tokens {n:>8}  spikes {s:>5}  rate {1e4*s/n:8.2f}")
print()
print("spike rate by PROMPT LENGTH of the containing assistant message (bin=16384)")
for b in sorted(pl_tokens):
    n, s = pl_tokens[b], pl_spikes[b]
    print(f"  [{b*BIN:>6},{(b+1)*BIN:>6}): tokens {n:>8}  spikes {s:>5}  rate {1e4*s/n:8.2f}")
print()
print("fine bins (4096) for position >= 98304")
for b in sorted(fine_bins):
    n, s = fine_bins[b], fine_spikes[b]
    print(f"  [{b*4096:>6},{(b+1)*4096:>6}): tokens {n:>8}  spikes {s:>5}  rate {1e4*s/n:8.2f}")
print()
print("spike rate by offset within the assistant message")
for b in ["0", "1-3", "4-15", "16-63", "64-255", "256-1023", "1024+"]:
    n, s = inmsg_tokens.get(b, 0), inmsg_spikes.get(b, 0)
    if n:
        print(f"  in_msg {b:>9}: tokens {n:>8}  spikes {s:>5}  rate {1e4*s/n:8.2f}")
print()
print("per-sequence clustering (err>2 sequences): n_asst_msgs, msgs_with_spike, first/last spike msg#, n_spikes")
for sid, e, n_asst, nsm, first, last, ns, L, sm in sorted(per_seq, key=lambda t: -t[1]):
    if e <= 2.0:
        continue
    span = (last - first + 1) if first is not None else 0
    # how many assistant messages between first and last spike message, and how many of those have spikes
    print(f"  {sid[:8]} err {e:<9.3g} L {L:<7} asst_msgs {n_asst:<4} msgs_with_spike {nsm:<4} first {first} last {last} span {span:<4} n_spikes {ns}")

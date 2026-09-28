"""Do tokens of flagged assistant messages (invalid tool call / malformed thinking) get a different advantage?

Joins per-message flags from the rollout jsonl with per-token advantages in the token-level .pt dumps.
"""
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
MAX_PT = int(os.environ.get("MAX_PT", "9999"))
MAX_JL = int(os.environ.get("MAX_JL", "9999"))

msgs_by_sid = {}
DUMPS = os.environ.get("DUMPS_DIR") or os.path.join(RUN, "dumps")
for jl in sorted(glob.glob(os.path.join(DUMPS, "rollouts/*.jsonl")))[:MAX_JL]:
    with open(jl) as f:
        for line in f:
            rec = json.loads(line)
            msgs_by_sid[rec["sample_id"]] = [
                (m["role"], int(m["n_tokens"]), bool(m.get("is_invalid_tool_call", False)), bool(m.get("has_malformed_thinking", False)))
                for m in rec["messages"]
            ]

n_flag_msgs = n_seq_with_flags = 0
examples = []
flag_adv_values = Counter()
diff_summary = []  # (sid, reward-ish adv of unflagged tokens, adv of flagged tokens)
for pt in sorted(glob.glob(os.path.join(DUMPS, "token_level/*.pt")))[:MAX_PT]:
    d = pt_numpy.load(pt)
    sids = d["sample_ids"]
    lens = d["input_lengths"].astype(np.int64)
    mask = d["token_mask"].astype(bool)
    adv = d["advantages"].astype(np.float64)
    rewards = d["rewards"].astype(np.float64)
    off = int(d["logprob_offset"])
    tok_start = lp_start = 0
    for i, sid in enumerate(sids):
        L = int(lens[i]); nlp = L - off
        msgs = msgs_by_sid.get(sid)
        if msgs is None:
            tok_start += L; lp_start += nlp
            continue
        gm = mask[tok_start : tok_start + L][off:]
        a = adv[lp_start : lp_start + nlp]
        bounds = np.cumsum([0] + [n for _, n, _, _ in msgs])
        flagged = np.zeros(nlp, dtype=bool)
        kinds = []
        for mi, (role, n, inv, mal) in enumerate(msgs):
            if role == "assistant" and (inv or mal):
                s, e = int(bounds[mi]), int(bounds[mi + 1])
                # logprob/advantage index = token position - off
                flagged[max(s - off, 0) : max(e - off, 0)] = True
                kinds.append(("invalid" if inv else "") + ("+malformed" if mal else ""))
        n_flag_msgs += len(kinds)
        if kinds:
            n_seq_with_flags += 1
            fa = a[flagged & gm]; ua = a[(~flagged) & gm]
            fvals = Counter(np.round(fa, 3).tolist()); uvals = Counter(np.round(ua, 3).tolist())
            flag_adv_values.update(fvals)
            diff_summary.append((sid[:8], float(rewards[i]), uvals.most_common(2), fvals.most_common(3), kinds[:3]))
        tok_start += L; lp_start += nlp

print(f"{os.path.basename(RUN)}: sequences with jsonl {len(msgs_by_sid)}; flagged assistant messages {n_flag_msgs} in {n_seq_with_flags} sequences")
print("distinct advantage values on flagged-message tokens (rounded):", flag_adv_values.most_common(8))
print("per-sequence: sid, reward, unflagged-token advantages (top2), flagged-token advantages (top3), kinds")
for row in diff_summary[:25]:
    print("  ", row)

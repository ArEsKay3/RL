"""Agent-behaviour flags (invalid tool call / malformed thinking) vs context position, and vs spikes.

Usage: python behavior_vs_position.py RUN_DIR [--with-pt]
"""
import glob
import json
import os
import sys

import numpy as np

RUN = sys.argv[1]
WITH_PT = "--with-pt" in sys.argv
BIN = 16384

sys.path.insert(
    0,
    "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools",
)

# ---- per-assistant-message flags by prompt-length bucket ----
bucket = {}  # b -> [n, invalid, malformed]
msgs_by_sid = {}
reward_by_sid = {}
turns_by_sid = {}
for jl in sorted(glob.glob(os.path.join(RUN, "dumps/rollouts/*.jsonl")))[: int(os.environ.get("MAX_JL", "9999"))]:
    with open(jl) as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec["messages"]
            msgs_by_sid[rec["sample_id"]] = msgs
            reward_by_sid[rec["sample_id"]] = rec.get("reward")
            turns_by_sid[rec["sample_id"]] = rec.get("num_assistant_turns")
            acc = 0
            for m in msgs:
                if m["role"] == "assistant":
                    b = acc // BIN
                    e = bucket.setdefault(b, [0, 0, 0])
                    e[0] += 1
                    e[1] += int(bool(m.get("is_invalid_tool_call", False)))
                    e[2] += int(bool(m.get("has_malformed_thinking", False)))
                acc += int(m["n_tokens"])

print(f"{RUN.rsplit('/',1)[-1]}: assistant messages by prompt-length bucket -> invalid_tool_call %, malformed_thinking %")
for b in sorted(bucket):
    n, inv, mal = bucket[b]
    print(f"  [{b*BIN:>6},{(b+1)*BIN:>6}): msgs {n:>6}  invalid {100*inv/n:6.2f}%  malformed {100*mal/n:6.2f}%")

if not WITH_PT:
    sys.exit(0)

import pt_numpy  # noqa: E402

SPIKE = 3.0
spiked_msgs = {}  # (sid, mi) -> n_spikes
seq_rows = []  # (sid, err, L, reward, turns)
for pt in sorted(glob.glob(os.path.join(RUN, "dumps/token_level/*.pt"))):
    d = pt_numpy.load(pt)
    sids = d["sample_ids"]
    lens = d["input_lengths"].astype(np.int64)
    mask = d["token_mask"].astype(bool)
    gen_lp = d["generation_logprobs"].astype(np.float64)
    prev_lp = d["prev_logprobs"].astype(np.float64)
    err = d["seq_mult_prob_error"].astype(np.float64)
    rewards = d["rewards"].astype(np.float64)
    off = int(d["logprob_offset"])
    tok_start = lp_start = 0
    for i, sid in enumerate(sids):
        L = int(lens[i])
        nlp = L - off
        gm = mask[tok_start : tok_start + L][off:]
        diff = np.abs(gen_lp[lp_start : lp_start + nlp] - prev_lp[lp_start : lp_start + nlp])
        msgs = msgs_by_sid[sid]
        bounds = np.cumsum([0] + [int(m["n_tokens"]) for m in msgs])
        gen_idx = np.nonzero(gm)[0]
        pos = gen_idx + off
        sp = diff[gen_idx] > SPIKE
        mi = np.searchsorted(bounds, pos[sp], side="right") - 1
        for m_ in mi:
            spiked_msgs[(sid, int(m_))] = spiked_msgs.get((sid, int(m_)), 0) + 1
        seq_rows.append((sid, float(err[i]), L, float(rewards[i]), turns_by_sid.get(sid)))
        tok_start += L
        lp_start += nlp

# messages with spikes vs without (restricted to prompt_len >= 131072 so the populations are comparable)
n_s = inv_s = mal_s = 0
n_c = inv_c = mal_c = 0
for sid, msgs in msgs_by_sid.items():
    if sid not in {r[0] for r in seq_rows}:
        continue
    acc = 0
    for mi, m in enumerate(msgs):
        if m["role"] == "assistant" and acc >= 131072:
            inv = int(bool(m.get("is_invalid_tool_call", False)))
            mal = int(bool(m.get("has_malformed_thinking", False)))
            if (sid, mi) in spiked_msgs:
                n_s += 1; inv_s += inv; mal_s += mal
            else:
                n_c += 1; inv_c += inv; mal_c += mal
        acc += int(m["n_tokens"])
print()
print(f"assistant messages at prompt_len >= 131072: with spikes n={n_s} invalid {100*inv_s/max(n_s,1):.2f}% malformed {100*mal_s/max(n_s,1):.2f}%  |  without spikes n={n_c} invalid {100*inv_c/max(n_c,1):.2f}% malformed {100*mal_c/max(n_c,1):.2f}%")

long_bad = [r for r in seq_rows if r[2] >= 131072 and r[1] > 2.0]
long_ok = [r for r in seq_rows if r[2] >= 131072 and r[1] <= 2.0]
print(f"sequences with L >= 131072: err>2 n={len(long_bad)} mean reward {np.mean([r[3] for r in long_bad]) if long_bad else float('nan'):.3f}; err<=2 n={len(long_ok)} mean reward {np.mean([r[3] for r in long_ok]) if long_ok else float('nan'):.3f}")
allb = [r for r in seq_rows if r[1] > 2.0]; allo = [r for r in seq_rows if r[1] <= 2.0]
print(f"all sequences: err>2 n={len(allb)} mean reward {np.mean([r[3] for r in allb]):.3f}; err<=2 n={len(allo)} mean reward {np.mean([r[3] for r in allo]):.3f}")

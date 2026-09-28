"""Locate where engine-vs-trainer logprob disagreement begins inside corrupted sequences.

Joins the token-level .pt dumps (engine `generation_logprobs` vs trainer `prev_logprobs`)
with the rollout jsonl per-message token counts to place the onset on turn boundaries.
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
ERR_THRESH = float(os.environ.get("ERR_THRESH", "2.0"))
TOK_THRESH = float(os.environ.get("TOK_THRESH", "1.0"))  # nats, per-token |diff| considered catastrophic

# ---- rollout jsonl: sample_id -> messages (role, n_tokens) ----
msgs_by_sid = {}
turns_by_sid = {}
for jl in sorted(glob.glob(os.path.join(RUN, "dumps/rollouts/*.jsonl"))):
    with open(jl) as f:
        for line in f:
            rec = json.loads(line)
            msgs_by_sid[rec["sample_id"]] = [(m["role"], int(m["n_tokens"])) for m in rec["messages"]]
            turns_by_sid[rec["sample_id"]] = int(rec["num_assistant_turns"])

rows = []
for pt in sorted(glob.glob(os.path.join(RUN, "dumps/token_level/*.pt"))):
    d = pt_numpy.load(pt)
    sids = d["sample_ids"]
    lens = d["input_lengths"].astype(np.int64)
    mask = d["token_mask"].astype(bool)
    gen_lp = d["generation_logprobs"].astype(np.float64)
    prev_lp = d["prev_logprobs"].astype(np.float64)
    err = d["seq_mult_prob_error"].astype(np.float64)
    off = int(d["logprob_offset"])
    tok_start = 0
    lp_start = 0
    for i, sid in enumerate(sids):
        L = int(lens[i])
        m = mask[tok_start : tok_start + L]
        # logprob for token t (t>=off) lives at lp index t-off within the sample block
        nlp = L - off
        g = gen_lp[lp_start : lp_start + nlp]
        p = prev_lp[lp_start : lp_start + nlp]
        gm = m[off:]  # mask aligned to logprob positions
        diff = np.abs(g - p)
        diff_gen = np.where(gm, diff, 0.0)
        msgs = msgs_by_sid.get(sid)
        rows.append(
            dict(
                sid=sid,
                step=int(d["step"]),
                L=L,
                err=float(err[i]),
                turns=turns_by_sid.get(sid),
                diff_gen=diff_gen,
                gm=gm,
                msgs=msgs,
            )
        )
        tok_start += L
        lp_start += nlp

print(f"sequences: {len(rows)}; corrupted (err>{ERR_THRESH}): {sum(r['err'] > ERR_THRESH for r in rows)}")

# sanity: message token sums vs input_lengths
mismatch = [(r["sid"], r["L"], sum(n for _, n in r["msgs"])) for r in rows if r["msgs"] and sum(n for _, n in r["msgs"]) != r["L"]]
print(f"message-sum != input_length for {len(mismatch)} sequences" + (f", e.g. {mismatch[:3]}" if mismatch else ""))


def onset(r):
    """First logprob position whose |diff| exceeds TOK_THRESH and whose following 64 generated tokens average > 0.5."""
    dg, gm = r["diff_gen"], r["gm"]
    idx = np.nonzero(dg > TOK_THRESH)[0]
    for t in idx:
        window = dg[t : t + 400]
        wm = gm[t : t + 400]
        if wm.sum() >= 16 and window[wm].mean() > 0.5:
            return int(t) + 1  # convert logprob index back to token position
    return None


def locate(r, pos):
    """Map a token position to (message index, offset within message, prompt tokens before that message)."""
    acc = 0
    for mi, (role, n) in enumerate(r["msgs"]):
        if pos < acc + n:
            return mi, role, pos - acc, acc
        acc += n
    return None, None, None, acc


hdr = "err       turns  L        onset_pos  msg#  role       in_msg  prompt_at_onset  P%256  P%16384  asst_turn#  tail_bad_frac  first_gen_bad"
print(hdr)
print("-" * len(hdr))
summary = []
for r in sorted(rows, key=lambda r: -r["err"]):
    if r["err"] <= ERR_THRESH or not r["msgs"]:
        continue
    pos = onset(r)
    if pos is None:
        print(f"{r['err']:<9.3g} {r['turns']:<6} {r['L']:<8} (no catastrophic onset found; max tok diff {r['diff_gen'].max():.2f})")
        continue
    mi, role, in_msg, P = locate(r, pos)
    asst_turn = sum(1 for role_, _ in r["msgs"][: mi + 1] if role_ == "assistant")
    # fraction of generated tokens after onset with |diff| > 0.5
    dg, gm = r["diff_gen"], r["gm"]
    tail = dg[pos - 1 :][gm[pos - 1 :]]
    tail_bad = float((tail > 0.5).mean()) if tail.size else float("nan")
    # is the very first generated token of that assistant message already bad?
    first_gen_bad = bool(dg[P - 1 + 0] > TOK_THRESH) if role == "assistant" and P >= 1 else False
    print(
        f"{r['err']:<9.3g} {r['turns']:<6} {r['L']:<8} {pos:<10} {mi:<5} {role:<10} {in_msg:<7} {P:<16} {P % 256:<6} {P % 16384:<8} {asst_turn:<11} {tail_bad:<14.2f} {first_gen_bad}"
    )
    summary.append(dict(err=r["err"], turns=r["turns"], P=P, in_msg=in_msg, role=role, mi=mi, tail_bad=tail_bad, first_gen_bad=first_gen_bad, sid=r["sid"]))

# aggregate
if summary:
    in_msg0 = sum(1 for s in summary if s["in_msg"] == 0)
    asst = sum(1 for s in summary if s["role"] == "assistant")
    print()
    print(f"onsets: {len(summary)}; in assistant messages: {asst}; at first token of message: {in_msg0}; first generated token already bad: {sum(s['first_gen_bad'] for s in summary)}")
    print(f"tail_bad_frac median: {np.median([s['tail_bad'] for s in summary]):.2f}")
    Ps = np.array([s["P"] for s in summary])
    print(f"prompt length at onset: min {Ps.min()} p25 {np.percentile(Ps,25):.0f} median {np.median(Ps):.0f} p75 {np.percentile(Ps,75):.0f} max {Ps.max()}")

# healthy reference
healthy = [r for r in rows if r["err"] <= ERR_THRESH and r["msgs"]]
if healthy:
    allh = np.concatenate([r["diff_gen"][r["gm"]] for r in healthy])
    print(f"healthy sequences: {len(healthy)}; per-token |diff| p50 {np.percentile(allh,50):.4f} p99 {np.percentile(allh,99):.3f} max {allh.max():.2f}; frac>1.0: {(allh>1.0).mean():.5f}")
    maxL = max(r["L"] for r in healthy)
    print(f"longest healthy sequence: {maxL} tokens, {max(r['turns'] for r in healthy)} turns")

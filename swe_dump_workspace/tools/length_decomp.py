"""Decompose generation length by run and step: turns, tokens per turn, thinking vs non-thinking tokens, tails.

Sources: analysis/rollouts/*.summary.jsonl (per rollout), analysis/tokens/*.rows.csv (trained rows), analysis/think_stop/*.npz (per-turn spans).
Writes analysis/length_decomp.md.
"""
from __future__ import annotations

import glob, json, math, os
from collections import defaultdict

import numpy as np, pandas as pd

W = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump"
SHORT = {"nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918": "runA_vllm", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918": "runB_minf",
         "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-nopg-noprefix-20260919": "chainC_minf", "nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919": "chainD_vllm",
         "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-20260920": "chainE_minf", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920": "chainF_minf_fromvllm10", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920": "chainG_minf_from0_nopg"}
RUNS = ["runA_vllm", "chainD_vllm", "runB_minf", "chainC_minf", "chainE_minf", "chainF_minf_fromvllm10", "chainG_minf_from0_nopg"]


def load_R():
    recs = []
    for f in glob.glob(f"{W}/analysis/rollouts/*.summary.jsonl"):
        with open(f) as fh:
            recs.extend(json.loads(l) for l in fh if l.strip())
    R = pd.DataFrame(recs); R = R[R.sample_id.notna()].copy()
    R["step"] = R.target_step.astype(int) + 1; R["run"] = R.run.map(SHORT)
    tok = set()
    for f in glob.glob(f"{W}/analysis/tokens/*.rows.csv"):
        tok.update(pd.read_csv(f, usecols=["sample_id"]).sample_id)
    R = R[R.sample_id.isin(tok)].copy()
    R["err"] = R.agent_error_kind.fillna("none"); R["truncated"] = R.truncated.fillna(False).astype(bool)
    return R


def load_spans():
    out = defaultdict(lambda: dict(span=[], think=[], reward=[]))
    for f in glob.glob(f"{W}/analysis/think_stop/*.thinkaudit.npz"):
        j = json.load(open(f[:-4] + ".json")); run = SHORT.get(j["run"]); z = np.load(f)
        d = out[(run, j["step"])]; d["span"].append(z["span_len"]); d["think"].append(z["think_len"]); d["reward"].append(z["span_reward"])
    return {k: {kk: np.concatenate(v) for kk, v in d.items()} for k, d in out.items()}


def main():
    R = load_R(); S = load_spans()
    L = ["# Generation-length decomposition", "", "Rows = trained rollouts (present in both dumps). gen_len = generated tokens per rollout. turns = assistant turns per rollout. "
         "longest_msg = longest single assistant message in the rollout (tokens). Spans = per-turn generated token runs from the token dumps; think = tokens up to and including `</think>` within a turn.", ""]
    hdr = ("| run | step | rollouts | mean gen_len | median | p90 | p99 | max | mean turns | median turns | share 200 turns | share ctx-window end | mean tokens/turn | "
           "mean longest_msg | share longest_msg>4k | >8k | >16k | mean gen_len excl. rollouts with a msg>8k | top-5% rollouts' share of tokens |")
    L += [hdr, "|" + "---|" * 20]
    per = {}
    for run in RUNS:
        for step in sorted(R[R.run == run].step.unique()):
            d = R[(R.run == run) & (R.step == step)]
            if len(d) < 256: continue
            g = d.gen_len.values.astype(float); t = d.turns.values.astype(float); m = d.max_asst_tokens.values.astype(float)
            top = np.sort(g)[-max(1, len(g) // 20):].sum() / g.sum()
            excl = g[m <= 8192].mean() if (m <= 8192).any() else float("nan")
            per[(run, step)] = dict(n=len(d), mean=g.mean(), med=np.median(g), p90=np.percentile(g, 90), p99=np.percentile(g, 99), mx=g.max(), turns=t.mean(), tmed=np.median(t),
                                    s200=(t >= 200).mean(), sctx=(d.err == "context_window").mean(), tpt=g.sum() / t.sum(), lm=m.mean(), l4=(m > 4096).mean(), l8=(m > 8192).mean(),
                                    l16=(m > 16384).mean(), excl=excl, top=top)
            p = per[(run, step)]
            L.append(f"| {run} | {step} | {p['n']} | {p['mean']:.0f} | {p['med']:.0f} | {p['p90']:.0f} | {p['p99']:.0f} | {p['mx']:.0f} | {p['turns']:.1f} | {p['tmed']:.0f} | {p['s200']:.3f} | {p['sctx']:.3f} | "
                     f"{p['tpt']:.0f} | {p['lm']:.0f} | {p['l4']:.3f} | {p['l8']:.3f} | {p['l16']:.3f} | {p['excl']:.0f} | {p['top']:.3f} |")
    L += ["", "## Per-turn spans (token dumps)", "",
          "| run | step | turns | mean span | median span | p99 span | share spans>4k | >8k | >16k | tokens in spans>8k / all tokens | mean think tokens per turn | mean non-think tokens per turn | tokens in think>4k / all tokens | spans without </think> |",
          "|" + "---|" * 14]
    sp = {}
    for run in RUNS:
        for (r, step), d in sorted(S.items()):
            if r != run: continue
            s = d["span"].astype(float); th = d["think"].astype(float); ok = th > 0
            think_tok = np.where(ok, th, 0.0); non = s - think_tok
            sp[(run, step)] = dict(n=len(s), mean=s.mean(), med=np.median(s), p99=np.percentile(s, 99), s4=(s > 4096).mean(), s8=(s > 8192).mean(), s16=(s > 16384).mean(),
                                   tok8=s[s > 8192].sum() / s.sum(), think=think_tok.mean(), non=non.mean(), think4=think_tok[think_tok > 4096].sum() / s.sum(), nothink=(~ok).mean())
            p = sp[(run, step)]
            L.append(f"| {run} | {step} | {p['n']} | {p['mean']:.0f} | {p['med']:.0f} | {p['p99']:.0f} | {p['s4']:.4f} | {p['s8']:.4f} | {p['s16']:.4f} | {p['tok8']:.3f} | {p['think']:.0f} | {p['non']:.0f} | {p['think4']:.3f} | {p['nothink']:.4f} |")
    # pooled comparison steps 17-24 run A vs run B
    L += ["", "## Pooled steps 17-24: run A vLLM vs run B MINF", ""]
    for name, keyset in (("rollouts", per), ("spans", sp)):
        pass
    A = R[(R.run == "runA_vllm") & (R.step.between(17, 24))]; B = R[(R.run == "runB_minf") & (R.step.between(17, 24))]
    def agg(d):
        g = d.gen_len.astype(float); t = d.turns.astype(float); m = d.max_asst_tokens.astype(float)
        return dict(n=len(d), mean=g.mean(), med=g.median(), p90=g.quantile(.9), p99=g.quantile(.99), turns=t.mean(), tpt=g.sum() / t.sum(), lm=m.mean(), l8=(m > 8192).mean(),
                    excl8=g[m <= 8192].mean(), top5=np.sort(g.values)[-len(g) // 20:].sum() / g.sum(), fail_mean=g[d.reward <= 0].mean(), succ_mean=g[d.reward > 0].mean(),
                    fail_turns=t[d.reward <= 0].mean(), succ_turns=t[d.reward > 0].mean(), pass_rate=(d.reward > 0).mean(), s200=(t >= 200).mean(), ctx=(d.err == "context_window").mean())
    a, b = agg(A), agg(B)
    L.append("| quantity | run A vLLM | run B MINF | B minus A |"); L.append("|---|---|---|---|")
    for k, lab in [("n", "trained rollouts"), ("mean", "mean gen_len (tokens)"), ("med", "median gen_len"), ("p90", "p90 gen_len"), ("p99", "p99 gen_len"), ("turns", "mean turns"), ("tpt", "mean tokens per turn"),
                   ("lm", "mean longest message (tokens)"), ("l8", "share of rollouts with a message > 8k"), ("excl8", "mean gen_len excluding rollouts with a message > 8k"),
                   ("top5", "share of all tokens in the longest 5% of rollouts"), ("pass_rate", "pass rate"), ("fail_mean", "mean gen_len, failed rollouts"), ("succ_mean", "mean gen_len, successful rollouts"),
                   ("fail_turns", "mean turns, failed"), ("succ_turns", "mean turns, successful"), ("s200", "share hitting 200 turns"), ("ctx", "share ending on context window")]:
        va, vb = a[k], b[k]
        fmt = (lambda v: f"{v:.3f}") if k in ("l8", "top5", "pass_rate", "s200", "ctx") else (lambda v: f"{v:.0f}")
        L.append(f"| {lab} | {fmt(va)} | {fmt(vb)} | {fmt(vb - va) if k != 'n' else ''} |")
    # turn-count vs tokens-per-turn attribution
    dT = (b["turns"] - a["turns"]) * a["tpt"]; dP = (b["tpt"] - a["tpt"]) * b["turns"]
    L.append(f"\nAttribution of the mean gen_len difference ({b['mean'] - a['mean']:+.0f} tokens): turns effect (delta turns x run A tokens/turn) {dT:+.0f}; tokens-per-turn effect (delta tokens/turn x run B turns) {dP:+.0f}.")
    sa = {k: v for k, v in sp.items() if k[0] == "runA_vllm" and 17 <= k[1] <= 24}; sb = {k: v for k, v in sp.items() if k[0] == "runB_minf" and 17 <= k[1] <= 24}
    def pool(S_, key):
        return np.average([v[key] for v in S_.values()], weights=[v["n"] for v in S_.values()])
    L.append("\n| per-turn quantity (pooled 17-24) | run A vLLM | run B MINF |"); L.append("|---|---|---|")
    for k, lab in [("mean", "mean span (tokens per turn)"), ("think", "mean think tokens per turn"), ("non", "mean non-think tokens per turn"), ("s4", "share of turns > 4k tokens"), ("s8", "share of turns > 8k"),
                   ("tok8", "share of all generated tokens in turns > 8k"), ("think4", "share of all generated tokens in think blocks > 4k"), ("nothink", "share of turns without </think>")]:
        L.append(f"| {lab} | {pool(sa, k):.4f} | {pool(sb, k):.4f} |")
    rep = "\n".join(L) + "\n"; open(f"{W}/analysis/length_decomp.md", "w").write(rep)
    print(rep)


if __name__ == "__main__":
    main()

"""Uncertainty of the sampled-</think> probability statistics.

Per (run, step): geometric-mean engine probability of the sampled </think> tokens and the
low-probability share (engine logprob < -0.1), with a naive token-level SE, a rollout-level
cluster bootstrap (512 rollouts) and a prompt-level cluster bootstrap (32 prompts).
Between arms at the same train step: paired differences with prompts matched by SWE instance id
(prompt-level bootstrap), then pooled over steps with within-step and between-step SEs.
Writes analysis/close_prob_uncertainty.md and .json.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pt_numpy import load  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
RUNS = {
    "runA_vllm": ("nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918", "run A (vLLM from scratch)"),
    "runB_minf": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918", "run B (MINF from MINF step 10)"),
    "chainD_vllm": ("nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919", "chain D (vLLM from MINF step 10)"),
    "chainF_minf_fromvllm10": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920", "chain F (MINF from vLLM step 10)"),
    "chainC_minf": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-nopg-noprefix-20260919", "chain C (MINF from MINF step 10, no prefix cache)"),
}
PAIRS = [("chainD_vllm", "runB_minf"), ("chainD_vllm", "runA_vllm"), ("runB_minf", "runA_vllm"),
         ("chainF_minf_fromvllm10", "runA_vllm"), ("chainF_minf_fromvllm10", "runB_minf"), ("chainC_minf", "runB_minf")]
STEPS = list(range(11, 36))
POOL_STEPS = list(range(15, 25))
THINK_END = 13
NBOOT = int(os.environ.get("NBOOT", 4000))
rng = np.random.default_rng(20260920)


def instance_map(exp):
    m = {}
    for f in glob.glob(f"{W}/analysis/rollouts/{exp}__target_step_*.summary.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("sample_id") and r.get("instance_id"):
                m[r["sample_id"]] = r["instance_id"]
    return m


def per_seq(args):
    run, exp, step = args
    files = sorted(glob.glob(f"{U}/{exp}/dumps/token_level/step_{step:05d}_chunk_*.pt"))
    if not files:
        return run, step, None
    rows = {}
    for path in files:
        d = load(path)
        lengths = np.asarray(d["input_lengths"]).astype(np.int64)
        ids_off = np.concatenate([[0], np.cumsum(lengths)]); lp_off = np.concatenate([[0], np.cumsum(lengths - 1)])
        ids_all = np.asarray(d["input_ids"]).astype(np.int64); mask_all = np.asarray(d["token_mask"]).astype(bool)
        gen_all = np.asarray(d["generation_logprobs"]).astype(np.float64)
        sids = list(d["sample_ids"])
        for i in range(len(lengths)):
            m = mask_all[ids_off[i] + 1:ids_off[i + 1]]; tok = ids_all[ids_off[i] + 1:ids_off[i + 1]]
            g = gen_all[lp_off[i]:lp_off[i + 1]]
            lp = g[m & (tok == THINK_END)]
            rows[sids[i]] = (int(lp.size), float(lp.sum()), int((lp < -0.1).sum()), float((lp * lp).sum()))
    return run, step, rows


def ratio_boot(n, s, l, size=None):
    """Bootstrap of mean logprob (sum s / sum n) and low share (sum l / sum n) over clusters."""
    k = len(n); size = size or k
    idx = rng.integers(0, k, (NBOOT, size))
    nb = n[idx].sum(1); sb = s[idx].sum(1); lb = l[idx].sum(1)
    ok = nb > 0
    return sb[ok] / nb[ok], lb[ok] / nb[ok]


def summarize(vals):
    vals = np.asarray(vals)
    return float(vals.std(ddof=1)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    imaps = {run: instance_map(exp) for run, (exp, _) in RUNS.items()}
    jobs = [(run, exp, step) for run, (exp, _) in RUNS.items() for step in STEPS]
    data = {}
    with Pool(int(os.environ.get("NPROC", 16))) as pool:
        for run, step, rows in pool.imap_unordered(per_seq, jobs):
            if rows:
                data[(run, step)] = rows
                print(f"loaded {run} step {step}: {len(rows)} rollouts", flush=True)

    # per (run, step) arrays and per-instance aggregates
    per = {}
    for (run, step), rows in data.items():
        sids = list(rows)
        n = np.array([rows[s][0] for s in sids], float); s_ = np.array([rows[s][1] for s in sids]); l = np.array([rows[s][2] for s in sids], float)
        sq = np.array([rows[s][3] for s in sids])
        inst = [imaps[run].get(s, s.rsplit("_g", 1)[0]) for s in sids]
        groups = {}
        for i, g in enumerate(inst):
            groups.setdefault(g, []).append(i)
        gn = np.array([n[ix].sum() for ix in groups.values()]); gs = np.array([s_[ix].sum() for ix in groups.values()]); gl = np.array([l[ix].sum() for ix in groups.values()])
        N, S, L, SQ = n.sum(), s_.sum(), l.sum(), sq.sum()
        mean_lp = S / N; var_tok = SQ / N - mean_lp ** 2
        se_naive = math.sqrt(max(var_tok, 0) / N)
        r_lp, r_low = ratio_boot(n, s_, l)
        g_lp, g_low = ratio_boot(gn, gs, gl)
        per[(run, step)] = {
            "n_rollouts": len(sids), "n_prompts": len(groups), "n_closes": int(N), "unmatched_instance": sum(1 for s in sids if s not in imaps[run]),
            "mean_lp": mean_lp, "geoP": math.exp(mean_lp), "low": L / N,
            "se_naive_lp": se_naive, "se_rollout_lp": summarize(r_lp)[0], "se_prompt_lp": summarize(g_lp)[0],
            "ci_prompt_P": [math.exp(summarize(g_lp)[1]), math.exp(summarize(g_lp)[2])],
            "se_rollout_low": summarize(r_low)[0], "se_prompt_low": summarize(g_low)[0], "ci_prompt_low": summarize(g_low)[1:],
            "_groups": {g: (gn[j], gs[j], gl[j]) for j, g in enumerate(groups)},
        }

    # paired differences per step, prompts matched by instance id
    pairs = {}
    for x, y in PAIRS:
        for step in STEPS:
            if (x, step) not in per or (y, step) not in per:
                continue
            gx, gy = per[(x, step)]["_groups"], per[(y, step)]["_groups"]
            common = sorted(set(gx) & set(gy))
            if len(common) < 4:
                continue
            nx = np.array([gx[g][0] for g in common]); sx = np.array([gx[g][1] for g in common]); lx = np.array([gx[g][2] for g in common])
            ny = np.array([gy[g][0] for g in common]); sy = np.array([gy[g][1] for g in common]); ly = np.array([gy[g][2] for g in common])
            k = len(common); idx = rng.integers(0, k, (NBOOT, k))
            nxb, sxb, lxb = nx[idx].sum(1), sx[idx].sum(1), lx[idx].sum(1); nyb, syb, lyb = ny[idx].sum(1), sy[idx].sum(1), ly[idx].sum(1)
            dP = np.exp(sxb / nxb) - np.exp(syb / nyb); dlow = lxb / nxb - lyb / nyb
            pairs[(x, y, step)] = {
                "matched_prompts": k, "dP": math.exp(sx.sum() / nx.sum()) - math.exp(sy.sum() / ny.sum()), "dP_se": summarize(dP)[0], "dP_ci": summarize(dP)[1:],
                "dlow": lx.sum() / nx.sum() - ly.sum() / ny.sum(), "dlow_se": summarize(dlow)[0], "dlow_ci": summarize(dlow)[1:],
                "_dP_boot": dP, "_dlow_boot": dlow,
            }

    pooled = {}
    for x, y in PAIRS:
        steps = [s for s in POOL_STEPS if (x, y, s) in pairs]
        if len(steps) < 3:
            continue
        d = np.array([pairs[(x, y, s)]["dP"] for s in steps]); dl = np.array([pairs[(x, y, s)]["dlow"] for s in steps])
        within = math.sqrt(sum(pairs[(x, y, s)]["dP_se"] ** 2 for s in steps)) / len(steps)
        within_l = math.sqrt(sum(pairs[(x, y, s)]["dlow_se"] ** 2 for s in steps)) / len(steps)
        between = d.std(ddof=1) / math.sqrt(len(steps)); between_l = dl.std(ddof=1) / math.sqrt(len(steps))
        # two-level bootstrap: resample steps, then take one within-step bootstrap draw per chosen step
        B = np.stack([pairs[(x, y, s)]["_dP_boot"][:NBOOT] for s in steps]); Bl = np.stack([pairs[(x, y, s)]["_dlow_boot"][:NBOOT] for s in steps])
        nb = min(B.shape[1], Bl.shape[1])
        sidx = rng.integers(0, len(steps), (nb, len(steps))); bidx = rng.integers(0, nb, (nb, len(steps)))
        two = B[sidx, bidx].mean(1); two_l = Bl[sidx, bidx].mean(1)
        pooled[(x, y)] = {"steps": steps, "mean_dP": float(d.mean()), "within_se": within, "between_se": between, "t_between": float(d.mean() / between) if between else float("nan"),
                          "pos": int((d > 0).sum()), "ci_two_level": summarize(two)[1:],
                          "mean_dlow": float(dl.mean()), "within_se_low": within_l, "between_se_low": between_l, "t_between_low": float(dl.mean() / between_l) if between_l else float("nan"),
                          "pos_low": int((dl > 0).sum()), "ci_two_level_low": summarize(two_l)[1:]}

    lines = ["# Uncertainty of the sampled-</think> probability statistics", "",
             f"Bootstrap draws: {NBOOT}. geo-mean P = exp(mean engine logprob of the sampled </think> tokens in the step's trained rollouts). "
             "low share = fraction of those tokens with engine logprob < -0.1 (P < 0.905). "
             "SE naive = token-level sd/sqrt(n) treating every close as independent. SE rollout = cluster bootstrap over rollouts (about 512). "
             "SE prompt = cluster bootstrap over prompts (about 32; 16 rollouts each). Design effect = (SE prompt / SE naive)^2. "
             "Paired differences match prompts across arms by SWE instance id at the same train step and bootstrap over matched prompts.", ""]
    lines += ["## Per run and step", "", "| run | step | prompts | rollouts | closes | geo-mean P | SE naive (P) | SE rollout (P) | SE prompt (P) | 95% CI prompt | design effect | low share | SE prompt (low) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (run, step), p in sorted(per.items(), key=lambda kv: (list(RUNS).index(kv[0][0]), kv[0][1])):
        P = p["geoP"]
        lines.append(f"| {RUNS[run][1]} | {step} | {p['n_prompts']} | {p['n_rollouts']} | {p['n_closes']} | {P:.4f} | {P * p['se_naive_lp']:.4f} | {P * p['se_rollout_lp']:.4f} | {P * p['se_prompt_lp']:.4f} | "
                     f"{p['ci_prompt_P'][0]:.4f}-{p['ci_prompt_P'][1]:.4f} | {(p['se_prompt_lp'] / p['se_naive_lp']) ** 2:.0f} | {100 * p['low']:.2f}% | {100 * p['se_prompt_low']:.2f} |")
    lines += ["", "## Paired differences at the same step (first arm minus second; prompts matched by instance id)", "",
              "| pair | step | matched prompts | dP (prob) | 95% CI | dlow (points) | 95% CI |", "|---|---|---|---|---|---|---|"]
    for (x, y, step), q in sorted(pairs.items(), key=lambda kv: (PAIRS.index((kv[0][0], kv[0][1])), kv[0][2])):
        lines.append(f"| {RUNS[x][1]} minus {RUNS[y][1]} | {step} | {q['matched_prompts']} | {q['dP']:+.4f} | {q['dP_ci'][0]:+.4f} to {q['dP_ci'][1]:+.4f} | {100 * q['dlow']:+.2f} | {100 * q['dlow_ci'][0]:+.2f} to {100 * q['dlow_ci'][1]:+.2f} |")
    lines += ["", f"## Pooled over steps {POOL_STEPS[0]}-{POOL_STEPS[-1]} (mean of per-step paired differences)", "",
              "| pair | steps | mean dP | within-step SE | between-step SE | t (between) | steps with dP>0 | 95% CI two-level bootstrap | mean dlow (points) | between-step SE | t | 95% CI |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (x, y), q in pooled.items():
        lines.append(f"| {RUNS[x][1]} minus {RUNS[y][1]} | {len(q['steps'])} | {q['mean_dP']:+.4f} | {q['within_se']:.4f} | {q['between_se']:.4f} | {q['t_between']:+.2f} | {q['pos']}/{len(q['steps'])} | {q['ci_two_level'][0]:+.4f} to {q['ci_two_level'][1]:+.4f} | "
                     f"{100 * q['mean_dlow']:+.2f} | {100 * q['between_se_low']:.2f} | {q['t_between_low']:+.2f} | {100 * q['ci_two_level_low'][0]:+.2f} to {100 * q['ci_two_level_low'][1]:+.2f} |")
    lines.append("")
    open(os.path.join(W, "analysis", "close_prob_uncertainty.md"), "w").write("\n".join(lines))
    dump = {"per": {f"{r}|{s}": {k: v for k, v in p.items() if not k.startswith("_")} for (r, s), p in per.items()},
            "pairs": {f"{x}|{y}|{s}": {k: v for k, v in q.items() if not k.startswith("_")} for (x, y, s), q in pairs.items()},
            "pooled": {f"{x}|{y}": q for (x, y), q in pooled.items()}}
    json.dump(dump, open(os.path.join(W, "analysis", "close_prob_uncertainty.json"), "w"), default=float, indent=1)
    print("wrote analysis/close_prob_uncertainty.md")


if __name__ == "__main__":
    main()

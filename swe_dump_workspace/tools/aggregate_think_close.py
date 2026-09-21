"""Aggregate think_close_by_step outputs into analysis/think_close_by_step.md."""
from __future__ import annotations

import glob, os
from collections import defaultdict

import numpy as np

W = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump"
SHORT = {"vllm_dump-20260918": "runA_vllm", "minf_dump-from10-20260918": "runB_minf", "minf_dump-nopg-noprefix-20260919": "chainC_minf",
         "vllm_dump-from10-nopg-20260919": "chainD_vllm", "minf_dump-from0-20260920": "chainE_minf", "minf_dump-fromvllm10-20260920": "chainF_minf_fromvllm10", "minf_dump-from0-nopg-noprefix-20260920": "chainG_minf_from0_nopg",
         "minf_dump-from0-prefix-nopg-r1-20260921": "chainI_minf_from0_prefix_r1", "minf_dump-from0-prefix-nopg-r2-20260921": "chainJ_minf_from0_prefix_r2"}


def main():
    per = defaultdict(lambda: dict(g=[], p=[], lag=[], ref=np.zeros((4, 3))))
    for f in sorted(glob.glob(f"{W}/analysis/think_close/*.tc.npz")):
        run = SHORT.get(os.path.basename(f).split("__")[0].split("64n-")[1], "?"); z = np.load(f)
        a = per[(run, int(z["step"]))]; a["g"].append(z["gen"]); a["p"].append(z["prev"]); a["lag"].append(z["lag"]); a["ref"] += z["ref"]
    L = ["# `</think>` trainer-minus-engine gap by step", "",
         "d = trainer logprob - engine logprob of the sampled `</think>`. Hesitant = engine logprob < -0.1. Lag 0 = same weights; lag 1 clean = one update, rollout finished before the next refit. 'frac d<0' = share of closes the trainer likes less than the engine did.", "",
         "| run | step | closes | hesitant share | engine lp mean / p10 | d all | d confident | d hesitant | frac d<0 (hesitant) | d hesitant lag0 | d hesitant lag1 clean | frac d<0 lag1 clean | ref: d all hesitant tokens lag0 / lag1 | ref frac d<0 lag1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (run, step), a in sorted(per.items()):
        g = np.concatenate(a["g"]); p = np.concatenate(a["p"]); lag = np.concatenate(a["lag"]); d = p - g
        hes = g < -0.1; conf = ~hes
        f = lambda x: f"{x.mean():+.4f}" if x.size else "-"
        fr = lambda x: f"{(x < 0).mean():.2f}" if x.size else "-"
        r = a["ref"]
        ref0 = f"{r[0, 1] / r[0, 0]:+.4f}" if r[0, 0] else "-"; ref1 = f"{r[1, 1] / r[1, 0]:+.4f}" if r[1, 0] else "-"; ref1f = f"{r[1, 2] / r[1, 0]:.2f}" if r[1, 0] else "-"
        L.append(f"| {run} | {step} | {len(d)} | {hes.mean():.3f} | {g.mean():+.4f} / {np.percentile(g, 10):+.3f} | {f(d)} | {f(d[conf])} | {f(d[hes])} | {fr(d[hes])} | "
                 f"{f(d[hes & (lag == 0)])} | {f(d[hes & (lag == 1)])} | {fr(d[hes & (lag == 1)])} | {ref0} / {ref1} | {ref1f} |")
    rep = "\n".join(L) + "\n"; open(f"{W}/analysis/think_close_by_step.md", "w").write(rep); print(rep)


if __name__ == "__main__":
    main()

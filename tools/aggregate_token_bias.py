"""Aggregate token_bias_by_id outputs into analysis/token_bias_report.md."""
from __future__ import annotations

import glob, json, os
from collections import defaultdict

import numpy as np

W = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump"
IN = f"{W}/analysis/token_bias"
TOK = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf/tokenizer.json"
SHORT = {"vllm_dump-20260918": "runA_vllm", "minf_dump-from10-20260918": "runB_minf", "minf_dump-nopg-noprefix-20260919": "chainC_minf",
         "vllm_dump-from10-nopg-20260919": "chainD_vllm", "minf_dump-from0-20260920": "chainE_minf", "minf_dump-fromvllm10-20260920": "chainF_minf_fromvllm10", "minf_dump-from0-nopg-noprefix-20260920": "chainG_minf_from0_nopg", "minf_dump-from0-prefix-nopg-r1-20260921": "chainI_minf_from0_prefix_r1", "minf_dump-from0-prefix-nopg-r2-20260921": "chainJ_minf_from0_prefix_r2", "vllm_dump-from0-seed1234-20260922": "chainK_vllm_from0_seed1234", "minf_dump-fromk10-prefix-nopg-20260923": "chainL_minf_fromk10", "vllm_dump-from0-nopg-r1-20260923": "chainM_vllm_from0_nopg_r1", "vllm_dump-from0-nopg-r2-20260923": "chainN_vllm_from0_nopg_r2", "minf_dump-from0-prefix-nopg-r3-20260923": "chainO_minf_from0_prefix_r3"}
PHASES = {"runA_vllm": {"early(1-4)": range(1, 5), "late(21-24)": range(21, 25)}, "runB_minf": {"early(11-14)": range(11, 15), "late(25-28)": range(25, 29)},
          "chainC_minf": {"early(11-14)": range(11, 15)}}
THINK_END, IM_END = 13, 11
KEYS = ["n", "sum_d", "sum_absd", "sum_gen", "sum_d2"] + [f"n_b{b}" for b in range(4)] + [f"sum_d_b{b}" for b in range(4)] + ["te_tl_n", "te_tl_sum_d", "te_tl_sum_gen"] \
    + [f"n_lag{l}" for l in range(4)] + [f"sum_d_lag{l}" for l in range(4)] \
    + [f"lag{l}_b{b}_{w}_{q}" for l in range(4) for b in range(4) for w in ("all", "te") for q in ("n", "sum_d")]


def vocab():
    j = json.load(open(TOK)); inv = {v: k for k, v in j["model"]["vocab"].items()}
    for t in j.get("added_tokens", []):
        inv[t["id"]] = t["content"]
    return inv


def main():
    inv = vocab()
    dec = lambda i: inv.get(int(i), f"<{i}>").replace("Ċ", "\\n").replace("Ġ", "␣").replace("ĉ", "\\t")
    per = defaultdict(dict)
    for f in sorted(glob.glob(f"{IN}/*.bias.npz")):
        run = SHORT.get(f.split("__")[0].split("64n-")[1], "?"); z = np.load(f); step = int(z["step"])
        acc = per[run].setdefault(step, {k: None for k in KEYS})
        for k in KEYS:
            acc[k] = z[k].astype(np.float64) if acc[k] is None else acc[k] + z[k]
    L = ["# Trainer-minus-engine logprob gap by token id", "",
         "d = prev_logprob (trainer, current weights) - generation_logprob (engine at sampling time), over generated tokens only; negative = trainer assigns less probability than the engine did.", ""]
    newline_id = next(i for i, s in inv.items() if s == "Ċ")
    # per-step table for </think> vs reference tokens
    L.append("## Per step: mean d at `</think>` vs all tokens, `\\n`, `<|im_end|>` (SE = standard error of the </think> mean)")
    L.append("| run | step | n </think> | d(</think>) | SE | d(all tokens) | d(\\n) | d(<|im_end|>) | mean gen_lp(</think>) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for run in sorted(per):
        for step in sorted(per[run]):
            a = per[run][step]; n = a["n"]; te = THINK_END
            var = a["sum_d2"][te] / n[te] - (a["sum_d"][te] / n[te]) ** 2
            L.append(f"| {run} | {step} | {int(n[te])} | {a['sum_d'][te] / n[te]:+.5f} | {np.sqrt(max(var, 0) / n[te]):.5f} | {a['sum_d'].sum() / n.sum():+.5f} | "
                     f"{a['sum_d'][newline_id] / n[newline_id]:+.5f} | {a['sum_d'][IM_END] / n[IM_END]:+.5f} | {a['sum_gen'][te] / n[te]:+.4f} |")
    # per-id ranking by phase
    for run, phases in PHASES.items():
        if run not in per: continue
        for ph, steps in phases.items():
            steps = [s for s in steps if s in per[run]]
            if not steps: continue
            tot = {k: sum(per[run][s][k] for s in steps) for k in KEYS}
            n = tot["n"]; ok = n >= 5000
            md = np.where(ok, tot["sum_d"] / np.maximum(n, 1), np.nan); mad = tot["sum_absd"] / np.maximum(n, 1); mg = tot["sum_gen"] / np.maximum(n, 1)
            ids = np.where(ok)[0]; order = ids[np.argsort(md[ids])]
            rank13 = int(np.where(order == THINK_END)[0][0]) + 1 if THINK_END in ids else None
            L.append(""); L.append(f"## {run} {ph}: {len(ids)} token ids with n>=5000, {int(n.sum()):,} generated tokens; `</think>` ranks #{rank13} most negative of {len(ids)}")
            L.append("| rank | token | id | n | mean d | mean |d| | mean gen_lp | share of all tokens |"); L.append("|---|---|---|---|---|---|---|---|")
            for r, i in enumerate(order[:12], 1):
                L.append(f"| {r} | `{dec(i)}` | {i} | {int(n[i])} | {md[i]:+.5f} | {mad[i]:.4f} | {mg[i]:+.4f} | {n[i] / n.sum():.5f} |")
            L.append("| ... most positive ... |  |  |  |  |  |  |  |")
            for i in order[-5:][::-1]:
                L.append(f"|  | `{dec(i)}` | {i} | {int(n[i])} | {md[i]:+.5f} | {mad[i]:.4f} | {mg[i]:+.4f} | {n[i] / n.sum():.5f} |")
            w = n[ids] / n[ids].sum(); mdv = md[ids]; srt = np.argsort(mdv); cw = np.cumsum(w[srt])
            pct = lambda q: mdv[srt][np.searchsorted(cw, q)]
            L.append(f"\nweighted percentiles of per-id mean d: p1 {pct(.01):+.5f}, p10 {pct(.10):+.5f}, p50 {pct(.5):+.5f}, p90 {pct(.9):+.5f}, p99 {pct(.99):+.5f}; all-token mean {tot['sum_d'].sum() / n.sum():+.5f}")
            # certainty buckets
            L.append("\ncertainty buckets by engine logprob of the sampled token: mean d for `</think>` vs all tokens")
            L.append("| bucket | n </think> | share | d(</think>) | n all | share | d(all) |"); L.append("|---|---|---|---|---|---|---|")
            names = ["gen_lp < -1", "[-1, -0.1)", "[-0.1, -0.01)", "[-0.01, 0]"]
            for b in range(4):
                nb = tot[f"n_b{b}"]; sd = tot[f"sum_d_b{b}"]
                L.append(f"| {names[b]} | {int(nb[THINK_END])} | {nb[THINK_END] / n[THINK_END]:.4f} | {sd[THINK_END] / max(nb[THINK_END], 1):+.5f} | {int(nb.sum())} | {nb.sum() / n.sum():.4f} | {sd.sum() / max(nb.sum(), 1):+.5f} |")
            L.append("\nby lag class (trainer version at the advantage stage minus the rollout's START weight version; class 1 = rollout finished before the next refit, class 2 = rollout straddled a refit so its later turns were generated under the newer weights)")
            L.append("| class | all tokens n | share | d(all) | </think> n | d(</think>) | </think> uncertain (gen_lp<-0.1) n | d | all uncertain n | d |"); L.append("|---|---|---|---|---|---|---|---|---|---|")
            LAGNAME = {0: "0 same weights", 1: "1 clean", 2: "1 straddled refit", 3: "other/unknown"}
            for l in range(4):
                nl = tot[f"n_lag{l}"]; sdl = tot[f"sum_d_lag{l}"]
                te_u_n = tot[f"lag{l}_b0_te_n"] + tot[f"lag{l}_b1_te_n"]; te_u_d = tot[f"lag{l}_b0_te_sum_d"] + tot[f"lag{l}_b1_te_sum_d"]
                all_u_n = tot[f"lag{l}_b0_all_n"] + tot[f"lag{l}_b1_all_n"]; all_u_d = tot[f"lag{l}_b0_all_sum_d"] + tot[f"lag{l}_b1_all_sum_d"]
                if nl.sum() == 0: continue
                L.append(f"| {LAGNAME[l]} | {int(nl.sum())} | {nl.sum() / n.sum():.3f} | {sdl.sum() / nl.sum():+.5f} | {int(nl[THINK_END])} | {sdl[THINK_END] / max(nl[THINK_END], 1):+.5f} | {int(te_u_n)} | {te_u_d / max(te_u_n, 1):+.5f} | {int(all_u_n)} | {all_u_d / max(all_u_n, 1):+.5f} |")
            L.append("\n`</think>` by think length (tokens before the close): n, mean d, mean gen_lp")
            L.append("| think length | n | mean d | mean gen_lp |"); L.append("|---|---|---|---|")
            for b, nm in enumerate(["<100", "100-1k", "1k-4k", ">4k"]):
                nn = tot["te_tl_n"][b]
                L.append(f"| {nm} | {int(nn)} | {tot['te_tl_sum_d'][b] / max(nn, 1):+.5f} | {tot['te_tl_sum_gen'][b] / max(nn, 1):+.4f} |")
    rep = "\n".join(L) + "\n"; open(f"{W}/analysis/token_bias_report.md", "w").write(rep); print(rep)


if __name__ == "__main__":
    main()

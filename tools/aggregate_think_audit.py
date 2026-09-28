"""Aggregate think_stop_audit outputs per run/step into analysis/think_stop_report.md."""
from __future__ import annotations

import glob, json, os, sys
from collections import Counter, defaultdict

import numpy as np

W = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump"
IN = f"{W}/analysis/think_stop"
TOK = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf/tokenizer.json"
SHORT = {"nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918": "runA_vllm", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918": "runB_minf",
         "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-nopg-noprefix-20260919": "chainC_minf", "nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919": "chainD_vllm",
         "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-20260920": "chainE_minf", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920": "chainF_minf_fromvllm10", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920": "chainG_minf_from0_nopg", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-20260921": "chainI_minf_from0_prefix_r1", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921": "chainJ_minf_from0_prefix_r2", "nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-seed1234-20260922": "chainK_vllm_from0_seed1234", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromk10-prefix-nopg-20260923": "chainL_minf_fromk10", "nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923": "chainM_vllm_from0_nopg_r1", "nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r2-20260923": "chainN_vllm_from0_nopg_r2", "nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r3-20260923": "chainO_minf_from0_prefix_r3"}


def vocab():
    j = json.load(open(TOK)); inv = {v: k for k, v in j["model"]["vocab"].items()}
    for t in j.get("added_tokens", []):
        inv[t["id"]] = t["content"]
    return inv


def main():
    inv = vocab()
    dec = lambda ids: "".join(inv.get(int(i), f"<{i}>") for i in ids).replace("Ċ", "\\n").replace("Ġ", " ")
    per = defaultdict(lambda: dict(js=[], tl=[], teg=[], tep=[], ieg=[], iep=[]))
    for f in sorted(glob.glob(f"{IN}/*.thinkaudit.json")):
        j = json.load(open(f)); key = (SHORT.get(j["run"], j["run"]), j["step"])
        per[key]["js"].append(j)
        z = np.load(f[:-5] + ".npz")
        per[key]["tl"].append(z["think_len"]); per[key]["teg"].append(z["think_end_gen_lp"]); per[key]["tep"].append(z["think_end_prev_lp"])
        per[key]["ieg"].append(z["im_end_gen_lp"]); per[key]["iep"].append(z["im_end_prev_lp"])
    lines = ["# Generation-mask / stop-token / think-close audit", "",
             "Span = maximal run of token_mask==1 (one assistant turn). think_len = tokens from span start to the first `</think>` inclusive.", ""]
    lines.append("| run | step | seqs | spans | end=im_end | end!=im_end | >1 im_end | no im_end | no </think> | tool_call before </think> | <think> inside span | mask on im_start | trunc seqs | think_len med / p95 / max | think>8k share | </think> gen_lp mean / med / p10 | </think> prev-gen | im_end gen_lp mean |")
    lines.append("|" + "---|" * 18)
    pats = defaultdict(lambda: dict(before=Counter(), after=Counter(), end=Counter(), after_think=Counter()))
    for (run, step), d in sorted(per.items()):
        js = d["js"]; S = lambda k: sum(j[k] for j in js)
        tl = np.concatenate(d["tl"]); tl_ok = tl[tl > 0]
        teg = np.concatenate(d["teg"]); teg = teg[~np.isnan(teg)]; tep = np.concatenate(d["tep"]); tep = tep[~np.isnan(tep)]
        ieg = np.concatenate(d["ieg"]); ieg = ieg[~np.isnan(ieg)]
        n_sp = S("n_spans")
        for j in js:
            for k, c in j["before"]: pats[run]["before"][tuple(k)] += c
            for k, c in j["after"]: pats[run]["after"][tuple(k)] += c
            for k, c in j["end_tok"]: pats[run]["end"][k] += c
            for k, c in j["after_think_first_tok"]: pats[run]["after_think"][k] += c
        end_im = sum(c for k, c in Counter(dict((k, c) for j in js for k, c in j["end_tok"])).items() if k == 11)
        end_im = sum(c for j in js for k, c in j["end_tok"] if k == 11)
        lines.append(f"| {run} | {step} | {S('n_seq')} | {n_sp} | {end_im / n_sp:.4f} | {S('spans_end_not_im_end')} | {S('spans_multi_im_end')} | {S('spans_no_im_end')} | "
                     f"{S('spans_no_think_end')} ({S('spans_no_think_end') / n_sp:.4f}) | {S('spans_toolcall_before_think_end')} | {S('spans_think_tok_inside')} | {S('mask_on_im_start')} | {S('seq_trunc')} | "
                     f"{np.median(tl_ok):.0f} / {np.percentile(tl_ok, 95):.0f} / {tl_ok.max()} | {(tl_ok > 8192).mean():.4f} | "
                     f"{teg.mean():+.4f} / {np.median(teg):+.4f} / {np.percentile(teg, 10):+.4f} | {(tep - teg).mean():+.5f} | {ieg.mean():+.5f} |")
    lines.append("")
    lines.append("## Boundary patterns (all steps pooled)")
    for run, p in sorted(pats.items()):
        lines.append(f"\n### {run}")
        lines.append("tokens just before a span: " + "; ".join(f"{dec(k)!r} x{c}" for k, c in p['before'].most_common(4)))
        lines.append("tokens just after a span: " + "; ".join(f"{dec(k)!r} x{c}" for k, c in p['after'].most_common(4)))
        lines.append("last token of a span: " + "; ".join(f"{dec([k])!r} x{c}" for k, c in p['end'].most_common(5)))
        lines.append("first token after </think>: " + "; ".join(f"{dec([k])!r} x{c}" for k, c in p['after_think'].most_common(4)))
    rep = "\n".join(lines) + "\n"
    open(f"{W}/analysis/think_stop_report.md", "w").write(rep)
    print(rep)


if __name__ == "__main__":
    main()

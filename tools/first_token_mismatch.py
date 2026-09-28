"""Trainer-vs-engine logprob mismatch by position within each assistant turn.

A turn (span) is a maximal run of token_mask == 1. For every span the first generated token's
engine logprob (generation_logprobs) and trainer logprob (prev_logprobs) are recorded, and
per-position statistics (positions 1..8 within the span, plus the rest) are accumulated per lag
class (0 = trainer version == rollout weight version, 1 = one refit behind and not straddling,
2 = straddled a refit, 3 = other). d = trainer logprob - engine logprob.
Writes analysis/first_token/<run>__<chunk>.ft.npz and analysis/first_token_report.md.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pt_numpy import load  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
RUNS = {
    "runA_vllm": ("nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918", "run A (vLLM from scratch)"),
    "runB_minf": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918", "run B (MINF from MINF step 10)"),
    "chainC_minf": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-nopg-noprefix-20260919", "chain C (MINF from MINF step 10, no prefix cache)"),
    "chainD_vllm": ("nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919", "chain D (vLLM from MINF step 10)"),
    "chainE_minf": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-20260920", "chain E (MINF from scratch, prefix cache on)"),
    "chainF_minf_fromvllm10": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920", "chain F (MINF from vLLM step 10)"),
    "chainG_minf_from0_nopg": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920", "chain G (MINF from scratch, no prefix cache)"),
}
EXP2RUN = {exp: run for run, (exp, _) in RUNS.items()}
NPOS = 8  # positions 1..8 tracked individually; index 8 = rest of the span
OUT = os.path.join(W, "analysis", "first_token")
BIN_EDGES = [0.05, 0.3, 1.0, 3.0]  # on -engine logprob: P>0.951, 0.741-0.951, 0.368-0.741, 0.050-0.368, <0.050
BIN_LABELS = ["P > 0.951", "0.741 < P <= 0.951", "0.368 < P <= 0.741", "0.050 < P <= 0.368", "P <= 0.050"]


def _acc(binned, lag, cls, gv, dv):
    b = np.digitize(-gv, BIN_EDGES)
    np.add.at(binned[lag, cls], (b, 0), 1); np.add.at(binned[lag, cls], (b, 1), dv)
    np.add.at(binned[lag, cls], (b, 2), np.abs(dv)); np.add.at(binned[lag, cls], (b, 3), (np.abs(dv) > 1).astype(float))


def process(path):
    exp = path.split("/runs/")[1].split("/")[0]; run = EXP2RUN[exp]
    name = os.path.basename(path)[:-3]; step = int(name.split("_")[1])
    out_path = os.path.join(OUT, f"{run}__{name}.ft.npz")
    if os.path.exists(out_path) and os.path.getmtime(out_path) > os.path.getmtime(path):
        return path, "cached"
    wv_map = json.load(open(os.path.join(W, "analysis", "token_bias", "wv_map.json")))
    d = load(path)
    lengths = np.asarray(d["input_lengths"]).astype(np.int64)
    ids_off = np.concatenate([[0], np.cumsum(lengths)]); lp_off = np.concatenate([[0], np.cumsum(lengths - 1)])
    ids_all = np.asarray(d["input_ids"]).astype(np.int64); mask_all = np.asarray(d["token_mask"]).astype(bool)
    gen_all = np.asarray(d["generation_logprobs"]).astype(np.float64); prev_all = np.asarray(d["prev_logprobs"]).astype(np.float64)
    trainer_v = d.get("trainer_version"); tags = d.get("tags") or [{}] * len(lengths); sids = list(d["sample_ids"])
    first_g, first_p, first_tok, first_lag, span_len, first_prev_tok = [], [], [], [], [], []
    pos = np.zeros((4, NPOS + 1, 4))  # lag x position x [n, sum d, sum |d|, n |d|>1]
    binned = np.zeros((4, 3, len(BIN_EDGES) + 1, 4))  # lag x {pos1, pos2-8, rest} x engine-lp bin x [n, sum d, sum |d|, n |d|>1]
    for i in range(len(lengths)):
        wv = (tags[i] or {}).get("weight_version"); se = wv_map.get(sids[i])
        raw = None if trainer_v is None or wv is None else int(trainer_v) - int(wv)
        lag = 0 if raw == 0 else (1 if raw == 1 and se is not None and se[0] == se[1] else (2 if raw == 1 and se is not None else 3))
        m = mask_all[ids_off[i] + 1:ids_off[i + 1]]; tok = ids_all[ids_off[i] + 1:ids_off[i + 1]]
        prev_tok = ids_all[ids_off[i]:ids_off[i + 1] - 1]
        g = gen_all[lp_off[i]:lp_off[i + 1]]; p = prev_all[lp_off[i]:lp_off[i + 1]]
        if not m.any():
            continue
        starts = np.flatnonzero(m & ~np.concatenate(([False], m[:-1])))
        ends = np.flatnonzero(m & ~np.concatenate((m[1:], [False])))
        dd = p - g
        first_g.append(g[starts]); first_p.append(p[starts]); first_tok.append(tok[starts]); first_prev_tok.append(prev_tok[starts])
        first_lag.append(np.full(len(starts), lag, np.int8)); span_len.append(ends - starts + 1)
        covered = np.zeros(len(m), bool)
        for k in range(NPOS):
            idx = starts + k; ok = idx <= ends; idx = idx[ok]
            if idx.size:
                v = dd[idx]; pos[lag, k] += [idx.size, v.sum(), np.abs(v).sum(), (np.abs(v) > 1).sum()]; covered[idx] = True
                _acc(binned, lag, 0 if k == 0 else 1, g[idx], v)
        rest = m & ~covered
        if rest.any():
            v = dd[rest]; pos[lag, NPOS] += [rest.sum(), v.sum(), np.abs(v).sum(), (np.abs(v) > 1).sum()]
            _acc(binned, lag, 2, g[rest], v)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(out_path, step=np.array(step), first_g=np.concatenate(first_g), first_p=np.concatenate(first_p),
                        first_tok=np.concatenate(first_tok), first_prev_tok=np.concatenate(first_prev_tok),
                        first_lag=np.concatenate(first_lag), span_len=np.concatenate(span_len), pos=pos, binned=binned)
    return path, "done"


def aggregate():
    per_run_lag = {}; per_step = {}; tok_counter = {}; binned_acc = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*.ft.npz"))):
        run = os.path.basename(f).split("__")[0]; z = np.load(f); step = int(z["step"])
        g, p, lag, tok, ptok = z["first_g"], z["first_p"], z["first_lag"], z["first_tok"], z["first_prev_tok"]
        d = p - g
        for L in (0, 1, 2, 3):
            sel = lag == L
            if not sel.any():
                continue
            a = per_run_lag.setdefault((run, L), {"n": 0, "sd": 0.0, "sad": 0.0, "n1": 0, "n05": 0, "sg": 0.0, "sp": 0.0, "pos": np.zeros((NPOS + 1, 4)), "n_gpos": 0, "n_gneg": 0})
            a["n"] += int(sel.sum()); a["sd"] += float(d[sel].sum()); a["sad"] += float(np.abs(d[sel]).sum()); a["n1"] += int((np.abs(d[sel]) > 1).sum()); a["n05"] += int((np.abs(d[sel]) > 0.5).sum())
            a["sg"] += float(g[sel].sum()); a["sp"] += float(p[sel].sum()); a["pos"] += z["pos"][L]; a["n_gpos"] += int((d[sel] > 0).sum()); a["n_gneg"] += int((d[sel] < 0).sum())
            s = per_step.setdefault((run, step, L), {"n": 0, "sd": 0.0, "sad": 0.0, "n1": 0, "sg": 0.0, "rest_n": 0, "rest_sad": 0.0, "rest_sd": 0.0, "rest_n1": 0})
            s["n"] += int(sel.sum()); s["sd"] += float(d[sel].sum()); s["sad"] += float(np.abs(d[sel]).sum()); s["n1"] += int((np.abs(d[sel]) > 1).sum()); s["sg"] += float(g[sel].sum())
            r = z["pos"][L][NPOS]; s["rest_n"] += r[0]; s["rest_sd"] += r[1]; s["rest_sad"] += r[2]; s["rest_n1"] += r[3]
        c = tok_counter.setdefault(run, Counter()); c.update(Counter(zip(ptok.tolist(), tok.tolist())))
        if "binned" in z.files:
            for L in (0, 1, 2, 3):
                binned_acc[(run, L)] = binned_acc.get((run, L), 0) + z["binned"][L]
    lines = ["# Trainer-vs-engine logprob mismatch at the first generated token of each assistant turn", "",
             "d = trainer logprob minus engine logprob of the sampled token. Span = maximal run of generated tokens = one assistant turn. "
             "Position 1 = first generated token of the turn (its logits come from the prefill of the prompt/tool-output prefix; MINF: chunked prefill + prefix cache, vLLM: prefix cache), positions 2+ come from decode steps. "
             "Lag 0 = trainer and engine at identical weights (pure numerical mismatch); lag 1 = engine one refit behind the trainer (includes the policy update); lag 2 = rollout straddled a refit. "
             "Pooled over all audited steps of each run.", ""]
    lines += ["## First token of each turn versus the rest of the turn (pooled over steps)", "",
              "| run | lag | turns | mean d (pos 1) | mean abs d (pos 1) | share abs d > 0.5 | share abs d > 1 | d > 0 : d < 0 | engine lp mean (pos 1) | trainer lp mean (pos 1) | mean abs d (pos 2) | mean abs d (pos 3-8) | mean abs d (rest) | share abs d > 1 (rest) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (run, L), a in sorted(per_run_lag.items(), key=lambda kv: (list(RUNS).index(kv[0][0]), kv[0][1])):
        if a["n"] < 100:
            continue
        P = a["pos"]; p2 = P[1]; p38 = P[2:NPOS].sum(0); rest = P[NPOS]
        lines.append(f"| {RUNS[run][1]} | {L} | {a['n']} | {a['sd'] / a['n']:+.4f} | {a['sad'] / a['n']:.4f} | {100 * a['n05'] / a['n']:.2f}% | {100 * a['n1'] / a['n']:.2f}% | {a['n_gpos']}:{a['n_gneg']} | {a['sg'] / a['n']:.4f} | {a['sp'] / a['n']:.4f} | "
                     f"{p2[2] / max(p2[0], 1):.4f} | {p38[2] / max(p38[0], 1):.4f} | {rest[2] / max(rest[0], 1):.4f} | {100 * rest[3] / max(rest[0], 1):.2f}% |")
    lines += ["", "## Mean abs d by position within the turn (pooled over steps)", "", "| run | lag | " + " | ".join(f"pos {k}" for k in range(1, NPOS + 1)) + " | rest |", "|---|---|" + "---|" * (NPOS + 1)]
    for (run, L), a in sorted(per_run_lag.items(), key=lambda kv: (list(RUNS).index(kv[0][0]), kv[0][1])):
        if a["n"] < 100:
            continue
        P = a["pos"]
        lines.append(f"| {RUNS[run][1]} | {L} | " + " | ".join(f"{P[k][2] / max(P[k][0], 1):.4f}" for k in range(NPOS + 1)) + " |")
    lines += ["", "## Per step, lag 1 (engine one refit behind) and lag 0 where available", "",
              "| run | step | lag | turns | mean d (pos 1) | mean abs d (pos 1) | share abs d > 1 (pos 1) | engine lp mean (pos 1) | mean abs d (rest) | share abs d > 1 (rest) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (run, step, L), s in sorted(per_step.items(), key=lambda kv: (list(RUNS).index(kv[0][0]), kv[0][1], kv[0][2])):
        if L not in (0, 1) or s["n"] < 50:
            continue
        lines.append(f"| {RUNS[run][1]} | {step} | {L} | {s['n']} | {s['sd'] / s['n']:+.4f} | {s['sad'] / s['n']:.4f} | {100 * s['n1'] / s['n']:.2f}% | {s['sg'] / s['n']:.4f} | {s['rest_sad'] / max(s['rest_n'], 1):.4f} | {100 * s['rest_n1'] / max(s['rest_n'], 1):.2f}% |")
    lines += ["", "## Mean abs d by engine probability of the sampled token (controls for entropy), lag 0 and lag 1 pooled over steps", "",
              "| run | lag | engine P bin | n pos 1 | mean abs d pos 1 | mean d pos 1 | n pos 2-8 | mean abs d pos 2-8 | n rest | mean abs d rest | mean d rest |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (run, L), B in sorted(binned_acc.items(), key=lambda kv: (list(RUNS).index(kv[0][0]), kv[0][1])):
        if L not in (0, 1) or not isinstance(B, np.ndarray):
            continue
        for bi, lab in enumerate(BIN_LABELS):
            c1, c2, c3 = B[0, bi], B[1, bi], B[2, bi]
            if c1[0] < 100:
                continue
            lines.append(f"| {RUNS[run][1]} | {L} | {lab} | {int(c1[0])} | {c1[2] / c1[0]:.4f} | {c1[1] / c1[0]:+.4f} | {int(c2[0])} | {c2[2] / max(c2[0], 1):.4f} | {int(c3[0])} | {c3[2] / max(c3[0], 1):.4f} | {c3[1] / max(c3[0], 1):+.4f} |")
    lines += ["", "## Most common (last prompt token id, first generated token id) pairs per run", ""]
    for run, c in tok_counter.items():
        tot = sum(c.values())
        lines.append(f"- {RUNS[run][1]}: " + ", ".join(f"({a}->{b}) {100 * n / tot:.1f}%" for (a, b), n in c.most_common(6)))
    lines.append("")
    open(os.path.join(W, "analysis", "first_token_report.md"), "w").write("\n".join(lines))
    print("wrote analysis/first_token_report.md")


def main():
    files = []
    for exp in EXP2RUN:
        files += sorted(glob.glob(f"{U}/{exp}/dumps/token_level/step_*_chunk_*.pt"))
    with Pool(int(os.environ.get("NPROC", 16))) as pool:
        for path, status in pool.imap_unordered(process, files):
            print(status, path.split("/runs/")[1], flush=True)
    aggregate()


if __name__ == "__main__":
    main()

"""Per-step mean generation length next to sampled-</think> probability statistics.

Reads analysis/think_close_by_step.md, analysis/think_stop_report.md and the
token-level row summaries under analysis/tokens; writes analysis/close_vs_length.md
and analysis/close_vs_length.svg (dependency-free SVG).
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import sys

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.path.join(W, "analysis")
RUNS = {
    "runA_vllm": ("nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918", "run A: vLLM engine, from scratch", "#1f77b4"),
    "runB_minf": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918", "run B: MINF engine, from MINF step_10", "#d62728"),
    "chainD_vllm": ("nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919", "chain D: vLLM engine, from MINF step_10", "#ff7f0e"),
    "chainF_minf_fromvllm10": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920", "chain F: MINF engine, from vLLM step_10", "#9467bd"),
    "chainC_minf": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-nopg-noprefix-20260919", "chain C: MINF engine, from MINF step_10, no prefix cache / no overlap", "#7f7f7f"),
    "chainG_minf_from0_nopg": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920", "chain G: MINF engine, from scratch, no prefix cache / no overlap", "#8c564b"),
    "chainE_minf": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-20260920", "chain E: MINF engine, from scratch (cancelled)", "#bcbd22"),
    "chainI_minf_from0_prefix_r1": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-20260921", "chain I: MINF engine, from scratch, prefix cache on / no overlap, replica 1", "#17becf"),
    "chainJ_minf_from0_prefix_r2": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921", "chain J: MINF engine, from scratch, prefix cache on / no overlap, replica 2", "#e377c2"),
    "chainK_vllm_from0_seed1234": ("nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-seed1234-20260922", "chain K: vLLM engine, from scratch, seed 1234", "#17becf"),
    "chainL_minf_fromk10": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromk10-prefix-nopg-20260923", "chain L: MINF engine, from chain K (vLLM seed 1234) step_10, prefix cache on / no overlap", "#8c564b"),
    "chainM_vllm_from0_nopg_r1": ("nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923", "chain M: vLLM engine, from scratch, no overlap, replica 1", "#ff7f0e"),
    "chainN_vllm_from0_nopg_r2": ("nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r2-20260923", "chain N: vLLM engine, from scratch, no overlap, replica 2", "#9467bd"),
    "chainO_minf_from0_prefix_r3": ("nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r3-20260923", "chain O: MINF engine, from scratch, prefix cache on / no overlap, replica 3", "#bcbd22"),
}


def read_close():
    out = {}
    for line in open(os.path.join(AN, "think_close_by_step.md")):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0] not in RUNS or not cells[1].isdigit():
            continue
        mean_lp, p10_lp = [float(x) for x in cells[4].split("/")]
        out[(cells[0], int(cells[1]))] = {
            "closes": int(cells[2]), "low_share": float(cells[3]), "lp_mean": mean_lp, "lp_p10": p10_lp}
    return out


def read_think_p95():
    out = {}
    for line in open(os.path.join(AN, "think_stop_report.md")):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 15 or cells[0] not in RUNS or not cells[1].isdigit():
            continue
        m = re.match(r"(\d+) / (\d+) / (\d+)", cells[13])
        if m:
            out[(cells[0], int(cells[1]))] = int(m.group(2))
    return out


def read_lengths():
    out = {}
    for short, (exp, _, _) in RUNS.items():
        steps = {}
        for f in glob.glob(os.path.join(AN, "tokens", f"{exp}__step_*_chunk_*.rows.csv")):
            step = int(re.search(r"step_(\d+)_chunk", f).group(1))
            for row in csv.DictReader(open(f)):
                try:
                    L = float(row["rollout_generation_length"]); r = float(row["reward"])
                except (ValueError, KeyError):
                    continue
                t = str(row.get("rollout_truncated")).lower() in ("true", "1")
                s = steps.setdefault(step, [0.0, 0.0, 0, 0])
                s[0] += L; s[1] += r; s[2] += t; s[3] += 1
        for step, (L, r, t, n) in steps.items():
            if n:
                out[(short, step)] = {"n": n, "len": L / n, "reward": r / n, "trunc": t / n}
    return out


def svg(series, panels, path):
    width, ph, left, right, top, gap = 1000, 230, 90, 30, 40, 50
    height = top + len(panels) * (ph + gap) + 60
    xs = sorted({x for s in series.values() for x in s})
    x0, x1 = min(xs), max(xs)
    def X(x):
        return left + (x - x0) / max(1, x1 - x0) * (width - left - right)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="13">',
             f'<rect width="{width}" height="{height}" fill="white"/>']
    for pi, (key, title, fmt, ylo, yhi) in enumerate(panels):
        py = top + pi * (ph + gap)
        vals = [v[key] for s in series.values() for v in s.values() if v.get(key) is not None]
        lo = ylo if ylo is not None else min(vals)
        hi = yhi if yhi is not None else max(vals)
        pad = (hi - lo) * 0.08 or 1
        lo, hi = lo - pad, hi + pad
        def Y(y, lo=lo, hi=hi, py=py):
            return py + ph - (y - lo) / (hi - lo) * ph
        parts.append(f'<text x="{left}" y="{py - 12}" font-size="15" font-weight="bold">{title}</text>')
        parts.append(f'<rect x="{left}" y="{py}" width="{width - left - right}" height="{ph}" fill="none" stroke="#999"/>')
        for k in range(6):
            y = lo + (hi - lo) * k / 5
            parts.append(f'<line x1="{left}" x2="{width - right}" y1="{Y(y):.1f}" y2="{Y(y):.1f}" stroke="#eee"/>')
            parts.append(f'<text x="{left - 8}" y="{Y(y) + 4:.1f}" text-anchor="end">{fmt(y)}</text>')
        for x in xs:
            parts.append(f'<text x="{X(x):.1f}" y="{py + ph + 16}" text-anchor="middle" font-size="11">{x}</text>')
            parts.append(f'<line x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{py}" y2="{py + ph}" stroke="#f3f3f3"/>')
        parts.append(f'<text x="{(left + width - right) / 2}" y="{py + ph + 32}" text-anchor="middle">train step</text>')
        for short, s in series.items():
            color = RUNS[short][2]
            pts = [(X(x), Y(v[key])) for x, v in sorted(s.items()) if v.get(key) is not None]
            if not pts:
                continue
            d = " ".join(f"{px:.1f},{py_:.1f}" for px, py_ in pts)
            parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
            for px, py_ in pts:
                parts.append(f'<circle cx="{px:.1f}" cy="{py_:.1f}" r="3" fill="{color}"/>')
    ly = height - 45
    lx = left
    for short in series:
        label = RUNS[short][1]
        parts.append(f'<line x1="{lx}" x2="{lx + 24}" y1="{ly}" y2="{ly}" stroke="{RUNS[short][2]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 30}" y="{ly + 4}">{label}</text>')
        lx += 30 + 7 * len(label) + 20
        if lx > width - 250:
            lx = left; ly += 20
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))


def main():
    close, p95, lengths = read_close(), read_think_p95(), read_lengths()
    series = {}
    for short in RUNS:
        s = {}
        for (r, step), c in close.items():
            if r != short:
                continue
            d = {"geo_p": math.exp(c["lp_mean"]), "p10": math.exp(c["lp_p10"]), "low_pct": 100 * c["low_share"],
                 "closes": c["closes"], "think_p95": p95.get((short, step))}
            d.update(lengths.get((short, step), {}))
            s[step] = d
        if s:
            series[short] = s
    order = [k for k in ["runA_vllm", "runB_minf", "chainD_vllm", "chainF_minf_fromvllm10", "chainC_minf", "chainG_minf_from0_nopg", "chainI_minf_from0_prefix_r1", "chainJ_minf_from0_prefix_r2", "chainK_vllm_from0_seed1234", "chainL_minf_fromk10", "chainM_vllm_from0_nopg_r1", "chainN_vllm_from0_nopg_r2", "chainO_minf_from0_prefix_r3", "chainE_minf"] if k in series]
    series = {k: series[k] for k in order}
    lines = ["# Mean generation length vs sampled-</think> probability, per train step", "",
             "Sources: analysis/think_close_by_step.md (engine logprob of every sampled </think> in the 512 trained rollouts of the step), "
             "analysis/think_stop_report.md (thinking tokens per assistant turn), analysis/tokens/*.rows.csv (rollout_generation_length, reward, rollout_truncated tags).", "",
             "Definitions: len = mean generated tokens per rollout; geo-mean P = exp(mean engine logprob of the sampled </think>); "
             "10th-pct P = exp(10th percentile of those logprobs); low share = fraction of sampled </think> with engine logprob < -0.1 (P < 0.905); "
             "think p95 = 95th percentile of thinking tokens per assistant turn; trunc = share of rollouts that hit the context window.", ""]
    for short in order:
        lines += [f"## {RUNS[short][1]}  ({short})", "",
                  "| step | n rollouts | len | reward | trunc | closes | geo-mean P | 10th-pct P | low share | think p95 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for step, d in sorted(series[short].items()):
            lines.append(f"| {step} | {d.get('n', '-')} | {d['len']:.0f} | {d['reward']:.3f} | {100 * d['trunc']:.1f}% | {d['closes']} | {d['geo_p']:.3f} | {d['p10']:.3f} | {d['low_pct']:.1f}% | {d['think_p95']} |"
                         if "len" in d else f"| {step} | - | - | - | - | {d['closes']} | {d['geo_p']:.3f} | {d['p10']:.3f} | {d['low_pct']:.1f}% | {d['think_p95']} |")
        lines.append("")
    trio = ["runA_vllm", "runB_minf", "chainD_vllm", "chainF_minf_fromvllm10"]
    lines += ["## Side by side, steps 11-24 (len in tokens; geo-mean P of the sampled </think>; low share in %)", "",
              "| step | len A | len B | len D | len F | P A | P B | P D | P F | low% A | low% B | low% D | low% F |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for step in range(11, 25):
        row = [str(step)]
        for key, f in (("len", "{:.0f}"), ("geo_p", "{:.3f}"), ("low_pct", "{:.1f}")):
            for short in trio:
                d = series.get(short, {}).get(step)
                row.append(f.format(d[key]) if d and d.get(key) is not None else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    open(os.path.join(AN, "close_vs_length.md"), "w").write("\n".join(lines))
    panels = [("len", "Mean generated tokens per rollout", lambda y: f"{y / 1000:.0f}k", None, None),
              ("geo_p", "Geometric-mean engine probability of the sampled </think>", lambda y: f"{y:.3f}", None, None),
              ("low_pct", "Share of sampled </think> with engine P < 0.905 (%)", lambda y: f"{y:.1f}", None, None),
              ("think_p95", "Thinking tokens per assistant turn, 95th percentile", lambda y: f"{y:.0f}", None, None)]
    svg(series, panels, os.path.join(AN, "close_vs_length.svg"))
    print("wrote", os.path.join(AN, "close_vs_length.md"), "and .svg")


if __name__ == "__main__":
    main()

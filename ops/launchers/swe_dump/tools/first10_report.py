"""First-ten-steps distribution study, five arms: run A (vLLM from scratch), chain G (MINF from scratch, no prefix cache),
chain I (MINF from scratch, prefix cache on, replica 1), chain J (MINF from scratch, prefix cache on, replica 2),
chain K (vLLM from scratch, seed 1234). Paired comparisons use run A (vLLM from scratch) as the reference, one block per other arm.

Stages:
  collect  numpy only; joins token-level rows with rollout summaries, scans the token dumps for per-token,
           per-span and per-rollout statistics, parses trainer metrics from the SingleController logs;
           writes analysis/first10/{rollouts.csv,spans_<arm>_<step>.npz,hists.npz,sc_metrics.json}
  plot     matplotlib; reads the collect outputs, computes matched-prompt statistics with cluster bootstraps
           and writes analysis/first10/first10_report.pdf, page PNGs and report.md
Usage: first10_report.py collect|plot|all   (NPROC, NBOOT env)
"""
from __future__ import annotations

import csv, glob, json, math, os, re, sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pt_numpy import load  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
OUT = f"{W}/analysis/first10"
STEPS = list(range(1, 11))
THINK_END = 13
GROUP = 16
MAX_TURNS = 200
ARMS = {
    "A": dict(exp="nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918", label="run A (vLLM from scratch)", color="#1f77b4",
              sc_logs=["ray_logs/3847571-logs", "ray_logs/3847610-logs"]),
    "G": dict(exp="nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920",
              label="chain G (MINF from scratch, no prefix cache)", color="#d62728", sc_logs=["ray_logs/3880119-logs", "ray_logs/3880142-logs"]),
    "I": dict(exp="nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-20260921",
              label="chain I (MINF from scratch, prefix cache on, replica 1)", color="#17becf", sc_logs=["ray_logs/3901530-1-logs"]),
    "J": dict(exp="nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921",
              label="chain J (MINF from scratch, prefix cache on, replica 2)", color="#e377c2", sc_logs=["ray_logs/3901613-logs"]),
    "K": dict(exp="nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-seed1234-20260922",
              label="chain K (vLLM from scratch, seed 1234)", color="#2ca02c", sc_logs=["ray_logs/3925335-1-logs"]),
}
REF = "A"
OTHERS = [a for a in ARMS if a != REF]
LP_EDGES = np.array([-np.inf, -30, -25, -20, -17.5, -15, -12.5, -10, -9, -8, -7, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1.25, -1,
                     -0.75, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.075, -0.05, -0.03, -0.02, -0.01, -0.005, -0.001, 1e-9])
ABSD_EDGES = np.array([0, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, np.inf])
SD_EDGES = np.array([-np.inf, -20, -10, -5, -2, -1, -0.5, -0.2, -0.1, -0.05, -0.02, -0.01, -0.001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, np.inf])
SC_KEYS = ["reward", "rollout_length/swe_agents_train/mean", "rollout_length/swe_agents_train/p50", "rollout_length/swe_agents_train/p95",
           "rollout_length/swe_agents_train/max", "rollout_length/swe_agents_train/truncation_rate", "gen_kl_error", "approx_entropy",
           "token_mult_prob_error", "mean_seq_mult_prob_error", "max_seq_mult_prob_error", "num_masked_seqs_by_logprob_error", "advantages/mean",
           "is_oob_ratio", "grad_norm", "loss", "probs_ratio", "sampling_importance_ratio", "policy_kl_error", "masked_correct_pct"]
ROLLOUT_COLS = ["arm", "step", "sample_id", "group_id", "instance_id", "reward", "gen_len", "input_len", "truncated", "adv", "sample_mask_after",
                "seq_mult_prob_error", "gen_lp_mean", "gen_lp_min", "prev_lp_mean", "lp_err_mean", "lp_err_p99", "lp_err_max", "n_err_gt1",
                "n_err_gt5", "n_genlp_lt_m10", "turns", "n_invalid_tool_call", "n_malformed_thinking", "n_gen_without_think_close",
                "n_gen_without_im_end", "agent_error_kind", "resp_incomplete", "max_asst_tokens", "n_function_call", "n_finish", "resolved",
                "patch_exists", "last_gen_has_im_end", "agent_timed_out", "n_spans", "n_closes", "sum_close_lp", "n_low_close", "sum_think",
                "max_think", "n_think_gt2k", "n_no_close", "max_span"]


# ----------------------------------------------------------------------------- collect
def read_summary(exp, step):
    path = f"{W}/analysis/rollouts/{exp}__target_step_{step - 1:05d}.summary.jsonl"
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("sample_id"):
            out[r["sample_id"]] = r
    return out


def read_token_rows(exp, step):
    out = {}
    for f in sorted(glob.glob(f"{W}/analysis/tokens/{exp}__step_{step:05d}_chunk_*.rows.csv")):
        for r in csv.DictReader(open(f)):
            out[r["sample_id"]] = r
    return out


def scan_chunk(args):
    arm, step, path = args
    d = load(path)
    lengths = np.asarray(d["input_lengths"]).astype(np.int64)
    ids_off = np.concatenate([[0], np.cumsum(lengths)]); lp_off = np.concatenate([[0], np.cumsum(lengths - 1)])
    ids_all = np.asarray(d["input_ids"]).astype(np.int64); mask_all = np.asarray(d["token_mask"]).astype(bool)
    gen_all = np.asarray(d["generation_logprobs"]).astype(np.float64); prev_all = np.asarray(d["prev_logprobs"]).astype(np.float64)
    sids = list(d["sample_ids"])
    m_all = np.zeros(gen_all.shape[0], dtype=bool)
    for i in range(len(lengths)):
        m_all[lp_off[i]:lp_off[i + 1]] = mask_all[ids_off[i] + 1:ids_off[i + 1]]
    g = gen_all[m_all]; p = prev_all[m_all]
    hist = dict(gen=np.histogram(g, LP_EDGES)[0], absd=np.histogram(np.abs(p - g), ABSD_EDGES)[0], sd=np.histogram(p - g, SD_EDGES)[0],
                n_tok=np.array([g.size]), sum_gen=np.array([g.sum()]), sum_d=np.array([(p - g).sum()]), sum_absd=np.array([np.abs(p - g).sum()]))
    per_roll = {}
    think_len, close_g, close_p, span_len, span_sid = [], [], [], [], []
    for i in range(len(lengths)):
        m = mask_all[ids_off[i] + 1:ids_off[i + 1]]; tok = ids_all[ids_off[i] + 1:ids_off[i + 1]]
        gg = gen_all[lp_off[i]:lp_off[i + 1]]; pp = prev_all[lp_off[i]:lp_off[i + 1]]
        dm = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        starts = np.where(dm == 1)[0]; ends = np.where(dm == -1)[0]
        n_closes = n_low = n_no = n_gt2k = 0; s_lp = s_think = 0.0; mx_think = mx_span = 0
        for s, e in zip(starts, ends):
            span = tok[s:e]; L = int(e - s); mx_span = max(mx_span, L)
            w = np.where(span == THINK_END)[0]
            if len(w):
                j = int(w[0]); tl = j + 1
                think_len.append(tl); close_g.append(float(gg[s + j])); close_p.append(float(pp[s + j]))
                n_closes += 1; s_lp += float(gg[s + j]); n_low += int(gg[s + j] < -0.1); s_think += tl; mx_think = max(mx_think, tl); n_gt2k += int(tl > 2000)
            else:
                think_len.append(-1); close_g.append(np.nan); close_p.append(np.nan); n_no += 1
            span_len.append(L); span_sid.append(i)
        per_roll[sids[i]] = (len(starts), n_closes, s_lp, n_low, s_think, mx_think, n_gt2k, n_no, mx_span)
    spans = dict(think_len=np.asarray(think_len, np.int32), close_gen=np.asarray(close_g, np.float32), close_prev=np.asarray(close_p, np.float32),
                 span_len=np.asarray(span_len, np.int32), span_sid=np.asarray(span_sid, np.int32), sids=np.asarray(sids))
    return arm, step, os.path.basename(path), hist, per_roll, spans


def parse_sc_metrics():
    out = {}
    for arm, a in ARMS.items():
        out[arm] = {}
        for rel in a["sc_logs"]:
            cands = glob.glob(f"{U}/{a['exp']}/{rel}/ray/session_*/logs/worker-*.out")
            sc = None
            for c in cands:
                with open(c, "rb") as fh:
                    head = fh.read(2_000_000)
                if b"SingleControllerActor" in head:
                    sc = c; break
            if sc is None:
                continue
            lines = open(sc, errors="replace").read().splitlines()
            for k, line in enumerate(lines):
                mm = re.search(r"train step (\d+)/", line)
                if not mm or k == 0:
                    continue
                step = int(mm.group(1)); prev = lines[k - 1]; rec = {}
                for key in SC_KEYS:
                    m2 = re.search(r"'(?:train/)?" + re.escape(key) + r"': ([-0-9.eE+]+|nan|inf)", prev)
                    if m2:
                        try:
                            rec[key] = float(m2.group(1))
                        except ValueError:
                            pass
                if rec:
                    out[arm][str(step)] = rec
    return out


def collect():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for arm, a in ARMS.items():
        for step in STEPS:
            jobs += [(arm, step, f) for f in sorted(glob.glob(f"{U}/{a['exp']}/dumps/token_level/step_{step:05d}_chunk_*.pt"))]
    print(f"scanning {len(jobs)} token chunks", flush=True)
    hists = defaultdict(lambda: None); per_roll = defaultdict(dict); spans = defaultdict(list)
    with Pool(int(os.environ.get("NPROC", 8))) as pool:
        for arm, step, name, h, pr, sp in pool.imap_unordered(scan_chunk, jobs):
            key = (arm, step)
            if hists[key] is None:
                hists[key] = h
            else:
                hists[key] = {k: hists[key][k] + h[k] for k in h}
            per_roll[key].update(pr); spans[key].append(sp)
            print(f"  done {arm} step {step} {name} tokens {int(h['n_tok'][0])}", flush=True)
    np.savez_compressed(f"{OUT}/hists.npz", lp_edges=LP_EDGES, absd_edges=ABSD_EDGES, sd_edges=SD_EDGES,
                        **{f"{arm}_{step}_{k}": v for (arm, step), h in hists.items() if h for k, v in h.items()})
    for (arm, step), lst in spans.items():
        offs = 0; sid_idx = []; sids = []
        for sp in lst:
            sid_idx.append(sp["span_sid"] + offs); sids.extend(sp["sids"].tolist()); offs += len(sp["sids"])
        np.savez_compressed(f"{OUT}/spans_{arm}_{step}.npz", think_len=np.concatenate([sp["think_len"] for sp in lst]),
                            close_gen=np.concatenate([sp["close_gen"] for sp in lst]), close_prev=np.concatenate([sp["close_prev"] for sp in lst]),
                            span_len=np.concatenate([sp["span_len"] for sp in lst]), span_sid=np.concatenate(sid_idx), sids=np.asarray(sids))
    with open(f"{OUT}/rollouts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROLLOUT_COLS); w.writeheader()
        for arm, a in ARMS.items():
            for step in STEPS:
                tok = read_token_rows(a["exp"], step); summ = read_summary(a["exp"], step); pr = per_roll[(arm, step)]
                miss = 0
                for sid, r in tok.items():
                    s = summ.get(sid)
                    if s is None:
                        miss += 1; s = {}
                    x = pr.get(sid, (0,) * 9)
                    w.writerow(dict(arm=arm, step=step, sample_id=sid, group_id=r["group_id"], instance_id=s.get("instance_id"), reward=r["reward"],
                                    gen_len=r["n_gen_tokens"], input_len=r["input_length"], truncated=int(r["rollout_truncated"] in ("True", "1", "1.0")),
                                    adv=r["adv"], sample_mask_after=r["sample_mask_after"], seq_mult_prob_error=r["seq_mult_prob_error"],
                                    gen_lp_mean=r["gen_lp_mean"], gen_lp_min=r["gen_lp_min"], prev_lp_mean=r["prev_lp_mean"], lp_err_mean=r["lp_err_mean"],
                                    lp_err_p99=r["lp_err_p99"], lp_err_max=r["lp_err_max"], n_err_gt1=r["n_err_gt1"], n_err_gt5=r["n_err_gt5"],
                                    n_genlp_lt_m10=r["n_genlp_lt_m10"], turns=s.get("turns"), n_invalid_tool_call=s.get("n_invalid_tool_call"),
                                    n_malformed_thinking=s.get("n_malformed_thinking"), n_gen_without_think_close=s.get("n_gen_without_think_close"),
                                    n_gen_without_im_end=s.get("n_gen_without_im_end"), agent_error_kind=s.get("agent_error_kind") or "none",
                                    resp_incomplete=int(s.get("resp_incomplete") is not None), max_asst_tokens=s.get("max_asst_tokens"),
                                    n_function_call=s.get("n_function_call"), n_finish=s.get("n_finish"), resolved=s.get("resolved"),
                                    patch_exists=s.get("patch_exists"), last_gen_has_im_end=s.get("last_gen_has_im_end"), agent_timed_out=s.get("agent_timed_out"),
                                    n_spans=x[0], n_closes=x[1], sum_close_lp=x[2], n_low_close=x[3], sum_think=x[4], max_think=x[5], n_think_gt2k=x[6],
                                    n_no_close=x[7], max_span=x[8]))
                print(f"rollouts {arm} step {step}: {len(tok)} rows, {miss} without summary", flush=True)
    json.dump(parse_sc_metrics(), open(f"{OUT}/sc_metrics.json", "w"), indent=1)
    print("COLLECT DONE", flush=True)


# ----------------------------------------------------------------------------- stats helpers
NBOOT = int(os.environ.get("NBOOT", 2000))
rng = np.random.default_rng(20260921)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def cluster_boot_mean(vals, clusters, weights=None):
    """Mean of vals (optionally a ratio sum(vals)/sum(weights)) with a cluster bootstrap; returns mean, lo, hi."""
    vals = np.asarray(vals, float); clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True); k = len(uniq)
    wsum = np.bincount(inv, weights=np.ones_like(vals) if weights is None else np.asarray(weights, float), minlength=k)
    vsum = np.bincount(inv, weights=vals, minlength=k)
    idx = rng.integers(0, k, (NBOOT, k)); num = vsum[idx].sum(1); den = wsum[idx].sum(1)
    b = num[den > 0] / den[den > 0]
    return float(vsum.sum() / wsum.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_prompt_diff(dfA, dfG, col, agg=np.mean):
    """Per (step, instance) aggregate in each arm, paired difference (second arm minus first arm) with a pair bootstrap and a sign count."""
    a = defaultdict(list); g = defaultdict(list)
    for r in dfA:
        a[(r["step"], r["instance_id"])].append(r[col])
    for r in dfG:
        g[(r["step"], r["instance_id"])].append(r[col])
    keys = sorted(set(a) & set(g)); d = np.array([agg(g[k]) - agg(a[k]) for k in keys])
    if d.size == 0:
        return dict(n=0)
    idx = rng.integers(0, d.size, (NBOOT, d.size)); b = d[idx].mean(1)
    return dict(n=int(d.size), mean=float(d.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                pos=int((d > 0).sum()), neg=int((d < 0).sum()), per_step={s: float(np.mean([x for k, x in zip(keys, d) if k[0] == s])) for s in STEPS})


def ks_stat(x, y):
    x = np.sort(np.asarray(x, float)); y = np.sort(np.asarray(y, float)); allv = np.concatenate([x, y])
    cx = np.searchsorted(x, allv, side="right") / x.size; cy = np.searchsorted(y, allv, side="right") / y.size
    return float(np.abs(cx - cy).max())


def ks_perm_p(dfA, dfG, col, nperm=1000):
    """KS statistic between arms with a permutation p-value that swaps arm labels within matched (step, instance) pairs."""
    a = defaultdict(list); g = defaultdict(list)
    for r in dfA:
        a[(r["step"], r["instance_id"])].append(float(r[col]))
    for r in dfG:
        g[(r["step"], r["instance_id"])].append(float(r[col]))
    keys = sorted(set(a) & set(g)); A = [np.asarray(a[k]) for k in keys]; G = [np.asarray(g[k]) for k in keys]
    obs = ks_stat(np.concatenate(A), np.concatenate(G)); cnt = 0
    for _ in range(nperm):
        flip = rng.random(len(keys)) < 0.5
        xa = np.concatenate([G[i] if f else A[i] for i, f in enumerate(flip)]); xg = np.concatenate([A[i] if f else G[i] for i, f in enumerate(flip)])
        cnt += ks_stat(xa, xg) >= obs
    return obs, (cnt + 1) / (nperm + 1)


def pct(x, q):
    x = np.asarray(x, float)
    return float(np.percentile(x, q)) if x.size else float("nan")


# ----------------------------------------------------------------------------- plot
def load_rollouts():
    rows = list(csv.DictReader(open(f"{OUT}/rollouts.csv")))
    num = ["step", "reward", "gen_len", "input_len", "truncated", "adv", "sample_mask_after", "seq_mult_prob_error", "gen_lp_mean", "gen_lp_min",
           "prev_lp_mean", "lp_err_mean", "lp_err_p99", "lp_err_max", "n_err_gt1", "n_err_gt5", "n_genlp_lt_m10", "turns", "n_invalid_tool_call",
           "n_malformed_thinking", "n_gen_without_think_close", "n_gen_without_im_end", "resp_incomplete", "max_asst_tokens", "n_function_call",
           "n_finish", "n_spans", "n_closes", "sum_close_lp", "n_low_close", "sum_think", "max_think", "n_think_gt2k", "n_no_close", "max_span"]
    for r in rows:
        for k in num:
            v = r.get(k)
            try:
                r[k] = float(v) if v not in (None, "", "None") else float("nan")
            except ValueError:
                r[k] = float("nan")
        r["step"] = int(r["step"])
        r["resolved"] = r.get("resolved") in ("True", "1", "1.0")
    return rows


class _Dummy:
    """Stand-in for matplotlib objects so the statistics and markdown tables can be produced without matplotlib."""

    def __getattr__(self, name):
        return _Dummy()

    def __call__(self, *a, **k):
        return _Dummy()

    def __getitem__(self, key):
        return _Dummy()

    def __iter__(self):
        return iter(())


def plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        have_mpl = True
    except ImportError:
        have_mpl = False
        plt = _Dummy(); plt.subplots = lambda *a, **k: (_Dummy(), _Dummy()); PdfPages = lambda *a, **k: _Dummy()
        print("matplotlib not available: writing tables only", flush=True)
    def render(V):
        ARMS = V["arms"]; by = V["by"]; SP = V["SP"]; H = V["H"]; sc = V["sc"]; REF = V["ref"]; OTHERS = [a for a in ARMS if a != REF]
        C = {a: ARMS[a]["color"] for a in ARMS}; LBL = {a: ARMS[a]["label"] for a in ARMS}
        NA = len(ARMS); BW = 0.8 / NA
        off = lambda i: (i - (NA - 1) / 2) * BW
        sfx = V["suffix"]

        def finish(fig, name, title):
            if not have_mpl:
                return
            fig.suptitle(V["title_prefix"] + title, fontsize=13, fontweight="bold"); fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig); fig.savefig(f"{OUT}/{name}{sfx}.png", dpi=110); plt.close(fig); pages.append(name + sfx)

        def per_step(arm, col, fn=np.mean):
            return [fn([r[col] for r in by[arm] if r["step"] == s]) for s in STEPS]

        def prompt_ci(arm, col, fn=np.mean):
            lo, hi = [], []
            for s in STEPS:
                sub = [r for r in by[arm] if r["step"] == s]
                m, l, h = cluster_boot_mean([r[col] for r in sub], [r["instance_id"] for r in sub]); lo.append(l); hi.append(h)
            return np.array(lo), np.array(hi)

        md.extend(V["heading"])
        # ---- page 1: per-step overview
        fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
        panels = [("reward", "mean reward (share resolved)", None), ("gen_len", "mean generated tokens per rollout", None),
                  ("truncated", "context-window truncation rate", None), ("turns", "mean assistant turns per rollout", None)]
        for k, (col, ttl, _) in enumerate(panels):
            a = ax.flat[k]
            for arm in ARMS:
                y = per_step(arm, col); lo, hi = prompt_ci(arm, col)
                a.plot(STEPS, y, "o-", color=C[arm], label=LBL[arm]); a.fill_between(STEPS, lo, hi, color=C[arm], alpha=0.15)
            a.set_title(ttl); a.set_xlabel("train step"); a.set_xticks(STEPS); a.grid(alpha=0.3)
        a = ax.flat[4]
        for arm in ARMS:
            y = [sc.get(arm, {}).get(str(s), {}).get("gen_kl_error", np.nan) for s in STEPS]
            a.plot(STEPS, y, "o-", color=C[arm], label=LBL[arm])
        a.set_title("gen_kl_error (trainer, from controller log)" + V["sc_note"]); a.set_xlabel("train step"); a.set_xticks(STEPS); a.grid(alpha=0.3)
        a = ax.flat[5]
        for arm in ARMS:
            y = [sc.get(arm, {}).get(str(s), {}).get("approx_entropy", np.nan) for s in STEPS]
            a.plot(STEPS, y, "o-", color=C[arm], label=LBL[arm])
        a.set_title("approx_entropy (trainer, from controller log)" + V["sc_note"]); a.set_xlabel("train step"); a.set_xticks(STEPS); a.grid(alpha=0.3)
        ax.flat[0].legend(fontsize=8, loc="best")
        finish(fig, "p1_overview", "Per-step overview, steps 1-10 (bands = 95% prompt-cluster bootstrap CI of the step mean)")
        md.append("## Per-step trainer metrics (controller logs)" + V["sc_note"] + "\n")
        md.append("Cell order in every column: " + " / ".join(LBL[a] for a in ARMS) + ".\n")
        md.append("| step | reward | gen tokens (mean) | truncation rate | gen_kl_error | approx_entropy |")
        md.append("|---|---|---|---|---|---|")
        for s in STEPS:
            f = lambda arm, key, fmt: (fmt % sc[arm][str(s)][key]) if key in sc.get(arm, {}).get(str(s), {}) else "-"
            cell = lambda key, fmt: " / ".join(f(a, key, fmt) for a in ARMS)
            md.append(f"| {s} | {cell('reward','%.3f')} | {cell('rollout_length/swe_agents_train/mean','%.0f')} | {cell('rollout_length/swe_agents_train/truncation_rate','%.3f')} | "
                      f"{cell('gen_kl_error','%.5f')} | {cell('approx_entropy','%.4f')} |")
        md.append("")

        # ---- page 2: length distribution incl. tail
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        L = {arm: np.array([r["gen_len"] for r in by[arm]]) for arm in ARMS}
        for arm in ARMS:
            x = np.sort(L[arm]); ax[0, 0].plot(x, np.arange(1, x.size + 1) / x.size, color=C[arm], label=LBL[arm])
            ax[0, 1].plot(x, 1 - np.arange(0, x.size) / x.size, color=C[arm], label=LBL[arm])
        ax[0, 0].set_xscale("log"); ax[0, 0].set_title("CDF of generated tokens per rollout (steps 1-10 pooled)"); ax[0, 0].set_xlabel("generated tokens"); ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize=8)
        ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log"); ax[0, 1].set_title("Tail: survival function P(length > x)"); ax[0, 1].set_xlabel("generated tokens"); ax[0, 1].grid(alpha=0.3)
        ax[0, 1].axvline(196608, color="k", ls=":", lw=1); ax[0, 1].text(196608, 0.5, " context limit", fontsize=8)
        for q, ls in ((50, "-"), (90, "--"), (99, ":")):
            for arm in ARMS:
                ax[1, 0].plot(STEPS, per_step(arm, "gen_len", lambda v: np.percentile(v, q)), ls, color=C[arm], marker="o", ms=3, label=LBL[arm] if q == 50 else None)
        ax[1, 0].set_title("Per-step length percentiles: solid p50, dashed p90, dotted p99"); ax[1, 0].set_xlabel("train step"); ax[1, 0].set_xticks(STEPS); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=7)
        edges = [0, 10e3, 20e3, 30e3, 40e3, 50e3, 75e3, 100e3, 150e3, 200e3]
        for i, arm in enumerate(ARMS):
            h, _ = np.histogram(L[arm], edges); ax[1, 1].bar(np.arange(len(h)) + off(i), h / L[arm].size, BW, color=C[arm], label=LBL[arm])
            for k, c in enumerate(h):
                if c:
                    ax[1, 1].text(k + off(i), c / L[arm].size * 1.15, str(int(c)), ha="center", fontsize=5.5, color=C[arm])
        ax[1, 1].set_xticks(np.arange(len(edges) - 1)); ax[1, 1].set_xticklabels([f"{int(edges[i]/1e3)}-{int(edges[i+1]/1e3)}k" for i in range(len(edges) - 1)], rotation=30, fontsize=8)
        ax[1, 1].set_yscale("log"); ax[1, 1].set_title("Length histogram (share of rollouts, log scale)"); ax[1, 1].grid(alpha=0.3, axis="y"); ax[1, 1].legend(fontsize=8)
        finish(fig, "p2_length", "Generation length distribution including the tail")
        md.append("## Generation length per rollout (steps 1-10 pooled; " + ", ".join(f"{LBL[a]} {len(by[a]):,}" for a in ARMS) + " rollouts)\n")
        md.append("| arm | mean | p50 | p90 | p95 | p99 | max | share > 50k | share > 100k | share > 150k | truncated (context window) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            x = L[arm]; tr = np.array([r["truncated"] for r in by[arm]])
            md.append(f"| {LBL[arm]} | {x.mean():.0f} | {pct(x,50):.0f} | {pct(x,90):.0f} | {pct(x,95):.0f} | {pct(x,99):.0f} | {x.max():.0f} | {(x>50e3).mean():.3f} | {(x>100e3).mean():.3f} | {(x>150e3).mean():.3f} | {tr.mean():.3f} |")
        md.append(f"\nPaired by prompt, other arm minus {LBL[REF]} (n = matched prompt-steps):\n")
        for o in OTHERS:
            ks, p = ks_perm_p(by[REF], by[o], "gen_len"); d = paired_prompt_diff(by[REF], by[o], "gen_len"); d99 = paired_prompt_diff(by[REF], by[o], "gen_len", agg=lambda v: np.max(v))
            md.append(f"- {LBL[o]}: mean length {d['mean']:+.0f} tokens, 95% CI [{d['lo']:+.0f}, {d['hi']:+.0f}] (n={d['n']}; longer in {d['pos']}, shorter in {d['neg']}); per step: "
                      + ", ".join(f"s{s} {v:+.0f}" for s, v in d["per_step"].items()) + f"; per-prompt MAX length {d99['mean']:+.0f} [{d99['lo']:+.0f}, {d99['hi']:+.0f}]; KS distance {ks:.3f}, paired-permutation p = {p:.3f}.")
        md.append("")

        # ---- page 3: reward and group structure
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        grp = {}
        for arm in ARMS:
            g = defaultdict(list)
            for r in by[arm]:
                g[(r["step"], r["group_id"])].append(r["reward"])
            grp[arm] = {k: np.array(v) for k, v in g.items()}
        for arm in ARMS:
            st_ = np.array([v.std() for v in grp[arm].values()]); ax[0, 0].hist(st_, bins=np.linspace(0, 0.5, 26), weights=np.full(st_.size, 1 / st_.size), histtype="step", lw=1.5, color=C[arm], label=f"{LBL[arm]} (n={st_.size:,} groups)")
        ax[0, 0].set_title("Within-group reward std (16 rollouts per group, ddof=0)"); ax[0, 0].set_xlabel("std"); ax[0, 0].set_ylabel("share of groups"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)
        for arm in ARMS:
            ks_ = [np.array([int(v.sum()) for k, v in grp[arm].items() if k[0] == s]) for s in STEPS]
            ax[0, 1].plot(STEPS, [np.mean(k == 0) for k in ks_], "v-", color=C[arm], label=f"{LBL[arm]} all-fail groups")
            ax[0, 1].plot(STEPS, [np.mean(k == GROUP) for k in ks_], "^--", color=C[arm], label=f"{LBL[arm]} all-pass groups")
        ax[0, 1].set_title("Share of degenerate groups (zero advantage signal)"); ax[0, 1].set_xlabel("train step"); ax[0, 1].set_xticks(STEPS); ax[0, 1].legend(fontsize=7); ax[0, 1].grid(alpha=0.3)
        pref = defaultdict(list)
        for r in by[REF]:
            pref[(r["step"], r["instance_id"])].append(r["reward"])
        diffs = {}
        for o in OTHERS:
            po = defaultdict(list)
            for r in by[o]:
                po[(r["step"], r["instance_id"])].append(r["reward"])
            keys = sorted(set(pref) & set(po)); xa = np.array([np.mean(pref[k]) for k in keys]); xo = np.array([np.mean(po[k]) for k in keys]); diffs[o] = xo - xa
            ax[1, 0].scatter(xa + rng.normal(0, 0.005, xa.size), xo + rng.normal(0, 0.005, xo.size), s=8, alpha=0.45, color=C[o], label=LBL[o])
        ax[1, 0].plot([0, 1], [0, 1], "k:", lw=1)
        ax[1, 0].set_xlabel(f"pass rate, {LBL[REF]}"); ax[1, 0].set_ylabel("pass rate, other arm"); ax[1, 0].set_title(f"Per-prompt pass rate vs {LBL[REF]} (matched prompt-steps)"); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=7)
        for o in OTHERS:
            ax[1, 1].hist(diffs[o], bins=np.linspace(-1, 1, 33), histtype="step", color=C[o], label=f"{LBL[o]} minus {LBL[REF]}")
        ax[1, 1].set_title(f"Paired pass-rate difference per prompt, other arm minus {LBL[REF]}"); ax[1, 1].set_xlabel("difference"); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=7)
        finish(fig, "p3_reward", "Reward and GRPO group structure")
        md.append("## Reward and group structure\n")
        md.append("| arm | mean reward | mean within-group std | groups all-fail | groups all-pass | mixed groups | mean adv. (trainer) |")
        md.append("|---|---|---|---|---|---|---|")
        for arm in ARMS:
            vals = list(grp[arm].values()); ks_ = np.array([int(v.sum()) for v in vals])
            adv = np.nanmean([sc[arm][str(s)].get("advantages/mean", np.nan) for s in STEPS if str(s) in sc.get(arm, {})])
            md.append(f"| {LBL[arm]} | {np.mean([r['reward'] for r in by[arm]]):.3f} | {np.mean([v.std() for v in vals]):.3f} | {np.mean(ks_==0):.3f} | {np.mean(ks_==GROUP):.3f} | {np.mean((ks_>0)&(ks_<GROUP)):.3f} | {adv:+.4f} |")
        md.append(f"\nPaired pass-rate and within-group-std differences, other arm minus {LBL[REF]}:\n")
        for o in OTHERS:
            d = paired_prompt_diff(by[REF], by[o], "reward"); dstd = paired_prompt_diff(by[REF], by[o], "reward", agg=lambda v: np.std(v))
            md.append(f"- {LBL[o]}: pass rate {d['mean']:+.4f}, 95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}] (higher in {d['pos']}, lower in {d['neg']}, equal in {d['n']-d['pos']-d['neg']} of {d['n']} prompt-steps); within-group reward std {dstd['mean']:+.4f}, 95% CI [{dstd['lo']:+.4f}, {dstd['hi']:+.4f}].")
        md.append("")

        # ---- page 4: end-think probability
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        CL = {arm: np.concatenate([SP[(arm, s)]["close_gen"] for s in STEPS if (arm, s) in SP]) for arm in ARMS}
        CL = {arm: v[~np.isnan(v)] for arm, v in CL.items()}
        bins = np.logspace(-4, 1.7, 60)
        for arm in ARMS:
            ax[0, 0].hist(-CL[arm], bins=bins, histtype="step", color=C[arm], label=LBL[arm], density=True)
        ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("-log P(</think>) under the engine (sampled closes)"); ax[0, 0].set_title("Distribution of the engine's </think> logprob"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)
        for arm in ARMS:
            x = np.sort(-CL[arm]); ax[0, 1].plot(x, 1 - np.arange(0, x.size) / x.size, color=C[arm], label=LBL[arm])
        ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log"); ax[0, 1].set_xlabel("-log P(</think>)"); ax[0, 1].set_title("Tail: share of closes with -log P > x"); ax[0, 1].grid(alpha=0.3)
        for thr, name in ((0.1, "P<0.905"), (0.693, "P<0.5"), (2.303, "P<0.1")):
            ax[0, 1].axvline(thr, color="k", ls=":", lw=0.8); ax[0, 1].text(thr, 0.9, " " + name, fontsize=7, rotation=90, va="top")
        for arm in ARMS:
            geo, lo, hi = [], [], []
            for s in STEPS:
                sub = [r for r in by[arm] if r["step"] == s]
                m, l, h = cluster_boot_mean([r["sum_close_lp"] for r in sub], [r["instance_id"] for r in sub], weights=[r["n_closes"] for r in sub])
                geo.append(math.exp(m)); lo.append(math.exp(l)); hi.append(math.exp(h))
            ax[1, 0].plot(STEPS, geo, "o-", color=C[arm], label=LBL[arm]); ax[1, 0].fill_between(STEPS, lo, hi, color=C[arm], alpha=0.15)
        ax[1, 0].set_title("Geo-mean P(</think>) per step (95% prompt-cluster CI)"); ax[1, 0].set_xlabel("train step"); ax[1, 0].set_xticks(STEPS); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=8)
        for arm in ARMS:
            low, lo, hi = [], [], []
            for s in STEPS:
                sub = [r for r in by[arm] if r["step"] == s]
                m, l, h = cluster_boot_mean([r["n_low_close"] for r in sub], [r["instance_id"] for r in sub], weights=[r["n_closes"] for r in sub])
                low.append(m); lo.append(l); hi.append(h)
            ax[1, 1].plot(STEPS, low, "o-", color=C[arm], label=LBL[arm]); ax[1, 1].fill_between(STEPS, lo, hi, color=C[arm], alpha=0.15)
        ax[1, 1].set_title("Low-confidence close share (engine logprob < -0.1, P < 0.905)"); ax[1, 1].set_xlabel("train step"); ax[1, 1].set_xticks(STEPS); ax[1, 1].grid(alpha=0.3)
        finish(fig, "p4_endthink", "End-of-thinking token: engine probability of the sampled </think>")
        md.append("## `</think>` close probability (engine logprob of the sampled close)\n")
        md.append("| arm | closes | geo-mean P | 10th-pct P | 1st-pct P | share P<0.905 | share P<0.5 | share P<0.1 | turns without </think> |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            x = CL[arm]; nn = np.concatenate([SP[(arm, s)]["think_len"] for s in STEPS if (arm, s) in SP])
            md.append(f"| {LBL[arm]} | {x.size} | {math.exp(x.mean()):.4f} | {math.exp(pct(x,10)):.3f} | {math.exp(pct(x,1)):.3f} | {(x<-0.1).mean():.4f} | {(x<-0.693).mean():.4f} | {(x<-2.303).mean():.5f} | {(nn<0).mean():.5f} |")
        for r in rows:
            r["mean_close_lp"] = r["sum_close_lp"] / r["n_closes"] if r["n_closes"] > 0 else np.nan
            r["low_close_rate"] = r["n_low_close"] / r["n_closes"] if r["n_closes"] > 0 else np.nan
        dcl = {arm: [r for r in by[arm] if r["n_closes"] > 0] for arm in ARMS}
        md.append(f"\nPaired per-prompt mean close logprob, other arm minus {LBL[REF]}:\n")
        for o in OTHERS:
            d = paired_prompt_diff(dcl[REF], dcl[o], "mean_close_lp", agg=np.nanmean)
            md.append(f"- {LBL[o]}: {d['mean']:+.4f} (multiplicative P ratio {math.exp(d['mean']):.4f}), 95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]; higher in {d['pos']}, lower in {d['neg']} of {d['n']} prompt-steps; per step: "
                      + ", ".join(f"s{s} {v:+.4f}" for s, v in d["per_step"].items()) + ".")
        md.append("")

        # ---- page 5: reasoning length per turn
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        TL = {arm: np.concatenate([SP[(arm, s)]["think_len"] for s in STEPS if (arm, s) in SP]) for arm in ARMS}
        TL = {arm: v[v > 0] for arm, v in TL.items()}
        SL = {arm: np.concatenate([SP[(arm, s)]["span_len"] for s in STEPS if (arm, s) in SP]) for arm in ARMS}
        for arm in ARMS:
            x = np.sort(TL[arm]); ax[0, 0].plot(x, np.arange(1, x.size + 1) / x.size, color=C[arm], label=LBL[arm])
            ax[0, 1].plot(x, 1 - np.arange(0, x.size) / x.size, color=C[arm], label=LBL[arm])
        ax[0, 0].set_xscale("log"); ax[0, 0].set_title("CDF of thinking tokens per assistant turn"); ax[0, 0].set_xlabel("tokens between <think> and </think>"); ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize=8)
        ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log"); ax[0, 1].set_title("Tail: share of turns with thinking > x tokens"); ax[0, 1].set_xlabel("thinking tokens"); ax[0, 1].grid(alpha=0.3)
        for q, ls in ((50, "-"), (95, "--"), (99, ":")):
            for arm in ARMS:
                ax[1, 0].plot(STEPS, [pct(SP[(arm, s)]["think_len"][SP[(arm, s)]["think_len"] > 0], q) if (arm, s) in SP else np.nan for s in STEPS], ls, color=C[arm], marker="o", ms=3, label=LBL[arm] if q == 50 else None)
        ax[1, 0].set_title("Per-step thinking-length percentiles: solid p50, dashed p95, dotted p99"); ax[1, 0].set_xlabel("train step"); ax[1, 0].set_xticks(STEPS); ax[1, 0].set_yscale("log"); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=7)
        for arm in ARMS:
            x = np.sort(SL[arm]); ax[1, 1].plot(x, 1 - np.arange(0, x.size) / x.size, color=C[arm], label=LBL[arm])
        ax[1, 1].set_xscale("log"); ax[1, 1].set_yscale("log"); ax[1, 1].set_title("Tail of generated tokens per assistant turn (whole turn incl. tool call)"); ax[1, 1].set_xlabel("tokens per turn"); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=8)
        finish(fig, "p5_reasoning", "Reasoning length per turn")
        md.append("## Thinking tokens per assistant turn\n")
        md.append("| arm | turns | mean | p50 | p90 | p95 | p99 | max | share > 1k | share > 2k | share > 5k | turn tokens p50 / p99 / max |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            x = TL[arm]; y = SL[arm]
            md.append(f"| {LBL[arm]} | {x.size} | {x.mean():.0f} | {pct(x,50):.0f} | {pct(x,90):.0f} | {pct(x,95):.0f} | {pct(x,99):.0f} | {x.max()} | {(x>1000).mean():.4f} | {(x>2000).mean():.4f} | {(x>5000).mean():.5f} | {pct(y,50):.0f} / {pct(y,99):.0f} / {y.max()} |")
        for r in rows:
            r["mean_think"] = r["sum_think"] / r["n_closes"] if r["n_closes"] > 0 else np.nan
        md.append(f"\nPaired per-prompt thinking length, other arm minus {LBL[REF]}:\n")
        for o in OTHERS:
            d = paired_prompt_diff(dcl[REF], dcl[o], "mean_think", agg=np.nanmean); dm = paired_prompt_diff(by[REF], by[o], "max_think", agg=np.nanmean)
            md.append(f"- {LBL[o]}: mean thinking tokens per turn {d['mean']:+.1f}, 95% CI [{d['lo']:+.1f}, {d['hi']:+.1f}] (higher in {d['pos']}/{d['n']}); per-rollout longest thinking span {dm['mean']:+.0f}, 95% CI [{dm['lo']:+.0f}, {dm['hi']:+.0f}].")
        md.append("")

        # ---- page 6: token logprob distribution
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        lpe = H["lp_edges"]; centers = np.array([(lpe[i] if np.isfinite(lpe[i]) else lpe[i + 1] - 5) for i in range(len(lpe) - 1)])
        G_ = {arm: sum(H[f"{arm}_{s}_gen"] for s in STEPS if f"{arm}_{s}_gen" in H) for arm in ARMS}
        NT = {arm: sum(int(H[f"{arm}_{s}_n_tok"][0]) for s in STEPS if f"{arm}_{s}_n_tok" in H) for arm in ARMS}
        for arm in ARMS:
            ax[0, 0].step(-centers, G_[arm] / NT[arm], where="post", color=C[arm], label=LBL[arm])
        ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("-log P(sampled token) under the engine (bin lower edge)"); ax[0, 0].set_title("Engine logprob of sampled tokens, share per bin"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)
        for arm in ARMS:
            cum = np.cumsum(G_[arm]) / NT[arm]
            ax[0, 1].step(-lpe[1:], cum, where="pre", color=C[arm], label=LBL[arm])
        ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log"); ax[0, 1].set_xlabel("-log P threshold x"); ax[0, 1].set_title("Tail: share of tokens with -log P >= x"); ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize=8)
        lpm_bins = np.linspace(min(r["gen_lp_mean"] for r in rows), max(r["gen_lp_mean"] for r in rows), 61)
        for arm in ARMS:
            ax[1, 0].hist([r["gen_lp_mean"] for r in by[arm]], bins=lpm_bins, histtype="step", weights=np.full(len(by[arm]), 1 / len(by[arm])), color=C[arm], label=f"{LBL[arm]} (n={len(by[arm]):,} rollouts)")
        ax[1, 0].set_title("Per-rollout mean engine logprob of generated tokens (share of rollouts)"); ax[1, 0].set_xlabel("mean logprob"); ax[1, 0].set_ylabel("share of rollouts"); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=8)
        for arm in ARMS:
            y = []
            for s in STEPS:
                k = f"{arm}_{s}_gen"
                if k in H:
                    h = H[k]; y.append(h[lpe[1:] <= -10].sum() / H[f"{arm}_{s}_n_tok"][0] * 1e6)
                else:
                    y.append(np.nan)
            ax[1, 1].plot(STEPS, y, "o-", color=C[arm], label=LBL[arm])
        ax[1, 1].set_title("Tokens with engine logprob < -10, per million generated tokens"); ax[1, 1].set_xlabel("train step"); ax[1, 1].set_xticks(STEPS); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=8)
        finish(fig, "p6_logprob", "Token-level engine logprob distribution including the tail")
        md.append("## Engine logprob of sampled tokens (all generated tokens, steps 1-10)\n")
        md.append("| arm | tokens | mean logprob | share < -1 | share < -5 | per-million < -10 | per-million < -15 | per-million < -20 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            h = G_[arm]; n = NT[arm]; sg = sum(float(H[f"{arm}_{s}_sum_gen"][0]) for s in STEPS if f"{arm}_{s}_sum_gen" in H)
            tail = lambda thr: h[lpe[1:] <= thr].sum() / n
            md.append(f"| {LBL[arm]} | {n:,} | {sg/n:.4f} | {tail(-1):.4f} | {tail(-5):.5f} | {tail(-10)*1e6:.1f} | {tail(-15)*1e6:.2f} | {tail(-20)*1e6:.2f} |")
        md.append(f"\nPaired per-prompt engine-logprob statistics, other arm minus {LBL[REF]}:\n")
        for o in OTHERS:
            d = paired_prompt_diff(by[REF], by[o], "gen_lp_mean"); dmin = paired_prompt_diff(by[REF], by[o], "gen_lp_min"); dr = paired_prompt_diff(by[REF], by[o], "n_genlp_lt_m10")
            md.append(f"- {LBL[o]}: mean logprob {d['mean']:+.4f}, 95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]; per-rollout minimum logprob {dmin['mean']:+.2f}, CI [{dmin['lo']:+.2f}, {dmin['hi']:+.2f}]; tokens below -10 per rollout {dr['mean']:+.2f}, CI [{dr['lo']:+.2f}, {dr['hi']:+.2f}].")
        md.append("")

        # ---- page 7: train/inference mismatch
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        ae = H["absd_edges"]; se = H["sd_edges"]
        AD = {arm: sum(H[f"{arm}_{s}_absd"] for s in STEPS if f"{arm}_{s}_absd" in H) for arm in ARMS}
        SD_ = {arm: sum(H[f"{arm}_{s}_sd"] for s in STEPS if f"{arm}_{s}_sd" in H) for arm in ARMS}
        for arm in ARMS:
            cum = 1 - np.cumsum(AD[arm]) / NT[arm]; ax[0, 0].step(ae[1:-1], cum[:-1], where="post", color=C[arm], label=LBL[arm])
        ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("|trainer logprob - engine logprob| threshold x"); ax[0, 0].set_title("Tail: share of tokens with |d| > x"); ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize=8)
        sc_ = np.array([(se[i] if np.isfinite(se[i]) else (se[i + 1] - 10)) for i in range(len(se) - 1)])
        for i, arm in enumerate(ARMS):
            ax[0, 1].bar(np.arange(len(sc_)) + off(i), SD_[arm] / NT[arm], BW, color=C[arm], label=LBL[arm])
        ax[0, 1].set_xticks(np.arange(len(sc_))); ax[0, 1].set_xticklabels([f"{se[i]:g}" for i in range(len(se) - 1)], rotation=90, fontsize=7); ax[0, 1].set_yscale("log")
        ax[0, 1].set_title("Signed d = trainer - engine logprob (bin lower edges; leftmost bin = d < -20)"); ax[0, 1].grid(alpha=0.3, axis="y"); ax[0, 1].legend(fontsize=8)
        for i, arm in enumerate(ARMS):
            for k in (0, 1, len(sc_) - 3, len(sc_) - 2):
                c = int(SD_[arm][k])
                if c:
                    ax[0, 1].text(k + off(i), SD_[arm][k] / NT[arm] * 1.6, str(c), ha="center", fontsize=5.5, color=C[arm])
        err_bins = np.linspace(np.log10(max(1e-6, min(r["lp_err_mean"] for r in rows))), np.log10(max(r["lp_err_mean"] for r in rows)), 51)
        for arm in ARMS:
            ax[1, 0].hist(np.log10(np.clip([r["lp_err_mean"] for r in by[arm]], 1e-6, None)), bins=err_bins, histtype="step", weights=np.full(len(by[arm]), 1 / len(by[arm])), color=C[arm], label=f"{LBL[arm]} (n={len(by[arm]):,} rollouts)")
        ax[1, 0].set_title("Per-rollout mean |d| (log10), share of rollouts"); ax[1, 0].set_ylabel("share of rollouts"); ax[1, 0].set_xlabel("log10 mean |trainer - engine| logprob"); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=8)
        for arm in ARMS:
            x = np.sort([r["seq_mult_prob_error"] for r in by[arm]]); ax[1, 1].plot(x, 1 - np.arange(0, x.size) / x.size, color=C[arm], label=LBL[arm])
        ax[1, 1].axvline(2.0, color="k", ls=":", lw=1); ax[1, 1].text(2.0, 0.5, " mask threshold 2.0", fontsize=8)
        ax[1, 1].set_xscale("log"); ax[1, 1].set_yscale("log"); ax[1, 1].set_title("Tail of per-sequence multiplicative prob error"); ax[1, 1].set_xlabel("seq_mult_prob_error"); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=8)
        finish(fig, "p7_mismatch", "Trainer vs engine logprob mismatch")
        md.append("## Trainer-minus-engine logprob mismatch d (all generated tokens)\n")
        md.append("| arm | mean d | mean abs d | share abs d > 0.5 | share > 1 | per-million > 5 | per-million > 10 | rollouts masked (seq error > 2) | seq error p99 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            n = NT[arm]; sd_sum = sum(float(H[f"{arm}_{s}_sum_d"][0]) for s in STEPS if f"{arm}_{s}_sum_d" in H); ad_sum = sum(float(H[f"{arm}_{s}_sum_absd"][0]) for s in STEPS if f"{arm}_{s}_sum_absd" in H)
            h = AD[arm]; tail = lambda thr: h[ae[:-1] >= thr].sum() / n
            masked = np.mean([r["sample_mask_after"] == 0 for r in by[arm]]); sme = np.array([r["seq_mult_prob_error"] for r in by[arm]])
            md.append(f"| {LBL[arm]} | {sd_sum/n:+.5f} | {ad_sum/n:.5f} | {tail(0.5):.5f} | {tail(1):.5f} | {tail(5)*1e6:.2f} | {tail(10)*1e6:.2f} | {masked:.4f} | {pct(sme,99):.3f} |")
        md.append(f"\nPaired per-prompt mean |d|, other arm minus {LBL[REF]}:\n")
        for o in OTHERS:
            d = paired_prompt_diff(by[REF], by[o], "lp_err_mean")
            md.append(f"- {LBL[o]}: {d['mean']:+.5f}, 95% CI [{d['lo']:+.5f}, {d['hi']:+.5f}]; higher in {d['pos']}/{d['n']} prompt-steps.")
        md.append("")

        # ---- page 8: failure modes and limits
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        kinds = sorted({r["agent_error_kind"] for r in rows})
        for i, arm in enumerate(ARMS):
            cnt = np.array([sum(1 for r in by[arm] if r["agent_error_kind"] == k) for k in kinds]) / len(by[arm])
            ax[0, 0].bar(np.arange(len(kinds)) + off(i), cnt, BW, color=C[arm], label=LBL[arm])
        ax[0, 0].set_xticks(np.arange(len(kinds))); ax[0, 0].set_xticklabels(kinds, rotation=20, fontsize=8); ax[0, 0].set_yscale("log"); ax[0, 0].set_title("Agent termination / error kind (share of rollouts)"); ax[0, 0].grid(alpha=0.3, axis="y"); ax[0, 0].legend(fontsize=8)
        rate_cols = [("truncated", "context-window truncation (trainer flag)"), ("ctx_kind", "context_window error kind (harness)"), ("max_turns", "hit max turns (200)"), ("inv_any", ">=1 invalid tool call"),
                     ("malformed_any", ">=1 malformed thinking"), ("noclose_any", ">=1 turn without </think>"), ("noimend_any", ">=1 turn without <|im_end|>"), ("resp_incomplete", "incomplete response")]
        for r in rows:
            r["max_turns"] = float(r["turns"] >= MAX_TURNS) if not np.isnan(r["turns"]) else np.nan
            r["ctx_kind"] = float(r["agent_error_kind"] == "context_window")
            r["inv_any"] = float(r["n_invalid_tool_call"] > 0); r["malformed_any"] = float(r["n_malformed_thinking"] > 0)
            r["noclose_any"] = float(r["n_gen_without_think_close"] > 0); r["noimend_any"] = float(r["n_gen_without_im_end"] > 0)
        for i, arm in enumerate(ARMS):
            vals, lo, hi = [], [], []
            for col, _ in rate_cols:
                v = np.array([r[col] for r in by[arm]]); v = v[~np.isnan(v)]; p_, l, h = wilson(int(v.sum()), v.size); vals.append(p_); lo.append(p_ - l); hi.append(h - p_)
            ax[0, 1].bar(np.arange(len(rate_cols)) + off(i), vals, BW, yerr=[lo, hi], color=C[arm], label=LBL[arm], capsize=1.5)
        ax[0, 1].set_xticks(np.arange(len(rate_cols))); ax[0, 1].set_xticklabels([n for _, n in rate_cols], rotation=25, fontsize=8, ha="right"); ax[0, 1].set_title("Limit and failure rates per rollout (95% Wilson CI)"); ax[0, 1].grid(alpha=0.3, axis="y"); ax[0, 1].legend(fontsize=8)
        for arm in ARMS:
            x = np.sort([r["turns"] for r in by[arm] if not np.isnan(r["turns"])]); ax[1, 0].plot(x, np.arange(1, x.size + 1) / x.size, color=C[arm], label=LBL[arm])
        ax[1, 0].set_title("CDF of assistant turns per rollout"); ax[1, 0].set_xlabel("turns"); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=8)
        for arm in ARMS:
            num = per_step(arm, "n_invalid_tool_call", np.nansum); den = per_step(arm, "turns", np.nansum)
            ax[1, 1].plot(STEPS, np.array(num) / np.array(den) * 1000, "o-", color=C[arm], label=f"{LBL[arm]} invalid tool calls")
            num2 = per_step(arm, "n_gen_without_think_close", np.nansum)
            ax[1, 1].plot(STEPS, np.array(num2) / np.array(den) * 1000, "s--", color=C[arm], label=f"{LBL[arm]} turns without </think>")
        ax[1, 1].set_title("Per-turn rates, per 1,000 assistant turns"); ax[1, 1].set_xlabel("train step"); ax[1, 1].set_xticks(STEPS); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=7)
        finish(fig, "p8_failures", "Invalid tool calls, malformed thinking, generation limits and termination kinds")
        md.append("## Failure modes and generation limits (share of rollouts, 95% Wilson CI)\n")
        md.append("| metric | " + " | ".join(LBL[a] for a in ARMS) + " | " + " | ".join(f"paired diff {LBL[o]} minus {LBL[REF]} (95% CI)" for o in OTHERS) + " |")
        md.append("|---|" + "---|" * len(ARMS) + "---|" * len(OTHERS))
        for col, name in rate_cols:
            cells = []
            for arm in ARMS:
                v = np.array([r[col] for r in by[arm]]); v = v[~np.isnan(v)]; p_, l, h = wilson(int(v.sum()), v.size); cells.append(f"{p_:.4f} [{l:.4f}, {h:.4f}] (n={int(v.sum())})")
            dd = []
            for o in OTHERS:
                d = paired_prompt_diff(by[REF], by[o], col, agg=np.nanmean); dd.append(f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
            md.append(f"| {name} | " + " | ".join(cells) + " | " + " | ".join(dd) + " |")
        md.append("")
        md.append("| termination kind | " + " | ".join(LBL[a] for a in ARMS) + " |")
        md.append("|---|" + "---|" * len(ARMS))
        for k in kinds:
            md.append(f"| {k} | " + " | ".join(f"{sum(1 for r in by[a] if r['agent_error_kind']==k)} ({sum(1 for r in by[a] if r['agent_error_kind']==k)/len(by[a]):.4f})" for a in ARMS) + " |")
        md.append("")
        for arm in ARMS:
            it = np.nansum([r["n_invalid_tool_call"] for r in by[arm]]); tt = np.nansum([r["turns"] for r in by[arm]]); nc = np.nansum([r["n_gen_without_think_close"] for r in by[arm]])
            md.append(f"- {LBL[arm]}: {int(it)} invalid tool calls over {int(tt)} assistant turns ({it/tt*1000:.2f} per 1,000 turns); {int(nc)} generations without `</think>` ({nc/tt*1000:.2f} per 1,000 turns).")
        md.append("")

    rows = load_rollouts(); by = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    sc = json.load(open(f"{OUT}/sc_metrics.json")); H = dict(np.load(f"{OUT}/hists.npz"))
    SP = {(arm, step): dict(np.load(f"{OUT}/spans_{arm}_{step}.npz")) for arm in ARMS for step in STEPS if os.path.exists(f"{OUT}/spans_{arm}_{step}.npz")}
    md = []; pdf = PdfPages(f"{OUT}/first10_report.pdf"); pages = []
    five = dict(arms={a: dict(label=ARMS[a]["label"], color=ARMS[a]["color"]) for a in ARMS}, by=by, SP=SP, H=H, sc=sc, ref=REF, suffix="", title_prefix="", sc_note="",
                heading=["# First ten steps, five arms: " + ", ".join(ARMS[a]["label"] for a in ARMS), "",
                         "All arms start from the same base model and see the same 32 prompts x 16 rollouts per step (in_order sampler); every paired comparison below pairs prompts by SWE instance id within the same step. Rollout counts: "
                         + ", ".join(f"{ARMS[a]['label']} {len(by[a])}" for a in ARMS) + f". Bootstrap draws: {NBOOT}.", "",
                         f"Pairing choice: {ARMS[REF]['label']} is the reference; every paired statistic is reported as (other arm minus {ARMS[REF]['label']}), one block per other arm. Distribution panels show all five arms. Histograms of per-rollout or per-group quantities are shares (normalized), not counts.", "",
                         "Caveats. (1) A two-line guard was applied live to the mounted Megatron-LM dynamic_engine.py on 2026-09-22 07:47: chain I (MINF from scratch, prefix cache on, replica 1) ran unguarded for steps 1-15 and guarded from step 16, so its steps 1-10 are unguarded like run A (vLLM from scratch) and chain G (MINF from scratch, no prefix cache) and this window is internally consistent; chain J (MINF from scratch, prefix cache on, replica 2) is guarded throughout. (2) Step accounting uses the SingleController worker log (worker-*.out containing SingleControllerActor); the driver log is unreliable for this family. (3) chain K (vLLM from scratch, seed 1234) is the only arm not on grpo.seed 42.", ""])
    # engine-pooled view: MINF = chain G + chain I + chain J, vLLM = run A + chain K
    GROUPS = {"vLLM": dict(members=["A", "K"], label="vLLM engine (run A + chain K, both from scratch)", color="#1f77b4"),
              "MINF": dict(members=["G", "I", "J"], label="MINF engine (chain G + chain I + chain J, all from scratch)", color="#d62728")}
    by_e = {g: [r for m in G_["members"] for r in by[m]] for g, G_ in GROUPS.items()}
    SP_e = {}
    for g, G_ in GROUPS.items():
        for s in STEPS:
            parts = [SP[(m, s)] for m in G_["members"] if (m, s) in SP]
            if parts:
                SP_e[(g, s)] = {k: np.concatenate([p[k] for p in parts]) for k in ("think_len", "close_gen", "close_prev", "span_len")}
    H_e = {k: H[k] for k in ("lp_edges", "absd_edges", "sd_edges")}
    for g, G_ in GROUPS.items():
        for s in STEPS:
            for stat in ("gen", "absd", "sd", "n_tok", "sum_gen", "sum_d", "sum_absd"):
                parts = [H[f"{m}_{s}_{stat}"] for m in G_["members"] if f"{m}_{s}_{stat}" in H]
                if parts:
                    H_e[f"{g}_{s}_{stat}"] = sum(parts)
    sc_e = {}
    for g, G_ in GROUPS.items():
        sc_e[g] = {}
        for s in STEPS:
            recs = [sc[m][str(s)] for m in G_["members"] if str(s) in sc.get(m, {})]
            keys = set().union(*[set(r) for r in recs]) if recs else set()
            sc_e[g][str(s)] = {k: float(np.mean([r[k] for r in recs if k in r])) for k in keys}
    engine = dict(arms={g: dict(label=G_["label"], color=G_["color"]) for g, G_ in GROUPS.items()}, by=by_e, SP=SP_e, H=H_e, sc=sc_e, ref="vLLM", suffix="_engine",
                  title_prefix="[engine-pooled] ", sc_note=" (mean over the pooled arms)",
                  heading=["", "# Engine-pooled view: MINF engine (chain G + chain I + chain J) vs vLLM engine (run A + chain K), steps 1-10", "",
                           "The same eight pages with the from-scratch arms pooled by serving engine: MINF engine = chain G (MINF from scratch, no prefix cache) + chain I (MINF from scratch, prefix cache on, replica 1) + chain J (MINF from scratch, prefix cache on, replica 2), 15,360 rollouts; vLLM engine = run A (vLLM from scratch) + chain K (vLLM from scratch, seed 1234), 10,240 rollouts. Every per-rollout, per-group or per-token histogram is a share (normalized within the pool), so the unequal pool sizes do not affect the shapes; paired statistics pair by (step, SWE instance) with all pooled rollouts of that prompt on each side and are reported as MINF minus vLLM. Trainer metrics from the controller logs are averaged over the pooled arms per step. The per-prompt MAX-length comparison is biased toward the larger pool (48 vs 32 rollouts per prompt-step) and is descriptive only.", ""])
    render(five); render(engine)
    if have_mpl:
        pdf.close()
    open(f"{OUT}/report.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("pages:", pages)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("collect", "all"):
        collect()
    if stage in ("plot", "all"):
        plot()

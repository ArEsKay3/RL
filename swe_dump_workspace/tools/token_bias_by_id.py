"""Per-token-id trainer-minus-engine logprob statistics over generated tokens.

Usage: token_bias_by_id.py OUT_DIR FILE [FILE ...]   (NPROC env = workers)
Per chunk writes OUT_DIR/<run>__<file>.bias.npz with, per token id: n, sum_d (prev-gen), sum_absd, sum_gen,
plus the same split by certainty bucket of the engine logprob, and </think> by think-length bucket.
"""
from __future__ import annotations

import json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pt_numpy import load  # noqa: E402

V = 140000
THINK_END = 13
GEN_EDGES = np.array([-np.inf, -1.0, -0.1, -0.01, 0.0 + 1e-9])  # bucket by engine logprob of the sampled token
TL_EDGES = np.array([0, 100, 1000, 4000, 10 ** 9])


def process(args):
    out_dir, path = args
    run = path.split("/runs/")[1].split("/")[0]; name = os.path.basename(path)[:-3]; step = int(name.split("_")[1])
    d = load(path)
    lengths = np.asarray(d["input_lengths"]).astype(np.int64)
    ids_off = np.concatenate([[0], np.cumsum(lengths)]); lp_off = np.concatenate([[0], np.cumsum(lengths - 1)])
    ids_all = np.asarray(d["input_ids"]).astype(np.int64); mask_all = np.asarray(d["token_mask"]).astype(bool)
    gen_all = np.asarray(d["generation_logprobs"]).astype(np.float64); prev_all = np.asarray(d["prev_logprobs"]).astype(np.float64)
    trainer_v = d.get("trainer_version"); tags = d.get("tags") or [{}] * len(lengths)
    wv_map = json.load(open(os.path.join(out_dir, "wv_map.json"))) if os.path.exists(os.path.join(out_dir, "wv_map.json")) else {}
    sample_ids = list(d["sample_ids"])
    toks, gens, prevs, tlens, lags = [], [], [], [], []
    for i in range(len(lengths)):
        wv = (tags[i] or {}).get("weight_version")
        se = wv_map.get(sample_ids[i])
        raw_lag = None if trainer_v is None or wv is None else int(trainer_v) - int(wv)
        if raw_lag == 0:
            lag_i = 0
        elif raw_lag == 1 and se is not None and se[0] == se[1]:
            lag_i = 1
        elif raw_lag == 1 and se is not None:
            lag_i = 2
        else:
            lag_i = 3
        L = int(lengths[i]); ids = ids_all[ids_off[i]:ids_off[i + 1]]
        m = mask_all[ids_off[i] + 1:ids_off[i + 1]]                # for token positions 1..L-1
        g = gen_all[lp_off[i]:lp_off[i + 1]]; p = prev_all[lp_off[i]:lp_off[i + 1]]
        tok = ids[1:]
        toks.append(tok[m]); gens.append(g[m]); prevs.append(p[m])
        # think length for each generated token: tokens since span start (span = run of mask)
        mm = m.astype(np.int8); dm = np.diff(np.concatenate([[0], mm, [0]]))
        starts = np.where(dm == 1)[0]; ends = np.where(dm == -1)[0]
        tl = np.zeros(len(tok), np.int64)
        for s, e in zip(starts, ends):
            tl[s:e] = np.arange(1, e - s + 1)
        tlens.append(tl[m]); lags.append(np.full(int(m.sum()), lag_i, np.int8))
    tok = np.concatenate(toks); g = np.concatenate(gens); p = np.concatenate(prevs); tl = np.concatenate(tlens); lag = np.concatenate(lags)
    assert tok.max() < V, tok.max()
    dd = p - g
    out = dict(step=np.array(step), n=np.bincount(tok, minlength=V), sum_d=np.bincount(tok, weights=dd, minlength=V),
               sum_absd=np.bincount(tok, weights=np.abs(dd), minlength=V), sum_gen=np.bincount(tok, weights=g, minlength=V),
               sum_d2=np.bincount(tok, weights=dd * dd, minlength=V))
    gb = np.digitize(g, GEN_EDGES) - 1  # 0: <-1, 1: [-1,-0.1), 2: [-0.1,-0.01), 3: [-0.01, 0]
    for b in range(4):
        sel = gb == b
        out[f"n_b{b}"] = np.bincount(tok[sel], minlength=V); out[f"sum_d_b{b}"] = np.bincount(tok[sel], weights=dd[sel], minlength=V)
    te = tok == THINK_END
    for l in range(4):
        sl = lag == l
        out[f"n_lag{l}"] = np.bincount(tok[sl], minlength=V); out[f"sum_d_lag{l}"] = np.bincount(tok[sl], weights=dd[sl], minlength=V)
        for b in range(4):
            sb = sl & (gb == b)
            out[f"lag{l}_b{b}_all_n"] = np.array(int(sb.sum())); out[f"lag{l}_b{b}_all_sum_d"] = np.array(float(dd[sb].sum()))
            out[f"lag{l}_b{b}_te_n"] = np.array(int((sb & te).sum())); out[f"lag{l}_b{b}_te_sum_d"] = np.array(float(dd[sb & te].sum()))
    tb = np.digitize(tl[te], TL_EDGES) - 1
    out["te_tl_n"] = np.bincount(tb, minlength=4); out["te_tl_sum_d"] = np.bincount(tb, weights=dd[te], minlength=4)
    out["te_tl_sum_gen"] = np.bincount(tb, weights=g[te], minlength=4)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{run}__{name}.bias.npz"), **out)
    return path, int(te.sum())


def main():
    out_dir = sys.argv[1]; files = sys.argv[2:]
    with Pool(min(int(os.environ.get("NPROC", 8)), len(files))) as pool:
        for path, n in pool.imap_unordered(process, [(out_dir, f) for f in files]):
            print(f"done think_end={n:6d}  {path}", flush=True)


if __name__ == "__main__":
    main()

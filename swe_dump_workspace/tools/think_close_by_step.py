"""Per-token records for </think>: gap (trainer - engine), engine logprob, lag class; plus reference stats for hesitant tokens.

Usage: think_close_by_step.py OUT_DIR FILE [FILE ...]  (NPROC env; reads OUT_DIR/../token_bias/wv_map.json)
"""
from __future__ import annotations

import json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pt_numpy import load  # noqa: E402

THINK_END = 13


def process(args):
    out_dir, path = args
    run = path.split("/runs/")[1].split("/")[0]; name = os.path.basename(path)[:-3]; step = int(name.split("_")[1])
    wv_path = os.path.join(os.path.dirname(out_dir.rstrip("/")), "token_bias", "wv_map.json")
    wv_map = json.load(open(wv_path)) if os.path.exists(wv_path) else {}
    d = load(path)
    lengths = np.asarray(d["input_lengths"]).astype(np.int64)
    ids_off = np.concatenate([[0], np.cumsum(lengths)]); lp_off = np.concatenate([[0], np.cumsum(lengths - 1)])
    ids_all = np.asarray(d["input_ids"]).astype(np.int64); mask_all = np.asarray(d["token_mask"]).astype(bool)
    gen_all = np.asarray(d["generation_logprobs"]).astype(np.float64); prev_all = np.asarray(d["prev_logprobs"]).astype(np.float64)
    trainer_v = d.get("trainer_version"); tags = d.get("tags") or [{}] * len(lengths); sids = list(d["sample_ids"])
    te_g, te_p, te_lag = [], [], []
    ref = np.zeros((4, 3))  # per lag class: n hesitant tokens, sum d, n with d<0
    for i in range(len(lengths)):
        wv = (tags[i] or {}).get("weight_version"); se = wv_map.get(sids[i])
        raw = None if trainer_v is None or wv is None else int(trainer_v) - int(wv)
        lag = 0 if raw == 0 else (1 if raw == 1 and se is not None and se[0] == se[1] else (2 if raw == 1 and se is not None else 3))
        m = mask_all[ids_off[i] + 1:ids_off[i + 1]]; tok = ids_all[ids_off[i] + 1:ids_off[i + 1]]
        g = gen_all[lp_off[i]:lp_off[i + 1]]; p = prev_all[lp_off[i]:lp_off[i + 1]]
        te = m & (tok == THINK_END)
        te_g.append(g[te]); te_p.append(p[te]); te_lag.append(np.full(int(te.sum()), lag, np.int8))
        hes = m & (g < -0.1)
        dd = p[hes] - g[hes]
        ref[lag] += [hes.sum(), dd.sum(), (dd < 0).sum()]
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{run}__{name}.tc.npz"), step=np.array(step), gen=np.concatenate(te_g), prev=np.concatenate(te_p),
                        lag=np.concatenate(te_lag), ref=ref)
    return path, int(sum(len(x) for x in te_g))


def main():
    out_dir = sys.argv[1]; files = sys.argv[2:]
    with Pool(min(int(os.environ.get("NPROC", 8)), len(files))) as pool:
        for path, n in pool.imap_unordered(process, [(out_dir, f) for f in files]):
            print(f"done {n:6d} closes  {path}", flush=True)


if __name__ == "__main__":
    main()

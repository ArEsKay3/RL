"""Per-row statistics from dumps/token_level/step_*_chunk_*.pt files (torch-free).

Usage: token_level_summary.py OUT_DIR FILE [FILE ...]
Writes OUT_DIR/<run>__<file>.rows.csv (one line per sequence) and
OUT_DIR/<run>__<file>.hist.json (lp_error and generation-logprob histograms).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pt_numpy import load  # noqa: E402

LP_ERR_BINS = [0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, np.inf]
GEN_LP_BINS = [-np.inf, -20, -15, -10, -8, -6, -4, -2, -1, -0.5, -0.1, 0]


def process(args):
    out_dir, path = args
    run = path.split("/runs/")[1].split("/")[0]
    name = os.path.basename(path).replace(".pt", "")
    obj = load(path)
    lengths = np.asarray(obj["input_lengths"]).astype(np.int64)
    n = int(lengths.shape[0])
    ids_off = np.concatenate([[0], np.cumsum(lengths)])
    lp_off = np.concatenate([[0], np.cumsum(lengths - 1)])
    mask = np.asarray(obj["token_mask"]).astype(np.uint8)
    gen = np.asarray(obj["generation_logprobs"]).astype(np.float32)
    prev = np.asarray(obj["prev_logprobs"]).astype(np.float32)
    adv = np.asarray(obj["advantages"]).astype(np.float32)
    ids = np.asarray(obj["input_ids"])
    rewards = np.asarray(obj["rewards"]).astype(np.float32)
    smb = np.asarray(obj["sample_mask_before"]).astype(np.float32)
    sma = np.asarray(obj["sample_mask_after"]).astype(np.float32)
    smpe = np.asarray(obj["seq_mult_prob_error"]).astype(np.float32)
    sample_ids = obj["sample_ids"]
    tags = obj.get("tags") or [{}] * n

    lp_err_hist = np.zeros(len(LP_ERR_BINS) - 1, dtype=np.int64)
    gen_lp_hist = np.zeros(len(GEN_LP_BINS) - 1, dtype=np.int64)
    rows = []
    for i in range(n):
        L = int(lengths[i])
        m = mask[ids_off[i] + 1 : ids_off[i + 1]].astype(bool)  # positions 1..L-1
        g = gen[lp_off[i] : lp_off[i + 1]]
        p = prev[lp_off[i] : lp_off[i + 1]]
        a = adv[lp_off[i] : lp_off[i + 1]]
        tok = ids[ids_off[i] + 1 : ids_off[i + 1]]
        gm, pm, am = g[m], p[m], a[m]
        err = np.abs(gm - pm)
        n_gen = int(m.sum())
        if n_gen:
            lp_err_hist += np.histogram(err, bins=LP_ERR_BINS)[0]
            gen_lp_hist += np.histogram(gm, bins=GEN_LP_BINS)[0]
        # last generated token id and whether the sequence hit the context limit
        last_gen_tok = int(tok[m][-1]) if n_gen else -1
        rows.append({
            "run": run, "file": name, "sample_id": sample_ids[i],
            "group_id": sample_ids[i].rsplit("_g", 1)[0],
            "weight_version": (tags[i] or {}).get("weight_version"),
            "rollout_truncated": (tags[i] or {}).get("rollout_truncated"),
            "rollout_generation_length": (tags[i] or {}).get("rollout_generation_length"),
            "input_length": L, "n_gen_tokens": n_gen,
            "reward": float(rewards[i]), "sample_mask_before": float(smb[i]), "sample_mask_after": float(sma[i]),
            "seq_mult_prob_error": float(smpe[i]),
            "adv": float(am[0]) if n_gen else float("nan"),
            "adv_std_within_row": float(am.std()) if n_gen else float("nan"),
            "gen_lp_mean": float(gm.mean()) if n_gen else float("nan"),
            "gen_lp_min": float(gm.min()) if n_gen else float("nan"),
            "prev_lp_mean": float(pm.mean()) if n_gen else float("nan"),
            "lp_err_mean": float(err.mean()) if n_gen else float("nan"),
            "lp_err_p99": float(np.percentile(err, 99)) if n_gen else float("nan"),
            "lp_err_max": float(err.max()) if n_gen else float("nan"),
            "n_err_gt1": int((err > 1).sum()), "n_err_gt5": int((err > 5).sum()),
            "n_genlp_lt_m10": int((gm < -10).sum()),
            "exp_err_mean": float(np.exp(err).mean()) if n_gen else float("nan"),
            "last_gen_token": last_gen_tok,
        })
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{run}__{name}.rows.csv")
    with open(csv_path + ".tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(csv_path + ".tmp", csv_path)
    hist = {
        "run": run, "file": name, "step": obj.get("step"), "chunk_index": obj.get("chunk_index"),
        "trainer_version": obj.get("trainer_version"), "n_rows": n,
        "n_gen_tokens": int(sum(r["n_gen_tokens"] for r in rows)),
        "lp_err_bins": [str(b) for b in LP_ERR_BINS], "lp_err_hist": lp_err_hist.tolist(),
        "gen_lp_bins": [str(b) for b in GEN_LP_BINS], "gen_lp_hist": gen_lp_hist.tolist(),
    }
    with open(os.path.join(out_dir, f"{run}__{name}.hist.json"), "w") as fh:
        json.dump(hist, fh)
    return path, n


def main():
    out_dir = sys.argv[1]
    files = sys.argv[2:]
    with Pool(min(int(os.environ.get("NPROC", 4)), len(files))) as pool:
        for path, n in pool.imap_unordered(process, [(out_dir, f) for f in files]):
            print(f"done {n:4d} rows  {path}", flush=True)


if __name__ == "__main__":
    main()

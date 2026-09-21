"""Deep per-token statistics from dumps/token_level/step_*_chunk_*.pt (torch-free).

Usage: token_deep.py OUT_DIR TOKENIZER_JSON FILE [FILE ...]
Per file writes OUT_DIR/<run>__<file>.deep.json (accumulators) and
OUT_DIR/<run>__<file>.deeprows.csv (one light record per sequence).
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

CATS = ["other", "im_end", "think_end", "tool_call", "tool_call_end", "newline", "im_start"]
POS_BINS = np.array([0, 4096, 16384, 32768, 65536, 131072, 1 << 30])
TURN_BINS = np.array([0, 1, 5, 20, 50, 100, 10**6])
GENLP_BINS = np.array([-np.inf, -12, -8, -4, -2, -1, -0.5, -0.1, 1e-4])
TLEN_BINS = np.array([0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 1 << 30])
CLIP_LO, CLIP_HI = 0.2, 5.0
TOK = {}


def token_ids(tok_path):
    j = json.load(open(tok_path))
    vocab = dict(j["model"]["vocab"])
    for t in j.get("added_tokens", []):
        vocab[t["content"]] = t["id"]
    return {
        "im_end": vocab["<|im_end|>"], "think_end": vocab["</think>"], "tool_call": vocab["<tool_call>"],
        "tool_call_end": vocab.get("</tool_call>", -1), "newline": vocab.get("Ċ", -1), "im_start": vocab.get("<|im_start|>", -1),
    }


def acc_new(n):
    return {"n": np.zeros(n), "d": np.zeros(n), "abs": np.zeros(n), "sq": np.zeros(n), "gt1": np.zeros(n),
            "gen": np.zeros(n), "prev": np.zeros(n), "wlo": np.zeros(n), "whi": np.zeros(n), "w": np.zeros(n),
            "pos_adv_n": np.zeros(n), "neg_adv_n": np.zeros(n), "pos_adv_w": np.zeros(n), "neg_adv_w": np.zeros(n)}


def acc_add(acc, code, d, g, p, w, a):
    n = len(acc["n"])
    absd = np.abs(d)
    acc["n"] += np.bincount(code, minlength=n)
    acc["d"] += np.bincount(code, weights=d, minlength=n)
    acc["abs"] += np.bincount(code, weights=absd, minlength=n)
    acc["sq"] += np.bincount(code, weights=d * d, minlength=n)
    acc["gt1"] += np.bincount(code, weights=(absd > 1).astype(np.float64), minlength=n)
    acc["gen"] += np.bincount(code, weights=g, minlength=n)
    acc["prev"] += np.bincount(code, weights=p, minlength=n)
    acc["wlo"] += np.bincount(code, weights=(w < CLIP_LO).astype(np.float64), minlength=n)
    acc["whi"] += np.bincount(code, weights=(w > CLIP_HI).astype(np.float64), minlength=n)
    acc["w"] += np.bincount(code, weights=np.clip(w, CLIP_LO, CLIP_HI), minlength=n)
    pos = a > 0
    neg = a < 0
    acc["pos_adv_n"] += np.bincount(code[pos], minlength=n)
    acc["neg_adv_n"] += np.bincount(code[neg], minlength=n)
    acc["pos_adv_w"] += np.bincount(code[pos], weights=np.clip(w[pos], CLIP_LO, CLIP_HI), minlength=n)
    acc["neg_adv_w"] += np.bincount(code[neg], weights=np.clip(w[neg], CLIP_LO, CLIP_HI), minlength=n)


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
    gen = np.asarray(obj["generation_logprobs"]).astype(np.float64)
    prev = np.asarray(obj["prev_logprobs"]).astype(np.float64)
    adv = np.asarray(obj["advantages"]).astype(np.float64)
    ids = np.asarray(obj["input_ids"])
    rewards = np.asarray(obj["rewards"]).astype(np.float64)
    sma = np.asarray(obj["sample_mask_after"]).astype(np.float64)
    sample_ids = obj["sample_ids"]
    tags = obj.get("tags") or [{}] * n
    trainer_version = obj.get("trainer_version")

    by = {"cat": acc_new(len(CATS)), "pos": acc_new(len(POS_BINS) - 1), "turn": acc_new(len(TURN_BINS) - 1),
          "genlp": acc_new(len(GENLP_BINS) - 1), "lag": acc_new(4), "eos_by_turn": acc_new(len(TURN_BINS) - 1),
          "last_tok_of_turn": acc_new(len(CATS))}
    tlen_hist = {"all": np.zeros(len(TLEN_BINS) - 1), "pos_adv": np.zeros(len(TLEN_BINS) - 1), "neg_adv": np.zeros(len(TLEN_BINS) - 1),
                 "zero_adv": np.zeros(len(TLEN_BINS) - 1)}
    rows = []
    for i in range(n):
        L = int(lengths[i])
        m = mask[ids_off[i] + 1: ids_off[i + 1]].astype(bool)
        g = gen[lp_off[i]: lp_off[i + 1]][m]
        p = prev[lp_off[i]: lp_off[i + 1]][m]
        a = adv[lp_off[i]: lp_off[i + 1]][m]
        tok = ids[ids_off[i] + 1: ids_off[i + 1]][m].astype(np.int64)
        posn = np.where(m)[0] + 1
        ng = int(m.sum())
        if ng == 0:
            continue
        d = p - g
        w = np.exp(np.clip(d, -30, 30))
        row_adv = float(a[0])
        wv = (tags[i] or {}).get("weight_version")
        lag = int(trainer_version) - int(wv) if (wv is not None and trainer_version is not None) else -1
        lag_code = np.full(ng, min(max(lag, 0), 2) if lag >= 0 else 3)
        cat = np.zeros(ng, dtype=np.int64)
        for ci, cname in enumerate(CATS[1:], start=1):
            tid = TOK.get(cname, -1)
            if tid >= 0:
                cat[tok == tid] = ci
        starts = np.concatenate([[True], m[1:] & ~m[:-1]]) if False else None
        # turn index: count of mask run starts up to each gen token
        m_int = m.astype(np.int8)
        run_start = np.concatenate([[m_int[0]], (m_int[1:] - m_int[:-1] == 1).astype(np.int8)])
        turn_idx_full = np.cumsum(run_start)
        turn_idx = turn_idx_full[m]
        n_turns = int(turn_idx.max())
        turn_code = np.clip(np.searchsorted(TURN_BINS, turn_idx, side="right") - 1, 0, len(TURN_BINS) - 2)
        pos_code = np.clip(np.searchsorted(POS_BINS, posn, side="right") - 1, 0, len(POS_BINS) - 2)
        genlp_code = np.clip(np.searchsorted(GENLP_BINS, g, side="right") - 1, 0, len(GENLP_BINS) - 2)
        acc_add(by["cat"], cat, d, g, p, w, a)
        acc_add(by["pos"], pos_code, d, g, p, w, a)
        acc_add(by["turn"], turn_code, d, g, p, w, a)
        acc_add(by["genlp"], genlp_code, d, g, p, w, a)
        acc_add(by["lag"], lag_code, d, g, p, w, a)
        eos = cat == 1
        if eos.any():
            acc_add(by["eos_by_turn"], turn_code[eos], d[eos], g[eos], p[eos], w[eos], a[eos])
        # last generated token of each turn (what ended the turn)
        last_of_turn = np.concatenate([turn_idx[1:] != turn_idx[:-1], [True]])
        acc_add(by["last_tok_of_turn"], cat[last_of_turn], d[last_of_turn], g[last_of_turn], p[last_of_turn], w[last_of_turn], a[last_of_turn])
        tlens = np.bincount(turn_idx)[1:]
        h = np.histogram(tlens, bins=TLEN_BINS)[0]
        tlen_hist["all"] += h
        tlen_hist["pos_adv" if row_adv > 0 else ("neg_adv" if row_adv < 0 else "zero_adv")] += h
        wc = np.clip(w, CLIP_LO, CLIP_HI)
        rows.append({
            "run": run, "file": name, "step": obj.get("step"), "sample_id": sample_ids[i], "group_id": sample_ids[i].rsplit("_g", 1)[0],
            "lag": lag, "reward": float(rewards[i]), "adv": row_adv, "sample_mask_after": float(sma[i]),
            "n_gen": ng, "n_turns": n_turns, "first_turn": int(tlens[0]), "last_turn": int(tlens[-1]), "max_turn": int(tlens.max()),
            "d_mean": float(d.mean()), "abs_mean": float(np.abs(d).mean()), "d_max": float(d.max()), "d_min": float(d.min()),
            "err_gt1": int((np.abs(d) > 1).sum()), "n_eos": int(eos.sum()),
            "eos_d_mean": float(d[eos].mean()) if eos.any() else float("nan"),
            "eos_gen_mean": float(g[eos].mean()) if eos.any() else float("nan"),
            "last_turn_ends_eos": bool(cat[-1] == 1),
            "frac_w_lo": float((w < CLIP_LO).mean()), "frac_w_hi": float((w > CLIP_HI).mean()), "w_mean": float(wc.mean()),
            "eff_weight": float((wc * np.sign(row_adv)).sum()) if row_adv != 0 else 0.0,
            "trunc": (tags[i] or {}).get("rollout_truncated"),
            "gen_lp_mean": float(g.mean()), "prev_lp_mean": float(p.mean()),
            "n_genlp_lt_m8": int((g < -8).sum()), "prev_at_tail_mean": float(p[g < -8].mean()) if (g < -8).any() else float("nan"),
            "gen_at_tail_mean": float(g[g < -8].mean()) if (g < -8).any() else float("nan"),
        })
    os.makedirs(out_dir, exist_ok=True)
    out = {"run": run, "file": name, "step": obj.get("step"), "trainer_version": trainer_version, "n_rows": n,
           "cats": CATS, "pos_bins": POS_BINS.tolist(), "turn_bins": TURN_BINS.tolist(), "genlp_bins": [str(x) for x in GENLP_BINS],
           "tlen_bins": TLEN_BINS.tolist(), "by": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in by.items()},
           "tlen_hist": {k: v.tolist() for k, v in tlen_hist.items()}}
    with open(os.path.join(out_dir, f"{run}__{name}.deep.json"), "w") as fh:
        json.dump(out, fh)
    if rows:
        with open(os.path.join(out_dir, f"{run}__{name}.deeprows.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return path, n


def main():
    global TOK
    out_dir, tok_path, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    TOK = token_ids(tok_path)
    print("token ids:", TOK, flush=True)
    nproc = min(int(os.environ.get("NPROC", "8")), len(files))
    with Pool(nproc) as pool:
        for path, n in pool.imap_unordered(process, [(out_dir, f) for f in files]):
            print(f"done {n:4d} rows  {path}", flush=True)


if __name__ == "__main__":
    main()

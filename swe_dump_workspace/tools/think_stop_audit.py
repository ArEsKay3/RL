"""Audit generation-mask spans in token-level dumps: boundaries, stop tokens, </think> placement and logprobs.

Usage: think_stop_audit.py OUT_DIR FILE [FILE ...]   (NPROC env = worker count)
Per chunk writes OUT_DIR/<run>__<file>.thinkaudit.json and .npz (per-span arrays).
"""
from __future__ import annotations

import collections, json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pt_numpy import load  # noqa: E402

IM_START, IM_END, THINK, THINK_END, TOOL_CALL = 10, 11, 12, 13, 14


def process(args):
    out_dir, path = args
    run = path.split("/runs/")[1].split("/")[0]
    name = os.path.basename(path)[:-3]
    step = int(name.split("_")[1])
    d = load(path)
    lengths = np.asarray(d["input_lengths"]).astype(np.int64)
    ids_off = np.concatenate([[0], np.cumsum(lengths)])
    lp_off = np.concatenate([[0], np.cumsum(lengths - 1)])
    ids_all = np.asarray(d["input_ids"])
    mask_all = np.asarray(d["token_mask"]).astype(np.int8)
    gen_all = np.asarray(d["generation_logprobs"]).astype(np.float32)
    prev_all = np.asarray(d["prev_logprobs"]).astype(np.float32)
    rewards = np.asarray(d["rewards"]).astype(np.float32)
    tags = d.get("tags") or [{}] * len(lengths)
    C = collections.Counter
    acc = dict(n_seq=int(len(lengths)), n_spans=0, spans_multi_im_end=0, spans_no_im_end=0, spans_no_think_end=0,
               spans_toolcall_before_think_end=0, spans_think_tok_inside=0, spans_end_not_im_end=0,
               last_span_ends_im_end=0, seq_trunc=0, mask_on_im_start=0, masked_tokens=0, gen_tokens_tag=0)
    end_tok, before, after, after_think = C(), C(), C(), C()
    think_len, te_g, te_p, ie_g, ie_p, span_len, span_step_reward = [], [], [], [], [], [], []
    for i in range(len(lengths)):
        L = int(lengths[i])
        ids = ids_all[ids_off[i]:ids_off[i + 1]]
        m = np.zeros(L, dtype=np.int8); m[1:] = mask_all[ids_off[i] + 1:ids_off[i + 1]]
        g = np.full(L, np.nan, np.float32); g[1:] = gen_all[lp_off[i]:lp_off[i + 1]]
        p = np.full(L, np.nan, np.float32); p[1:] = prev_all[lp_off[i]:lp_off[i + 1]]
        acc["masked_tokens"] += int(m.sum())
        acc["gen_tokens_tag"] += int((tags[i] or {}).get("rollout_generation_length") or 0)
        acc["seq_trunc"] += int(bool((tags[i] or {}).get("rollout_truncated", False)))
        acc["mask_on_im_start"] += int(((ids == IM_START) & (m == 1)).sum())
        dm = np.diff(np.concatenate([[0], m, [0]]))
        starts = np.where(dm == 1)[0]; ends = np.where(dm == -1)[0]
        for k, (s, e) in enumerate(zip(starts, ends)):
            span = ids[s:e]
            acc["n_spans"] += 1; span_len.append(int(e - s)); span_step_reward.append(float(rewards[i]))
            end_tok[int(span[-1])] += 1
            before[tuple(int(x) for x in ids[max(0, s - 4):s])] += 1
            after[tuple(int(x) for x in ids[e:e + 3])] += 1
            n11 = int((span == IM_END).sum())
            acc["spans_multi_im_end"] += int(n11 > 1); acc["spans_no_im_end"] += int(n11 == 0)
            acc["spans_end_not_im_end"] += int(span[-1] != IM_END)
            acc["spans_think_tok_inside"] += int((span == THINK).any())
            w = np.where(span == THINK_END)[0]
            if len(w):
                j = int(w[0]); think_len.append(j + 1); te_g.append(float(g[s + j])); te_p.append(float(p[s + j]))
                acc["spans_toolcall_before_think_end"] += int((span[:j] == TOOL_CALL).any())
                if s + j + 1 < e:
                    after_think[int(span[j + 1])] += 1
            else:
                think_len.append(-1); te_g.append(np.nan); te_p.append(np.nan)
                acc["spans_no_think_end"] += 1
            if n11:
                j = int(np.where(span == IM_END)[0][-1]); ie_g.append(float(g[s + j])); ie_p.append(float(p[s + j]))
            if k == len(starts) - 1:
                acc["last_span_ends_im_end"] += int(span[-1] == IM_END)
    out = dict(run=run, file=name, step=step, **acc)
    out["end_tok"] = end_tok.most_common(8)
    out["before"] = [[list(k), c] for k, c in before.most_common(6)]
    out["after"] = [[list(k), c] for k, c in after.most_common(6)]
    out["after_think_first_tok"] = after_think.most_common(6)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{run}__{name}")
    json.dump(out, open(base + ".thinkaudit.json", "w"))
    np.savez_compressed(base + ".thinkaudit.npz", think_len=np.asarray(think_len, np.int32), think_end_gen_lp=np.asarray(te_g, np.float32),
                        think_end_prev_lp=np.asarray(te_p, np.float32), im_end_gen_lp=np.asarray(ie_g, np.float32), im_end_prev_lp=np.asarray(ie_p, np.float32),
                        span_len=np.asarray(span_len, np.int32), span_reward=np.asarray(span_step_reward, np.float32))
    return path, acc["n_spans"]


def main():
    out_dir = sys.argv[1]; files = sys.argv[2:]
    with Pool(min(int(os.environ.get("NPROC", 8)), len(files))) as pool:
        for path, n in pool.imap_unordered(process, [(out_dir, f) for f in files]):
            print(f"done {n:6d} spans  {path}", flush=True)


if __name__ == "__main__":
    main()

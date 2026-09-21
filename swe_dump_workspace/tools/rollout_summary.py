"""Stream dumps/rollouts/target_step_*.jsonl files into compact per-row summaries.

Usage: rollout_summary.py OUT_DIR FILE [FILE ...]
Writes OUT_DIR/<run>__<file>.summary.jsonl, one small JSON object per rollout.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

try:
    import orjson as _fastjson

    def loads(s):
        return _fastjson.loads(s)
except Exception:  # noqa: BLE001
    def loads(s):
        return json.loads(s)


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def summarize_row(row: dict) -> dict:
    fr = row.get("full_result") or {}
    resp = fr.get("response") or {}
    out = resp.get("output") or []
    ic = fr.get("instance_config") or {}
    ptm = fr.get("per_turn_metrics") or {}
    usage = ptm.get("accumulated_token_usage") or {}
    msgs = row.get("messages") or []
    assistant_tokens = [m.get("n_tokens", 0) for m in msgs if m.get("role") == "assistant"]
    user_tokens = [m.get("n_tokens", 0) for m in msgs if m.get("role") == "user"]

    types = Counter(o.get("type") for o in out if isinstance(o, dict))
    fc = [o for o in out if isinstance(o, dict) and o.get("type") == "function_call"]
    fc_names = Counter(o.get("name") for o in fc)
    gens = [o.get("generation_str") or "" for o in fc]
    msg_items = [o for o in out if isinstance(o, dict) and o.get("type") == "message"]

    def msg_text(o):
        c = o.get("content")
        if isinstance(c, list):
            return "".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
        return str(c or "")

    msg_texts = [msg_text(o) for o in msg_items]
    last = out[-1] if out else {}
    last_gen = gens[-1] if gens else ""
    statuses = Counter(o.get("status") for o in out if isinstance(o, dict) and o.get("status") is not None)
    rm = row.get("rollout_metrics") or {}
    latencies = ptm.get("response_latencies") or []
    lat_vals = [x.get("latency", x) if isinstance(x, dict) else x for x in latencies] if isinstance(latencies, list) else []
    lat_vals = [v for v in lat_vals if isinstance(v, (int, float))]

    return {
        "sample_id": row.get("sample_id"),
        "group_id": row.get("group_id"),
        "target_step": row.get("target_step"),
        "completion_index": row.get("completion_index"),
        "prompt_idx": row.get("prompt_idx"),
        "instance_id": (row.get("prompt_metadata") or {}).get("instance_id"),
        "dataset": (row.get("prompt_metadata") or {}).get("dataset_name"),
        "start_wv": row.get("start_weight_version"),
        "end_wv": row.get("end_weight_version"),
        "reward": row.get("reward"),
        "truncated": row.get("truncated"),
        "turns": row.get("num_assistant_turns"),
        "gen_len": row.get("generation_length"),
        "total_tokens": row.get("total_tokens"),
        "prompt_tokens": msgs[0].get("n_tokens") if msgs else None,
        "max_asst_tokens": max(assistant_tokens) if assistant_tokens else 0,
        "sum_user_tokens": sum(user_tokens),
        "n_invalid_tool_call": sum(1 for m in msgs if m.get("is_invalid_tool_call")),
        "n_malformed_thinking": sum(1 for m in msgs if m.get("has_malformed_thinking")),
        "resolved": fr.get("resolved"),
        "patch_exists": fr.get("patch_exists"),
        "model_patch_len": len(fr.get("model_patch") or ""),
        "agent_error_kind": fr.get("agent_error_kind"),
        "agent_timed_out": fr.get("agent_timed_out"),
        "eval_timed_out": fr.get("eval_timed_out"),
        "oom_killed": fr.get("oom_killed"),
        "eval_oom_killed": fr.get("eval_oom_killed"),
        "openhands_run_time": _num(fr.get("openhands_run_time")),
        "total_model_call_time": _num(fr.get("total_model_call_time")),
        "total_command_exec_time": _num(fr.get("total_command_exec_time")),
        "final_eval_time": _num(fr.get("final_eval_time")),
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
        "usage_cache_read_tokens": usage.get("cache_read_tokens"),
        "ptm_keys": sorted(ptm.keys()) if isinstance(ptm, dict) else None,
        "n_latencies": len(lat_vals),
        "latency_max": max(lat_vals) if lat_vals else None,
        "latency_sum": sum(lat_vals) if lat_vals else None,
        "resp_status": resp.get("status"),
        "resp_error": None if resp.get("error") is None else str(resp.get("error"))[:200],
        "resp_incomplete": None if resp.get("incomplete_details") is None else str(resp.get("incomplete_details"))[:200],
        "mask_sample": ic.get("mask_sample"),
        "agent_max_turns": ic.get("agent_max_turns"),
        "n_out_items": len(out),
        "n_reasoning": types.get("reasoning", 0),
        "n_message": types.get("message", 0),
        "n_message_nonblank": sum(1 for t in msg_texts if t.strip()),
        "n_function_call": types.get("function_call", 0),
        "n_function_call_output": types.get("function_call_output", 0),
        "fc_names": dict(fc_names),
        "n_finish": fc_names.get("finish", 0),
        "last_item_type": last.get("type") if isinstance(last, dict) else None,
        "last_item_name": last.get("name") if isinstance(last, dict) else None,
        "last_gen_has_tool_call": "<tool_call>" in last_gen,
        "last_gen_has_im_end": last_gen.rstrip().endswith("<|im_end|>"),
        "n_gen_without_im_end": sum(1 for g in gens if not g.rstrip().endswith("<|im_end|>")),
        "n_gen_without_think_close": sum(1 for g in gens if "</think>" not in g),
        "max_gen_chars": max((len(g) for g in gens), default=0),
        "item_statuses": dict(statuses),
        "rm_natural_termination_rate": rm.get("natural_termination_rate"),
        "rm_truncation_rate": rm.get("truncation_rate"),
        "rm_max_turns_reached_rate": rm.get("max_turns_reached_rate"),
        "rm_turns_mean": rm.get("turns_per_sample/mean"),
        "rm_max_gen_tokens_per_turn_max": rm.get("max_gen_tokens_per_turn/max"),
        "committed_at": row.get("committed_at"),
    }


def process(args):
    out_dir, path = args
    run = path.split("/runs/")[1].split("/")[0]
    name = os.path.basename(path).replace(".jsonl", "")
    out_path = os.path.join(out_dir, f"{run}__{name}.summary.jsonl")
    tmp = out_path + ".tmp"
    n = 0
    with open(path, "rb") as fh, open(tmp, "w") as out:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = loads(line)
                rec = summarize_row(row)
            except Exception as e:  # noqa: BLE001
                rec = {"parse_error": f"{type(e).__name__}: {e}"[:200]}
            rec["run"] = run
            rec["file"] = name
            out.write(json.dumps(rec) + "\n")
            n += 1
    os.replace(tmp, out_path)
    return path, n


def main():
    out_dir = sys.argv[1]
    files = sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    with Pool(min(int(os.environ.get("NPROC", 6)), len(files))) as pool:
        for path, n in pool.imap_unordered(process, [(out_dir, f) for f in files]):
            print(f"done {n:5d} rows  {path}", flush=True)


if __name__ == "__main__":
    main()

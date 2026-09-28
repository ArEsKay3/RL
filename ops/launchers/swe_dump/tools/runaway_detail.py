"""Targeted pass over dump JSONL files: per-turn shape of long / truncated / errored rollouts.

Usage: runaway_detail.py OUT_CSV IDS_CSV FILE...
IDS_CSV has columns file,sample_id (file = basename without .jsonl).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import zlib
from multiprocessing import Pool

try:
    import orjson

    loads = orjson.loads
except Exception:  # noqa: BLE001
    loads = json.loads

IDS: dict[str, set[str]] = {}


def rep_ratio(s: str) -> float:
    tail = s[-20000:].encode()
    return len(zlib.compress(tail, 6)) / max(1, len(tail))


def describe(row: dict) -> dict:
    fr = row.get("full_result") or {}
    out = (fr.get("response") or {}).get("output") or []
    msgs = row.get("messages") or []
    asst = [(i, m.get("n_tokens", 0)) for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    gens = [o.get("generation_str") or "" for o in out if isinstance(o, dict) and o.get("type") == "function_call" and o.get("generation_str")]
    longest = max(gens, key=len) if gens else ""
    last = gens[-1] if gens else ""
    li = max(range(len(asst)), key=lambda k: asst[k][1]) if asst else -1
    err_keys = {k: str(v)[:300] for k, v in fr.items() if "error" in k.lower() and v not in (None, "", False, [])}
    return {
        "sample_id": row.get("sample_id"), "reward": row.get("reward"), "truncated": row.get("truncated"),
        "turns": row.get("num_assistant_turns"), "total_tokens": row.get("total_tokens"),
        "agent_error_kind": fr.get("agent_error_kind"),
        "longest_turn_idx": li, "longest_turn_tokens": asst[li][1] if li >= 0 else 0, "n_asst": len(asst),
        "longest_is_last": li == len(asst) - 1 if asst else None,
        "longest_chars": len(longest), "longest_has_think_close": "</think>" in longest,
        "longest_has_tool_call": "<tool_call>" in longest, "longest_ends_im_end": longest.rstrip().endswith("<|im_end|>"),
        "longest_rep_ratio": round(rep_ratio(longest), 3) if longest else None,
        "longest_tail": longest[-160:].replace("\n", "\\n") if longest else "",
        "last_chars": len(last), "last_ends_im_end": last.rstrip().endswith("<|im_end|>"),
        "last_has_think_close": "</think>" in last, "last_rep_ratio": round(rep_ratio(last), 3) if last else None,
        "last_tail": last[-160:].replace("\n", "\\n") if last else "",
        "last_item_type": out[-1].get("type") if out else None,
        "err_keys": json.dumps(err_keys)[:600],
    }


def process(path: str):
    name = os.path.basename(path).replace(".jsonl", "")
    run = path.split("/runs/")[1].split("/")[0]
    want = IDS.get(f"{run}/{name}") or set()
    res = []
    if not want:
        return path, res
    needles = [s.encode() for s in want]
    with open(path, "rb") as fh:
        for line in fh:
            head = line[:400]
            if any(n in head for n in needles):
                row = loads(line)
                if row.get("sample_id") in want:
                    d = describe(row)
                    d["run"] = run
                    d["file"] = name
                    res.append(d)
                    if len(res) == len(want):
                        break
    return path, res


def main():
    out_csv, ids_csv, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    with open(ids_csv) as fh:
        for r in csv.DictReader(fh):
            IDS.setdefault(f"{r['run']}/{r['file']}", set()).add(r["sample_id"])
    rows = []
    with Pool(min(6, len(files))) as pool:
        for path, res in pool.imap_unordered(process, files):
            rows.extend(res)
            print(f"{len(res):4d} rows from {path}", flush=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_csv, len(rows))


if __name__ == "__main__":
    main()

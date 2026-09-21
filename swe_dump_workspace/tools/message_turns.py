"""Turn structure around Responses 'message' items in dump rollouts.

Usage: message_turns.py OUT_JSON FILE...
For every rollout: counts of message items by role/blankness, message-only turns
(assistant message with no function_call before the next reasoning item), what
followed them, and the last item when it is a message.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

try:
    import orjson

    loads = orjson.loads
except Exception:  # noqa: BLE001
    loads = json.loads


def text_of(o):
    c = o.get("content")
    if isinstance(c, list):
        return "".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return str(c or "")


def process(path):
    run = path.split("/runs/")[1].split("/")[0]
    agg = Counter()
    examples = []
    with open(path, "rb") as fh:
        for line in fh:
            row = loads(line)
            out = (row.get("full_result") or {}).get("response", {}).get("output") or []
            agg["rows"] += 1
            n = len(out)
            for i, o in enumerate(out):
                if o.get("type") != "message":
                    continue
                role = o.get("role", "?")
                blank = not text_of(o).strip()
                agg[f"msg_{role}_{'blank' if blank else 'text'}"] += 1
                # what comes after this message before the next reasoning/message item
                j = i + 1
                followed = "end"
                while j < n:
                    t = out[j].get("type")
                    if t == "function_call":
                        followed = "function_call"
                        break
                    if t in ("reasoning", "message"):
                        followed = f"next_{t}"
                        break
                    followed = t
                    j += 1
                agg[f"msg_{role}_{'blank' if blank else 'text'}_then_{followed}"] += 1
                if followed != "function_call" and len(examples) < 6:
                    examples.append({
                        "sample_id": row.get("sample_id"), "idx": i, "n": n, "role": role, "blank": blank,
                        "text_head": text_of(o)[:160], "followed": followed,
                        "prev_type": out[i - 1].get("type") if i else None,
                        "next_types": [x.get("type") for x in out[i + 1 : i + 4]],
                        "reward": row.get("reward"), "turns": row.get("num_assistant_turns"),
                        "error_kind": (row.get("full_result") or {}).get("agent_error_kind"),
                    })
    return run, os.path.basename(path), dict(agg), examples


def main():
    out_json, files = sys.argv[1], sys.argv[2:]
    res = {}
    with Pool(min(4, len(files))) as pool:
        for run, name, agg, ex in pool.imap_unordered(process, files):
            res[f"{run}/{name}"] = {"agg": agg, "examples": ex}
            print("done", run, name, {k: v for k, v in agg.items() if "then" not in k}, flush=True)
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()

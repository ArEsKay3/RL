"""Classify user-role message items in dump rollouts by what preceded them; show blank-assistant-message-then-message cases.

Usage: user_message_context.py FILE...
"""
import json
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
    c = Counter()
    cases = []
    with open(path, "rb") as fh:
        for line in fh:
            row = loads(line)
            out = (row.get("full_result") or {}).get("response", {}).get("output") or []
            for i, o in enumerate(out):
                if o.get("type") == "message" and o.get("role") == "user":
                    prev = out[i - 1] if i else {}
                    pt = prev.get("type")
                    if pt == "message":
                        pt += "_blank" if not text_of(prev).strip() else "_text"
                    c[f"user_msg_after_{pt}"] += 1
                    c["user_msg_text:" + text_of(o)[:60].replace("\n", " ")] += 1
                    nxt = out[i + 1].get("type") if i + 1 < len(out) else "END"
                    c[f"user_msg_then_{nxt}"] += 1
                if o.get("type") == "message" and o.get("role") == "assistant" and not text_of(o).strip():
                    nxt = out[i + 1] if i + 1 < len(out) else {}
                    if nxt.get("type") != "function_call":
                        prev = out[i - 1] if i else {}
                        cases.append({
                            "sample_id": row.get("sample_id"), "idx": i, "n": len(out), "reward": row.get("reward"),
                            "turns": row.get("num_assistant_turns"), "error_kind": (row.get("full_result") or {}).get("agent_error_kind"),
                            "prev": prev.get("type"), "prev_reasoning_tail": text_of(prev)[-120:] if prev.get("type") == "message" else "".join(s.get("text", "") for s in (prev.get("summary") or []))[-120:],
                            "next": {k: (v if k != "content" else text_of(nxt)[:100]) for k, v in nxt.items() if k in ("type", "role", "content", "name")},
                            "after_next": [x.get("type") for x in out[i + 2 : i + 5]],
                        })
    return run, path.rsplit("/", 1)[1], c, cases


if __name__ == "__main__":
    files = sys.argv[1:]
    tot = {}
    with Pool(min(4, len(files))) as pool:
        for run, name, c, cases in pool.imap_unordered(process, files):
            eng = "vllm" if "vllm_dump" in run else "minf"
            t = tot.setdefault(eng, Counter())
            t.update(c)
            for cs in cases:
                print(f"[{eng} {name}] blank assistant message not followed by a tool call:", json.dumps(cs)[:700], flush=True)
    for eng, c in tot.items():
        print(f"\n== {eng}")
        for k, v in sorted(c.items()):
            print(f"  {v:6d}  {k}")

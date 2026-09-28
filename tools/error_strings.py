"""Print the harness error strings recorded for selected rollouts.

Usage: error_strings.py IDS_CSV FILE...   (IDS_CSV columns: run,file,sample_id,label)
"""
import csv
import json
import sys
from multiprocessing import Pool

try:
    import orjson

    loads = orjson.loads
except Exception:  # noqa: BLE001
    loads = json.loads

WANT = {}


def find_errors(obj, path="", out=None, depth=0):
    if out is None:
        out = []
    if depth > 4:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{path}.{k}" if path else k
            if isinstance(v, str) and any(t in k.lower() for t in ("error", "exception", "status", "reason", "finish")) and v:
                out.append((kp, v[:400]))
            elif isinstance(v, (dict, list)) and k not in ("output", "input", "messages", "prompt_input", "instance_config", "tools"):
                find_errors(v, kp, out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:20]):
            find_errors(v, f"{path}[{i}]", out, depth + 1)
    return out


def process(path):
    run = path.split("/runs/")[1].split("/")[0]
    name = path.rsplit("/", 1)[1].replace(".jsonl", "")
    want = WANT.get(f"{run}/{name}", {})
    res = []
    if not want:
        return res
    needles = {s.encode(): s for s in want}
    with open(path, "rb") as fh:
        for line in fh:
            head = line[:400]
            hit = next((s for n, s in needles.items() if n in head), None)
            if not hit:
                continue
            row = loads(line)
            if row.get("sample_id") != hit:
                continue
            fr = row.get("full_result") or {}
            out = (fr.get("response") or {}).get("output") or []
            res.append({
                "label": want[hit], "engine": "vllm" if "vllm_dump" in run else "minf", "sample_id": hit,
                "reward": row.get("reward"), "truncated": row.get("truncated"), "total_tokens": row.get("total_tokens"),
                "agent_error_kind": fr.get("agent_error_kind"), "last_item": out[-1].get("type") if out else None,
                "last_item_role": out[-1].get("role") if out else None,
                "fr_keys": sorted(k for k in fr if k not in ("response",)),
                "errors": find_errors({k: v for k, v in fr.items() if k != "response"}),
                "resp_status": (fr.get("response") or {}).get("status"),
                "resp_incomplete": (fr.get("response") or {}).get("incomplete_details"),
            })
            if len(res) == len(want):
                break
    return res


if __name__ == "__main__":
    ids_csv, files = sys.argv[1], sys.argv[2:]
    with open(ids_csv) as fh:
        for r in csv.DictReader(fh):
            WANT.setdefault(f"{r['run']}/{r['file']}", {})[r["sample_id"]] = r["label"]
    with Pool(min(6, len(files))) as pool:
        for res in pool.imap_unordered(process, files):
            for d in res:
                print(json.dumps(d)[:1600], flush=True)

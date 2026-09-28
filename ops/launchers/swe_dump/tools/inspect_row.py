"""Print the item sequence of one rollout around function_call items whose generation lacks </think>.

Usage: inspect_row.py DUMP_JSONL SAMPLE_ID
"""
import json
import sys

path, sid = sys.argv[1], sys.argv[2]
needle = sid.encode()
row = None
with open(path, "rb") as fh:
    for line in fh:
        if needle in line[:400]:
            row = json.loads(line)
            if row.get("sample_id") == sid:
                break
            row = None
if row is None:
    sys.exit(f"{sid} not found in {path}")
fr = row["full_result"]
out = fr["response"]["output"]
pi = row.get("prompt_input") or {}
print("prompt_input keys:", sorted(pi.keys()) if isinstance(pi, dict) else type(pi).__name__)
for k in ("max_output_tokens", "max_tokens", "max_completion_tokens", "temperature", "top_p", "reasoning", "tool_choice", "parallel_tool_calls", "truncation"):
    if isinstance(pi, dict) and k in pi:
        print(f"  {k} = {str(pi[k])[:120]}")
print("reward", row["reward"], "turns", row["num_assistant_turns"], "truncated", row["truncated"], "n_items", len(out))
print("output item types in order (first 40):", [o.get("type", "?")[:3] for o in out[:40]])
for i, o in enumerate(out):
    if o.get("type") == "function_call" and "</think>" not in (o.get("generation_str") or ""):
        g = o.get("generation_str") or ""
        print(f"\n=== item {i}: function_call name={o.get('name')} gen_len_chars={len(g)} keys={sorted(o.keys())}")
        print("  arguments head:", repr((o.get("arguments") or "")[:200]))
        print("  generation_str head:", repr(g[:300]))
        print("  generation_str tail:", repr(g[-300:]))
        for j in range(max(0, i - 3), min(len(out), i + 3)):
            p = out[j]
            t = p.get("type")
            if t == "reasoning":
                txt = "".join(c.get("text", "") for c in (p.get("summary") or p.get("content") or []) if isinstance(c, dict))
                print(f"  [{j}] reasoning len={len(txt)} head={txt[:120]!r} tail={txt[-80:]!r}")
            elif t == "function_call":
                gg = p.get("generation_str") or ""
                print(f"  [{j}] function_call {p.get('name')} gen_len={len(gg)} same_gen_as_target={gg == g} has_think_close={'</think>' in gg} ends_im_end={gg.rstrip().endswith('<|im_end|>')}")
            elif t == "message":
                c = p.get("content")
                txt = "".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else str(c)
                print(f"  [{j}] message len={len(txt)} head={txt[:120]!r}")
            else:
                print(f"  [{j}] {t} {str(p.get('output', ''))[:80]!r}")
        break

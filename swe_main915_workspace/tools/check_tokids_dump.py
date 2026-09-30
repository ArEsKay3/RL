"""Verify raw per-turn token ids in rollout dump rows: presence, sizes, and consistency with the per-message ids."""
import glob, json, os, sys
RUN = sys.argv[1]
MAX_ROWS = int(os.environ.get("MAX_ROWS", "64"))
files = sorted(glob.glob(os.path.join(RUN, "dumps/rollouts/*.jsonl")))
rows = 0; items = 0; with_ids = 0; gen_match = 0; gen_mismatch = 0; prefix_match = 0; prefix_mismatch = 0; bytes_total = 0
prompt_id_total = 0; gen_id_total = 0; examples = []
for f in files:
    with open(f) as fh:
        for line in fh:
            if rows >= MAX_ROWS: break
            rows += 1; bytes_total += len(line)
            rec = json.loads(line)
            msgs = rec.get("messages") or []
            out = ((rec.get("full_result") or {}).get("response") or {}).get("output") or []
            trainable = [o for o in out if isinstance(o, dict) and ("generation_str" in o or "prompt_token_ids" in o)]
            assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
            # cumulative prefix from per-message ids
            run_ids = []; k = 0
            for m in msgs:
                if m.get("role") == "assistant":
                    if k < len(trainable):
                        o = trainable[k]; items += 1
                        p = o.get("prompt_token_ids"); g = o.get("generation_token_ids")
                        if isinstance(p, list) and isinstance(g, list):
                            with_ids += 1; prompt_id_total += len(p); gen_id_total += len(g)
                            if g == m.get("token_ids"): gen_match += 1
                            else: gen_mismatch += 1
                            if p == run_ids: prefix_match += 1
                            else:
                                prefix_mismatch += 1
                                if len(examples) < 3:
                                    examples.append({"file": os.path.basename(f), "sample_id": rec.get("sample_id"), "turn": k, "len_prompt": len(p), "len_prefix": len(run_ids), "first_diff": next((i for i,(a,b) in enumerate(zip(p, run_ids)) if a != b), min(len(p), len(run_ids)))})
                    k += 1
                run_ids = run_ids + list(m.get("token_ids") or [])
print(f"rows={rows} trainable_items={items} items_with_raw_ids={with_ids}")
print(f"generation_token_ids == assistant message token_ids: {gen_match} match / {gen_mismatch} mismatch")
print(f"prompt_token_ids == concat(previous message token_ids): {prefix_match} match / {prefix_mismatch} mismatch")
print(f"raw ids per row: prompt {prompt_id_total/max(rows,1):,.0f}  generation {gen_id_total/max(rows,1):,.0f}; mean row size {bytes_total/max(rows,1)/1e6:.1f} MB")
for e in examples: print("  mismatch example:", e)

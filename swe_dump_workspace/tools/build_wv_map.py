"""Build analysis/token_bias/wv_map.json (sample_id -> [start_wv, end_wv]) from the rollout summaries."""
import glob, json, os

W = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump"
m = {}
for f in glob.glob(f"{W}/analysis/rollouts/*.summary.jsonl"):
    for line in open(f):
        r = json.loads(line)
        if r.get("sample_id") and r.get("start_wv") is not None:
            m[r["sample_id"]] = [r["start_wv"], r["end_wv"]]
os.makedirs(f"{W}/analysis/token_bias", exist_ok=True)
json.dump(m, open(f"{W}/analysis/token_bias/wv_map.json", "w"))
print("wv_map entries", len(m))

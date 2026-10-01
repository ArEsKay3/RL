"""Pool shard_*/**/results.jsonl for a run dir and assert 500 problems x 5 repeats = 2500 distinct rows.

Usage: pooled_check.py RUN_DIR   -> prints json {pass1, problems, rows, distinct_pairs, per_shard}
"""
import glob, json, sys

rd = sys.argv[1]
rows, pairs, per_shard = [], set(), {}
for f in glob.glob(f"{rd}/shard_*/harbor___swebench_verified_1_0/*/results.jsonl"):
    shard = f[len(rd) + 1:].split("/")[0]
    for line in open(f, errors="ignore"):
        d = json.loads(line)
        r = d.get("reward")
        if isinstance(r, (int, float)):
            rows.append(r); per_shard[shard] = per_shard.get(shard, 0) + 1
            pairs.add((d.get("problem_idx"), d.get("repeat_idx", d.get("repeat"))))
problems = {p for p, _ in pairs}
out = dict(run_dir=rd, pass1=round(sum(rows) / len(rows), 4) if rows else None, problems=len(problems), rows=len(rows),
           distinct_pairs=len(pairs), per_shard=dict(sorted(per_shard.items())),
           complete=(len(problems) == 500 and len(rows) == 2500 and len(pairs) == 2500))
print(json.dumps(out))
sys.exit(0 if out["complete"] else 1)

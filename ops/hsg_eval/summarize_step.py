"""Print one results.md row for a merged run dir: pass@1 [CI], pass@3 (unbiased), pass@5 [bootstrap CI], attempts, solver failures graded 0 by class.

Usage: summarize_step.py STEP RUN_DIR
"""
import collections, glob, json, sys
from math import comb

step, rd = sys.argv[1:3]
rid = rd.rstrip("/").split("/")[-1]
ej = glob.glob(f"{rd}/swebench-verified_1.0/eval-*.json")[0]
sc = json.load(open(ej))["benchmark"]["scores"]
p1 = sc["pass@1"]; p5 = sc.get("pass@5", {})
per = collections.defaultdict(list); fail = collections.Counter(); n = 0
for f in glob.glob(f"{rd}/shard_*/harbor___swebench_verified_1_0/*/results.jsonl"):
    for l in open(f, errors="ignore"):
        d = json.loads(l); r = d.get("reward")
        if not isinstance(r, (int, float)): continue
        n += 1; per[d["problem_idx"]].append(r)
        sd = d.get("scoring_details") or {}
        if sd.get("method") == "solve_failed": fail[sd.get("error_category") or sd.get("error") or "solve_failed"] += 1
assert n == 2500 and len(per) == 500, (n, len(per))
k3 = sum(1 - comb(5 - sum(1 for r in v if r > 0), 3) / comb(5, 3) for v in per.values()) / 500
k5 = sum(1 for v in per.values() if any(r > 0 for r in v)) / 500
val = p1.get("value", p1.get("mean"))
if val is None:
    val = sum(sum(v) for v in per.values()) / n
fails = ", ".join(f"{k} {c}" for k, c in sorted(fail.items())) or "none"
print(f"| {step} | {rid} | {val:.4f} [{p1['ci_lower']:.4f}, {p1['ci_upper']:.4f}] | {k3:.4f} | {k5:.3f} [{p5.get('bootstrap_ci_lower', float('nan')):.3f}, {p5.get('bootstrap_ci_upper', float('nan')):.3f}] | {n} | {sum(fail.values())} ({fails}) |")

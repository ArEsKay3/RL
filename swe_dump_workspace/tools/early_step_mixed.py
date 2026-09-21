"""Mixed-group breakdown for the early-step engine comparison (imports early_step_compare)."""
import statistics as st, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import early_step_compare as E

rows = E.load_runA()
inst_step = {r["instance"]: r["tstep"] for r in rows if r["instance"]}
minf, _, _ = E.load_minf(inst_step)
A = defaultdict(list); M = {}
for r in rows:
    A[(r["step"], r["instance"])].append(r)
for inst, s in minf.items():
    M[(s[0]["step"], inst)] = s

def kind(g):
    k = sum(x["reward"] > 0 for x in g)
    return "mixed" if 0 < k < len(g) else ("all_pass" if k == len(g) else "all_fail")

def desc(v):
    return f"n={len(v):4d} mean={st.mean(v):6.0f} median={st.median(v):6.0f}" if v else "n=0"

print("Pooled steps 1-8 by group type and outcome:")
for eng, G in (("vLLM", A), ("MINF", M)):
    b = defaultdict(list)
    for g in G.values():
        kd = kind(g)
        for x in g:
            b[(kd, "resolved" if x["reward"] > 0 else "failed")].append(x["L"])
    for key in sorted(b):
        print(f"  {eng} {key[0]:8s} {key[1]:8s} {desc(b[key])}")

print("\nMixed-group failures by error kind:")
for eng, G in (("vLLM", A), ("MINF", M)):
    b = defaultdict(list); t = defaultdict(list)
    for g in G.values():
        if kind(g) != "mixed": continue
        for x in g:
            if x["reward"] <= 0:
                b[x["err"]].append(x["L"]); 
                if x["turns"] is not None: t[x["err"]].append(x["turns"])
    for k in sorted(b):
        print(f"  {eng} {k:15s} {desc(b[k])} turns={st.mean(t[k]) if t[k] else float('nan'):5.0f}")

print("\nMixed-group successes by error kind:")
for eng, G in (("vLLM", A), ("MINF", M)):
    b = defaultdict(list)
    for g in G.values():
        if kind(g) != "mixed": continue
        for x in g:
            if x["reward"] > 0: b[x["err"]].append(x["L"])
    for k in sorted(b):
        print(f"  {eng} {k:15s} {desc(b[k])}")

print("\nPaired prompts mixed in both engines:")
df = []; ds = []; per_step = defaultdict(lambda: [0, 0, 0, 0])
for key in A:
    if key not in M or kind(A[key]) != "mixed" or kind(M[key]) != "mixed": continue
    fa = st.mean(x["L"] for x in A[key] if x["reward"] <= 0); fm = st.mean(x["L"] for x in M[key] if x["reward"] <= 0)
    sa = st.mean(x["L"] for x in A[key] if x["reward"] > 0); sm = st.mean(x["L"] for x in M[key] if x["reward"] > 0)
    df.append(fm - fa); ds.append(sm - sa)
    p = per_step[key[0]]; p[0] += 1; p[1] += fm < fa; p[2] += sm > sa; p[3] += (fm - sm) < (fa - sa)
print(f"  failures : MINF shorter in {sum(d < 0 for d in df)}/{len(df)}  mean diff {st.mean(df):+.0f}  median diff {st.median(df):+.0f}")
print(f"  successes: MINF longer  in {sum(d > 0 for d in ds)}/{len(ds)}  mean diff {st.mean(ds):+.0f}  median diff {st.median(ds):+.0f}")
print("  step | prompts | MINF failures shorter | MINF successes longer | MINF gap smaller")
for s in sorted(per_step):
    p = per_step[s]; print(f"  {s:4d} | {p[0]:7d} | {p[1]:21d} | {p[2]:21d} | {p[3]:16d}")

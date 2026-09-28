import glob,csv,collections,re
import numpy as np
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
rows=[]
for fn in glob.glob(f"{L}/mainchain/*.csv"):
    rows+=list(csv.DictReader(open(fn)))
print(f"main MINF chain (nano35-swe-v2-stream128-inorder1-cmh-64n-minf): {len(rows)} rollout result dirs read from {len(glob.glob(f'{L}/mainchain/*.csv'))} Gym results dirs")
def q(v,p): return float(np.percentile(v,p)) if len(v) else float('nan')
by=collections.defaultdict(list)
for r in rows:
    if r["step"]: by[int(r["step"])].append(r)
unm=sum(1 for r in rows if not r["step"]); print(f"rollouts whose instance is not in the step map (steps > 36 or unknown): {unm}")
print("\nMAIN MINF CHAIN per train step (from OpenHands per-turn token usage; 'turn' = one assistant message incl. its thinking):")
print(f"{'step':>4} {'n':>5} {'resolved':>8} {'compl tok mean':>14} {'p50':>7} {'longest turn p50':>16} {'p99':>8} {'max':>8} | {'rollouts w/ turn >=8k':>21} {'>=30k':>6} {'>=100k':>7}")
for s in sorted(by):
    R=by[s]; ct=[int(r["completion_tokens"]) for r in R]; mt=[int(r["max_turn"]) for r in R]
    print(f"{s:>4} {len(R):>5} {np.mean([int(r['resolved']) for r in R]):>8.3f} {np.mean(ct):>14,.0f} {q(ct,50):>7,.0f} {q(mt,50):>16,.0f} {q(mt,99):>8,.0f} {max(mt):>8,} | {sum(1 for r in R if int(r['turns_ge8k'])>0):>21} {sum(1 for r in R if int(r['turns_ge30k'])>0):>6} {sum(1 for r in R if int(r['turns_ge100k'])>0):>7}")
# comparables from token dumps: per rollout longest reasoning block
print("\nDUMP ARMS per train step (from token dumps; 'block' = one reasoning block, i.e. the thinking part of a turn), 512 trained rollouts per step:")
ARMS=["G","I","A"]; NAMES={"G":"chain G MINF from0 no-pfx","I":"chain I MINF from0 pfx-on","A":"run A vLLM from0"}
P=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0]))  # (arm,step)->sid->[max_blk, reason_tok]
for arm in ARMS:
    for fn in glob.glob(f"{L}/parts/{arm}__s*_c*.items.csv"):
        with open(fn) as fh:
            rd=csv.reader(fh); next(rd)
            for row in rd:
                step=int(row[1]); ntok=int(row[9]); e=P[(arm,step)][row[2]]; e[0]=max(e[0],ntok); e[1]+=ntok
steps=sorted({s for (_,s) in P})
print(f"{'step':>4} " + " ".join(f"{NAMES[a]+': blk p99':>28} {'>=8k':>5} {'>=30k':>6} {'>=100k':>7}" for a in ARMS))
for s in steps:
    cells=[]
    for a in ARMS:
        d=P.get((a,s))
        if not d: cells.append(f"{'':>28} {'':>5} {'':>6} {'':>7}"); continue
        mb=[v[0] for v in d.values()]
        cells.append(f"{q(mb,99):>28,.0f} {sum(1 for x in mb if x>=8000):>5} {sum(1 for x in mb if x>=30000):>6} {sum(1 for x in mb if x>=100000):>7}")
    print(f"{s:>4} "+" ".join(cells))

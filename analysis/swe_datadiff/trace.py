import glob,csv,collections,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
def f(x,d=np.nan):
    try: return float(x)
    except: return d
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return (c-h,c+h)
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
SA={}; TA={}
for a,o in ARMS:
    S=[]; T=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.samples.csv")):
        with open(fn) as fh: S+=list(csv.DictReader(fh))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.turns.csv")):
        with open(fn) as fh: T+=list(csv.DictReader(fh))
    SA[a]=S; TA[a]=T
P("TRACE 1. success rate of rollouts by their LONGEST turn (tokens), steps 1-10. n = rollouts, rate = mean reward, [95% CI]")
bins=[(0,1000),(1000,2000),(2000,4000),(4000,8000),(8000,16000),(16000,10**9),(4000,16000)]
P(f"{'arm':<4}{'out':<5} "+" ".join(f"{(str(lo)+'-'+(str(hi) if hi<10**9 else 'inf')):>22}" for lo,hi in bins))
for a,o in ARMS:
    cells=[]
    for lo,hi in bins:
        rs=[r for r in SA[a] if lo<=f(r["max_turn"])<hi]; k=sum(1 for r in rs if f(r["reward"])>0); n=len(rs); ci=wilson(k,n)
        cells.append(f"{k:>3}/{n:<4}={k/max(1,n):.3f}[{ci[0]:.2f},{ci[1]:.2f}]")
    P(f"{a:<4}{o:<5} "+" ".join(f"{c:>22}" for c in cells))
P("")
P("TRACE 2. rollouts with at least one 4k-16k turn: how many, how many long turns each, success rate; and success rate of the OTHER rollouts")
P(f"{'arm':<4}{'out':<5} {'n rollouts':>10} {'long turns/rollout':>18} {'success':>8} {'[CI]':>13} | {'others success':>14} | {'success by # long turns: 1':>26} {'2':>6} {'3+':>6}")
for a,o in ARMS:
    per=collections.Counter(r["sample_id"] for r in TA[a] if 4000<=f(r["n_tok"])<16000); S={r["sample_id"]:r for r in SA[a]}
    rs=[S[s] for s in per]; k=sum(1 for r in rs if f(r["reward"])>0); n=len(rs); ci=wilson(k,n)
    others=[r for s,r in S.items() if s not in per]; ko=sum(1 for r in others if f(r["reward"])>0)
    by={c:[S[s] for s,cnt in per.items() if (cnt==c if c<3 else cnt>=3)] for c in (1,2,3)}
    P(f"{a:<4}{o:<5} {n:>10} {np.mean(list(per.values())):>18.2f} {k/n:>8.3f} [{ci[0]:.3f},{ci[1]:.3f}] | {ko/len(others):>14.3f} | "+" ".join(f"{sum(1 for r in by[c] if f(r['reward'])>0)/max(1,len(by[c])):.3f}({len(by[c])})" for c in (1,2,3)))
P("")
P("TRACE 3. per step: rewarded rollouts with a 4k-16k turn (count) and mean number of 4k-16k turns per rewarded rollout")
P(f"{'arm':<4}{'out':<5} "+" ".join(f"{s:>7}" for s in range(1,11))+f" {'total':>6}")
for a,o in ARMS:
    S={r["sample_id"]:r for r in SA[a]}; per=collections.defaultdict(int)
    for r in TA[a]:
        if 4000<=f(r["n_tok"])<16000: per[r["sample_id"]]+=1
    cells=[]; tot=0
    for s in range(1,11):
        rs=[S[x] for x in per if int(f(S[x]["step"]))==s and f(S[x]["reward"])>0]; tot+=len(rs); cells.append(f"{len(rs):>3}/{sum(per[x['sample_id']] for x in rs):<3}")
    P(f"{a:<4}{o:<5} "+" ".join(f"{c:>7}" for c in cells)+f" {tot:>6}")
P("   (rewarded rollouts with a long turn / long turns they contain)")
P("")
P("TRACE 4. where do rewarded long turns sit? think share, zlib, relative position, tool called next (from turn features), compared with punished long turns")
for a,o in ARMS:
    lg=[r for r in TA[a] if 4000<=f(r["n_tok"])<16000]
    for sign,lab in ((1,"rewarded"),(-1,"punished")):
        rs=[r for r in lg if (f(r["adv"])>1e-9 if sign==1 else f(r["adv"])<-1e-9)]
        if not rs: continue
        P(f"{a:<2}{o:<5}{lab:<9} n {len(rs):>4} mean tok {np.mean([f(r['n_tok']) for r in rs]):>6.0f} think share {np.mean([f(r['think_tok'])/f(r['n_tok']) for r in rs]):.2f} zlib mean {np.nanmean([f(r['zlib_reason']) for r in rs]):.3f} share zlib<0.20 {np.mean([f(r['zlib_reason'],1)<0.20 for r in rs]):.2f} rel pos {np.mean([f(r['turn'])/max(1,f(r['n_turns'])) for r in rs]):.2f} is last turn {np.mean([f(r['turn'])==f(r['n_turns'])-1 for r in rs]):.2f} no-close {np.mean([f(r['has_close'])==0 for r in rs]):.3f} mean engine lp {np.mean([f(r['mean_gen_lp']) for r in rs]):.3f} d_close {np.nanmean([f(r['d_close']) for r in rs]):+.4f}")
open(f"{D}/trace_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

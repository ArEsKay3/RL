import glob,csv,collections,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","MINF","BAD"),("I","MINF","BAD"),("J","MINF","good"),("A","vLLM","good"),("K","vLLM","good"),("M","vLLM","good"),("N","vLLM","good")]
def f(x,d=np.nan):
    try: return float(x)
    except: return d
def wil(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return (c-h,c+h)
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("WHAT HAPPENS AFTER A LONG THINK TURN (4k-16k tokens), steps 1-10, per arm. Each long turn that is not the rollout's last turn is an event; 'before' = the 3 turns preceding it, 'after' = the 3 turns following it.")
rows_all={}
for a,eng,o in ARMS:
    T=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s0*_c*.turns.csv")):
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                if int(r["step"])<=10: T.append(r)
    by=collections.defaultdict(list)
    for r in T: by[r["sample_id"]].append(r)
    ev=[]
    for sid,rs in by.items():
        rs.sort(key=lambda r:f(r["turn"])); n=len(rs); lens=[f(r["n_tok"]) for r in rs]; zs=[f(r["zlib_reason"],1) for r in rs]; th=[f(r["think_tok"]) for r in rs]
        reward=f(rs[0]["reward"]); adv=f(rs[0]["adv"]); last=rs[-1]; runaway=(f(last["has_close"])==0 and f(last["ends_seq"])==1)
        for k in range(n-1):
            if 4000<=lens[k]<16000:
                before=lens[max(0,k-3):k]; after=lens[k+1:k+4]; rest=lens[k+1:]
                ev.append(dict(k=k,n=n,reward=reward,adv=adv,len=lens[k],next1=lens[k+1],after=np.mean(after),before=np.mean(before) if before else np.nan,
                               rest_max=max(rest),rest_long=sum(1 for x in rest if x>=4000),rest_long3=sum(1 for x in after if x>=4000),rest_n=len(rest),rest_frac2k=np.mean([x>=2000 for x in rest]),
                               later_loop=any(th[j]>=1000 and zs[j]<0.10 for j in range(k+1,n)),later_near=any(th[j]>=1000 and zs[j]<0.25 for j in range(k+1,n)),
                               runaway=runaway,z=zs[k],next_z=zs[k+1] if th[k+1]>=1000 else np.nan,ends_within3=(n-1-k)<=3))
    rows_all[a]=ev
P(f"{'arm':<3}{'eng':<5}{'out':<5} {'events':>6} {'next turn tok':>13} {'after/before':>12} {'next also >=4k':>14} {'>=4k in next 3':>14} {'more >=4k later':>15} {'rest max tok':>12} {'rest frac>=2k':>13} {'later loop':>10} {'later near-rep':>14} {'ends runaway':>12} {'ends within 3':>13} {'success':>8}")
for a,eng,o in ARMS:
    ev=rows_all[a]; n=len(ev)
    P(f"{a:<3}{eng:<5}{o:<5} {n:>6} {np.mean([e['next1'] for e in ev]):>13.0f} {np.nanmean([e['after']/e['before'] for e in ev if e['before']>0]):>12.2f} {np.mean([e['next1']>=4000 for e in ev]):>14.3f} {np.mean([e['rest_long3']>0 for e in ev]):>14.3f} {np.mean([e['rest_long']>0 for e in ev]):>15.3f} {np.mean([e['rest_max'] for e in ev]):>12.0f} {np.mean([e['rest_frac2k'] for e in ev]):>13.3f} {np.mean([e['later_loop'] for e in ev]):>10.3f} {np.mean([e['later_near'] for e in ev]):>14.3f} {np.mean([e['runaway'] for e in ev]):>12.3f} {np.mean([e['ends_within3'] for e in ev]):>13.3f} {np.mean([e['reward'] for e in ev]):>8.3f}")
P("")
P("pooled by engine (steps 1-10), with 95% CI where binary")
for eng in ("MINF","vLLM"):
    ev=[e for a,g,o in ARMS if g==eng for e in rows_all[a]]; n=len(ev)
    def b(key): k=sum(1 for e in ev if e[key]); lo,hi=wil(k,n); return f"{k/n:.3f} [{lo:.3f},{hi:.3f}]"
    P(f"  {eng}: events {n}; next turn >=4k {b('next1_ge4k') if False else ''}{np.mean([e['next1']>=4000 for e in ev]):.3f}; another >=4k later {np.mean([e['rest_long']>0 for e in ev]):.3f}; later loop {b('later_loop')}; later near-rep {b('later_near')}; ends runaway {b('runaway')}; success {np.mean([e['reward'] for e in ev]):.3f}; after/before length ratio {np.nanmean([e['after']/e['before'] for e in ev if e['before']>0]):.2f}; rest max tok {np.mean([e['rest_max'] for e in ev]):.0f}")
P("")
P("SPLIT BY OUTCOME: the same after-turn statistics for long turns in FAILED vs SUCCESSFUL rollouts (does a long think in a failing rollout look like the start of a degradation more often in vLLM data?)")
P(f"{'arm':<3}{'eng':<5}{'out':<5} | {'failed: events':>14} {'next>=4k':>8} {'later loop':>10} {'later near':>10} {'runaway':>8} {'rest max':>8} {'after/before':>12} | {'succ: events':>12} {'next>=4k':>8} {'later near':>10} {'rest max':>8} {'after/before':>12}")
for a,eng,o in ARMS:
    ev=rows_all[a]
    for lab,sel in (("failed",lambda e:e["reward"]<=0),("succ",lambda e:e["reward"]>0)):
        es=[e for e in ev if sel(e)]
        if lab=="failed": s1=f"{len(es):>14} {np.mean([e['next1']>=4000 for e in es]):>8.3f} {np.mean([e['later_loop'] for e in es]):>10.3f} {np.mean([e['later_near'] for e in es]):>10.3f} {np.mean([e['runaway'] for e in es]):>8.3f} {np.mean([e['rest_max'] for e in es]):>8.0f} {np.nanmean([e['after']/e['before'] for e in es if e['before']>0]):>12.2f}"
        else: s2=f"{len(es):>12} {np.mean([e['next1']>=4000 for e in es]):>8.3f} {np.mean([e['later_near'] for e in es]):>10.3f} {np.mean([e['rest_max'] for e in es]):>8.0f} {np.nanmean([e['after']/e['before'] for e in es if e['before']>0]):>12.2f}"
    P(f"{a:<3}{eng:<5}{o:<5} | {s1} | {s2}")
P("")
P("TRANSITION: given a long think turn, probability that the rollout's remaining part contains a FULL loop or ends in runaway ('degrades'), by arm, and the success rate of long-turn rollouts that did NOT degrade")
P(f"{'arm':<3}{'eng':<5}{'out':<5} {'P(degrade | long turn)':>23} {'success | no degrade':>20} {'success | degrade':>18} {'n no-degrade':>12}")
for a,eng,o in ARMS:
    ev=rows_all[a]; deg=[e for e in ev if e["later_loop"] or e["runaway"]]; nd=[e for e in ev if not (e["later_loop"] or e["runaway"])]
    lo,hi=wil(len(deg),len(ev))
    P(f"{a:<3}{eng:<5}{o:<5} {len(deg)/len(ev):>12.3f} [{lo:.3f},{hi:.3f}] {np.mean([e['reward'] for e in nd]):>20.3f} {np.mean([e['reward'] for e in deg]) if deg else float('nan'):>18.3f} {len(nd):>12}")
open(f"{D}/after_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

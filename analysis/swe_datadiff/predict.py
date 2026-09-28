import glob,csv,math,sys
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
# known-outcome reference arms (loop share at steps 20-30 decides): bad = looped, good = stayed healthy
REF=[("G","Minf",             "bad"),("I","Minf","bad"),
     ("A","VLLM","good"),("K","VLLM","good"),("M","VLLM","good"),("N","VLLM","good"),("J","VLLM (healthy Minf run)","good"),
     ("P4","Minf (no cache invalidation)","good"),("Q4","VLLM (no prefix caching)","good")]
NEW=[("P42","Minf (no cache invalidation), seed 1234"),("Q42","VLLM (no prefix caching), seed 1234")]
CLASSES=[("E(think 4k-16k)","long"),("E(near-rep)","near"),("E(turns 1k-16k)","l1")]
def loadcsv(p):
    rows=[]
    for fn in sorted(glob.glob(p)):
        with open(fn) as fh: rows+=list(csv.DictReader(fh))
    return rows
def col(rows,k):
    o=np.empty(len(rows))
    for i,r in enumerate(rows):
        try: o[i]=float(r[k]) if r[k]!="" else np.nan
        except: o[i]=np.nan
    return o
rng=np.random.default_rng(5); NB=2000
data={}
for a,_,*_ in REF+NEW:
    t=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv")
    if not t: continue
    data[a]=dict(step=col(t,"step"),adv=col(t,"adv"),n=col(t,"n_tok"),think=col(t,"think_tok"),z=col(t,"zlib_reason"),gid=np.array([r["sample_id"].rsplit("_g",1)[0] for r in t]))
def sel_of(d,cls):
    if cls=="long": return (d["n"]>=4000)&(d["n"]<16000)
    if cls=="near": return (d["think"]>=1000)&(d["z"]<0.25)
    if cls=="l1": return (d["n"]>=1000)&(d["n"]<16000)
def E(a,k,cls,lo=1):
    d=data[a]; m=(d["step"]>=lo)&(d["step"]<=k)
    if not m.any(): return None
    u,g=np.unique(d["gid"][m],return_inverse=True); ng=len(u); sel=sel_of(d,cls)&m; gi=np.searchsorted(u,d["gid"][sel])
    N=np.bincount(g,weights=d["n"][m],minlength=ng); S=np.bincount(gi,weights=(d["adv"]*d["n"])[sel],minlength=ng)
    W=rng.multinomial(ng,np.ones(ng)/ng,size=NB).astype(float); b=1e3*(W@S)/(W@N)
    return 1e3*S.sum()/N.sum(),b
def loopshare(a,lo,hi):
    d=data[a]; m=(d["step"]>=lo)&(d["step"]<=hi)
    if not m.any(): return float("nan")
    lp=m&(d["think"]>=1000)&(d["z"]<0.10); return 100*d["n"][lp].sum()/d["n"][m].sum()
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
maxstep={a:int(np.nanmax(data[a]["step"])) for a in data}
P("EARLY-WARNING CALIBRATION. Statistic: E over training steps 1..k of a class of think turns (1000 x sum(advantage x tokens) / all generated tokens). Reference arms: bad = looped later (G, I), good = stayed healthy. Question: at which k does the statistic separate the outcomes, and where do the new seed-1234 arms fall?")
P("outcome check for the reference arms (loop-token share %, think turns >= 1k with zlib < 0.10): "+"; ".join(f"{a} 11-20: {loopshare(a,11,20):.1f}, 21-30: {loopshare(a,21,30):.1f}, 31-40: {loopshare(a,31,40):.1f}" for a,_,_ in REF if a in data))
for cname,cls in CLASSES:
    P(""); P(f"--- {cname}, steps 1..k")
    P(f"{'k':>3} | "+" ".join(f"{a:>7}" for a,_,_ in REF)+" | "+f"{'bad max':>8} {'good min':>8} {'gap':>6} {'LOO acc':>7} | "+" ".join(f"{a:>16}" for a,_ in NEW))
    for k in (3,5,7,10,13,15,20):
        vals={}; bs={}
        for a,_,_ in REF:
            if a in data and maxstep[a]>=k:
                r=E(a,k,cls); vals[a]=r[0]; bs[a]=r[1]
        if len([a for a,_,o in REF if o=="bad" and a in vals])<2: continue
        bad=[vals[a] for a,_,o in REF if o=="bad" and a in vals]; good=[vals[a] for a,_,o in REF if o=="good" and a in vals]
        gap=min(bad)-max(good)
        # leave-one-out: threshold = midpoint between mean(bad) and mean(good) of the others; classify held-out arm
        correct=0; tot=0
        for a,_,o in REF:
            if a not in vals: continue
            ob=[vals[x] for x,_,oo in REF if oo=="bad" and x in vals and x!=a]; og=[vals[x] for x,_,oo in REF if oo=="good" and x in vals and x!=a]
            if not ob or not og: continue
            thr=(np.mean(ob)+np.mean(og))/2; pred="bad" if vals[a]>thr else "good"; correct+=int(pred==o); tot+=1
        newcells=[]
        for a,_ in NEW:
            if a in data and maxstep[a]>=k:
                o,b=E(a,k,cls); newcells.append(f"{o:+6.2f} [{np.percentile(b,2.5):+5.1f},{np.percentile(b,97.5):+5.1f}]")
            else: newcells.append(f"{'.':>16}")
        P(f"{k:>3} | "+" ".join(f"{vals[a]:>+7.2f}" if a in vals else f"{'.':>7}" for a,_,_ in REF)+f" | {max(bad):>+8.2f} {min(good):>+8.2f} {gap:>+6.2f} {correct:>3}/{tot:<3} | "+" ".join(newcells))
P("")
P("PREDICTION for the new arms at their current step count. For each class: value at k = steps available, the good arms' mean and sd at the same k, the bad arms' values, z vs good, and a two-class score. Score = log-likelihood ratio under Gaussians fitted to the arm-level values at that k (variance = between-arm sd^2 + this arm's bootstrap variance), prior 50/50; P(bad) is rough with only two bad arms.")
for a,lab in NEW:
    if a not in data: P(f"{lab}: no features yet"); continue
    k=maxstep[a]; P(f"\n{lab}: steps available 1-{k}; loop share 1-{k}: {loopshare(a,1,k):.2f} %" + (f", 11-{k}: {loopshare(a,11,k):.2f} %" if k>10 else ""))
    P(f"  {'class':<18} {'value [95% CI]':>22} {'good mean +/- sd (n)':>22} {'bad values':>16} {'z vs good':>9} {'P(bad)':>7}")
    llr_tot=0.0
    for cname,cls in CLASSES:
        o,b=E(a,k,cls); ref={x:E(x,k,cls) for x,_,_ in REF if x in data and maxstep[x]>=k}
        good=np.array([ref[x][0] for x,_,oo in REF if oo=="good" and x in ref]); bad=np.array([ref[x][0] for x,_,oo in REF if oo=="bad" and x in ref])
        sg=math.sqrt(good.std(ddof=1)**2+b.var()); sb=math.sqrt((bad.std(ddof=1) if len(bad)>1 else good.std(ddof=1))**2+b.var())
        lg=-0.5*((o-good.mean())/sg)**2-math.log(sg); lb=-0.5*((o-bad.mean())/sb)**2-math.log(sb); llr=lb-lg; llr_tot+=llr
        pb=1/(1+math.exp(-llr))
        P(f"  {cname:<18} {o:>+7.2f} [{np.percentile(b,2.5):+6.2f},{np.percentile(b,97.5):+6.2f}] {good.mean():>+8.2f} +/- {good.std(ddof=1):4.2f} ({len(good)}) {' '.join(f'{v:+.2f}' for v in bad):>16} {(o-good.mean())/good.std(ddof=1):>+9.2f} {pb:>7.2f}")
    P(f"  combined (three classes treated as independent, which overstates certainty): P(bad) = {1/(1+math.exp(-llr_tot)):.2f}")
P("")
P("Decision rule used earlier (steps 1-10): E(think 4k-16k) above about -1.5 or E(near-rep) above about -2 puts a run with the two arms that later looped.")
open(f"{D}/predict_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

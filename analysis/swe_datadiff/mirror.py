import glob,csv,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
# first-segment step ranges (from segments_report.txt); Q4 = single segment so far
RUNS=[("Q4","VLLM (no prefix caching)","cache: none",(1,10)),("A","VLLM","cache kept",(1,8)),("K","VLLM","cache kept",(1,14)),("M","VLLM","cache kept",(1,16)),("N","VLLM","cache kept",(1,18)),("P4","Minf (no cache invalidation)","cache kept",(1,15)),("G","Minf","cache: none",(1,6)),("I","Minf","cache invalidated",(1,13)),("J","VLLM (the healthy Minf run)","cache invalidated",(1,12))]
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
def ols(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); b=np.polyfit(x,y,1); r=y-(b[0]*x+b[1]); se=math.sqrt((r@r)/max(1,len(x)-2)/((x-x.mean())@(x-x.mean()))) if len(x)>2 else float("nan"); return b[0],se
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("MIRROR TEST: does disabling vLLM's prefix cache remove the growing engine-trainer mismatch? Steps 1-10 restricted to each run's first Slurm segment (fresh engine at step 1). |d| x1e3 = token-weighted mean |trainer - engine logprob| over short turns (< 500 tokens); turn 0 = first assistant turn of each rollout. Slope = OLS per step with se.")
P(f"{'run':<28}{'cache':<20} {'steps':>6} "+" ".join(f"{s:>5}" for s in range(1,11))+f" | {'slope all':>12} {'slope turn0':>12} | {'mean 6-10 minus 1-5: all':>24} {'turn0':>6}")
for a,lab,cache,(lo,hi) in RUNS:
    T=loadcsv(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv")+loadcsv(f"{D}/parts/{a}__s010_c*.turns.csv")
    if not T: P(f"{lab:<28}{cache:<20} no features"); continue
    step=col(T,"step").astype(int); n=col(T,"n_tok"); ad=col(T,"mean_abs_d"); tk=col(T,"turn"); short=(n<500)&(n>=20)&~np.isnan(ad)
    steps=[s for s in range(1,11) if lo<=s<=hi and (short&(step==s)).sum()>300]
    da=[1e3*np.average(ad[short&(step==s)],weights=n[short&(step==s)]) for s in steps]
    d0=[1e3*np.average(ad[short&(step==s)&(tk==0)],weights=n[short&(step==s)&(tk==0)]) for s in steps]
    sa,sea=ols(steps,da); s0,se0=ols(steps,d0)
    cells=[f"{da[steps.index(s)]:5.2f}" if s in steps else "    ." for s in range(1,11)]
    m15=np.mean([v for s,v in zip(steps,da) if s<=5]); m610=np.mean([v for s,v in zip(steps,da) if s>=6]) if any(s>=6 for s in steps) else float("nan")
    t15=np.mean([v for s,v in zip(steps,d0) if s<=5]); t610=np.mean([v for s,v in zip(steps,d0) if s>=6]) if any(s>=6 for s in steps) else float("nan")
    P(f"{lab:<28}{cache:<20} {steps[0]:>2}-{steps[-1]:<3} "+" ".join(cells)+f" | {sa:>+6.3f}±{sea:.3f} {s0:>+6.3f}±{se0:.3f} | {m610-m15:>+24.2f} {t610-t15:>+6.2f}")
P("")
P("Reference from the full 30-step runs (segments_report.txt): first-turn slope +0.19 ± 0.04 x1e-3 per step with the cache kept across refits, +0.01 ± 0.03 with the cache invalidated or absent.")
open(f"{D}/mirror_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

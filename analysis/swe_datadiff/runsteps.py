import glob,csv
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("A","A"),("K","K"),("M","M"),("N","N"),("J","J"),("Q4","Q⁗"),("Q42","Q⁗2"),("G","G"),("I","I"),("P4","P⁗"),("P42","P⁗2")]
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
res={}
for a,lab in ARMS:
    T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); step=col(T,"step").astype(int); n=col(T,"n_tok"); adv=col(T,"adv"); th=col(T,"think_tok"); z=col(T,"zlib_reason")
    r={}
    for s in sorted(set(step)):
        m=step==s; mc=step<=s; lg=m&(n>=4000)&(n<16000); lgc=mc&(n>=4000)&(n<16000); lp=m&(th>=1000)&(z<0.10)
        r[s]=(1e3*(adv[lg]*n[lg]).sum()/n[m].sum(),1e3*(adv[lgc]*n[lgc]).sum()/n[mc].sum(),100*n[lp].sum()/n[m].sum())
    res[a]=r
maxs=max(max(r) for r in res.values())
md=[]
def M(*x): md.append(" ".join(str(v) for v in x))
M("# E(think turns 4k-16k) per training step, every run (2026-09-27)\n")
M("E = 1000 x sum over 4k-16k-token think turns of (advantage x turn tokens) / all generated tokens of the step (per-step) or of steps 1..k (cumulative). Loop share = tokens in think turns >= 1k tokens with zlib < 0.10, % of the step's generated tokens. Runs: A, K, M, N = vLLM from scratch; J = MINF from scratch (stayed healthy); G, I = MINF from scratch (looped); P⁗, P⁗2 = MINF from scratch with the prefix cache kept across weight updates (seeds 42, 1234); Q⁗, Q⁗2 = vLLM from scratch with prefix caching off (seeds 42, 1234).\n")
for title,idx,fmt in (("## Per-step E(4k-16k)",0,"{:+.2f}"),("## Cumulative E(4k-16k), steps 1..k",1,"{:+.2f}"),("## Loop share per step, %",2,"{:.2f}")):
    M(title+"\n"); M("| step | "+" | ".join(lab for _,lab in ARMS)+" |"); M("|---|"+"---|"*len(ARMS))
    for s in range(1,maxs+1):
        M(f"| {s} | "+" | ".join(fmt.format(res[a][s][idx]) if s in res[a] else "" for a,_ in ARMS)+" |")
    M("")
open(f"{D}/runsteps.md","w").write("\n".join(md)+"\n"); print("\n".join(md))

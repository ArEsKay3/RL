import glob,csv,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","MINF"),("I","MINF"),("J","MINF"),("A","vLLM"),("K","vLLM"),("M","vLLM"),("N","vLLM")]
def loadcsv(pattern):
    rows=[]
    for fn in sorted(glob.glob(pattern)):
        with open(fn) as fh: rows+=list(csv.DictReader(fh))
    return rows
def col(rows,k):
    out=np.empty(len(rows))
    for i,r in enumerate(rows):
        try: out[i]=float(r[k]) if r[k]!="" else np.nan
        except: out[i]=np.nan
    return out
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
data={}
for a,e in ARMS:
    T=loadcsv(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv")+loadcsv(f"{D}/parts/{a}__s010_c*.turns.csv")
    data[a]=dict(z=col(T,"zlib_reason"),think=col(T,"think_tok"),n=col(T,"n_tok"),start=col(T,"start"),step=col(T,"step"),dc=col(T,"d_close"),gc=col(T,"g_close"),adv=col(T,"adv"),g=np.array([r["sample_id"].rsplit("_g",1)[0] for r in T]))
P("ENGINE FINGERPRINT DETAIL: share of long think turns (>=1k think tokens) that are near-repetitive (zlib < 0.25), by engine, by step window and by context position; and the trainer-minus-engine logprob at </think> by position. Group bootstrap CIs pooled by engine.")
rng=np.random.default_rng(5)
def pooled(engine,sel_fn,val_fn,nb=1000):
    vals=[];gids=[]
    for a,e in ARMS:
        if e!=engine: continue
        d=data[a]; m=sel_fn(d); vals.append(val_fn(d)[m]); gids.append(np.array([a+"|"+x for x in d["g"][m]]))
    v=np.concatenate(vals); g=np.concatenate(gids); u,inv=np.unique(g,return_inverse=True); ng=len(u)
    s=np.bincount(inv,weights=np.nan_to_num(v),minlength=ng); c=np.bincount(inv,weights=(~np.isnan(v)).astype(float),minlength=ng)
    W=rng.multinomial(ng,np.ones(ng)/ng,size=nb).astype(float); bs=(W@s)/(W@c); return np.nansum(v)/np.sum(~np.isnan(v)),np.percentile(bs,2.5),np.percentile(bs,97.5),int(np.sum(~np.isnan(v)))
def line(label,sel_fn,val_fn):
    r={e:pooled(e,sel_fn,val_fn) for e in ("MINF","vLLM")}
    P(f"  {label:<34} MINF {r['MINF'][0]:.4f} [{r['MINF'][1]:.4f},{r['MINF'][2]:.4f}] (n {r['MINF'][3]:>6})   vLLM {r['vLLM'][0]:.4f} [{r['vLLM'][1]:.4f},{r['vLLM'][2]:.4f}] (n {r['vLLM'][3]:>6})   diff {r['MINF'][0]-r['vLLM'][0]:+.4f}")
P(""); P("share zlib<0.25 among turns with >=1k think tokens")
line("steps 1-10",lambda d:d["think"]>=1000,lambda d:(d["z"]<0.25).astype(float))
line("steps 1-3 (weights ~ base)",lambda d:(d["think"]>=1000)&(d["step"]<=3),lambda d:(d["z"]<0.25).astype(float))
line("steps 4-10",lambda d:(d["think"]>=1000)&(d["step"]>=4),lambda d:(d["z"]<0.25).astype(float))
for lo,hi in [(0,32000),(32000,64000),(64000,128000),(128000,10**9)]:
    line(f"context {lo//1000}k-{hi//1000 if hi<10**9 else 'inf'}k",lambda d,lo=lo,hi=hi:(d["think"]>=1000)&(d["start"]>=lo)&(d["start"]<hi),lambda d:(d["z"]<0.25).astype(float))
for lo,hi in [(1000,2000),(2000,4000),(4000,8000),(8000,10**9)]:
    line(f"think tokens {lo}-{hi if hi<10**9 else 'inf'}",lambda d,lo=lo,hi=hi:(d["think"]>=lo)&(d["think"]<hi),lambda d:(d["z"]<0.25).astype(float))
P(""); P("mean zlib of think, turns >=1k think")
line("steps 1-10",lambda d:d["think"]>=1000,lambda d:d["z"])
line("rewarded turns only",lambda d:(d["think"]>=1000)&(d["adv"]>1e-9),lambda d:d["z"])
line("punished turns only",lambda d:(d["think"]>=1000)&(d["adv"]<-1e-9),lambda d:d["z"])
P(""); P("trainer - engine logprob at </think> (d_close)")
line("all turns",lambda d:~np.isnan(d["dc"]),lambda d:d["dc"])
for lo,hi in [(0,32000),(32000,64000),(64000,128000),(128000,10**9)]:
    line(f"context {lo//1000}k-{hi//1000 if hi<10**9 else 'inf'}k",lambda d,lo=lo,hi=hi:(~np.isnan(d["dc"]))&(d["start"]>=lo)&(d["start"]<hi),lambda d:d["dc"])
P(""); P("engine logprob of </think> (g_close)")
line("all turns",lambda d:~np.isnan(d["gc"]),lambda d:d["gc"])
line("long turns >=4k",lambda d:(~np.isnan(d["gc"]))&(d["n"]>=4000),lambda d:d["gc"])
open(f"{D}/fingerprint_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

import glob,csv,math,sys
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","MINF from scratch, no pfx","BAD"),("I","MINF from scratch, pfx r1","BAD"),("J","MINF from scratch, pfx r2","good"),("A","vLLM from scratch","good"),("K","vLLM from scratch, seed 1234","good"),("M","vLLM from scratch r1 (J args)","good"),("N","vLLM from scratch r2 (J args)","good so far")]
def loadcsv(pattern):
    rows=[]
    for fn in sorted(glob.glob(pattern)):
        with open(fn) as fh: rows+=list(csv.DictReader(fh))
    return rows
def col(rows,k):
    out=np.empty(len(rows))
    for i,r in enumerate(rows):
        v=r[k]
        try: out[i]=float(v) if v!="" else np.nan
        except: out[i]=np.nan
    return out
rng=np.random.default_rng(3); NB=2000
res={}
for a,desc,outc in ARMS:
    rows=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv")
    if not rows: continue
    gids=[r["sample_id"].rsplit("_g",1)[0] for r in rows]; gmap={g:i for i,g in enumerate(sorted(set(gids)))}; g=np.array([gmap[x] for x in gids]); ng=len(gmap)
    adv=col(rows,"adv"); n=col(rows,"n_tok"); think=col(rows,"think_tok"); z=col(rows,"zlib_reason"); start=col(rows,"start"); step=col(rows,"step")
    W=rng.multinomial(ng,np.ones(ng)/ng,size=NB).astype(float); Ng=np.bincount(g,weights=n,minlength=ng)
    def E(sel):
        Ag=np.bincount(g,weights=adv*n*sel,minlength=ng); return 1e3*Ag.sum()/Ng.sum(), 1e3*(W@Ag)/(W@Ng)
    classes={"long 4k-16k":(n>=4000)&(n<16000),"long >=4k":n>=4000,"long >=2k":n>=2000,"near-rep 0.10-0.25":(think>=1000)&(z>=0.10)&(z<0.25),"near-rep 0.20-0.25":(think>=1000)&(z>=0.20)&(z<0.25),"long@16-32k ctx":(n>=4000)&(start>=16000)&(start<32000),"loop <0.10":(think>=1000)&(z<0.10),"all turns":np.ones(len(n),bool),"short <500":n<500}
    r={}
    for k,sel in classes.items():
        o,b=E(sel); r[k]=(o,np.percentile(b,2.5),np.percentile(b,97.5),b)
    lg=n>=4000; r["rewarded long / long"]=(int((lg&(adv>1e-9)).sum()),int(lg.sum()))
    r["steps"]=sorted(set(step.astype(int))); r["n_turns"]=len(n); r["desc"]=desc; r["outcome"]=outc
    # per step E for long 4k-16k
    ps={}
    for s in r["steps"]:
        ms=step==s; sel=ms&(n>=4000)&(n<16000); ps[s]=1e3*(adv[sel]*n[sel]).sum()/n[ms].sum()
    r["per_step_long"]=ps
    res[a]=r
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("CROSS-ARM TEST, steps 1-10: token-weighted advantage exposure E per 1,000 generated tokens (group bootstrap 95% CI). Outcome = whether the arm developed the loop pathology later.")
keys=["long 4k-16k","long >=4k","near-rep 0.10-0.25","near-rep 0.20-0.25","long@16-32k ctx","loop <0.10","short <500","all turns"]
P(f"{'arm':<4}{'outcome':<13}{'steps':>6} "+" ".join(f"{k:>26}" for k in keys)+f" {'rew long/long':>14}")
for a,rr in res.items():
    P(f"{a:<4}{rr['outcome']:<13}{len(rr['steps']):>6} "+" ".join(f"{rr[k][0]:>+8.2f} [{rr[k][1]:>+6.2f},{rr[k][2]:>+6.2f}]" for k in keys)+f" {rr['rewarded long / long'][0]:>5}/{rr['rewarded long / long'][1]:<5} = {100*rr['rewarded long / long'][0]/max(1,rr['rewarded long / long'][1]):.1f}%")
P("")
P("per-step E(long 4k-16k):")
P(f"{'arm':<4}"+" ".join(f"{s:>7}" for s in range(1,11))+f" {'mean':>7}")
for a,rr in res.items():
    ps=rr["per_step_long"]; P(f"{a:<4}"+" ".join(f"{ps.get(s,float('nan')):>+7.2f}" for s in range(1,11))+f" {np.mean(list(ps.values())):>+7.2f}")
P("")
P("pooled bad (G,I) vs good (A,K,J,M,N): difference of means with bootstrap se")
for k in keys:
    bad=[res[a][k] for a in res if res[a]["outcome"]=="BAD"]; good=[res[a][k] for a in res if res[a]["outcome"]!="BAD"]
    if not bad or not good: continue
    ob=np.mean([x[0] for x in bad]); og=np.mean([x[0] for x in good]); bb=np.mean([x[3] for x in bad],axis=0); bg=np.mean([x[3] for x in good],axis=0); se=(bb-bg).std(); d=ob-og
    P(f"  {k:<22} bad {ob:>+7.2f}  good {og:>+7.2f}  diff {d:>+7.2f}  se {se:.2f}  p {math.erfc(abs(d/se)/math.sqrt(2)) if se>0 else 1:.3f}   per-arm: "+", ".join(f"{a} {res[a][k][0]:+.2f}" for a in res))
open(f"{D}/arms_exposure_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

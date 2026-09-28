import glob,csv,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("Pp","P′: MINF engine, frozen P step-20 weights"),("Qp","Q′: vLLM engine, frozen P step-20 weights")]
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
rng=np.random.default_rng(21); NB=2000
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("FROZEN-WEIGHT ENGINE COMPARISON: chains P′ (MINF) and Q′ (vLLM) generate from the same frozen weights (chain P step 20, lr 0), 15 steps x 512 rollouts each. Same prompts per step. Any difference in reward/advantage coupling here is engine + sampling noise only.")
res={}
for a,lab in ARMS:
    T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); S=loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv")
    if not T: P(f"{lab}: no features yet"); continue
    adv=col(T,"adv"); n=col(T,"n_tok"); think=col(T,"think_tok"); z=col(T,"zlib_reason"); step=col(T,"step"); rew=col(T,"reward")
    gids=[r["sample_id"].rsplit("_g",1)[0] for r in T]; u,g=np.unique(gids,return_inverse=True); ng=len(u); W=rng.multinomial(ng,np.ones(ng)/ng,size=NB).astype(float)
    Ng=np.bincount(g,weights=n,minlength=ng)
    def E(sel):
        Ag=np.bincount(g,weights=adv*n*sel,minlength=ng); return 1e3*Ag.sum()/Ng.sum(),1e3*(W@Ag)/(W@Ng)
    def share(sel):
        num=np.bincount(g,weights=rew*n*sel,minlength=ng); den=np.bincount(g,weights=n*sel,minlength=ng); return num.sum()/den.sum(),(W@num)/(W@den)
    lg=(n>=4000)&(n<16000); nr=(think>=1000)&(z<0.25); lp=(think>=1000)&(z<0.10); l1=(n>=1000)&(n<16000)
    r={}
    for k,sel in (("E long 4k-16k",lg),("E near-rep",nr),("E loop<0.10",lp),("E turns 1k-16k",l1)):
        o,b=E(sel); r[k]=(o,np.percentile(b,2.5),np.percentile(b,97.5),b)
    for k,sel in (("reward share, long tokens",lg),("reward share, near-rep tokens",nr)):
        o,b=share(sel); r[k]=(o,np.percentile(b,2.5),np.percentile(b,97.5),b)
    Sr=np.array([float(x["reward"]) for x in S]); r["reward rate"]=Sr.mean(); r["loop share %"]=100*n[lp].sum()/n.sum(); r["long share %"]=100*n[lg].sum()/n.sum(); r["near share %"]=100*n[nr].sum()/n.sum(); r["mean gen"]=np.mean([float(x["n_gen"]) for x in S]); r["steps"]=(int(step.min()),int(step.max())); r["n"]=len(S)
    res[a]=r
    P(f"\n{lab}: steps {r['steps'][0]}-{r['steps'][1]}, rollouts {r['n']}, reward rate {r['reward rate']:.3f}, mean generated {r['mean gen']:.0f}, token shares: loops {r['loop share %']:.2f} %, 4k-16k turns {r['long share %']:.2f} %, near-rep {r['near share %']:.2f} %")
    for k in ("E long 4k-16k","E near-rep","E loop<0.10","E turns 1k-16k","reward share, long tokens","reward share, near-rep tokens"):
        P(f"   {k:<30} {r[k][0]:>+8.3f}  [{r[k][1]:>+7.3f}, {r[k][2]:>+7.3f}]")
    # per step E long
    P("   per-step E(long 4k-16k): "+" ".join(f"s{int(s)}:{1e3*(adv[(step==s)&lg]*n[(step==s)&lg]).sum()/n[step==s].sum():+.1f}" for s in sorted(set(step))))
if len(res)==2:
    P("\nMINF minus vLLM at frozen weights (bootstrap):")
    for k in ("E long 4k-16k","E near-rep","E loop<0.10","E turns 1k-16k","reward share, long tokens","reward share, near-rep tokens"):
        d=res["Pp"][k][0]-res["Qp"][k][0]; bs=res["Pp"][k][3]-res["Qp"][k][3]; se=bs.std(); p=math.erfc(abs(d/se)/math.sqrt(2)) if se>0 else 1
        P(f"   {k:<30} diff {d:>+8.3f}  se {se:.3f}  p {p:.3f}")
open(f"{D}/frozen_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

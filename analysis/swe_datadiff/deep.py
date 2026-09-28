import glob,csv,math,collections,sys
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","bad"),("I","bad"),("P42","bad"),("A","good"),("K","good"),("M","good"),("N","good"),("Q4","good"),("J","good"),("P4","good")]
LO,HI=(int(sys.argv[1]),int(sys.argv[2])) if len(sys.argv)>2 else (1,10)
def col(rows,k):
    o=np.empty(len(rows))
    for i,r in enumerate(rows):
        try: o[i]=float(r[k]) if r[k]!="" else np.nan
        except: o[i]=np.nan
    return o
rng=np.random.default_rng(23); NB=2000
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
DATA={}
for a,o in ARMS:
    rows=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.eff.csv")):
        s=int(fn.split("__s")[1][:3])
        if LO<=s<=HI:
            with open(fn) as fh:
                for r in csv.DictReader(fh): r["step"]=s; rows.append(r)
    if not rows: continue
    d=dict(step=np.array([r["step"] for r in rows]),adv=col(rows,"adv"),n=col(rows,"n_tok"),think=col(rows,"think_tok"),cls=np.array([r["cls"] for r in rows]),s1=col(rows,"s1mp_think_tr"),s1deep=col(rows,"s1mp_deep_tr"),s1entry=col(rows,"s1mp_entry_tr"),gid=np.array([r["sample_id"].rsplit("_g",1)[0] for r in rows]))
    d["deepn"]=np.maximum(0,d["think"]-4096); DATA[a]=d
arms=[(a,o) for a,o in ARMS if a in DATA]
P(f"DEEP-THINK GRADIENT WEIGHT, steps {LO}-{HI}. R_deep = 1000 x sum over think tokens beyond position 4096 of adv x (1 - p_trainer) / all generated tokens. Group bootstrap (2000 resamples) for CIs; bad = G, I, P⁗2; good = A, K, M, N, Q⁗, J, P⁗.")
def boot(a,sel,field,denom_all=True):
    d=DATA[a]; u,g=np.unique(d["gid"],return_inverse=True); ng=len(u)
    S=np.bincount(g,weights=np.where(sel,d["adv"]*d[field],0.0),minlength=ng); N=np.bincount(g,weights=d["n"],minlength=ng)
    W=rng.multinomial(ng,np.ones(ng)/ng,size=NB).astype(float); return 1e3*S.sum()/N.sum(),1e3*(W@S)/(W@N)
P(""); P(f"{'arm':<5}{'out':<5} {'R_deep [95% CI]':>28} {'R_deep+ (rewarded)':>18} {'R_deep- (punished)':>18} {'R+/|R-|':>8} | {'deep tokens rew':>15} {'deep tokens pun':>15} {'mean(1-p) deep rew':>18} {'mean(1-p) deep pun':>18} | {'R_entry':>8}")
res={}
for a,o in arms:
    d=DATA[a]; sel=d["think"]>=1000; o_,b=boot(a,sel,"s1deep"); res[a]=(o_,b)
    pos=sel&(d["adv"]>0); neg=sel&(d["adv"]<0)
    rp=1e3*(d["adv"][pos]*d["s1deep"][pos]).sum()/d["n"].sum(); rn=1e3*(d["adv"][neg]*d["s1deep"][neg]).sum()/d["n"].sum()
    tr=d["deepn"][pos].sum(); tp=d["deepn"][neg].sum(); mr=d["s1deep"][pos].sum()/max(1,tr); mp=d["s1deep"][neg].sum()/max(1,tp)
    re_,_=boot(a,sel,"s1entry")
    P(f"{a:<5}{o:<5} {o_:>+8.4f} [{np.percentile(b,2.5):+.4f},{np.percentile(b,97.5):+.4f}] {rp:>+18.4f} {rn:>+18.4f} {rp/abs(rn) if rn else float('nan'):>8.3f} | {int(tr):>15} {int(tp):>15} {mr:>18.4f} {mp:>18.4f} | {re_:>+8.4f}")
bad=[res[a][0] for a,o in arms if o=="bad"]; good=[res[a][0] for a,o in arms if o=="good"]
bb=np.mean([res[a][1] for a,o in arms if o=="bad"],axis=0); bg=np.mean([res[a][1] for a,o in arms if o=="good"],axis=0)
if bad and good: P(f"\nbad mean {np.mean(bad):+.4f}, good mean {np.mean(good):+.4f}, difference {np.mean(bad)-np.mean(good):+.4f}, bootstrap se {(bb-bg).std():.4f}, z {(np.mean(bad)-np.mean(good))/(bb-bg).std():+.1f}; separation gap (bad min - good max) {min(bad)-max(good):+.4f}")
P("")
P("Deep tokens by class and advantage sign: share of the arm's deep think tokens (beyond 4096) that sit in loop-class turns, for punished and rewarded rollouts; mean (1-p_trainer) of deep tokens in near-class and normal-class turns")
P(f"{'arm':<5}{'out':<5} {'pun deep in loop turns':>22} {'rew deep in loop turns':>22} {'pun deep in near':>16} {'rew deep in near':>16} | {'mean(1-p) pun normal_long deep':>30} {'pun near deep':>13} {'rew normal_long deep':>20} {'rew near deep':>13}")
for a,o in arms:
    d=DATA[a]; sel=d["think"]>=1000; pos=sel&(d["adv"]>0); neg=sel&(d["adv"]<0)
    def sh(m,c): return d["deepn"][m&(d["cls"]==c)].sum()/max(1,d["deepn"][m].sum())
    def m1p(m,c): mm=m&(d["cls"]==c); return d["s1deep"][mm].sum()/max(1,d["deepn"][mm].sum())
    P(f"{a:<5}{o:<5} {sh(neg,'loop'):>22.3f} {sh(pos,'loop'):>22.3f} {sh(neg,'near'):>16.3f} {sh(pos,'near'):>16.3f} | {m1p(neg,'normal_long'):>30.4f} {m1p(neg,'near'):>13.4f} {m1p(pos,'normal_long'):>20.4f} {m1p(pos,'near'):>13.4f}")
P("")
P("Per-step R_deep (x1e3), all steps with data:")
allsteps=sorted({int(s) for a in DATA for s in set(DATA[a]["step"])})
P(f"{'step':>4} | "+" ".join(f"{a:>7}" for a,o in arms))
for s in allsteps:
    cells=[]
    for a,o in arms:
        d=DATA[a]; m=(d["step"]==s)&(d["think"]>=1000); tot=d["n"][d["step"]==s].sum()
        cells.append(f"{1e3*(d['adv'][m]*d['s1deep'][m]).sum()/tot:>+7.3f}" if tot>0 else f"{'.':>7}")
    P(f"{s:>4} | "+" ".join(cells))
P(""); P("Cumulative R_deep over steps 1..k:")
P(f"{'k':>4} | "+" ".join(f"{a:>7}" for a,o in arms))
for s in allsteps:
    cells=[]
    for a,o in arms:
        d=DATA[a]; m=(d["step"]<=s)&(d["think"]>=1000); tot=d["n"][d["step"]<=s].sum()
        cells.append(f"{1e3*(d['adv'][m]*d['s1deep'][m]).sum()/tot:>+7.3f}" if tot>0 and (d["step"]==s).any() else f"{'.':>7}")
    P(f"{s:>4} | "+" ".join(cells))
open(f"{D}/deep_{LO}_{HI}_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

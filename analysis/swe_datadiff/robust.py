import glob,csv,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=["G","I","J","A","K","M","N"]; BAD={"G","I"}
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
A={}
for a in ARMS:
    T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv")
    A[a]=dict(adv=col(T,"adv"),n=col(T,"n_tok"),think=col(T,"think_tok"),z=col(T,"zlib_reason"),step=col(T,"step"),start=col(T,"start"))
def E(a,sel,steps=None):
    d=A[a]; m=sel(d)
    if steps: m=m&(d["step"]>=steps[0])&(d["step"]<=steps[1]); tot=d["n"][(d["step"]>=steps[0])&(d["step"]<=steps[1])].sum()
    else: tot=d["n"].sum()
    return 1e3*(d["adv"][m]*d["n"][m]).sum()/tot
defs={}
for lo in (1000,1500,2000,3000,4000,6000,8000):
    for hi in (8000,16000,32000,10**9):
        if hi>lo: defs[f"turn {lo}-{hi if hi<10**9 else 'inf'}"]=lambda d,lo=lo,hi=hi:(d["n"]>=lo)&(d["n"]<hi)
for lo in (1000,2000,4000):
    for hi in (16000,10**9):
        defs[f"think {lo}-{hi if hi<10**9 else 'inf'}"]=lambda d,lo=lo,hi=hi:(d["think"]>=lo)&(d["think"]<hi)
for zlo,zhi in ((0.10,0.25),(0.10,0.30),(0.15,0.30),(0.20,0.30),(0.0,0.30),(0.25,0.35)):
    for tmin in (500,1000,2000):
        defs[f"zlib {zlo:.2f}-{zhi:.2f} think>={tmin}"]=lambda d,zlo=zlo,zhi=zhi,tmin=tmin:(d["think"]>=tmin)&(d["z"]>=zlo)&(d["z"]<zhi)
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("ROBUSTNESS: rank of the two bad arms (G, I) among the 7 from-scratch arms when E(class) is sorted from most reinforcing (rank 1) to most suppressing (rank 7), for many class definitions and step windows.")
P(f"{'class':<32} {'steps 1-10: G I ranks':>22} {'gap bad-good':>12} | {'steps 1-5':>14} {'gap':>7} | {'steps 6-10':>14} {'gap':>7} | values 1-10 (G I J A K M N)")
top2=0; total=0
for name,sel in defs.items():
    cells=[]
    for steps in ((1,10),(1,5),(6,10)):
        v={a:E(a,sel,steps) for a in ARMS}; order=sorted(ARMS,key=lambda a:-v[a]); rk={a:order.index(a)+1 for a in ARMS}
        gap=np.mean([v[a] for a in BAD])-np.mean([v[a] for a in ARMS if a not in BAD]); cells.append((rk["G"],rk["I"],gap,v))
    r=cells[0]; total+=1; top2+=int(set([r[0],r[1]])=={1,2})
    P(f"{name:<32} {str(r[0])+' '+str(r[1]):>22} {r[2]:>+12.2f} | {str(cells[1][0])+' '+str(cells[1][1]):>14} {cells[1][2]:>+7.2f} | {str(cells[2][0])+' '+str(cells[2][1]):>14} {cells[2][2]:>+7.2f} | "+" ".join(f"{r[3][a]:+.2f}" for a in ARMS))
P(f"\nbad arms occupy ranks {{1,2}} in {top2} of {total} definitions (chance for one definition: 1/21)")
open(f"{D}/robust_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

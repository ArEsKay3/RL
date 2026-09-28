import glob,csv,math,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","MINF from0 no-pfx","BAD",1,10),("I","MINF from0 pfx r1","BAD",1,10),("J","MINF from0 pfx r2","good",1,10),("A","vLLM from0","good",1,10),("K","vLLM from0 seed1234","good",1,10),("M","vLLM from0 r1","good",1,10),("N","vLLM from0 r2","good",1,10),
      ("B","MINF <- MINF10","BAD",11,20),("C","MINF <- MINF10 no-pfx","BAD?",11,20),("D","vLLM <- MINF10","BAD","11",20),("L","MINF <- K10","BAD",11,20),("F","MINF <- vLLM10","good",11,20),
      ("P","MINF live after M-replay","good",21,30),("Q","vLLM live after M-replay","good",21,30),("R","vLLM live after G-replay 1-10","BAD",11,20),("Rp","vLLM live after G-replay 1-16","BAD",17,26)]
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
P("ALL ARMS: token-weighted advantage exposure E per 1,000 generated tokens over the arm's first ten dumped steps (or the live steps of splice runs). Also the loop-token share in the SAME steps and in the arm's later steps (from loopfeedback tables when available).")
P(f"{'arm':<3} {'what':<32} {'outcome':<8} {'steps':>7} {'n':>6} {'E long4-16k':>12} {'E near .10-.25':>14} {'E loop<.10':>11} {'E all':>8} {'rew long/long':>14} {'loop% these steps':>17}")
rows=[]
for a,desc,outc,lo,hi in ARMS:
    lo=int(lo); T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv")
    if not T: continue
    step=col(T,"step"); m=(step>=lo)&(step<=hi); T=[r for r,k in zip(T,m) if k]
    if not T: continue
    adv=col(T,"adv"); n=col(T,"n_tok"); think=col(T,"think_tok"); z=col(T,"zlib_reason"); tot=n.sum()
    E=lambda sel: 1e3*(adv[sel]*n[sel]).sum()/tot
    lg=(n>=4000)&(n<16000); nr=(think>=1000)&(z>=0.10)&(z<0.25); lp=(think>=1000)&(z<0.10)
    S=loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv"); S=[r for r in S if lo<=float(r["step"])<=hi]
    loop_share=100*sum(float(r["loop_tok"]) for r in S)/sum(float(r["n_gen"]) for r in S)
    steps=sorted({int(x) for x in step[m]})
    rows.append((a,desc,outc,f"{steps[0]}-{steps[-1]}",len(S),E(lg),E(nr),E(lp),E(np.ones(len(n),bool)),int((lg&(adv>1e-9)).sum()),int(lg.sum()),loop_share))
for r in sorted(rows,key=lambda r:-r[5]):
    P(f"{r[0]:<3} {r[1]:<32} {r[2]:<8} {r[3]:>7} {r[4]:>6} {r[5]:>+12.2f} {r[6]:>+14.2f} {r[7]:>+11.2f} {r[8]:>+8.2f} {r[9]:>5}/{r[10]:<5}={100*r[9]/max(1,r[10]):>4.1f}% {r[11]:>17.2f}")
P("")
P("sorted by E(long 4k-16k), most reinforcing first. From-scratch arms (steps 1-10) are the clean comparison; transplant arms start from step-10 weights that may already differ; splice live phases start after replayed training.")
open(f"{D}/arms_exposure2_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

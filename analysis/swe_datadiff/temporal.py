import glob,csv,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
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
WIN=[(1,5),(6,10),(11,15),(16,20),(21,25),(26,30)]
P("TEMPORAL: token-weighted advantage exposure per 1,000 generated tokens by 5-step window, with the loop-token share of the same window. Does weak suppression of long/near-repetitive thinking precede loop growth?")
for name,selfn in [("E(long 4k-16k)",lambda d:(d["n"]>=4000)&(d["n"]<16000)),("E(near-rep zlib 0.10-0.25, think>=1k)",lambda d:(d["think"]>=1000)&(d["z"]>=0.10)&(d["z"]<0.25)),("E(full loops zlib<0.10, think>=1k)",lambda d:(d["think"]>=1000)&(d["z"]<0.10)),("E(all think tokens)",None)]:
    P(""); P(name)
    P(f"{'arm':<4}{'out':<5} "+" ".join(f"{lo:>2}-{hi:<5}" for lo,hi in WIN))
    for a,o in ARMS:
        T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); d=dict(adv=col(T,"adv"),n=col(T,"n_tok"),think=col(T,"think_tok"),z=col(T,"zlib_reason"),step=col(T,"step"))
        cells=[]
        for lo,hi in WIN:
            m=(d["step"]>=lo)&(d["step"]<=hi)
            if not m.any(): cells.append("   .    "); continue
            tot=d["n"][m].sum()
            if selfn is None: v=1e3*(d["adv"][m]*d["think"][m]).sum()/tot
            else: s=selfn(d)&m; v=1e3*(d["adv"][s]*d["n"][s]).sum()/tot
            cells.append(f"{v:>+8.2f}")
        P(f"{a:<4}{o:<5} "+" ".join(cells))
P(""); P("loop-token share (%) of the window (turns with >=1k think and zlib<0.10) and share of tokens in 4k-16k turns (%)")
P(f"{'arm':<4}{'out':<5} "+" ".join(f"{lo:>2}-{hi:<5}" for lo,hi in WIN)+" | long-turn token share: "+" ".join(f"{lo:>2}-{hi:<3}" for lo,hi in WIN))
for a,o in ARMS:
    T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); d=dict(n=col(T,"n_tok"),think=col(T,"think_tok"),z=col(T,"zlib_reason"),step=col(T,"step"))
    c1=[];c2=[]
    for lo,hi in WIN:
        m=(d["step"]>=lo)&(d["step"]<=hi)
        if not m.any(): c1.append("   .    "); c2.append("  .   "); continue
        tot=d["n"][m].sum(); lp=m&(d["think"]>=1000)&(d["z"]<0.10); lg=m&(d["n"]>=4000)&(d["n"]<16000)
        c1.append(f"{100*d['n'][lp].sum()/tot:>8.2f}"); c2.append(f"{100*d['n'][lg].sum()/tot:>6.2f}")
    P(f"{a:<4}{o:<5} "+" ".join(c1)+" | "+" ".join(c2))
open(f"{D}/temporal_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

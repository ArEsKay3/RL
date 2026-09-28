import glob,csv
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("P4","Minf (no cache invalidation), seed 42"),("Q4","VLLM (no prefix caching), seed 42"),("P42","Minf (no cache invalidation), seed 1234"),("G","Minf (looped), for reference"),("A","VLLM, for reference")]
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
WIN=[(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45),(46,50),(51,55),(56,60)]
print("OUTCOME BY 5-STEP WINDOW: reward, mean generated tokens, loop-token share % (think >= 1k, zlib < 0.10), 4k-16k-turn token share %, E(think 4k-16k), E(near-rep)")
for a,lab in ARMS:
    T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); S=loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv")
    if not T: print(f"\n{lab}: no features"); continue
    step=col(T,"step"); n=col(T,"n_tok"); adv=col(T,"adv"); think=col(T,"think_tok"); z=col(T,"zlib_reason"); ss=col(S,"step"); rew=col(S,"reward"); gen=col(S,"n_gen")
    print(f"\n{lab}: steps {int(step.min())}-{int(step.max())}")
    print(f"  {'window':<7} {'n':>5} {'reward':>6} {'gen':>6} {'loop%':>6} {'long%':>6} {'E long':>7} {'E near':>7}")
    for lo,hi in WIN:
        m=(step>=lo)&(step<=hi); ms=(ss>=lo)&(ss<=hi)
        if not m.any(): continue
        tot=n[m].sum(); lg=m&(n>=4000)&(n<16000); nr=m&(think>=1000)&(z<0.25); lp=m&(think>=1000)&(z<0.10)
        print(f"  {lo:>2}-{hi:<4} {int(ms.sum()):>5} {rew[ms].mean():>6.3f} {gen[ms].mean():>6.0f} {100*n[lp].sum()/tot:>6.2f} {100*n[lg].sum()/tot:>6.2f} {1e3*(adv[lg]*n[lg]).sum()/tot:>+7.2f} {1e3*(adv[nr]*n[nr]).sum()/tot:>+7.2f}")

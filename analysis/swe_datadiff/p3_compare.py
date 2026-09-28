import glob,csv,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("P3","P‴: MINF keep-prefix from P step 20, lr 3e-6"),("P","P: MINF cache-invalidating, live after M-replay (same seed lineage)"),("Q","Q: vLLM live after M-replay"),("Pp","P′: MINF frozen P20 weights (lr 0)"),("Qp","Q′: vLLM frozen P20 weights (lr 0)")]
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
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
WIN=[(21,25),(26,30),(31,35),(36,40),(41,45),(46,50)]
P("CHAIN P‴ (MINF, prefix cache kept across refits, from chain P step 20) against its lineage: chain P (MINF, cache invalidated per refit; live steps 21-31 after replaying chain M 1-20), chain Q (vLLM, same replay), and the frozen pair P′/Q′ (same step-20 weights, lr 0).")
P("Per 5-step window: reward, mean generated tokens, loop-token share %, 4k-16k-turn token share %, E(long 4k-16k), E(near-rep zlib<0.25), and the short-turn mismatch |d| x1e3 (all / first turn) with signed d x1e3.")
for a,lab in ARMS:
    t=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); s=loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv")
    if not t: P(f"\n{lab}: no features"); continue
    step=col(t,"step"); n=col(t,"n_tok"); adv=col(t,"adv"); think=col(t,"think_tok"); z=col(t,"zlib_reason"); ad=col(t,"mean_abs_d"); dd=col(t,"mean_d"); tk=col(t,"turn"); short=(n<500)&(n>=20)&~np.isnan(ad)
    ss=col(s,"step"); rew=col(s,"reward"); gen=col(s,"n_gen")
    P(f"\n{lab}: steps {int(step.min())}-{int(step.max())}, rollouts {len(s)}")
    P(f"  {'window':<7} {'n':>5} {'reward':>6} {'gen':>6} {'loop%':>6} {'long%':>6} | {'E long':>7} {'E near':>7} {'E loop':>7} | {'|d| all':>7} {'turn0':>7} {'signed d':>8}")
    base=None
    for lo,hi in WIN:
        m=(step>=lo)&(step<=hi); ms=(ss>=lo)&(ss<=hi)
        if not m.any(): continue
        tot=n[m].sum(); lg=m&(n>=4000)&(n<16000); nr=m&(think>=1000)&(z<0.25); lp=m&(think>=1000)&(z<0.10); sh=m&short
        d_all=np.average(ad[sh],weights=n[sh]); d0=np.average(ad[sh&(tk==0)],weights=n[sh&(tk==0)]); sd=np.average(dd[sh],weights=n[sh])
        if base is None: base=(d_all,d0)
        P(f"  {lo:>2}-{hi:<4} {int(ms.sum()):>5} {rew[ms].mean():>6.3f} {gen[ms].mean():>6.0f} {100*n[lp].sum()/tot:>6.2f} {100*n[lg].sum()/tot:>6.2f} | {1e3*(adv[lg]*n[lg]).sum()/tot:>+7.2f} {1e3*(adv[nr]*n[nr]).sum()/tot:>+7.2f} {1e3*(adv[lp]*n[lp]).sum()/tot:>+7.2f} | {d_all*1e3:>7.2f} {d0*1e3:>7.2f} {sd*1e3:>+8.3f}"+(f"   (growth vs first window: all {100*(d_all/base[0]-1):+.1f} %, turn0 {100*(d0/base[1]-1):+.1f} %)" if (lo,hi)!=WIN[0] else ""))
P("")
P("Reading guide: P′/Q′ show the drift of the mismatch that comes from prompt sets alone (weights frozen: +1-2 % all, +6-8 % turn 0 over 15 steps). Chain P (cache invalidated) vs chain P‴ (cache kept) on the same seed weights and lr isolates the knob: a stale cache should make P‴'s |d| grow with training like the vLLM arms.")
open(f"{D}/p3_compare_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

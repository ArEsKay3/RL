import glob,csv,json,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","MINF no-pfx"),("I","MINF pfx"),("J","MINF pfx"),("A","vLLM"),("K","vLLM"),("M","vLLM"),("N","vLLM")]
span=json.load(open(f"{D}/span_class.json"))
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
P("STALE PREFIX CACHE TEST. In a step whose rollouts straddled a refit (start weight version v, end v+1), early turns were generated under v and late turns under v+1; the trainer scores all turns with the newer weights. Weight drift alone makes early turns' trainer-minus-engine mismatch |d| larger in BOTH engines; a prefix cache that is NOT reset on refit additionally leaves the late turns of vLLM rollouts scored by the engine with stale (old-weight) KV for the cached prefix, so their |d| should stay elevated in vLLM arms but return to baseline in MINF arms.")
P("Per-turn mean |d| (token-mean within the turn, then mean over turns), steps 1-30, by step class (none / straddled) and by relative turn position (first 30 % / last 30 % of the rollout's turns).")
P(f"{'arm':<3}{'engine':<13} | {'none: early':>11} {'late':>8} | {'straddled: early':>16} {'late':>8} {'late-early':>10} | {'mixed: early':>12} {'late':>8} | {'straddled/none ratio late':>25} | {'mean d straddled late':>21} {'mean d none late':>17}")
for a,eng in ARMS:
    T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); step=col(T,"step").astype(int); ad=col(T,"mean_abs_d"); dd=col(T,"mean_d"); rel=col(T,"turn")/np.maximum(1,col(T,"n_turns")); n=col(T,"n_tok")
    cls=np.array([span[a].get(str(s),"?") for s in step])
    def m(c,lo,hi,arr=ad): sel=(cls==c)&(rel>=lo)&(rel<hi)&(n>=20); return np.nanmean(arr[sel])
    ne,nl=m("none",0,0.3),m("none",0.7,1.01); fe,fl=m("full",0,0.3),m("full",0.7,1.01); me,ml=m("mixed",0,0.3),m("mixed",0.7,1.01)
    P(f"{a:<3}{eng:<13} | {ne:>11.5f} {nl:>8.5f} | {fe:>16.5f} {fl:>8.5f} {fl-fe:>+10.5f} | {me:>12.5f} {ml:>8.5f} | {fl/nl:>25.3f} | {m('full',0.7,1.01,dd):>+21.5f} {m('none',0.7,1.01,dd):>+17.5f}")
P("")
P("Same, restricted to steps 1-10, and the tail: share of turns with max |d| > 1 by class and position")
P(f"{'arm':<3}{'engine':<13} | {'none early':>10} {'none late':>9} | {'straddled early':>15} {'straddled late':>14} || {'max|d|>1 none early':>19} {'none late':>9} {'strad early':>11} {'strad late':>10}")
for a,eng in ARMS:
    T=loadcsv(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv")+loadcsv(f"{D}/parts/{a}__s010_c*.turns.csv"); step=col(T,"step").astype(int); ad=col(T,"mean_abs_d"); mx=col(T,"max_abs_d"); rel=col(T,"turn")/np.maximum(1,col(T,"n_turns")); n=col(T,"n_tok")
    cls=np.array([span[a].get(str(s),"?") for s in step])
    def m(c,lo,hi,arr=ad): sel=(cls==c)&(rel>=lo)&(rel<hi)&(n>=20); return np.nanmean(arr[sel])
    def t(c,lo,hi): sel=(cls==c)&(rel>=lo)&(rel<hi)&(n>=20); return np.mean(mx[sel]>1)
    P(f"{a:<3}{eng:<13} | {m('none',0,0.3):>10.5f} {m('none',0.7,1.01):>9.5f} | {m('full',0,0.3):>15.5f} {m('full',0.7,1.01):>14.5f} || {t('none',0,0.3):>19.4f} {t('none',0.7,1.01):>9.4f} {t('full',0,0.3):>11.4f} {t('full',0.7,1.01):>10.4f}")
P("")
P("Per-sample view: seq_mult_prob_error and TIS-clipped tokens by step class")
P(f"{'arm':<3}{'engine':<13} | {'none: sme median':>16} {'p99':>7} {'clip/1M tok':>11} | {'straddled: sme median':>21} {'p99':>7} {'clip/1M tok':>11}")
for a,eng in ARMS:
    S=loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv"); step=col(S,"step").astype(int); sme=col(S,"seq_mult_prob_error"); clip=col(S,"n_clip_lo")+col(S,"n_clip_hi"); ng=col(S,"n_gen")
    cls=np.array([span[a].get(str(s),"?") for s in step])
    def q(c): sel=cls==c; return np.nanmedian(sme[sel]),np.nanpercentile(sme[sel],99),1e6*clip[sel].sum()/ng[sel].sum()
    a1=q("none"); a2=q("full")
    P(f"{a:<3}{eng:<13} | {a1[0]:>16.4f} {a1[1]:>7.3f} {a1[2]:>11.1f} | {a2[0]:>21.4f} {a2[1]:>7.3f} {a2[2]:>11.1f}")
open(f"{D}/stalecache_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

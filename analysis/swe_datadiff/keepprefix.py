import glob,csv,math,json,os,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
R="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
ARMS=[("P4","MINF keep-prefix from0","?"),("G","MINF from0 no-pfx","BAD"),("I","MINF from0 pfx","BAD"),("J","MINF from0 pfx","good"),("A","vLLM from0","good"),("K","vLLM from0 seed1234","good"),("M","vLLM from0 r1","good"),("N","vLLM from0 r2","good")]
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
rng=np.random.default_rng(7); NB=2000
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
T={};S={}
for a,_,_ in ARMS:
    T[a]=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); S[a]=loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv")
def stats(a,lo,hi):
    t=T[a]; step=col(t,"step"); m=(step>=lo)&(step<=hi)
    if not m.any(): return None
    adv=col(t,"adv")[m]; n=col(t,"n_tok")[m]; think=col(t,"think_tok")[m]; z=col(t,"zlib_reason")[m]; rew=col(t,"reward")[m]
    gids=np.array([r["sample_id"].rsplit("_g",1)[0] for r,k in zip(t,m) if k]); u,g=np.unique(gids,return_inverse=True); ng=len(u); W=rng.multinomial(ng,np.ones(ng)/ng,size=NB).astype(float); Ng=np.bincount(g,weights=n,minlength=ng)
    def E(sel):
        Ag=np.bincount(g,weights=adv*n*sel,minlength=ng); return 1e3*Ag.sum()/Ng.sum(),1e3*(W@Ag)/(W@Ng)
    def share(sel):
        num=np.bincount(g,weights=rew*n*sel,minlength=ng); den=np.bincount(g,weights=n*sel,minlength=ng); return num.sum()/max(1,den.sum()),(W@num)/np.maximum(1,(W@den))
    lg=(n>=4000)&(n<16000); nr=(think>=1000)&(z<0.25); lp=(think>=1000)&(z<0.10); l1=(n>=1000)&(n<16000)
    r={}
    for k,sel in (("E long 4k-16k",lg),("E near-rep",nr),("E loop<0.10",lp),("E turns 1k-16k",l1)):
        o,b=E(sel); r[k]=(o,np.percentile(b,2.5),np.percentile(b,97.5),b)
    for k,sel in (("rew share long",lg),("rew share near",nr)):
        o,b=share(sel); r[k]=(o,np.percentile(b,2.5),np.percentile(b,97.5),b)
    ss=[x for x in S[a] if lo<=float(x["step"])<=hi]
    r["reward"]=np.mean([float(x["reward"]) for x in ss]); r["gen"]=np.mean([float(x["n_gen"]) for x in ss]); r["loop%"]=100*n[lp].sum()/n.sum(); r["long%"]=100*n[lg].sum()/n.sum(); r["near%"]=100*n[nr].sum()/n.sum(); r["n"]=len(ss); r["steps"]=sorted(set(step[m].astype(int)))
    return r
P("CHAIN P⁗ (MINF from scratch, prefix cache KEPT across refits: invalidate_prefix_cache_on_weight_update=false, nvshmem, prefix caching on, opg off, seed 42) vs the seven from-scratch arms.")
P("E(class) = 1000 x sum(advantage x turn tokens in class) / all generated tokens; group bootstrap 95% CI (2000 resamples). Reference: bad arms G/I had E(long) -0.1/-0.8 and E(near-rep) -0.9/-1.3 in steps 1-10; good arms -2.3..-4.8 and -3.6..-6.4.")
for lo,hi in ((1,10),(1,5),(6,10),(11,20)):
    P(""); P(f"--- steps {lo}-{hi}")
    P(f"{'arm':<4}{'what':<24}{'out':<5}{'n':>5} {'reward':>6} {'gen':>6} {'loop%':>5} {'long%':>5} | {'E long 4k-16k':>22} {'E near-rep':>22} {'E loop<.10':>20} {'E 1k-16k':>20} | {'rew share long':>20} {'rew share near':>20}")
    for a,what,o in ARMS:
        r=stats(a,lo,hi)
        if not r: continue
        f=lambda k:f"{r[k][0]:>+6.2f} [{r[k][1]:>+6.2f},{r[k][2]:>+6.2f}]"; s=lambda k:f"{r[k][0]:>5.3f} [{r[k][1]:.3f},{r[k][2]:.3f}]"
        P(f"{a:<4}{what:<24}{o:<5}{r['n']:>5} {r['reward']:>6.3f} {r['gen']:>6.0f} {r['loop%']:>5.2f} {r['long%']:>5.2f} | {f('E long 4k-16k'):>22} {f('E near-rep'):>22} {f('E loop<0.10'):>20} {f('E turns 1k-16k'):>20} | {s('rew share long'):>20} {s('rew share near'):>20}")
# rank of P4 among from-scratch arms, steps 1-10
r10={a:stats(a,1,10) for a,_,_ in ARMS}
P(""); P("rank of chain P⁗ among the 8 from-scratch arms, steps 1-10 (rank 1 = most reinforcing / least negative):")
for k in ("E long 4k-16k","E near-rep","E turns 1k-16k","rew share near","rew share long"):
    order=sorted(ARMS,key=lambda x:-r10[x[0]][k][0]); P(f"  {k:<16} "+" > ".join(f"{a}({r10[a][k][0]:+.2f})" if not k.startswith('rew') else f"{a}({r10[a][k][0]:.3f})" for a,_,_ in order))
# z of P4 vs good arms
P(""); P("z-score of chain P⁗ against the five good arms (mean, sd across arms), and of G/I for reference:")
for k in ("E long 4k-16k","E near-rep"):
    g=np.array([r10[a][k][0] for a,_,o in ARMS if o=="good"]); m,s=g.mean(),g.std(ddof=1)
    P(f"  {k:<16} good mean {m:+.2f} sd {s:.2f} | P4 z {(r10['P4'][k][0]-m)/s:+.2f} | G z {(r10['G'][k][0]-m)/s:+.2f} | I z {(r10['I'][k][0]-m)/s:+.2f}")
# per-step E(long) and E(near)
P(""); P("per-step E(long 4k-16k) / E(near-rep), steps 1-20:")
P(f"{'arm':<4}"+" ".join(f"{s:>13}" for s in range(1,21)))
for a,_,_ in ARMS:
    t=T[a]; step=col(t,"step").astype(int); adv=col(t,"adv"); n=col(t,"n_tok"); think=col(t,"think_tok"); z=col(t,"zlib_reason")
    cells=[]
    for s in range(1,21):
        m=step==s
        if not m.any(): cells.append(f"{'.':>13}"); continue
        tot=n[m].sum(); lg=m&(n>=4000)&(n<16000); nr=m&(think>=1000)&(z<0.25)
        cells.append(f"{1e3*(adv[lg]*n[lg]).sum()/tot:>+5.1f}/{1e3*(adv[nr]*n[nr]).sum()/tot:>+6.1f}")
    P(f"{a:<4}"+" ".join(cells))
# stale-cache signature: |d| growth on short turns by window and by turn index
P(""); P("STALE-CACHE SIGNATURE: token-weighted mean |trainer - engine logprob| over short turns (< 500 tokens), by 5-step window, and growth from window 1-5 to the latest complete window; also the first turn of each rollout (context = cached system prompt + task).")
WIN=[(1,5),(6,10),(11,15),(16,20),(21,25),(26,30)]
P(f"{'arm':<4}{'what':<24} "+" ".join(f"{lo:>2}-{hi:<6}" for lo,hi in WIN)+f" {'growth':>7} | {'turn0: 1-5':>10} {'latest':>7} {'growth':>7} | {'signed d x1e3 1-5':>17} {'latest':>7}")
for a,what,o in ARMS:
    t=T[a]; step=col(t,"step"); n=col(t,"n_tok"); ad=col(t,"mean_abs_d"); dd=col(t,"mean_d"); tk=col(t,"turn"); short=(n<500)&(n>=20)&~np.isnan(ad)
    cells=[]; vals=[]; t0=[]; sd=[]
    for lo,hi in WIN:
        m=short&(step>=lo)&(step<=hi)
        if m.sum()<1000: cells.append(f"{'.':>9}"); continue
        v=np.average(ad[m],weights=n[m]); vals.append(v); cells.append(f"{v*1e3:>9.2f}")
        m0=m&(tk==0); t0.append(np.average(ad[m0],weights=n[m0])); sd.append(1e3*np.average(dd[m],weights=n[m]))
    if len(vals)<2: P(f"{a:<4}{what:<24} "+" ".join(cells)); continue
    P(f"{a:<4}{what:<24} "+" ".join(cells)+f" {100*(vals[-1]/vals[0]-1):>+6.1f}% | {t0[0]*1e3:>10.2f} {t0[-1]*1e3:>7.2f} {100*(t0[-1]/t0[0]-1):>+6.1f}% | {sd[0]:>+17.3f} {sd[-1]:>+7.3f}")
P("  (x1e-3 units for |d|; vLLM arms grew +8..+29 % overall and +15..+33 % on turn 0 from 1-5 to 26-30; MINF cache-invalidating arms +0..+15 % / -5..+7 %)")
# per-step |d| short turns for P4 vs A, G, J
P(""); P("per-step |d| x1e3 over short turns, steps 1-20:")
for a,_,_ in ARMS:
    t=T[a]; step=col(t,"step").astype(int); n=col(t,"n_tok"); ad=col(t,"mean_abs_d"); short=(n<500)&(n>=20)&~np.isnan(ad)
    P(f"{a:<4}"+" ".join(f"{np.average(ad[short&(step==s)],weights=n[short&(step==s)])*1e3:5.2f}" if (short&(step==s)).sum()>500 else "    ." for s in range(1,21)))
open(f"{D}/keepprefix_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

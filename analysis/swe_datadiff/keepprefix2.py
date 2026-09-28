import glob,csv,math,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("P4","MINF keep-prefix","?"),("G","MINF no-pfx","BAD"),("I","MINF pfx","BAD"),("J","MINF pfx","good"),("A","vLLM","good"),("K","vLLM","good"),("M","vLLM","good"),("N","vLLM","good")]
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
rng=np.random.default_rng(9); NB=2000
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("MATCHED-WINDOW MISMATCH GROWTH, steps 1-5 -> 16-20 (the range chain P⁗ has). |d| = |trainer - engine logprob|, token-weighted over short turns (< 500 tokens); growth = ratio - 1 with a group bootstrap 95% CI. Turn 0 = first assistant turn (context = cached system prompt + task).")
P(f"{'arm':<4}{'engine':<18}{'out':<5} {'|d| 1-5':>8} {'16-20':>8} {'growth [CI]':>24} | {'turn0 1-5':>9} {'16-20':>8} {'growth [CI]':>24} | {'signed d 1-5':>12} {'16-20':>8} | {'turns 1-2':>10} {'turns>=10':>10}")
T={}
for a,eng,o in ARMS:
    t=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); T[a]=t
    step=col(t,"step"); n=col(t,"n_tok"); ad=col(t,"mean_abs_d"); dd=col(t,"mean_d"); tk=col(t,"turn"); short=(n<500)&(n>=20)&~np.isnan(ad)
    gids=np.array([r["sample_id"].rsplit("_g",1)[0] for r in t]); u,g=np.unique(gids,return_inverse=True); ng=len(u); W=rng.multinomial(ng,np.ones(ng)/ng,size=NB).astype(float)
    def wmean(sel):
        num=np.bincount(g,weights=np.where(sel,ad*n,0),minlength=ng); den=np.bincount(g,weights=np.where(sel,n,0),minlength=ng); return num.sum()/den.sum(),(W@num)/(W@den)
    m1=short&(step>=1)&(step<=5); m2=short&(step>=16)&(step<=20)
    o1,b1=wmean(m1); o2,b2=wmean(m2); gr=b2/b1-1
    t1,tb1=wmean(m1&(tk==0)); t2,tb2=wmean(m2&(tk==0)); tg=tb2/tb1-1
    s1=np.average(dd[m1],weights=n[m1]); s2=np.average(dd[m2],weights=n[m2])
    e1=np.average(ad[m1&(tk>=1)&(tk<=2)],weights=n[m1&(tk>=1)&(tk<=2)]); e2=np.average(ad[m2&(tk>=1)&(tk<=2)],weights=n[m2&(tk>=1)&(tk<=2)])
    l1=np.average(ad[m1&(tk>=10)],weights=n[m1&(tk>=10)]); l2=np.average(ad[m2&(tk>=10)],weights=n[m2&(tk>=10)])
    P(f"{a:<4}{eng:<18}{o:<5} {o1*1e3:>8.2f} {o2*1e3:>8.2f} {100*(o2/o1-1):>+7.1f}% [{100*np.percentile(gr,2.5):>+5.1f},{100*np.percentile(gr,97.5):>+5.1f}] | {t1*1e3:>9.2f} {t2*1e3:>8.2f} {100*(t2/t1-1):>+7.1f}% [{100*np.percentile(tg,2.5):>+5.1f},{100*np.percentile(tg,97.5):>+5.1f}] | {s1*1e3:>+12.3f} {s2*1e3:>+8.3f} | {100*(e2/e1-1):>+9.1f}% {100*(l2/l1-1):>+9.1f}%")
P("")
P("PROMPT-PAIRED comparison of chain P⁗ with each arm, steps 1-10: per (step, prompt) E_g(long 4k-16k) and E_g(near-rep) (1000 x sum adv x tokens / arm total tokens); P⁗ minus other arm; sign test over prompts with any long turn in either arm.")
PH={}
for a,_,_ in ARMS:
    for fn in glob.glob(f"{D}/parts/{a}__s*_c*.prompts.csv"):
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                if int(r["step"])<=10: PH[(a,r["sample_id"])]=(int(r["step"]),r["prompt_hash"])
G={}
for a,_,_ in ARMS:
    t=[r for r in T[a] if int(r["step"])<=10]; tot=sum(float(r["n_tok"]) for r in t)
    g=collections.defaultdict(lambda:[0.0,0.0,0])
    for r in t:
        key=PH.get((a,r["sample_id"]))
        if not key: continue
        n=float(r["n_tok"]); adv=float(r["adv"]); th=float(r["think_tok"]); z=float(r["zlib_reason"]) if r["zlib_reason"] else 1.0
        if 4000<=n<16000: g[key][0]+=1e3*adv*n/tot; g[key][2]+=1
        if th>=1000 and z<0.25: g[key][1]+=1e3*adv*n/tot
    G[a]=g
P(f"{'vs arm':<6}{'out':<5} {'prompts':>7} | {'E long: mean diff':>17} {'se':>6} {'+/-':>9} {'sign p':>7} | {'E near: mean diff':>17} {'se':>6} {'+/-':>9} {'sign p':>7}")
for a,eng,o in ARMS[1:]:
    keys=sorted(set(G["P4"])&set(G[a])); keys=[k for k in keys if G["P4"][k][2]+G[a][k][2]>0]
    if not keys: P(f"{a:<6}{o:<5} no common prompts"); continue
    dl=np.array([G["P4"][k][0]-G[a][k][0] for k in keys]); dn=np.array([G["P4"][k][1]-G[a][k][1] for k in keys])
    def st(d):
        pos=int((d>1e-12).sum()); neg=int((d<-1e-12).sum()); nn=pos+neg; z=(pos-nn/2)/math.sqrt(nn/4) if nn else 0; return pos,neg,math.erfc(abs(z)/math.sqrt(2))
    pl,nl,ppl=st(dl); pn,nn_,ppn=st(dn)
    P(f"{a:<6}{o:<5} {len(keys):>7} | {dl.mean():>+17.4f} {dl.std(ddof=1)/math.sqrt(len(dl)):>6.4f} {str(pl)+'/'+str(nl):>9} {ppl:>7.3f} | {dn.mean():>+17.4f} {dn.std(ddof=1)/math.sqrt(len(dn)):>6.4f} {str(pn)+'/'+str(nn_):>9} {ppn:>7.3f}")
P("  (negative mean diff = P⁗ suppresses the class more than the other arm; the same statistic gave bad-minus-good +0.010/prompt with 101+/63- in the earlier 7-arm analysis)")
open(f"{D}/keepprefix2_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

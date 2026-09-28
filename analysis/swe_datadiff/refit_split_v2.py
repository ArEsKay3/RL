import glob,csv,json,math,collections,datetime
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
RUNS=[("A","VLLM"),("K","VLLM"),("M","VLLM"),("N","VLLM"),("G","Minf"),("I","Minf"),("J","VLLM"),("P4","Minf (no cache invalidation)")]
TREAT=["VLLM","Minf","Minf (no cache invalidation)"]
LABEL={"J":"VLLM (the healthy Minf run)"}
def loadcsv(p):
    rows=[]
    for fn in sorted(glob.glob(p)):
        with open(fn) as fh: rows+=list(csv.DictReader(fh))
    return rows
def f(x,d=np.nan):
    try: return float(x)
    except: return d
def iso(s):
    try: return datetime.datetime.fromisoformat(s).timestamp()
    except Exception: return np.nan
def fmt(t): return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).strftime("%H:%M:%S") if np.isfinite(t) else "-"
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
rng=np.random.default_rng(3)
P("TURN-LEVEL SPLIT AT THE WEIGHT UPDATE, steps 1-10. For each (run, step) the refit time is bracketed from the rollouts: T > commit time of every rollout that started and ended under the old version; T < commit time of every rollout that crossed (started old, ended new) and < first model call of every rollout that started under the new version. A turn is 'pre' if its model call ended before the lower bound, 'post' if it started after the upper bound; turns inside the bracket are dropped.")
allturns={a:{} for a,_ in RUNS}; brackets={}
for a,tr in RUNS:
    rows=loadcsv(f"{D}/parts/{a}__s*.refit_ts.csv")
    if not rows: P(f"{a} ({tr}): no timestamp file yet"); continue
    by=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows: by[int(r["step"])][r["sample_id"]].append(r)
    for step,rolls in sorted(by.items()):
        info={}
        for sid,rr in rolls.items():
            rr.sort(key=lambda x:int(x["turn"])); ts=np.array([iso(x["ts"]) for x in rr]); lat=np.array([f(x["latency"],0.0) for x in rr])
            info[sid]=dict(sv=int(rr[0]["start_v"]),ev=int(rr[0]["end_v"]),ca=f(rr[0]["committed_at"]),ts=ts,lat=lat)
        vs=sorted({v["sv"] for v in info.values()}|{v["ev"] for v in info.values()}); vmin=vs[0]; vmax=vs[-1]
        cross=[v for v in info.values() if v["sv"]<v["ev"]]; old=[v for v in info.values() if v["sv"]==v["ev"]==vmin]; new=[v for v in info.values() if v["sv"]==v["ev"]==vmax and vmax>vmin]
        if not cross and not new: brackets[(a,step)]=None; allturns[a][step]=(info,None); continue
        lo=max([v["ca"] for v in old],default=-np.inf); lo2=max([np.nanmin(v["ts"]-v["lat"]) for v in cross],default=-np.inf)
        hi=min([v["ca"] for v in cross],default=np.inf); hi2=min([np.nanmin(v["ts"]-v["lat"]) for v in new],default=np.inf)
        hi_eff=min(hi,hi2); lo_eff=max(lo,lo2) if lo2<hi_eff else lo
        brackets[(a,step)]=dict(lo=lo_eff,hi=hi_eff,n_cross=len(cross),n_old=len(old),n_new=len(new),lo2_ok=lo2<hi_eff); allturns[a][step]=(info,(lo_eff,hi_eff))
P("")
P("Refit brackets (UTC clock times): n_cross / n_old / n_new = rollouts that crossed / finished before / started after; width = upper - lower bound in seconds; share of turns before the lower bound (pre) and after the upper bound (post) among all turns of the step.")
P(f"{'run':<4}{'treatment':<30}{'step':>4} {'cross/old/new':>13} {'lower':>9} {'upper':>9} {'width s':>8} {'pre %':>6} {'post %':>7} {'mid %':>6}")
T={}
for a,tr in RUNS:
    t=loadcsv(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv")+loadcsv(f"{D}/parts/{a}__s010_c*.turns.csv")
    T[a]=t
    for step in range(1,11):
        b=brackets.get((a,step))
        if not b: continue
        info,_=allturns[a][step]; n_pre=n_post=n_mid=0
        for v in info.values():
            st=v["ts"]-v["lat"]; en=v["ts"]; n_pre+=int(np.sum(en<b["lo"])); n_post+=int(np.sum(st>b["hi"])); n_mid+=int(np.sum(~(en<b["lo"])&~(st>b["hi"])))
        tot=max(1,n_pre+n_post+n_mid)
        P(f"{a:<4}{LABEL.get(a,tr):<30}{step:>4} {b['n_cross']:>4}/{b['n_old']:>3}/{b['n_new']:>3} {fmt(b['lo']):>9} {fmt(b['hi']):>9} {b['hi']-b['lo'] if np.isfinite(b['hi']-b['lo']) else float('nan'):>8.0f} {100*n_pre/tot:>6.1f} {100*n_post/tot:>7.1f} {100*n_mid/tot:>6.1f}")
# join turns.csv with labels
P("")
lab_rows={a:[] for a,_ in RUNS}; mism=collections.Counter()
for a,tr in RUNS:
    byid=collections.defaultdict(list)
    for r in T[a]: byid[r["sample_id"]].append(r)
    for step,(info,br) in allturns[a].items():
        for sid,v in info.items():
            rr=byid.get(sid)
            if not rr: mism[(a,"no dump rows")]+=1; continue
            rr.sort(key=lambda x:int(f(x["turn"])))
            if len(rr)!=len(v["ts"]): mism[(a,"turn count mismatch")]+=1; continue
            for k,r in enumerate(rr):
                if br is None: lab="norefit"
                else:
                    st=v["ts"][k]-v["lat"][k]; en=v["ts"][k]
                    lab="pre" if en<br[0] else ("post" if st>br[1] else "mid")
                lab_rows[a].append((step,sid,k,lab,f(r["n_tok"]),f(r["adv"]),f(r["think_tok"]),f(r["zlib_reason"],1.0),f(r["mean_abs_d"]),f(r["reward"]),f(r["start"])))
    P(f"{a} ({tr}): labeled turns {len(lab_rows[a])}; skipped rollouts: {dict((k[1],v) for k,v in mism.items() if k[0]==a)}")
def arr(a):
    L=lab_rows[a]; return dict(step=np.array([x[0] for x in L]),sid=np.array([x[1] for x in L]),lab=np.array([x[3] for x in L]),n=np.array([x[4] for x in L]),adv=np.array([x[5] for x in L]),think=np.array([x[6] for x in L]),z=np.array([x[7] for x in L]),ad=np.array([x[8] for x in L]),rew=np.array([x[9] for x in L]),start=np.array([x[10] for x in L]))
A={a:arr(a) for a,_ in RUNS if lab_rows[a]}
P("")
P("(a) Engine-trainer mismatch |d| x1e3 on short turns (< 500 tokens), token-weighted, by label. A stale prefix (cache kept) predicts post > pre for the cache-retaining runs only.")
P(f"{'run':<4}{'treatment':<30} {'pre':>7} {'post':>7} {'post-pre':>9} {'no-refit steps':>14} {'n pre':>7} {'n post':>7}")
pool=collections.defaultdict(lambda:collections.defaultdict(lambda:[0.0,0.0]))
for a,tr in RUNS:
    if a not in A: continue
    d=A[a]; sh=(d["n"]<500)&(d["n"]>=20)&np.isfinite(d["ad"])
    def wm(l): m=sh&(d["lab"]==l); pool[tr][l][0]+=(d["ad"][m]*d["n"][m]).sum(); pool[tr][l][1]+=d["n"][m].sum(); return 1e3*np.average(d["ad"][m],weights=d["n"][m]) if m.sum()>100 else np.nan, int(m.sum())
    pre,npre=wm("pre"); post,npost=wm("post"); nr,_=wm("norefit")
    P(f"{a:<4}{LABEL.get(a,tr):<30} {pre:>7.2f} {post:>7.2f} {post-pre:>+9.2f} {nr:>14.2f} {npre:>7} {npost:>7}")
for tr in TREAT:
    if tr not in pool: continue
    g=lambda l: 1e3*pool[tr][l][0]/pool[tr][l][1] if pool[tr][l][1]>0 else np.nan
    P(f"{'':<4}{tr+', pooled':<30} {g('pre'):>7.2f} {g('post'):>7.2f} {g('post')-g('pre'):>+9.2f} {g('norefit'):>14.2f}")
P("")
P("(b) Advantage exposure of think turns 4k-16k, split by label, straddled steps only. E = 1000 x sum(adv x tokens) / all generated tokens of those steps; token share = class tokens in that label / all tokens of those steps.")
P(f"{'run':<4}{'treatment':<30} {'E pre':>7} {'E post':>7} {'E mid':>7} {'share pre %':>11} {'share post %':>12} | {'E near-rep pre':>14} {'E near-rep post':>15}")
pb=collections.defaultdict(lambda:collections.defaultdict(float))
for a,tr in RUNS:
    if a not in A: continue
    d=A[a]; strd=np.isin(d["step"],[s for (aa,s),b in brackets.items() if aa==a and b]); tot=d["n"][strd].sum()
    lg=strd&(d["n"]>=4000)&(d["n"]<16000); nr=strd&(d["think"]>=1000)&(d["z"]<0.25)
    def E(sel,l): m=sel&(d["lab"]==l); v=(d["adv"][m]*d["n"][m]).sum(); pb[tr][(l,'v' if sel is lg else 'vn')]+=v; pb[tr]["tot"]+= (tot if (l=="pre" and sel is lg) else 0); return 1e3*v/tot
    def sh(sel,l): m=sel&(d["lab"]==l); return 100*d["n"][m].sum()/tot
    P(f"{a:<4}{LABEL.get(a,tr):<30} {E(lg,'pre'):>+7.2f} {E(lg,'post'):>+7.2f} {E(lg,'mid'):>+7.2f} {sh(lg,'pre'):>11.2f} {sh(lg,'post'):>12.2f} | {E(nr,'pre'):>+14.2f} {E(nr,'post'):>+15.2f}")
for tr in TREAT:
    if tr not in pb: continue
    tot=pb[tr]["tot"]; P(f"{'':<4}{tr+', pooled':<30} {1e3*pb[tr][('pre','v')]/tot:>+7.2f} {1e3*pb[tr][('post','v')]/tot:>+7.2f} {1e3*pb[tr][('mid','v')]/tot:>+7.2f} {'':>11} {'':>12} | {1e3*pb[tr][('pre','vn')]/tot:>+14.2f} {1e3*pb[tr][('post','vn')]/tot:>+15.2f}")
P("")
P("(c) Within-group rank correlation of reward with the rollout's post-refit tokens, its pre-refit tokens, and its total tokens (straddled steps, groups with reward variance). If post-refit generation is worse in cache-retaining runs, the post-refit correlation should be more negative there than in Minf, beyond the ordinary length effect.")
def spearman(x,y):
    x=np.argsort(np.argsort(x)); y=np.argsort(np.argsort(y)); return np.corrcoef(x,y)[0,1] if x.std()>0 and y.std()>0 else np.nan
P(f"{'run':<4}{'treatment':<30} {'rho(r, post tok)':>16} {'rho(r, pre tok)':>15} {'rho(r, total)':>13} {'rho(r, post share)':>18} {'groups':>6}")
pc=collections.defaultdict(list)
for a,tr in RUNS:
    if a not in A: continue
    d=A[a]; strd=np.isin(d["step"],[s for (aa,s),b in brackets.items() if aa==a and b])
    per=collections.defaultdict(lambda:[0.0,0.0,0.0,np.nan])
    for i in np.where(strd)[0]:
        p=per[d["sid"][i]]; p[3]=d["rew"][i]; p[2]+=d["n"][i]
        if d["lab"][i]=="post": p[0]+=d["n"][i]
        elif d["lab"][i]=="pre": p[1]+=d["n"][i]
    groups=collections.defaultdict(list)
    for sid,p in per.items(): groups[sid.rsplit("_g",1)[0]].append(p)
    c=[[],[],[],[]]
    for g in groups.values():
        r=np.array([p[3] for p in g])
        if r.std()==0 or len(g)<4: continue
        c[0].append(spearman(r,np.array([p[0] for p in g]))); c[1].append(spearman(r,np.array([p[1] for p in g]))); c[2].append(spearman(r,np.array([p[2] for p in g]))); c[3].append(spearman(r,np.array([p[0]/max(1,p[2]) for p in g])))
    vals=[np.nanmean(x) for x in c]; pc[tr].append(vals)
    P(f"{a:<4}{LABEL.get(a,tr):<30} {vals[0]:>+16.3f} {vals[1]:>+15.3f} {vals[2]:>+13.3f} {vals[3]:>+18.3f} {len(c[0]):>6}")
for tr in TREAT:
    if tr not in pc: continue
    v=np.nanmean(np.array(pc[tr]),axis=0); P(f"{'':<4}{tr+', mean over runs':<30} {v[0]:>+16.3f} {v[1]:>+15.3f} {v[2]:>+13.3f} {v[3]:>+18.3f}")
open(f"{D}/refit_split_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

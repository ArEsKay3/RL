import glob,csv,os,re,subprocess,time,math,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
R="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
ARMS=[("A","VLLM","cache kept","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918"),
      ("K","VLLM","cache kept","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-seed1234-20260922"),
      ("M","VLLM","cache kept","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923"),
      ("N","VLLM","cache kept","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r2-20260923"),
      ("Q4","VLLM (no prefix caching)","no cache","nano35-swe-v2-from0-noprefix-vllm-20260926"),
      ("G","Minf","no cache","nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920"),
      ("I","Minf","cache invalidated","nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-20260921"),
      ("J","Minf","cache invalidated","nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921"),
      ("P4","Minf (no cache invalidation)","cache kept","nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925"),
      ("P42","Minf (no cache invalidation), seed 1234","cache kept","nano35-swe-v2-from0-nvshmem-keepprefix-minf-seed1234-20260926"),
      ("P3","Minf (no cache invalidation), from step 20","cache kept","nano35-swe-v2-fromP20-nvshmem-keepprefix-minf-20260925")]
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
def ep(s):
    try: return time.mktime(time.strptime(s,"%Y-%m-%dT%H:%M:%S"))
    except Exception: return None
def ols(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3: return float("nan"),float("nan")
    b=np.polyfit(x,y,1); r=y-(b[0]*x+b[1]); se=math.sqrt((r@r)/(len(x)-2)/((x-x.mean())@(x-x.mean()))); return b[0],se
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
runs={}
for a,lab,grp,e in ARMS:
    jobs=sorted({re.match(r"(\d+)",d).group(1) for d in os.listdir(f"{R}/{e}/ray_logs") if re.match(r"\d+",d)}) if os.path.isdir(f"{R}/{e}/ray_logs") else []
    segs=[]
    if jobs:
        txt=subprocess.check_output(["sacct","-j",",".join(jobs),"-X","-n","-P","-o","JobID,Start,End,State"],text=True,timeout=120)
        for line in txt.strip().splitlines():
            jid,st,en,state=line.split("|")[:4]; s=ep(st); e_=ep(en) if en not in ("Unknown","None") else time.time()
            if s: segs.append((s,e_,jid))
    segs.sort()
    mt={}
    for fn in glob.glob(f"{R}/{e}/dumps/token_level/step_*_chunk_*.pt"):
        s=int(re.search(r"step_(\d+)_",fn).group(1)); mt[s]=max(mt.get(s,0),os.path.getmtime(fn))
    def seg_of(t):
        for i,(s,e_,j) in enumerate(segs):
            if s-60<=t<=e_+120: return i
        prev=[i for i,(s,_,_) in enumerate(segs) if s<=t]; return prev[-1] if prev else -1
    T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv")
    step=col(T,"step").astype(int); n=col(T,"n_tok"); ad=col(T,"mean_abs_d"); tk=col(T,"turn"); short=(n<500)&(n>=20)&~np.isnan(ad)
    per=[]
    for s in sorted(set(step)):
        m=short&(step==s)
        if m.sum()<300: continue
        m0=m&(tk==0); per.append((s,seg_of(mt.get(s,0)),1e3*np.average(ad[m],weights=n[m]),1e3*np.average(ad[m0],weights=n[m0]) if m0.sum()>50 else np.nan))
    # group into segments (consecutive steps with the same segment id)
    seglist=[]; cur=None
    for s,g,da,d0 in per:
        if cur is None or g!=cur[0] or s!=cur[1][-1][0]+1: cur=(g,[]); seglist.append(cur)
        cur[1].append((s,da,d0))
    runs[a]=dict(lab=lab,grp=grp,segs=seglist,njobs=len(segs))
P("RISING ENGINE-TRAINER MISMATCH vs STEPS SINCE THE ENGINE STARTED. |d| = |trainer logprob - engine logprob| x1e3, token-weighted over short turns (< 500 tokens) so loops do not confound it; 'turn 0' = the first assistant turn of each rollout, whose context is almost entirely the shared, always-hot system prompt (the part of the prefix cache that is never evicted). Every Slurm segment restart recreates the engine, so k counts steps since the last restart; segments are found from job start/end times and dump-file times.")
P("")
P("Table A. Turn-0 |d| by steps since engine start (k), longest segment of each run. Slope = OLS per step over that segment (x1e-3/step), with se. 'rise 10' = value at k=10 minus value at k=0 when k=10 exists.")
K=list(range(0,21))
P(f"{'run':<44}{'cache':<19}{'seg steps':>10} | "+" ".join(f"{k:>5}" for k in K)+f" | {'slope/step':>13} {'rise 10':>7}")
pool=collections.defaultdict(list); pool_all=collections.defaultdict(list)
for a,lab,grp,e in ARMS:
    r=runs[a]; longest=max(r["segs"],key=lambda s:len(s[1]))
    steps=[s for s,_,_ in longest[1]]; d0=[x for _,_,x in longest[1]]; da=[x for _,x,_ in longest[1]]
    sl,se=ols(range(len(d0)),d0); rise=(d0[10]-d0[0]) if len(d0)>10 else float("nan")
    cells=[f"{d0[k]:5.1f}" if k<len(d0) else "    ." for k in K]
    P(f"{lab:<44}{grp:<19}{steps[0]:>4}-{steps[-1]:<5} | "+" ".join(cells)+f" | {sl:>+6.3f}±{se:5.3f} {rise:>+7.1f}")
    for g,seg in r["segs"]:
        if len(seg)>=6:
            for k,(s,x_all,x0) in enumerate(seg): pool[grp].append((k,x0)); pool_all[grp].append((k,x_all))
P("")
P("Table B. Same, all short turns (turn 0 and later), longest segment.")
P(f"{'run':<44}{'cache':<19}{'seg steps':>10} | "+" ".join(f"{k:>5}" for k in K)+f" | {'slope/step':>13} {'rise 10':>7}")
for a,lab,grp,e in ARMS:
    r=runs[a]; longest=max(r["segs"],key=lambda s:len(s[1]))
    steps=[s for s,_,_ in longest[1]]; da=[x for _,x,_ in longest[1]]
    sl,se=ols(range(len(da)),da); rise=(da[10]-da[0]) if len(da)>10 else float("nan")
    P(f"{lab:<44}{grp:<19}{steps[0]:>4}-{steps[-1]:<5} | "+" ".join(f"{da[k]:5.1f}" if k<len(da) else "    ." for k in K)+f" | {sl:>+6.3f}±{se:5.3f} {rise:>+7.1f}")
P("")
P("Table C. All segments of at least 6 steps, per run: turn-0 |d| at k=0 -> at the segment's last step, slope, and the drop at the next restart (first value of the next segment minus last value of this one).")
P(f"{'run':<44}{'cache':<19} {'segment':>9} {'k0':>5} {'last':>5} {'slope':>7} {'drop at restart':>15}")
drops=collections.defaultdict(list)
for a,lab,grp,e in ARMS:
    r=runs[a]
    for i,(g,seg) in enumerate(r["segs"]):
        if len(seg)<6: continue
        d0=[x for _,_,x in seg]; sl,se=ols(range(len(d0)),d0)
        nxt=r["segs"][i+1][1][0][2] if i+1<len(r["segs"]) else float("nan"); drop=nxt-d0[-1] if np.isfinite(nxt) else float("nan")
        if np.isfinite(drop): drops[grp].append(drop)
        P(f"{lab:<44}{grp:<19} {seg[0][0]:>4}-{seg[-1][0]:<4} {d0[0]:>5.1f} {d0[-1]:>5.1f} {sl:>+7.3f} {drop:>+15.1f}")
P("")
P("Table D. Pooled by cache treatment (all segments >= 6 steps): OLS slope of |d| on steps since engine start, x1e-3 per step, with se; mean drop at restarts.")
P(f"{'treatment':<20} {'runs':>4} {'arm-steps':>9} {'turn-0 slope':>14} {'all-short slope':>16} {'mean restart drop (turn 0)':>26}")
for grp in ("cache kept","cache invalidated","no cache"):
    pts=pool[grp]; pa=pool_all[grp]
    if not pts: continue
    x=np.array([k for k,_ in pts]); y=np.array([v for _,v in pts]); s0,e0=ols(x,y); xa=np.array([k for k,_ in pa]); ya=np.array([v for _,v in pa]); sa,ea=ols(xa,ya)
    nr=len({a for a,lab,g,e in ARMS if g==grp})
    P(f"{grp:<20} {nr:>4} {len(pts):>9} {s0:>+7.3f} ± {e0:.3f} {sa:>+9.3f} ± {ea:.3f} {np.mean(drops[grp]) if drops[grp] else float('nan'):>+26.1f}")
P("")
P("Reference: the frozen-weight pair (lr 0, same prompts) drifted +1-2 % overall and +6-8 % on turn 0 over 15 steps, i.e. about +0.05-0.1 x1e-3 per step on turn 0 from the changing prompt set alone.")
open(f"{D}/drift_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

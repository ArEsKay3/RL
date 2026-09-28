import glob,csv,os,re,subprocess,time,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
R="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
ARMS=[("P4","MINF keep-prefix from0","nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925"),("P3","MINF keep-prefix <-P20","nano35-swe-v2-fromP20-nvshmem-keepprefix-minf-20260925"),
      ("A","vLLM from0","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918"),("K","vLLM from0 s1234","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-seed1234-20260922"),("M","vLLM from0 r1","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923"),("N","vLLM from0 r2","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r2-20260923"),
      ("G","MINF from0 no-pfx","nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920"),("I","MINF from0 pfx","nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-20260921"),("J","MINF from0 pfx","nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921"),
      ("P","MINF <-M replay","nano35-swe-v2-splice-minf-runMdata-to20-20260924"),("Q","vLLM <-M replay","nano35-swe-v2-splice-vllm-runMdata-to20-20260924")]
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
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("SEGMENT STRUCTURE vs ENGINE-TRAINER MISMATCH. Every Slurm segment restart recreates the inference engine (empty prefix cache). Per step: short-turn |d| x1e3 (token-weighted, turns < 500 tokens) and first-turn |d| x1e3; '|' marks a new segment. Summary: mean per-step change of |d| inside segments vs mean jump across segment boundaries.")
summary=[]
for a,lab,e in ARMS:
    jobs=sorted({re.match(r"(\d+)",d).group(1) for d in os.listdir(f"{R}/{e}/ray_logs") if re.match(r"\d+",d)}) if os.path.isdir(f"{R}/{e}/ray_logs") else []
    segs=[]
    if jobs:
        try:
            txt=subprocess.check_output(["sacct","-j",",".join(jobs),"-X","-n","-P","-o","JobID,Start,End,State"],text=True,timeout=120)
            for line in txt.strip().splitlines():
                jid,st,en,state=line.split("|")[:4]; s=ep(st); en_=ep(en) if en not in ("Unknown","None") else time.time()
                if s: segs.append((s,en_,jid,state))
        except Exception as ex: P(f"  sacct failed for {a}: {ex}")
    segs.sort()
    # step -> mtime of latest chunk
    mt={}
    for fn in glob.glob(f"{R}/{e}/dumps/token_level/step_*_chunk_*.pt"):
        s=int(re.search(r"step_(\d+)_",fn).group(1)); mt[s]=max(mt.get(s,0),os.path.getmtime(fn))
    def seg_of(t):
        for i,(s,en_,jid,state) in enumerate(segs):
            if s-60<=t<=en_+120: return i
        prev=[i for i,(s,_,_,_) in enumerate(segs) if s<=t]; return prev[-1] if prev else -1
    T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv")
    if not T: continue
    step=col(T,"step").astype(int); n=col(T,"n_tok"); ad=col(T,"mean_abs_d"); tk=col(T,"turn"); short=(n<500)&(n>=20)&~np.isnan(ad)
    steps=sorted(set(step)); rows=[]
    for s in steps:
        m=short&(step==s)
        if m.sum()<300: continue
        d_all=1e3*np.average(ad[m],weights=n[m]); m0=m&(tk==0); d0=1e3*np.average(ad[m0],weights=n[m0]) if m0.sum()>50 else np.nan
        rows.append((s,seg_of(mt.get(s,0)) if s in mt else -1,d_all,d0))
    line=[]; prev=None
    for s,sg,d_all,d0 in rows:
        line.append(("| " if (prev is not None and sg!=prev) else "")+f"s{s}:{d_all:.1f}/{d0:.1f}"); prev=sg
    P(f"\n{a} {lab}: {len(segs)} segments ({', '.join(jid for _,_,jid,_ in segs)}); steps with dumps {steps[0]}-{steps[-1]}")
    P("  "+" ".join(line))
    within=[];jumps=[];within0=[];jumps0=[]
    for (s1,g1,d1,e1),(s2,g2,d2,e2) in zip(rows,rows[1:]):
        if s2!=s1+1: continue
        if g1==g2: within.append(d2-d1); within0.append(e2-e1)
        else: jumps.append(d2-d1); jumps0.append(e2-e1)
    summary.append((a,lab,len(segs),np.mean(within) if within else np.nan,len(within),np.mean(jumps) if jumps else np.nan,len(jumps),np.nanmean(within0) if within0 else np.nan,np.nanmean(jumps0) if jumps0 else np.nan))
P("")
P(f"{'arm':<4}{'what':<24}{'segs':>4} | {'within-seg d|d|/step':>20} {'n':>3} | {'jump at boundary':>16} {'n':>3} | {'turn0 within/step':>17} {'turn0 jump':>10}")
for a,lab,ns,w,nw,j,nj,w0,j0 in summary:
    P(f"{a:<4}{lab:<24}{ns:>4} | {w:>+20.3f} {nw:>3} | {j:>+16.3f} {nj:>3} | {w0:>+17.3f} {j0:>+10.3f}")
P("  (units: |d| x1e3 per step; a stale-cache mechanism predicts positive within-segment drift and negative jumps at boundaries in arms that keep the cache across refits: A, K, M, N, Q, P4, P3; ~0 both in G, I, J, P)")
open(f"{D}/segments_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

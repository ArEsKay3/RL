import glob,csv,json,os,math,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
R="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
RUNS=[("A","VLLM","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918"),("K","VLLM","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-seed1234-20260922"),("M","VLLM","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923"),("N","VLLM","nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r2-20260923"),
      ("G","Minf","nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920"),("I","Minf","nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-20260921"),("J","Minf","nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921"),
      ("P4","Minf (no cache invalidation)","nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925")]
TREAT=["VLLM","Minf","Minf (no cache invalidation)"]
CLASSES=[("E(think turns 4k-16k)","long"),("E(near-repetitive think)","near"),("E(long turns starting 16k-32k into context)","ctx")]
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
rng=np.random.default_rng(12); NB=3000
data={}
for a,tr,e in RUNS:
    t=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv"); s=loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv")
    gmean=collections.defaultdict(list)
    for r in s: gmean[r["group_id"]].append(float(r["reward"]))
    gmean={k:np.mean(v) for k,v in gmean.items()}
    gid=np.array([r["sample_id"].rsplit("_g",1)[0] for r in t])
    data[a]=dict(step=col(t,"step"),adv=col(t,"adv"),n=col(t,"n_tok"),think=col(t,"think_tok"),z=col(t,"zlib_reason"),start=col(t,"start"),rew=col(t,"reward"),gid=gid,gm=np.array([gmean.get(g,np.nan) for g in gid]))
def sel_of(d,cls):
    if cls=="long": return (d["n"]>=4000)&(d["n"]<16000)
    if cls=="near": return (d["think"]>=1000)&(d["z"]<0.25)
    if cls=="ctx": return (d["n"]>=4000)&(d["start"]>=16000)&(d["start"]<32000)
def group_sums(a,lo,hi,cls):
    d=data[a]; m=(d["step"]>=lo)&(d["step"]<=hi); sel=sel_of(d,cls)&m
    u,g=np.unique(d["gid"][m],return_inverse=True); ng=len(u)
    gi=np.searchsorted(u,d["gid"][sel])
    S=dict(N=np.bincount(g,weights=d["n"][m],minlength=ng),
           adv=np.bincount(gi,weights=(d["adv"]*d["n"])[sel],minlength=ng),
           cen=np.bincount(gi,weights=((d["rew"]-d["gm"])*d["n"])[sel],minlength=ng),
           rn=np.bincount(gi,weights=(d["rew"]*d["n"])[sel],minlength=ng),
           nsel=np.bincount(gi,weights=d["n"][sel],minlength=ng))
    return S
def est(S,W=None):
    if W is None:
        return dict(adv=1e3*S["adv"].sum()/S["N"].sum(),cen=1e3*S["cen"].sum()/S["N"].sum(),share=S["rn"].sum()/max(1,S["nsel"].sum()))
    return dict(adv=1e3*(W@S["adv"])/(W@S["N"]),cen=1e3*(W@S["cen"])/(W@S["N"]),share=(W@S["rn"])/np.maximum(1,W@S["nsel"]))
def concat(Ss):
    return {k:np.concatenate([S[k] for S in Ss]) for k in Ss[0]}
def boot(S):
    ng=len(S["N"]); W=rng.multinomial(ng,np.ones(ng)/ng,size=NB).astype(float); return est(S),est(S,W)
md=[]
def M(*x): md.append(" ".join(str(v) for v in x))
def ci(o,b,fmt="{:+.2f}"): return f"{fmt.format(o)} [{fmt.format(np.percentile(b,2.5))}, {fmt.format(np.percentile(b,97.5))}]"
M("# Advantage exposure tables (2026-09-26)\n")
M("Runs: VLLM = four vLLM-generation runs from scratch; Minf = three MINF-generation runs from scratch (prefix cache invalidated at every weight update, or no prefix cache); Minf (no cache invalidation) = one MINF run from scratch with the prefix cache kept across weight updates. Same prompts per step in every run (320 prompt groups of 16 per 10 steps).")
M("E(class) = 1000 x sum over turns in the class of (advantage x turn tokens) / all generated tokens in the window. Negative = the batch gradient suppresses that class of tokens on net. Brackets = 95 % group-bootstrap interval. Classes: think turns of 4k-16k tokens; near-repetitive think turns (>= 1k think tokens, zlib ratio of the think text < 0.25); long turns (>= 4k tokens) that start 16k-32k tokens into the context.\n")
for lo,hi in ((1,10),(11,20)):
    M(f"## Table {1 if lo==1 else 2}. Advantage exposure, training steps {lo}-{hi}\n")
    M("| run | "+" | ".join(c for c,_ in CLASSES)+" |"); M("|---|"+"---|"*len(CLASSES))
    per={}
    for a,tr,_ in RUNS:
        cells=[]; per[a]={}
        for cname,cls in CLASSES:
            S=group_sums(a,lo,hi,cls); o,b=boot(S); per[a][cls]=(S,o,b); cells.append(ci(o["adv"],b["adv"]))
        M(f"| {tr} | "+" | ".join(cells)+" |")
    M("| | | | |")
    pooled={}
    for tr in TREAT:
        cells=[]
        for cname,cls in CLASSES:
            S=concat([per[a][cls][0] for a,t,_ in RUNS if t==tr]); o,b=boot(S); pooled[(tr,cls)]=(o,b); cells.append(ci(o["adv"],b["adv"]))
        M(f"| **{tr}, pooled** | "+" | ".join(cells)+" |")
    M("")
    if lo==1:
        M("Separation, steps 1-10 (pooled Minf minus pooled VLLM; z = difference / bootstrap se):\n")
        M("| class | Minf - VLLM | z | Minf (no cache invalidation) - VLLM | z |"); M("|---|---|---|---|---|")
        for cname,cls in CLASSES:
            oV,bV=pooled[("VLLM",cls)]; oM,bM=pooled[("Minf",cls)]; oP,bP=pooled[("Minf (no cache invalidation)",cls)]
            d1=oM["adv"]-oV["adv"]; s1=(bM["adv"]-bV["adv"]).std(); d2=oP["adv"]-oV["adv"]; s2=(bP["adv"]-bV["adv"]).std()
            M(f"| {cname} | {d1:+.2f} | {d1/s1:+.1f} | {d2:+.2f} | {d2/s2:+.1f} |")
        M("")
        # ---- weighting comparison
        M("## Table 3. The same classes under three weightings, steps 1-10\n")
        M("Advantage = what the trainer uses: leave-one-out group normalization, (r_i - mean of the other 15) / std of the other 15. Centered reward = r_i - group mean, no normalization. Raw reward = share of the class's tokens that sit in rewarded (r = 1) rollouts, no group information at all. Relative gap = (Minf - VLLM) / |VLLM|.\n")
        for cname,cls in CLASSES:
            M(f"### {cname}\n")
            M("| run | advantage-weighted E | centered-reward E | raw reward share |"); M("|---|---|---|---|")
            for a,tr,_ in RUNS:
                S,o,b=per[a][cls]; M(f"| {tr} | {ci(o['adv'],b['adv'])} | {ci(o['cen'],b['cen'])} | {ci(o['share'],b['share'],'{:.3f}')} |")
            M("| | | | |")
            for tr in TREAT:
                o,b=pooled[(tr,cls)]; M(f"| **{tr}, pooled** | {ci(o['adv'],b['adv'])} | {ci(o['cen'],b['cen'])} | {ci(o['share'],b['share'],'{:.3f}')} |")
            oV,bV=pooled[("VLLM",cls)]; oM,bM=pooled[("Minf",cls)]
            rows=[]
            for w,lab in (("adv","advantage"),("cen","centered reward"),("share","raw reward share")):
                d=oM[w]-oV[w]; se=(bM[w]-bV[w]).std(); rel=100*d/abs(oV[w]) if oV[w]!=0 else float("nan")
                rows.append(f"| {lab} | {d:+.3f} | {d/se:+.1f} | {rel:+.0f} % |")
            M("| | | | |"); M("| **Minf - VLLM by weighting** | difference | z | relative gap |")
            for r in rows: M(r)
            M("")
        # ---- straddle test
        span=json.load(open(f"{D}/span_class.json"))
        if "P4" not in span:
            span["P4"]={}
            for ts in range(0,20):
                fn=f"{R}/nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925/dumps/rollouts/target_step_{ts:05d}.jsonl"
                if not os.path.exists(fn): continue
                vs=set(); k=0
                with open(fn,"rb") as fh:
                    for line in fh:
                        try: o=json.loads(line)
                        except Exception: break
                        vs.add((o["start_weight_version"],o["end_weight_version"])); k+=1
                        if k>=40: break
                span["P4"][str(ts+1)]="none" if all(s==e for s,e in vs) else ("full" if all(s!=e for s,e in vs) else "mixed")
            json.dump(span,open(f"{D}/span_class.json","w"))
        M("## Table 4. Is the suppression concentrated in steps whose rollouts ran past a weight update? (steps 1-10, E(think turns 4k-16k))\n")
        M("For each run, steps are split by that run's own rollouts: 'straddled' = the step's rollouts started under one weight version and finished under the next (their later turns were generated after the refit; in the VLLM and no-invalidation runs those turns reuse prefix KV from the older weights). Because the prompt set differs by step, each run's value is paired against the pooled Minf runs at the same steps (Minf recomputes the prefix after every update, so straddling has no cache effect there).\n")
        M("| run | straddled steps | E, straddled | E, not straddled | Minf at same steps: straddled / not | run - Minf, straddled | run - Minf, not straddled |"); M("|---|---|---|---|---|---|---|")
        def E_step(a,s,cls="long"):
            d=data[a]; m=d["step"]==s; sel=sel_of(d,cls)&m; return 1e3*(d["adv"][sel]*d["n"][sel]).sum()/d["n"][m].sum()
        def E_minf_step(s,cls="long"):
            num=0.0;den=0.0
            for a,tr,_ in RUNS:
                if tr!="Minf": continue
                d=data[a]; m=d["step"]==s; sel=sel_of(d,cls)&m; num+=(d["adv"][sel]*d["n"][sel]).sum(); den+=d["n"][m].sum()
            return 1e3*num/den
        agg=collections.defaultdict(list)
        for a,tr,_ in RUNS:
            if tr=="Minf": continue
            st=[s for s in range(1,11) if span.get(a,{}).get(str(s)) in ("full","mixed")]; ns=[s for s in range(1,11) if s not in st]
            Es=np.mean([E_step(a,s) for s in st]) if st else np.nan; En=np.mean([E_step(a,s) for s in ns]); Ms=np.mean([E_minf_step(s) for s in st]) if st else np.nan; Mn=np.mean([E_minf_step(s) for s in ns])
            agg[tr].append((Es-Ms,En-Mn))
            M(f"| {tr} | {', '.join(map(str,st)) or '-'} | {Es:+.2f} | {En:+.2f} | {Ms:+.2f} / {Mn:+.2f} | {Es-Ms:+.2f} | {En-Mn:+.2f} |")
        for tr,v in agg.items():
            v=np.array(v); M(f"| **{tr}, mean over runs** | | | | | {np.nanmean(v[:,0]):+.2f} | {np.nanmean(v[:,1]):+.2f} |")
        M("")
open(f"{D}/tables.md","w").write("\n".join(md)+"\n"); print("\n".join(md))

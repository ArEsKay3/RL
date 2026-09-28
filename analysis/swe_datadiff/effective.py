import glob,csv,json,math,collections,sys
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","bad","Minf"),("I","bad","Minf"),("P42","bad","Minf"),("A","good","VLLM"),("K","good","VLLM"),("M","good","VLLM"),("N","good","VLLM"),("Q4","good","VLLM"),("J","good","Minf"),("P4","good","Minf")]
LO,HI=(int(sys.argv[1]),int(sys.argv[2])) if len(sys.argv)>2 else (1,10)
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
rng=np.random.default_rng(17); NB=1500
def rank_line(name,vals,fmt="{:+.3f}",note=""):
    bad=[vals[a] for a,o,_ in ARMS if o=="bad" and a in vals]; good=[vals[a] for a,o,_ in ARMS if o=="good" and a in vals]
    order=sorted(vals,key=lambda a:-vals[a]); ranks=sorted(order.index(a)+1 for a,o,_ in ARMS if o=="bad" and a in vals)
    sep="YES" if bad and good and min(bad)>max(good) else ("rev" if bad and good and max(bad)<min(good) else "")
    P(f"{name:<64} "+" ".join(f"{a}:{fmt.format(vals[a])}" for a,o,_ in ARMS if a in vals)+f" | bad ranks {ranks} {sep} {note}")
DATA={}
for a,o,eng in ARMS:
    rows=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.eff.csv")):
        s=int(fn.split("__s")[1][:3])
        if LO<=s<=HI:
            with open(fn) as fh:
                for r in csv.DictReader(fh): r["step"]=s; rows.append(r)
    if not rows: continue
    d=dict(step=np.array([r["step"] for r in rows]),adv=col(rows,"adv"),rew=col(rows,"reward"),n=col(rows,"n_tok"),think=col(rows,"think_tok"),z=col(rows,"zlib"),cls=np.array([r["cls"] for r in rows]),
           s1=col(rows,"s1mp_think_tr"),s1e=col(rows,"s1mp_think_eng"),s1entry=col(rows,"s1mp_entry_tr"),s1deep=col(rows,"s1mp_deep_tr"),s1ans=col(rows,"s1mp_answer_tr"),pc=col(rows,"close_p_tr"),pce=col(rows,"close_p_eng"),lowp=col(rows,"n_lowp_think"),mptr=col(rows,"mean_ptr_think"),
           gid=np.array([r["sample_id"].rsplit("_g",1)[0] for r in rows]),sid=np.array([r["sample_id"] for r in rows]))
    DATA[a]=d
arms=[x for x in ARMS if x[0] in DATA]
P(f"EFFECTIVE (TRAINER-GRADIENT) EXPOSURE, steps {LO}-{HI}. For a sampled token the policy-gradient weight on its logit is advantage x (1 - p_trainer). R(class) = 1000 x sum over think tokens in the class of adv x (1 - p_trainer) / all generated tokens. Positive = the data pushes the trainer toward producing that kind of thinking; negative = away. Compare with E, which weights tokens equally.")
P("3 bad arms (G, I, P⁗2) vs 7 good (A, K, M, N, Q⁗, J, P⁗); 'YES' = perfect separation (chance 1/120).")
P("")
tot={a:DATA[a]["n"].sum() for a in DATA}
def R(a,sel,field="s1"): d=DATA[a]; return 1e3*(d["adv"][sel]*d[field][sel]).sum()/tot[a]
def E(a,sel): d=DATA[a]; return 1e3*(d["adv"][sel]*d["think"][sel]).sum()/tot[a]
P("--- A. by turn class (think part), R vs E")
for cls in ("short","normal_long","near","loop"):
    rank_line(f"R think tokens, class={cls}",{a:R(a,DATA[a]["cls"]==cls) for a in DATA})
    rank_line(f"E think tokens, class={cls}",{a:E(a,DATA[a]["cls"]==cls) for a in DATA},"{:+.2f}")
rank_line("R think tokens, all classes",{a:R(a,np.ones(len(DATA[a]["adv"]),bool)) for a in DATA})
rank_line("R think tokens, near+loop",{a:R(a,np.isin(DATA[a]["cls"],["near","loop"])) for a in DATA})
rank_line("R think tokens, near+loop+normal_long (>=1k think)",{a:R(a,np.isin(DATA[a]["cls"],["near","loop","normal_long"])) for a in DATA})
P("")
P("--- B. by depth into the think block (trainer-gradient weight), turns with >= 1k think tokens")
rank_line("R entry (first 256 think tokens), >=1k think turns",{a:R(a,DATA[a]["think"]>=1000,"s1entry") for a in DATA})
rank_line("R deep (think tokens beyond 4096), >=1k think turns",{a:R(a,DATA[a]["think"]>=1000,"s1deep") for a in DATA})
rank_line("R entry, near+loop turns",{a:R(a,np.isin(DATA[a]["cls"],["near","loop"]),"s1entry") for a in DATA})
rank_line("R deep, near+loop turns",{a:R(a,np.isin(DATA[a]["cls"],["near","loop"]),"s1deep") for a in DATA})
rank_line("R answer part (after </think>), all turns",{a:R(a,np.ones(len(DATA[a]["adv"]),bool),"s1ans") for a in DATA})
P("")
P("--- C. the close decision: Rclose = 1000 x sum over turns with </think> of adv x (1 - p_trainer(</think>)) / all tokens; positive = data teaches closing")
for lab,sel_fn in (("all turns",lambda d:np.isfinite(d["pc"])),("turns >=1k think",lambda d:np.isfinite(d["pc"])&(d["think"]>=1000)),("turns >=4k think",lambda d:np.isfinite(d["pc"])&(d["think"]>=4000)),("near+loop turns",lambda d:np.isfinite(d["pc"])&np.isin(d["cls"],["near","loop"]))):
    vals={}
    for a in DATA:
        d=DATA[a]; m=sel_fn(d); vals[a]=1e3*(d["adv"][m]*(1-d["pc"][m])).sum()/tot[a]
    rank_line(f"Rclose, {lab}",vals,"{:+.4f}")
    vals2={}
    for a in DATA:
        d=DATA[a]; m=sel_fn(d); vals2[a]=float(np.nanmean(d["pc"][m]))
    rank_line(f"mean p_trainer(</think>), {lab}",vals2,"{:.3f}")
P("")
P("--- D. positive vs negative parts of R for near+loop think tokens, and the ratio")
for a in DATA:
    pass
vp={a:1e3*(DATA[a]["adv"]*DATA[a]["s1"])[(DATA[a]["adv"]>0)&np.isin(DATA[a]["cls"],["near","loop"])].sum()/tot[a] for a in DATA}
vn={a:1e3*(DATA[a]["adv"]*DATA[a]["s1"])[(DATA[a]["adv"]<0)&np.isin(DATA[a]["cls"],["near","loop"])].sum()/tot[a] for a in DATA}
rank_line("R+ near+loop (rewarded)",vp); rank_line("R- near+loop (punished)",vn); rank_line("R+/|R-| near+loop",{a:vp[a]/abs(vn[a]) for a in DATA},"{:.3f}")
vp2={a:1e3*(DATA[a]["adv"]*DATA[a]["s1"])[(DATA[a]["adv"]>0)&(DATA[a]["think"]>=1000)].sum()/tot[a] for a in DATA}
vn2={a:1e3*(DATA[a]["adv"]*DATA[a]["s1"])[(DATA[a]["adv"]<0)&(DATA[a]["think"]>=1000)].sum()/tot[a] for a in DATA}
rank_line("R+ >=1k think (rewarded)",vp2); rank_line("R- >=1k think (punished)",vn2); rank_line("R+/|R-| >=1k think",{a:vp2[a]/abs(vn2[a]) for a in DATA},"{:.3f}")
P("")
P("--- E. how 'confident' the trainer is in the sampled thinking tokens: mean p_trainer over think tokens by class (engine fidelity / on-policy-ness of the data)")
for cls in ("short","normal_long","near","loop"):
    vals={}
    for a in DATA:
        d=DATA[a]; m=(d["cls"]==cls)&(d["think"]>0); vals[a]=float(np.average(d["mptr"][m],weights=d["think"][m])) if m.any() else float("nan")
    rank_line(f"mean p_trainer of think tokens, class={cls}",vals,"{:.4f}")
vals={a:float(DATA[a]["lowp"].sum()/DATA[a]["think"].sum()) for a in DATA}; rank_line("share of think tokens with p_trainer < 0.5",vals,"{:.4f}")
P("")
P("--- F. depth curves from the per-chunk aggregates: mean |d| (trainer-engine) and mean log p_trainer by depth into the think block, normal_long turns, MINF arms vs VLLM arms")
agg={}
for a,o,eng in arms:
    acc={}
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.effagg.json")):
        s=int(fn.split("__s")[1][:3])
        if not (LO<=s<=HI): continue
        for k,v in json.load(open(fn)).items(): acc[k]=acc.get(k,0.0)+v
    agg[a]=acc
P(f"{'arm':<5}{'eng':<5}{'out':<5} "+" ".join(f"{'depth '+b:>16}" for b in ("0","256","1024","4096","16384"))+"   (cells: mean|d| x1e3 / mean logp_trainer / share p<0.5)")
for cls in ("normal_long","near"):
    P(f"  class {cls}:")
    for a,o,eng in arms:
        cells=[]
        for b in ("0","256","1024","4096","16384"):
            n=agg[a].get(f"{cls}|{b}|n",0)
            cells.append(f"{1e3*agg[a][f'{cls}|{b}|absd']/n:5.1f}/{agg[a][f'{cls}|{b}|lptr']/n:6.3f}/{agg[a][f'{cls}|{b}|lowp']/n:.3f}" if n>1000 else f"{'.':>16}")
        P(f"  {a:<5}{eng:<5}{o:<5} "+" ".join(f"{c:>16}" for c in cells))
open(f"{D}/effective_{LO}_{HI}_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

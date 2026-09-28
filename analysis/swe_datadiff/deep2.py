import glob,csv,math,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","bad"),("I","bad"),("P42","bad"),("A","good"),("K","good"),("M","good"),("N","good"),("Q4","good"),("J","good"),("P4","good")]
def col(rows,k):
    o=np.empty(len(rows))
    for i,r in enumerate(rows):
        try: o[i]=float(r[k]) if r[k]!="" else np.nan
        except: o[i]=np.nan
    return o
rng=np.random.default_rng(29); NB=2000
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
DATA={}; PH={}
for a,o in ARMS:
    rows=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.eff.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.eff.csv")):
        s=int(fn.split("__s")[1][:3])
        with open(fn) as fh:
            for r in csv.DictReader(fh): r["step"]=s; rows.append(r)
    d=dict(step=np.array([r["step"] for r in rows]),adv=col(rows,"adv"),n=col(rows,"n_tok"),think=col(rows,"think_tok"),cls=np.array([r["cls"] for r in rows]),s1deep=col(rows,"s1mp_deep_tr"),sid=np.array([r["sample_id"] for r in rows]),turn=col(rows,"turn").astype(int))
    d["gid"]=np.array([s.rsplit("_g",1)[0] for s in d["sid"]]); d["deepn"]=np.maximum(0,d["think"]-4096); DATA[a]=d
    for fn in glob.glob(f"{D}/parts/{a}__s0*_c*.prompts.csv"):
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                if int(r["step"])<=10: PH[(a,r["sample_id"])]=(int(r["step"]),r["prompt_hash"])
tot={a:DATA[a]["n"].sum() for a in DATA}
def rank_line(name,vals,fmt="{:+.3f}"):
    bad=[vals[a] for a,o in ARMS if o=="bad"]; good=[vals[a] for a,o in ARMS if o=="good"]
    order=sorted(vals,key=lambda a:-vals[a]); ranks=sorted(order.index(a)+1 for a,o in ARMS if o=="bad")
    sep="YES" if min(bad)>max(good) else ("rev" if max(bad)<min(good) else "")
    P(f"{name:<62} "+" ".join(f"{a}:{fmt.format(vals[a])}" for a,o in ARMS)+f" | bad ranks {ranks} {sep}")
P("DECOMPOSITION OF THE REWARDED DEEP-THINK GRADIENT, steps 1-10 (deep = think tokens beyond position 4096; live weight = 1 - p_trainer). Per 1k generated tokens unless stated.")
S_plus={};S_minus={};Rp={};nroll={};madv={};live={}
for a,o in ARMS:
    d=DATA[a]; pos=(d["adv"]>0)&(d["deepn"]>0); neg=(d["adv"]<0)&(d["deepn"]>0)
    S_plus[a]=1e3*d["s1deep"][pos].sum()/tot[a]; S_minus[a]=1e3*d["s1deep"][neg].sum()/tot[a]; Rp[a]=1e3*(d["adv"][pos]*d["s1deep"][pos]).sum()/tot[a]
    sids=set(d["sid"][pos]); nroll[a]=len(sids); madv[a]=float(np.mean([d["adv"][d["sid"]==s][0] for s in sids])) if sids else float("nan"); live[a]=d["s1deep"][pos].sum()/max(1,d["deepn"][pos].sum())
rank_line("live deep tokens in REWARDED rollouts, sum(1-p) per 1k tok",S_plus)
rank_line("live deep tokens in PUNISHED rollouts, sum(1-p) per 1k tok",S_minus)
rank_line("R_deep+ = sum adv x (1-p) rewarded deep, per 1k tok",Rp)
rank_line("rewarded rollouts containing a deep think turn [count]",nroll,"{:.0f}")
rank_line("mean advantage of those rollouts",madv,"{:.3f}")
rank_line("mean (1-p) of rewarded deep tokens",live,"{:.4f}")
rank_line("deep tokens in rewarded rollouts per 1k tok",{a:1e3*DATA[a]["deepn"][DATA[a]["adv"]>0].sum()/tot[a] for a in DATA},"{:.2f}")
rank_line("deep tokens in punished rollouts per 1k tok",{a:1e3*DATA[a]["deepn"][DATA[a]["adv"]<0].sum()/tot[a] for a in DATA},"{:.2f}")
rank_line("live deep tokens, rewarded / punished ratio",{a:S_plus[a]/S_minus[a] for a in DATA},"{:.3f}")
P("")
P("Group bootstrap for S+ (live rewarded deep tokens per 1k) and R_deep+:")
bs={}
for a,o in ARMS:
    d=DATA[a]; u,g=np.unique(d["gid"],return_inverse=True); ng=len(u); pos=(d["adv"]>0)&(d["deepn"]>0)
    S=np.bincount(g,weights=np.where(pos,d["s1deep"],0.0),minlength=ng); Rr=np.bincount(g,weights=np.where(pos,d["adv"]*d["s1deep"],0.0),minlength=ng); N=np.bincount(g,weights=d["n"],minlength=ng)
    W=rng.multinomial(ng,np.ones(ng)/ng,size=NB).astype(float); bs[a]=(1e3*(W@S)/(W@N),1e3*(W@Rr)/(W@N))
    P(f"  {a:<4}{o:<5} S+ {S_plus[a]:+.3f} [{np.percentile(bs[a][0],2.5):+.3f},{np.percentile(bs[a][0],97.5):+.3f}]   R_deep+ {Rp[a]:+.3f} [{np.percentile(bs[a][1],2.5):+.3f},{np.percentile(bs[a][1],97.5):+.3f}]")
for name,idx,vals in (("S+",0,S_plus),("R_deep+",1,Rp)):
    bb=np.mean([bs[a][idx] for a,o in ARMS if o=="bad"],axis=0); bg=np.mean([bs[a][idx] for a,o in ARMS if o=="good"],axis=0)
    dlt=np.mean([vals[a] for a,o in ARMS if o=="bad"])-np.mean([vals[a] for a,o in ARMS if o=="good"]); se=(bb-bg).std()
    P(f"  {name}: bad mean - good mean = {dlt:+.3f}, bootstrap se {se:.3f}, z {dlt/se:+.1f}")
P("")
P("CONCENTRATION: share of R_deep+ carried by the top-N rewarded rollouts (ranked by adv x live deep tokens), and the top 8 rollouts of each bad arm (step, sample, adv, deep tokens, live deep, classes of its deep turns)")
for a,o in ARMS:
    d=DATA[a]; pos=(d["adv"]>0)&(d["deepn"]>0)
    per=collections.defaultdict(lambda:[0.0,0,0.0,set(),0])
    for i in np.where(pos)[0]:
        p=per[d["sid"][i]]; p[0]+=d["adv"][i]*d["s1deep"][i]; p[1]+=int(d["deepn"][i]); p[2]+=d["s1deep"][i]; p[3].add(d["cls"][i]); p[4]=d["step"][i]
    items=sorted(per.items(),key=lambda kv:-kv[1][0]); total=sum(v[0] for _,v in items)
    cum=lambda n: sum(v[0] for _,v in items[:n])/total if total else float("nan")
    line=f"{a:<4}{o:<5} rollouts {len(items):>4}; top-10 {100*cum(10):4.0f} %, top-25 {100*cum(25):4.0f} %, top-50 {100*cum(50):4.0f} %, top-100 {100*cum(100):4.0f} %"
    if o=="bad":
        line+="\n      "+"; ".join(f"s{v[4]} {sid[-8:]} adv {d['adv'][d['sid']==sid][0]:+.2f} deep {v[1]} live {v[2]:.0f} {'/'.join(sorted(v[3]))}" for sid,v in items[:8])
    P(line)
P("")
P("PROMPT-PAIRED (same step and prompt across all 10 arms): per-prompt R_deep (group value per 1k arm tokens); bad-arm mean minus good-arm mean; sign test; also for S+.")
key=collections.defaultdict(dict); keyS=collections.defaultdict(dict)
for a,o in ARMS:
    d=DATA[a]; groups=collections.defaultdict(lambda:[0.0,0.0])
    for i in range(len(d["adv"])):
        if d["deepn"][i]<=0: continue
        k=PH.get((a,d["sid"][i]))
        if not k: continue
        groups[k][0]+=d["adv"][i]*d["s1deep"][i]
        if d["adv"][i]>0: groups[k][1]+=d["s1deep"][i]
    # also register prompts with zero deep tokens
    for i in range(len(d["adv"])):
        k=PH.get((a,d["sid"][i]))
        if k and k not in groups: groups[k]=[0.0,0.0]
    for k,v in groups.items(): key[k][a]=1e3*v[0]/tot[a]; keyS[k][a]=1e3*v[1]/tot[a]
common=[k for k in key if len(key[k])==10]
P(f"  prompts present in all 10 arms: {len(common)}")
for name,dd in (("R_deep",key),("S+ (live rewarded deep)",keyS)):
    diffs=np.array([np.mean([dd[k][a] for a,o in ARMS if o=="bad"])-np.mean([dd[k][a] for a,o in ARMS if o=="good"]) for k in common])
    nz=diffs[np.abs(diffs)>1e-12]; pos=int((nz>0).sum()); neg=int((nz<0).sum()); z=(pos-len(nz)/2)/math.sqrt(len(nz)/4) if len(nz) else 0
    top=np.argsort(-np.abs(diffs))[:5]
    P(f"  {name}: mean diff {diffs.mean():+.4f} per prompt (sum {diffs.sum():+.3f}); prompts with bad>good {pos}, bad<good {neg} (z {z:+.2f}); top-5 |diff| prompts carry {100*np.abs(diffs[top]).sum()/np.abs(diffs).sum():.0f} % of total |diff|")
open(f"{D}/deep2_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

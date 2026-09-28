import glob,csv,collections,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
def f(x,d=0.0):
    try: return float(x)
    except: return d
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("REWARD-ONLY vs ADVANTAGE: does the separation of the bad arms survive when the advantage is replaced by simpler reward-based weights? Steps 1-10, class = think turns of 4k-16k tokens (and near-repetitive zlib<0.25 with >=1k think). Rank 1 = most reinforcing (least negative).")
variants=["A: full advantage (LOO, std-normalized)","B: LOO-centered reward, no std: r_i - mean(others)","C: group-centered reward: r_i - mean(group)","D: raw reward: r_i (share of class tokens in rewarded rollouts)","E: raw reward minus arm mean reward","F: sign of advantage only (+1/-1/0)","G: reward-centered, rollout-count weighted (not token weighted)"]
res={v:{} for v in variants}
for a,o in ARMS:
    S=[];T=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.samples.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.samples.csv")):
        with open(fn) as fh: S+=list(csv.DictReader(fh))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.turns.csv")):
        with open(fn) as fh: T+=list(csv.DictReader(fh))
    groups=collections.defaultdict(list)
    for r in S: groups[r["group_id"]].append(r)
    rew={r["sample_id"]:f(r["reward"]) for r in S}; adv={r["sample_id"]:f(r["adv"]) for r in S}
    gmean={}; loo={}
    for g in groups.values():
        rs=np.array([f(r["reward"]) for r in g]); m=rs.mean()
        for i,r in enumerate(g):
            gmean[r["sample_id"]]=m; loo[r["sample_id"]]=rs[i]-np.delete(rs,i).mean()
    armmean=np.mean(list(rew.values()))
    tot=sum(f(r["n_tok"]) for r in T)
    for cls,sel in (("long",lambda r:4000<=f(r["n_tok"])<16000),("near",lambda r:f(r["think_tok"])>=1000 and f(r["zlib_reason"],1)<0.25)):
        rows=[r for r in T if sel(r)]; ctok=sum(f(r["n_tok"]) for r in rows)
        W={variants[0]:lambda r:adv[r["sample_id"]],variants[1]:lambda r:loo[r["sample_id"]],variants[2]:lambda r:rew[r["sample_id"]]-gmean[r["sample_id"]],variants[3]:lambda r:rew[r["sample_id"]],variants[4]:lambda r:rew[r["sample_id"]]-armmean,variants[5]:lambda r:float(np.sign(adv[r["sample_id"]]))}
        for v,w in W.items():
            if v==variants[3]: val=sum(w(r)*f(r["n_tok"]) for r in rows)/ctok
            else: val=1e3*sum(w(r)*f(r["n_tok"]) for r in rows)/tot
            res[v][(a,cls)]=val
        # G: rollout-count weighted, group-centered reward over rollouts containing a class turn
        sids={r["sample_id"] for r in rows}; res[variants[6]][(a,cls)]=1e3*sum(rew[s]-gmean[s] for s in sids)/len(S)
for cls in ("long","near"):
    P(""); P(f"class = {cls}")
    P(f"{'weighting':<58} "+" ".join(f"{a:>8}" for a,_ in ARMS)+f" | {'bad ranks':>9} {'bad-good':>9}")
    for v in variants:
        vals={a:res[v][(a,cls)] for a,_ in ARMS}; order=sorted(ARMS,key=lambda x:-vals[x[0]]); rk={a:order.index((a,o))+1 for a,o in ARMS}
        bad=np.mean([vals[a] for a,o in ARMS if o=="BAD"]); good=np.mean([vals[a] for a,o in ARMS if o!="BAD"])
        fmt=(lambda x:f"{x:>8.4f}") if v==variants[3] else (lambda x:f"{x:>+8.3f}")
        P(f"{v:<58} "+" ".join(fmt(vals[a]) for a,_ in ARMS)+f" | {rk['G']:>4} {rk['I']:>4} {bad-good:>+9.3f}")
P("")
P("Reading: variant D is a pure reward statistic (fraction of the class's tokens that sit in successful rollouts); B/C remove the group baseline without normalizing; A is what the trainer uses.")
open(f"{D}/rewardonly_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

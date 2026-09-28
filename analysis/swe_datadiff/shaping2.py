import glob,csv,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
def f(x,d=0.0):
    try: return float(x)
    except: return d
def loo_adv(r):
    r=np.asarray(r,float); n=len(r); out=np.zeros(n)
    for i in range(n):
        o=np.delete(r,i); m=o.mean(); s=o.std(ddof=1) if n>2 else 0.0
        out[i]=(r[i]-m)/s if s>0 else (r[i]-m)
    return out
S={};T={}
for a,o in ARMS:
    S[a]=[];T[a]=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.samples.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.samples.csv")):
        with open(fn) as fh: S[a]+=list(csv.DictReader(fh))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.turns.csv")):
        with open(fn) as fh: T[a]+=list(csv.DictReader(fh))
# verify formula
groups=collections.defaultdict(list)
for r in S["M"]: groups[r["group_id"]].append(r)
err=max(np.abs(loo_adv([f(r["reward"]) for r in g])-np.array([f(r["adv"]) for r in g])).max() for g in groups.values())
print(f"advantage formula = leave-one-out: (r_i - mean of the other 15) / std_ddof1 of the other 15, unnormalized when that std is 0. Max |predicted - observed| over chain M groups: {err:.4f}")
print()
print("EXACT REWARD-SHAPING SIMULATION, steps 1-10: r' = r - lam x (near-repetitive + looping think tokens)/10k, LOO advantages recomputed. Cells: E(long 4k-16k) / E(near-rep zlib<0.25)  per 1k tokens; last column: share of groups that gain gradient (were zero-variance, now not)")
lams=[0.0,0.02,0.05,0.1,0.2,0.5]
print(f"{'arm':<4}{'out':<5} "+" ".join(f"{'lam='+str(l):>17}" for l in lams)+f" | {'groups gaining gradient (lam=0.05)':>34}")
for a,o in ARMS:
    groups=collections.defaultdict(list)
    for r in S[a]: groups[r["group_id"]].append(r)
    pen={r["sample_id"]:(f(r["loop_tok"])+f(r["near_loop_tok"]))/1e4 for r in S[a]}
    tot=sum(f(r["n_tok"]) for r in T[a]); cells=[]; gain=0
    for lam in lams:
        adv={}
        for gid,g in groups.items():
            rp=[f(r["reward"])-lam*pen[r["sample_id"]] for r in g]; av=loo_adv(rp)
            if lam==0.05 and np.ptp([f(r["reward"]) for r in g])==0 and np.ptp(rp)>0: gain+=1
            for r,v in zip(g,av): adv[r["sample_id"]]=v
        El=1e3*sum(adv[r["sample_id"]]*f(r["n_tok"]) for r in T[a] if 4000<=f(r["n_tok"])<16000)/tot
        En=1e3*sum(adv[r["sample_id"]]*f(r["n_tok"]) for r in T[a] if f(r["think_tok"])>=1000 and f(r["zlib_reason"],1)<0.25)/tot
        cells.append(f"{El:>+7.2f}/{En:>+8.2f}")
    print(f"{a:<4}{o:<5} "+" ".join(f"{c:>17}" for c in cells)+f" | {gain:>3} of {len(groups)}")
print()
print("Alternative shaping: penalize only FULL loops (zlib<0.10) -> does it also equalize? (lam=0.1)")
for a,o in ARMS:
    groups=collections.defaultdict(list)
    for r in S[a]: groups[r["group_id"]].append(r)
    pen={r["sample_id"]:f(r["loop_tok"])/1e4 for r in S[a]}; tot=sum(f(r["n_tok"]) for r in T[a]); adv={}
    for g in groups.values():
        rp=[f(r["reward"])-0.1*pen[r["sample_id"]] for r in g]
        for r,v in zip(g,loo_adv(rp)): adv[r["sample_id"]]=v
    El=1e3*sum(adv[r["sample_id"]]*f(r["n_tok"]) for r in T[a] if 4000<=f(r["n_tok"])<16000)/tot
    En=1e3*sum(adv[r["sample_id"]]*f(r["n_tok"]) for r in T[a] if f(r["think_tok"])>=1000 and f(r["zlib_reason"],1)<0.25)/tot
    print(f"  {a} {o}: E long {El:+.2f}  E near-rep {En:+.2f}")
print()
print("First three steps only (weights ~ base model in every arm): E(long 4k-16k) cumulative and E(near-rep), observed advantages")
for a,o in ARMS:
    tot=sum(f(r["n_tok"]) for r in T[a] if int(r["step"])<=3)
    El=1e3*sum(f(r["adv"])*f(r["n_tok"]) for r in T[a] if int(r["step"])<=3 and 4000<=f(r["n_tok"])<16000)/tot
    En=1e3*sum(f(r["adv"])*f(r["n_tok"]) for r in T[a] if int(r["step"])<=3 and f(r["think_tok"])>=1000 and f(r["zlib_reason"],1)<0.25)/tot
    E1k=1e3*sum(f(r["adv"])*f(r["n_tok"]) for r in T[a] if int(r["step"])<=3 and 1000<=f(r["n_tok"])<16000)/tot
    print(f"  {a} {o}: steps 1-3  E(long 4k-16k) {El:+.2f}   E(near-rep) {En:+.2f}   E(turns 1k-16k) {E1k:+.2f}")

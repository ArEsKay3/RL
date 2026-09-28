import glob,csv,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
def f(x,d=0.0):
    try: return float(x)
    except: return d
S={};T={}
for a,o in ARMS:
    S[a]=[];T[a]=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.samples.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.samples.csv")):
        with open(fn) as fh: S[a]+=list(csv.DictReader(fh))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.turns.csv")):
        with open(fn) as fh: T[a]+=list(csv.DictReader(fh))
# check advantage formula on observed data: adv = (r - mean)/(std_ddof1 + eps)
a="M"; groups=collections.defaultdict(list)
for r in S[a]: groups[r["group_id"]].append(r)
errs=[]
for g in groups.values():
    rw=np.array([f(r["reward"]) for r in g]); ad=np.array([f(r["adv"]) for r in g])
    if rw.std()==0: continue
    pred=(rw-rw.mean())/(rw.std(ddof=1)+0.014); errs.append(np.abs(pred-ad).max())
print(f"advantage formula check (chain M): max |predicted - observed| over groups = {max(errs):.3f} with adv=(r-mean)/(std_ddof1+0.014)")
print()
print("REWARD SHAPING SIMULATION on the existing steps 1-10 data: r' = r - lam x (tokens in near-repetitive or looping think turns, zlib<0.25 & >=1k think) / 10,000; advantages recomputed per group; E(long 4k-16k) and E(near-rep) per arm. Goal: does a small penalty make every arm's data suppress long/near-repetitive thinking?")
lams=[0.0,0.05,0.1,0.2,0.5,1.0]
print(f"{'arm':<4}{'out':<5} "+" ".join(f"{'lam='+str(l):>16}" for l in lams)+"   (each cell: E long4-16k / E near-rep)")
for a,o in ARMS:
    groups=collections.defaultdict(list)
    for r in S[a]: groups[r["group_id"]].append(r)
    pen={r["sample_id"]:(f(r["loop_tok"])+f(r["near_loop_tok"]))/1e4 for r in S[a]}
    tot=sum(f(r["n_tok"]) for r in T[a])
    cells=[]
    for lam in lams:
        adv={}
        for g in groups.values():
            rp=np.array([f(r["reward"])-lam*pen[r["sample_id"]] for r in g]); sd=rp.std(ddof=1)
            for r,v in zip(g,(rp-rp.mean())/(sd+0.014) if sd>0 else np.zeros(len(g))): adv[r["sample_id"]]=v
        El=1e3*sum(adv[r["sample_id"]]*f(r["n_tok"]) for r in T[a] if 4000<=f(r["n_tok"])<16000)/tot
        En=1e3*sum(adv[r["sample_id"]]*f(r["n_tok"]) for r in T[a] if f(r["think_tok"])>=1000 and f(r["zlib_reason"],1)<0.25)/tot
        cells.append(f"{El:>+7.2f}/{En:>+7.2f}")
    print(f"{a:<4}{o:<5} "+" ".join(f"{c:>16}" for c in cells))
print()
print("share of rollouts whose reward changes sign or whose group advantage pattern changes (lam=0.1): penalty per rollout = 0.1 x near/loop tokens/10k -> a 5k-token near-repetitive think costs 0.05 reward units")
for a,o in ARMS:
    pen=[(f(r["loop_tok"])+f(r["near_loop_tok"]))/1e4*0.1 for r in S[a]]; print(f"  {a}: rollouts with any penalty {np.mean([p>0 for p in pen]):.3f}; mean penalty among them {np.mean([p for p in pen if p>0]):.3f}; max {max(pen):.2f}")

import glob,csv,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
def f(x,d=np.nan):
    try: return float(x)
    except: return d
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("DECOMPOSITION of the long-turn exposure (turns 4k-16k tokens, steps 1-10). adv>0 = rewarded rollout, adv<0 = punished, adv=0 = zero-variance group (no gradient).")
P(f"{'arm':<4}{'out':<5} {'long turns':>10} {'tok(M)':>7} | {'n adv>0':>7} {'tok%':>5} {'mean adv':>8} | {'n adv<0':>7} {'tok%':>5} {'mean adv':>8} | {'n adv=0':>7} {'tok%':>5} | {'E+':>6} {'E-':>7} {'E':>7} | {'rollouts w/ long turn: rew%':>27} {'pun%':>5} {'zero%':>5}")
for a,o in ARMS:
    T=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.turns.csv")):
        with open(fn) as fh: T+=list(csv.DictReader(fh))
    S=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.samples.csv")):
        with open(fn) as fh: S+=list(csv.DictReader(fh))
    tot=sum(f(r["n_tok"]) for r in T)
    lg=[r for r in T if 4000<=f(r["n_tok"])<16000]
    pos=[r for r in lg if f(r["adv"])>1e-9]; neg=[r for r in lg if f(r["adv"])<-1e-9]; zer=[r for r in lg if abs(f(r["adv"]))<=1e-9]
    tk=lambda rs: sum(f(r["n_tok"]) for r in rs)
    Ep=1e3*sum(f(r["adv"])*f(r["n_tok"]) for r in pos)/tot; En=1e3*sum(f(r["adv"])*f(r["n_tok"]) for r in neg)/tot
    # rollouts containing a long turn, by adv sign
    sl=[r for r in S if f(r["max_turn"])>=4000]
    rp=sum(1 for r in sl if f(r["adv"])>1e-9); rn=sum(1 for r in sl if f(r["adv"])<-1e-9); rz=len(sl)-rp-rn
    P(f"{a:<4}{o:<5} {len(lg):>10} {tk(lg)/1e6:>7.2f} | {len(pos):>7} {100*tk(pos)/tk(lg):>5.1f} {np.mean([f(r['adv']) for r in pos]):>+8.3f} | {len(neg):>7} {100*tk(neg)/tk(lg):>5.1f} {np.mean([f(r['adv']) for r in neg]):>+8.3f} | {len(zer):>7} {100*tk(zer)/tk(lg):>5.1f} | {Ep:>+6.2f} {En:>+7.2f} {Ep+En:>+7.2f} | {100*rp/len(sl):>27.1f} {100*rn/len(sl):>5.1f} {100*rz/len(sl):>5.1f}")
P("")
P("Same for ALL generated tokens (the length-asymmetry statistic): share of tokens by adv sign and mean adv by sign")
P(f"{'arm':<4}{'out':<5} {'rollouts':>8} | {'adv>0 n':>7} {'tok%':>5} {'mean adv':>8} {'mean len':>8} | {'adv<0 n':>7} {'tok%':>5} {'mean adv':>8} {'mean len':>8} | {'adv=0 n':>7} {'tok%':>5} {'mean len':>8} | {'E all':>7}")
for a,o in ARMS:
    S=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.samples.csv")):
        with open(fn) as fh: S+=list(csv.DictReader(fh))
    tot=sum(f(r["n_gen"]) for r in S)
    pos=[r for r in S if f(r["adv"])>1e-9]; neg=[r for r in S if f(r["adv"])<-1e-9]; zer=[r for r in S if abs(f(r["adv"]))<=1e-9]
    tk=lambda rs: sum(f(r["n_gen"]) for r in rs); ml=lambda rs: np.mean([f(r["n_gen"]) for r in rs])
    E=1e3*sum(f(r["adv"])*f(r["n_gen"]) for r in S)/tot
    P(f"{a:<4}{o:<5} {len(S):>8} | {len(pos):>7} {100*tk(pos)/tot:>5.1f} {np.mean([f(r['adv']) for r in pos]):>+8.3f} {ml(pos):>8.0f} | {len(neg):>7} {100*tk(neg)/tot:>5.1f} {np.mean([f(r['adv']) for r in neg]):>+8.3f} {ml(neg):>8.0f} | {len(zer):>7} {100*tk(zer)/tot:>5.1f} {ml(zer):>8.0f} | {E:>+7.2f}")
P("")
P("Group structure: number of rewarded members per group (histogram over 320 groups) and mean generated length of rewarded vs punished members within mixed groups")
P(f"{'arm':<4}{'out':<5} "+" ".join(f"{k:>4}" for k in range(17))+f" | {'mixed groups':>12} {'rew len':>8} {'pun len':>8} {'ratio':>6}")
for a,o in ARMS:
    S=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.samples.csv")):
        with open(fn) as fh: S+=list(csv.DictReader(fh))
    groups=collections.defaultdict(list)
    for r in S: groups[r["group_id"]].append(r)
    h=collections.Counter(sum(1 for r in g if f(r["reward"])>0) for g in groups.values())
    mixed=[g for g in groups.values() if 0<sum(1 for r in g if f(r["reward"])>0)<len(g)]
    rl=np.mean([f(r["n_gen"]) for g in mixed for r in g if f(r["reward"])>0]); pl=np.mean([f(r["n_gen"]) for g in mixed for r in g if f(r["reward"])<=0])
    P(f"{a:<4}{o:<5} "+" ".join(f"{h.get(k,0):>4}" for k in range(17))+f" | {len(mixed):>12} {rl:>8.0f} {pl:>8.0f} {rl/pl:>6.3f}")
open(f"{D}/decompose_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

import glob,csv,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","bad"),("I","bad"),("P42","bad"),("A","good"),("K","good"),("M","good"),("N","good"),("Q4","good"),("J","good"),("P4","good")]
def f(x,d=np.nan):
    try: return float(x)
    except: return d
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
data={}
for a,o in ARMS:
    eff=[];turns={};samp={}
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.eff.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.eff.csv")):
        with open(fn) as fh:
            for r in csv.DictReader(fh): eff.append(r)
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.turns.csv")):
        with open(fn) as fh:
            for r in csv.DictReader(fh): turns[(r["sample_id"],int(r["turn"]))]=(int(f(r["has_close"])),int(f(r["ends_seq"])))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.samples.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.samples.csv")):
        with open(fn) as fh:
            for r in csv.DictReader(fh): samp[r["sample_id"]]=(int(f(r["truncated"])),int(f(r["ends_seq_noclose"])),int(f(r["loop_turns"])),int(f(r["n_gen"])),float(f(r["reward"])))
    sid=np.array([r["sample_id"] for r in eff]); adv=np.array([f(r["adv"]) for r in eff]); n=np.array([f(r["n_tok"]) for r in eff]); th=np.array([f(r["think_tok"]) for r in eff]); s1d=np.array([f(r["s1mp_deep_tr"]) for r in eff]); tk=np.array([int(r["turn"]) for r in eff])
    hc=np.array([turns.get((s,k),(1,0))[0] for s,k in zip(sid,tk)]); trunc=np.array([samp.get(s,(0,0,0,0,0))[0] for s in sid]); runaway=np.array([samp.get(s,(0,0,0,0,0))[1] for s in sid]); hasloop=np.array([samp.get(s,(0,0,0,0,0))[2]>0 for s in sid])
    data[a]=dict(adv=adv,n=n,th=th,s1d=s1d,hc=hc,trunc=trunc,runaway=runaway,hasloop=hasloop,samp=samp)
def stats(a,keep):
    d=data[a]; tot=d["n"].sum(); sel=keep&(d["th"]>=1000)
    R=1e3*(d["adv"][sel]*d["s1d"][sel]).sum()/tot; Rp=1e3*(d["adv"][sel&(d["adv"]>0)]*d["s1d"][sel&(d["adv"]>0)]).sum()/tot; Sp=1e3*d["s1d"][sel&(d["adv"]>0)].sum()/tot
    return R,Rp,Sp
def rank(vals):
    order=sorted(vals,key=lambda a:-vals[a]); ranks=sorted(order.index(a)+1 for a,o in ARMS if o=="bad"); bad=[vals[a] for a,o in ARMS if o=="bad"]; good=[vals[a] for a,o in ARMS if o=="good"]
    return ranks,("YES" if min(bad)>max(good) else "")
P("ROBUSTNESS OF THE DEEP-THINK STATISTICS TO CONTEXT-LIMIT / RUNAWAY / LOOP ROLLOUTS, steps 1-10 (x1e-3 per generated token). Bad = G, I, P⁗2; good = A, K, M, N, Q⁗, J, P⁗.")
P("")
P("How many context-limit rollouts are there, and how many of them are rewarded?")
P(f"{'arm':<5}{'out':<5} {'truncated':>9} {'rewarded':>8} {'runaway (unclosed think at end)':>31} {'rewarded':>8} {'with a full loop':>16} {'rewarded':>8} | {'deep tokens in truncated+runaway rollouts, rewarded':>50} {'punished':>8}")
for a,o in ARMS:
    d=data[a]; s=d["samp"]; tr=[v for v in s.values() if v[0]]; ru=[v for v in s.values() if v[1]]; lo=[v for v in s.values() if v[2]>0]
    deep=np.maximum(0,d["th"]-4096); m=(d["trunc"]==1)|(d["runaway"]==1)
    P(f"{a:<5}{o:<5} {len(tr):>9} {sum(1 for v in tr if v[4]>0):>8} {len(ru):>31} {sum(1 for v in ru if v[4]>0):>8} {len(lo):>16} {sum(1 for v in lo if v[4]>0):>8} | {int(deep[m&(d['adv']>0)].sum()):>50} {int(deep[m&(d['adv']<0)].sum()):>8}")
P("")
P("Statistics under exclusions (R_deep net / R_deep rewarded side / S+):")
P(f"{'exclusion':<52} "+" ".join(f"{a:>22}" for a,o in ARMS)+f" | {'bad ranks: net / R+ / S+':>28}")
excl=[("none (as reported)",lambda d:np.ones(len(d["adv"]),bool)),
      ("drop truncated rollouts",lambda d:d["trunc"]==0),
      ("drop rollouts ending in an unclosed think (runaway)",lambda d:d["runaway"]==0),
      ("drop truncated OR runaway rollouts",lambda d:(d["trunc"]==0)&(d["runaway"]==0)),
      ("count only think blocks that were closed (</think> present)",lambda d:d["hc"]==1),
      ("drop any rollout containing a full loop (zlib<0.10)",lambda d:~d["hasloop"]),
      ("closed blocks only AND no truncated/runaway/loop rollouts",lambda d:(d["hc"]==1)&(d["trunc"]==0)&(d["runaway"]==0)&(~d["hasloop"]))]
for name,fn in excl:
    vals={a:stats(a,fn(data[a])) for a,o in ARMS}
    r0,s0=rank({a:v[0] for a,v in vals.items()}); r1,s1=rank({a:v[1] for a,v in vals.items()}); r2,s2=rank({a:v[2] for a,v in vals.items()})
    P(f"{name:<52} "+" ".join(f"{v[0]:+.3f}/{v[1]:+.3f}/{v[2]:.3f}" for a,v in vals.items())+f" | {str(r0)+s0:>9} {str(r1)+s1:>9} {str(r2)+s2:>9}")
open(f"{D}/deep_robust_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

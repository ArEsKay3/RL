import glob,csv,math,collections,itertools
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","bad"),("I","bad"),("P42","bad"),("A","good"),("K","good"),("M","good"),("N","good"),("Q4","good"),("J","good"),("P4","good")]
def loadcsv(p):
    rows=[]
    for fn in sorted(glob.glob(p)):
        with open(fn) as fh: rows+=list(csv.DictReader(fh))
    return rows
def f(x,d=0.0):
    try: return float(x)
    except: return d
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
S={}
for a,o in ARMS:
    S[a]=[r for r in loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv") if 1<=int(float(r["step"]))<=10]
def rank_report(name,vals,fmt="{:+.3f}"):
    bad=[vals[a] for a,o in ARMS if o=="bad"]; good=[vals[a] for a,o in ARMS if o=="good"]
    order=sorted(vals,key=lambda a:-vals[a]); ranks={a:order.index(a)+1 for a in vals}
    sep="YES" if min(bad)>max(good) else ("yes(rev)" if max(bad)<min(good) else "")
    P(f"{name:<58} "+" ".join(f"{a}:{fmt.format(vals[a])}" for a,o in ARMS)+f" | bad ranks {sorted(ranks[a] for a,o in ARMS if o=='bad')} {sep}")
P("REWARD-SIDE CANDIDATES, steps 1-10, 3 bad arms (G, I, P⁗2) vs 7 good (A, K, M, N, Q⁗, J, P⁗). Rank 1 = largest value. 'YES' = every bad arm above every good arm (chance 1/120 per statistic).")
feats={}
for a,o in ARMS:
    s=S[a]; tot=sum(f(r["n_gen"]) for r in s)
    rew=[r for r in s if f(r["reward"])>0]
    feats.setdefault("reward rate",{})[a]=len(rew)/len(s)
    feats.setdefault("rewarded rollouts with a full loop (zlib<0.10, >=1k) [count]",{})[a]=sum(1 for r in rew if f(r["loop_tok"])>0)
    feats.setdefault("rewarded rollouts with near-loop (zlib<0.25) [count]",{})[a]=sum(1 for r in rew if f(r["near_loop_tok"])+f(r["loop_tok"])>0)
    feats.setdefault("rewarded rollouts ending in an unclosed think (runaway) [count]",{})[a]=sum(1 for r in rew if f(r["ends_seq_noclose"])>0)
    feats.setdefault("rewarded truncated rollouts [count]",{})[a]=sum(1 for r in rew if f(r["truncated"])>0)
    feats.setdefault("positive adv mass on loop tokens, per 1k tok",{})[a]=1e3*sum(f(r["adv"])*f(r["loop_tok"]) for r in rew)/tot
    feats.setdefault("positive adv mass on near+loop tokens, per 1k tok",{})[a]=1e3*sum(f(r["adv"])*(f(r["loop_tok"])+f(r["near_loop_tok"])) for r in rew)/tot
    feats.setdefault("negative adv mass on near+loop tokens, per 1k tok",{})[a]=1e3*sum(f(r["adv"])*(f(r["loop_tok"])+f(r["near_loop_tok"])) for r in s if f(r["adv"])<0)/tot
    feats.setdefault("net adv mass on near+loop tokens, per 1k tok",{})[a]=1e3*sum(f(r["adv"])*(f(r["loop_tok"])+f(r["near_loop_tok"])) for r in s)/tot
    posmass=sum(f(r["adv"])*f(r["n_gen"]) for r in rew)
    feats.setdefault("share of positive adv mass in rollouts with near/loop content",{})[a]=sum(f(r["adv"])*f(r["n_gen"]) for r in rew if f(r["loop_tok"])+f(r["near_loop_tok"])>0)/posmass
    feats.setdefault("share of positive adv mass in rollouts with a 4k+ turn",{})[a]=sum(f(r["adv"])*f(r["n_gen"]) for r in rew if f(r["max_turn"])>=4000)/posmass
    feats.setdefault("share of positive adv mass in rollouts > 50k tokens",{})[a]=sum(f(r["adv"])*f(r["n_gen"]) for r in rew if f(r["n_gen"])>50000)/posmass
    feats.setdefault("mean gen tokens of rewarded rollouts",{})[a]=np.mean([f(r["n_gen"]) for r in rew])
    feats.setdefault("mean gen tokens of punished rollouts",{})[a]=np.mean([f(r["n_gen"]) for r in s if f(r["adv"])<0])
    feats.setdefault("rewarded/punished mean length ratio",{})[a]=np.mean([f(r["n_gen"]) for r in rew])/np.mean([f(r["n_gen"]) for r in s if f(r["adv"])<0])
    feats.setdefault("success rate of rollouts with a 4k+ turn",{})[a]=np.mean([f(r["reward"]) for r in s if f(r["max_turn"])>=4000])
    feats.setdefault("success rate of rollouts with near/loop content",{})[a]=np.mean([f(r["reward"]) for r in s if f(r["loop_tok"])+f(r["near_loop_tok"])>0])
    feats.setdefault("success rate of rollouts > 50k tokens",{})[a]=np.mean([f(r["reward"]) for r in s if f(r["n_gen"])>50000])
    feats.setdefault("loop-token share %, steps 1-10",{})[a]=100*sum(f(r["loop_tok"]) for r in s)/tot
    feats.setdefault("near-loop-token share %, steps 1-10",{})[a]=100*sum(f(r["near_loop_tok"]) for r in s)/tot
    feats.setdefault("rollouts with 200 turns [count]",{})[a]=sum(1 for r in s if f(r["n_turns"])>=200)
    feats.setdefault("rewarded rollouts with 200 turns [count]",{})[a]=sum(1 for r in rew if f(r["n_turns"])>=200)
    feats.setdefault("zero-variance groups [count]",{})[a]=sum(1 for g,v in collections.Counter((r["group_id"],f(r["reward"])>0) for r in s).items() if v==16)
    feats.setdefault("mean openhands run time (s)",{})[a]=np.nanmean([f(r["openhands_run_time"],np.nan) for r in s])
    feats.setdefault("mean |adv| of rewarded rollouts",{})[a]=np.mean([f(r["adv"]) for r in rew])
    feats.setdefault("mean |adv| of punished rollouts",{})[a]=np.mean([-f(r["adv"]) for r in s if f(r["adv"])<0])
    feats.setdefault("mean seq_mult_prob_error",{})[a]=np.mean([f(r["seq_mult_prob_error"]) for r in s])
    feats.setdefault("TIS-clipped tokens per 1M",{})[a]=1e6*sum(f(r["n_clip_lo"])+f(r["n_clip_hi"]) for r in s)/tot
    feats.setdefault("masked-by-logprob-error samples [count]",{})[a]=sum(1 for r in s if f(r["mask_after"])==0)
for name,vals in feats.items():
    rank_report(name,vals,"{:+.3f}" if abs(np.mean(list(vals.values())))<50 else "{:+.0f}")
P("")
P("Top rewarded rollouts by positive advantage mass on near/loop tokens (adv x (loop+near tokens)), per arm, steps 1-10: step, sample, adv, loop tok, near tok, gen tok, turns")
for a,o in ARMS:
    rew=sorted([r for r in S[a] if f(r["adv"])>0 and f(r["loop_tok"])+f(r["near_loop_tok"])>0],key=lambda r:-f(r["adv"])*(f(r["loop_tok"])+f(r["near_loop_tok"])))
    P(f"{a} ({o}): {len(rew)} such rollouts; top 5: "+"; ".join(f"s{int(f(r['step']))} {r['sample_id'][-8:]} adv {f(r['adv']):+.2f} loop {int(f(r['loop_tok']))} near {int(f(r['near_loop_tok']))} gen {int(f(r['n_gen']))} turns {int(f(r['n_turns']))}" for r in rew[:5]))
open(f"{D}/rewardside_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

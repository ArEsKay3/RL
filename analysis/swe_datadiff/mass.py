import glob,csv,collections,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
def f(x,d=np.nan):
    try: return float(x)
    except: return d
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("CLOSE-TOKEN EXPOSURE: E(</think>) = 1000 x sum over turns with a close of adv / total generated tokens (one close token per turn); E(<|im_end|>) likewise; E(think tokens) = adv x think tokens; ratio think/close = how much the batch reinforces 'keep thinking' relative to 'close' (both negative in all arms; less negative = weaker suppression).")
P(f"{'arm':<4}{'out':<5} {'E close':>8} {'E im_end':>9} {'E think tok':>11} {'E answer tok':>12} | {'turns rew':>9} {'turns pun':>9} {'sum adv rew turns':>17} {'sum adv pun turns':>17} | {'E close, turns>=1k':>18} {'E think, turns>=1k':>18}")
for a,o in ARMS:
    T=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.turns.csv")):
        with open(fn) as fh: T+=list(csv.DictReader(fh))
    tot=sum(f(r["n_tok"]) for r in T)
    Ec=1e3*sum(f(r["adv"]) for r in T if f(r["has_close"])==1)/tot; Ei=1e3*sum(f(r["adv"]) for r in T if f(r["last_is_imend"])==1)/tot
    Et=1e3*sum(f(r["adv"])*f(r["think_tok"]) for r in T)/tot; Ea=1e3*sum(f(r["adv"])*(f(r["n_tok"])-f(r["think_tok"])) for r in T)/tot
    nr=sum(1 for r in T if f(r["adv"])>1e-9); npn=sum(1 for r in T if f(r["adv"])<-1e-9); sr=sum(f(r["adv"]) for r in T if f(r["adv"])>1e-9); sp=sum(f(r["adv"]) for r in T if f(r["adv"])<-1e-9)
    lg=[r for r in T if f(r["think_tok"])>=1000]; Ecl=1e3*sum(f(r["adv"]) for r in lg if f(r["has_close"])==1)/tot; Etl=1e3*sum(f(r["adv"])*f(r["think_tok"]) for r in lg)/tot
    P(f"{a:<4}{o:<5} {Ec:>+8.4f} {Ei:>+9.4f} {Et:>+11.2f} {Ea:>+12.2f} | {nr:>9} {npn:>9} {sr:>+17.0f} {sp:>+17.0f} | {Ecl:>+18.4f} {Etl:>+18.2f}")
P("")
P("GRADIENT MASS OUTLIERS: rollouts ranked by |adv| x generated tokens (the weight a sample carries in the token-level loss). Top 12 rewarded per arm with their long-think content.")
for a,o in ARMS:
    S=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.samples.csv")):
        with open(fn) as fh: S+=list(csv.DictReader(fh))
    tot=sum(f(r["n_gen"]) for r in S)
    rew=sorted([r for r in S if f(r["adv"])>1e-9],key=lambda r:-f(r["adv"])*f(r["n_gen"]))
    mass_rew=sum(f(r["adv"])*f(r["n_gen"]) for r in rew); mass_top=sum(f(r["adv"])*f(r["n_gen"]) for r in rew[:12])
    P(f"{a} {o}: rewarded rollouts {len(rew)}, total positive mass {1e3*mass_rew/tot:+.1f} per 1k tokens, top-12 share {100*mass_top/mass_rew:.0f}%, positive mass in rollouts with a 4k+ turn: {100*sum(f(r['adv'])*f(r['n_gen']) for r in rew if f(r['max_turn'])>=4000)/mass_rew:.1f}%, with near/loop tokens: {100*sum(f(r['adv'])*f(r['n_gen']) for r in rew if f(r['near_loop_tok'])+f(r['loop_tok'])>0)/mass_rew:.1f}%")
    for r in rew[:6]: P(f"    s{int(f(r['step'])):>2} {r['sample_id'][-8:]} adv {f(r['adv']):+.2f} gen {int(f(r['n_gen'])):>6} turns {int(f(r['n_turns'])):>3} max_turn {int(f(r['max_turn'])):>6} near_tok {int(f(r['near_loop_tok'])):>6} loop_tok {int(f(r['loop_tok'])):>6} mass {1e3*f(r['adv'])*f(r['n_gen'])/tot:.2f}")
P("")
P("NEGATIVE MASS: share of punished mass (|adv| x tokens) carried by rollouts with a 4k+ turn, and by rollouts with near/loop tokens")
for a,o in ARMS:
    S=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.samples.csv")):
        with open(fn) as fh: S+=list(csv.DictReader(fh))
    pun=[r for r in S if f(r["adv"])<-1e-9]; m=sum(f(r["adv"])*f(r["n_gen"]) for r in pun)
    P(f"{a} {o}: punished rollouts {len(pun)}; share of punished mass in rollouts with a 4k+ turn {100*sum(f(r['adv'])*f(r['n_gen']) for r in pun if f(r['max_turn'])>=4000)/m:.1f}%; with near/loop tokens {100*sum(f(r['adv'])*f(r['n_gen']) for r in pun if f(r['near_loop_tok'])+f(r['loop_tok'])>0)/m:.1f}%; mean |adv| of punished {np.mean([-f(r['adv']) for r in pun]):.3f}; mean gen {np.mean([f(r['n_gen']) for r in pun]):.0f}")
open(f"{D}/mass_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

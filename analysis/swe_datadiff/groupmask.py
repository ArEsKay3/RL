import glob,csv,collections
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
def f(x,d=0.0):
    try: return float(x)
    except: return d
def load(a):
    per=collections.defaultdict(lambda:dict(adv=0.0,rew=0.0,n=0.0,deep=0,live=0.0,long_tok=0.0,step=0))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.eff.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.eff.csv")):
        s=int(fn.split("__s")[1][:3])
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                p=per[r["sample_id"]]; p["adv"]=f(r["adv"]); p["rew"]=f(r["reward"]); p["n"]+=f(r["n_tok"]); p["step"]=s
                d=max(0,int(f(r["think_tok"]))-4096); nt=f(r["n_tok"])
                if d>0: p["deep"]+=d; p["live"]+=f(r["s1mp_deep_tr"])
                if 4000<=nt<16000: p["long_tok"]+=nt
    groups=collections.defaultdict(list)
    for sid,p in per.items(): p["sid"]=sid; groups[sid.rsplit("_g",1)[0]].append(p)
    return groups
def eff(groups,drop_sids):
    tot=sum(p["n"] for g in groups.values() for p in g); keep=[p for g in groups.values() for p in g if p["sid"] not in drop_sids]
    Rd=1e3*sum(p["adv"]*p["live"] for p in keep)/tot; Rp=1e3*sum(p["adv"]*p["live"] for p in keep if p["adv"]>0)/tot; E=1e3*sum(p["adv"]*p["long_tok"] for p in keep)/tot
    ntok=sum(p["n"] for p in keep)/tot
    return Rd,Rp,E,len(drop_sids),1-ntok
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("GROUP-LEVEL AND FILTER-LEVEL ALTERNATIVES TO THE ROLLOUT MASK, chain G steps 1-10 (5,120 rollouts, 320 groups). Effective statistics = over the rollouts that remain trainable; x1e-3 per generated token. Healthy arms at k=10: R_deep -0.052 .. -0.171 (mean -0.119), equal-weight E -2.3 .. -4.8; looped arms R_deep -0.019 .. -0.032.")
G=load("G"); allp=[p for g in G.values() for p in g]
poison_g={gid:g for gid,g in G.items() if any(p["rew"]>0 and p["deep"]>0 for p in g) and 0<sum(1 for p in g if p["rew"]>0)<len(g)}
net={gid:sum(p["adv"]*p["live"] for p in g if p["deep"]>0) for gid,g in poison_g.items()}
P("")
P(f"{'scheme':<78} {'rollouts masked':>15} {'data removed':>12} {'eff. R_deep':>11} {'eff. R_deep+':>12} {'eff. E(4k-16k)':>14}")
def row(name,drop): r=eff(G,drop); P(f"{name:<78} {r[3]:>15} {100*r[4]:>11.1f}% {r[0]:>+11.3f} {r[1]:>+12.3f} {r[2]:>+14.2f}")
row("none (chain G / chain R / chain Y as far as deep thinking goes)",set())
row("chain X: mask the 127 rewarded rollouts with a >4096-token think block",{p["sid"] for p in allp if p["rew"]>0 and p["deep"]>0})
row(f"mask all {len(poison_g)} groups that contain a rewarded deep-thinker (whole prompts)",{p["sid"] for g in poison_g.values() for p in g})
order=sorted(net,key=lambda k:-net[k])
for N in (5,10,15,20,25,30,40):
    row(f"mask the top {N} such groups by net deep-think gradient (whole prompts)",{p["sid"] for gid in order[:N] for p in poison_g[gid]})
pos=[gid for gid in order if net[gid]>0]
row(f"mask the {len(pos)} groups whose net deep-think gradient is positive (whole prompts)",{p["sid"] for gid in pos for p in poison_g[gid]})
row("length filter: mask EVERY rollout with a >4096-token think block, rewarded or not",{p["sid"] for p in allp if p["deep"]>0})
row("length filter, rewarded+punished deep-thinkers inside poison groups only",{p["sid"] for g in poison_g.values() for p in g if p["deep"]>0})
row("mask rewarded deep-thinkers AND their whole groups' punished deep-thinkers (siblings)",{p["sid"] for g in poison_g.values() for p in g if p["deep"]>0})
P("")
P("Per-step counts for the two whole-group schemes that land inside the healthy band (groups per train step 1..10):")
for N in (10,15,20):
    cnt=collections.Counter(poison_g[gid][0]["step"] for gid in order[:N]); P(f"  top {N} groups: "+" ".join(f"s{s}:{cnt.get(s,0)}" for s in range(1,11))+f"  (total {N} groups = {16*N} rollouts, {100*16*N/5120:.1f} % of data)")
cnt=collections.Counter(poison_g[gid][0]["step"] for gid in pos); P(f"  all positive-net groups ({len(pos)}): "+" ".join(f"s{s}:{cnt.get(s,0)}" for s in range(1,11)))
P("")
P("Note: whole-group masking also removes those prompts' punished deep-thinkers and all their other rollouts, so the remaining data's suppression comes only from the pure-suppression groups; the filter variants remove deep thinking from both sides and leave nothing to suppress or reinforce (R_deep = 0 by construction).")
# how much of X's extra suppression is 'synthetic': R- inside poison groups vs pure groups
Rm_poison=1e3*sum(p["adv"]*p["live"] for g in poison_g.values() for p in g if p["deep"]>0 and p["adv"]<0)/sum(p["n"] for p in allp)
P(f"Of chain X's effective R_deep (-0.246), {Rm_poison:+.3f} comes from punished deep-thinking siblings inside the poison groups (whose advantages were computed with the masked successes present) and {-0.246-Rm_poison:+.3f} from pure-suppression groups.")
open(f"{D}/groupmask_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

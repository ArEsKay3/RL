import glob,csv,collections,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","bad"),("I","bad"),("P42","bad"),("A","good"),("K","good"),("M","good"),("N","good"),("Q4","good"),("J","good"),("P4","good")]
def f(x,d=0.0):
    try: return float(x)
    except: return d
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
def rank(vals,fmt="{:+.3f}"):
    order=sorted(vals,key=lambda a:-vals[a]); ranks=sorted(order.index(a)+1 for a,o in ARMS if o=="bad"); bad=[vals[a] for a,o in ARMS if o=="bad"]; good=[vals[a] for a,o in ARMS if o=="good"]
    return " ".join(f"{a}:{fmt.format(vals[a])}" for a,o in ARMS)+f" | bad ranks {ranks} {'YES' if min(bad)>max(good) else ('rev' if max(bad)<min(good) else '')}"
G={}; PH={}
for a,o in ARMS:
    per=collections.defaultdict(lambda:dict(adv=0.0,rew=0.0,n=0.0,deep=0,live=0.0,loopdeep=0,turns=0,step=0))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.eff.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.eff.csv")):
        s=int(fn.split("__s")[1][:3])
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                p=per[r["sample_id"]]; p["adv"]=f(r["adv"]); p["rew"]=f(r["reward"]); p["n"]+=f(r["n_tok"]); p["turns"]+=1; p["step"]=s
                d=max(0,int(f(r["think_tok"]))-4096)
                if d>0: p["deep"]+=d; p["live"]+=f(r["s1mp_deep_tr"]); p["loopdeep"]+=d if r["cls"]=="loop" else 0
    for fn in glob.glob(f"{D}/parts/{a}__s0*_c*.prompts.csv"):
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                if int(r["step"])<=10: PH[(a,r["sample_id"])]=(int(r["step"]),r["prompt_hash"])
    groups=collections.defaultdict(list)
    for sid,p in per.items(): p["sid"]=sid; groups[sid.rsplit("_g",1)[0]].append(p)
    G[a]=groups
tot={a:sum(p["n"] for g in G[a].values() for p in g) for a in G}
P("WHAT THE SIBLINGS DO. Groups = 16 rollouts of one prompt. 'Poison group' = a group containing at least one rewarded rollout with a think block > 4096 tokens ('deep'). Advantage is leave-one-out within the group, so a rewarded deep-thinker's weight is large only when its siblings failed; the group's net gradient on deep thinking is R+ (its rewarded deep members) plus R- (its punished deep members). Steps 1-10, x1e-3 per generated token.")
P("")
P("A. Decomposition of R_deep by group type")
rows={}
for a,o in ARMS:
    acc=collections.defaultdict(float); cnt=collections.Counter()
    for gid,g in G[a].items():
        k=sum(1 for p in g if p["rew"]>0); rd=[p for p in g if p["rew"]>0 and p["deep"]>0]; pd=[p for p in g if p["rew"]<=0 and p["deep"]>0]
        if k==0 or k==len(g): typ="zero-variance"
        elif rd: typ="poison (rewarded deep-thinker present)"
        elif pd: typ="pure suppression (punished deep-thinkers only)"
        else: typ="no deep thinking"
        cnt[typ]+=1
        for p in g:
            if p["deep"]>0: acc[(typ,"R+" if p["adv"]>0 else "R-")]+=1e3*p["adv"]*p["live"]/tot[a]
    rows[a]=(acc,cnt)
P(f"{'arm':<5}{'out':<5} {'poison groups':>13} {'R+ in them':>10} {'R- in them':>10} {'net in them':>11} | {'pure-suppression groups':>23} {'R- in them':>10} | {'zero-var groups':>15} {'deep tok there (M)':>18}")
for a,o in ARMS:
    acc,cnt=rows[a]; pg="poison (rewarded deep-thinker present)"; ps="pure suppression (punished deep-thinkers only)"
    zv=sum(p["deep"] for gid,g in G[a].items() for p in g if (sum(1 for q in g if q["rew"]>0) in (0,len(g))))
    P(f"{a:<5}{o:<5} {cnt[pg]:>13} {acc[(pg,'R+')]:>+10.3f} {acc[(pg,'R-')]:>+10.3f} {acc[(pg,'R+')]+acc[(pg,'R-')]:>+11.3f} | {cnt[ps]:>23} {acc[(ps,'R-')]:>+10.3f} | {cnt['zero-variance']:>15} {zv/1e6:>18.2f}")
P("  rank check, net R_deep inside poison groups:      "+rank({a:rows[a][0][("poison (rewarded deep-thinker present)","R+")]+rows[a][0][("poison (rewarded deep-thinker present)","R-")] for a,o in ARMS}))
P("  rank check, R- from pure-suppression groups:      "+rank({a:rows[a][0][("pure suppression (punished deep-thinkers only)","R-")] for a,o in ARMS}))
P("  rank check, number of poison groups:              "+rank({a:float(rows[a][1]["poison (rewarded deep-thinker present)"]) for a,o in ARMS},"{:.0f}"))
P("  rank check, number of pure-suppression groups:    "+rank({a:float(rows[a][1]["pure suppression (punished deep-thinkers only)"]) for a,o in ARMS},"{:.0f}"))
P("")
P("B. Inside the poison groups: successes, advantages, and what the siblings did")
P(f"{'arm':<5}{'out':<5} {'groups':>6} {'mean k':>6} {'k<=3':>5} {'mean adv of poison':>18} | {'siblings: punished':>18} {'of which deep':>13} {'their deep tok (M)':>18} {'live share':>10} {'loop share of deep':>18} {'punished w/o deep: mean len':>27} | {'rewarded siblings w/o deep':>26}")
for a,o in ARMS:
    gs=[g for g in G[a].values() if any(p["rew"]>0 and p["deep"]>0 for p in g) and 0<sum(1 for p in g if p["rew"]>0)<len(g)]
    ks=[sum(1 for p in g if p["rew"]>0) for g in gs]; padv=[p["adv"] for g in gs for p in g if p["rew"]>0 and p["deep"]>0]
    pun=[p for g in gs for p in g if p["rew"]<=0]; pdeep=[p for p in pun if p["deep"]>0]; dt=sum(p["deep"] for p in pdeep); lv=sum(p["live"] for p in pdeep); ld=sum(p["loopdeep"] for p in pdeep)
    pnod=[p for p in pun if p["deep"]==0]; rnod=[p for g in gs for p in g if p["rew"]>0 and p["deep"]==0]
    P(f"{a:<5}{o:<5} {len(gs):>6} {np.mean(ks):>6.2f} {sum(1 for k in ks if k<=3):>5} {np.mean(padv):>18.3f} | {len(pun):>18} {len(pdeep):>13} {dt/1e6:>18.2f} {lv/max(1,dt):>10.4f} {ld/max(1,dt):>18.3f} {np.mean([p['n'] for p in pnod]) if pnod else float('nan'):>27.0f} | {len(rnod):>26}")
P("  live share = sum(1-p_trainer)/deep tokens of the punished deep-thinking siblings (how much counter-gradient they can supply); loop share = fraction of their deep tokens sitting in full-loop turns (inert).")
P("")
P("C. Per-group balance in poison groups: share of poison groups whose net R_deep is positive (the group teaches 'deep thinking wins')")
vals={}
for a,o in ARMS:
    gs=[g for g in G[a].values() if any(p["rew"]>0 and p["deep"]>0 for p in g) and 0<sum(1 for p in g if p["rew"]>0)<len(g)]
    net=[sum(p["adv"]*p["live"] for p in g if p["deep"]>0) for g in gs]; vals[a]=np.mean([v>0 for v in net])
    P(f"  {a:<5}{o:<5} poison groups {len(gs):>3}: net>0 in {100*vals[a]:.0f} %; mean net per group {1e3*np.mean(net)/tot[a]*len(gs):+.3f} (sum over groups, per 1k tok)")
P("  rank check, share of poison groups with net>0:   "+rank(vals,"{:.2f}"))
P("")
P("D. Same prompts, other arms: for every (step, prompt) where a BAD arm had a rewarded deep-thinker, what happened in the healthy arms on that prompt")
key=collections.defaultdict(dict)
for a,o in ARMS:
    for gid,g in G[a].items():
        kk=PH.get((a,g[0]["sid"]))
        if not kk: continue
        k=sum(1 for p in g if p["rew"]>0); rd=[p for p in g if p["rew"]>0 and p["deep"]>0]; pd=[p for p in g if p["rew"]<=0 and p["deep"]>0]
        key[kk][a]=dict(k=k,rd=len(rd),pd=len(pd),net=sum(p["adv"]*p["live"] for p in g if p["deep"]>0),live_rd=sum(p["live"] for p in rd),len_rd=np.mean([p["n"] for p in rd]) if rd else float("nan"))
for a0,o0 in ARMS:
    if o0!="bad": continue
    ps=[kk for kk,d in key.items() if a0 in d and d[a0]["rd"]>0 and len(d)==10]
    hk=[np.mean([d[a]["k"] for a,o in ARMS if o=="good"]) for kk in ps for d in [key[kk]]]
    hrd=[np.mean([d[a]["rd"]>0 for a,o in ARMS if o=="good"]) for kk in ps for d in [key[kk]]]
    bk=[key[kk][a0]["k"] for kk in ps]
    P(f"  {a0}: {len(ps)} such prompts; successes k: this arm mean {np.mean(bk):.2f}, healthy arms mean {np.mean(hk):.2f}; healthy arms also had a rewarded deep-thinker on the same prompt in {100*np.mean(hrd):.0f} % of (prompt, arm) cases; this arm's net R_deep on these prompts {1e3*sum(key[kk][a0]['net'] for kk in ps)/tot[a0]:+.3f} vs healthy-arm mean on the same prompts {1e3*np.mean([sum(key[kk][a]['net'] for kk in ps)/tot[a] for a,o in ARMS if o=='good']):+.3f}")
open(f"{D}/groups_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

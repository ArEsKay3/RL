import glob,csv,collections,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
def f(x,d=np.nan):
    try: return float(x)
    except: return d
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
S={};T={};PH={}
for a,o in ARMS:
    S[a]=[];T[a]=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.samples.csv")):
        with open(fn) as fh: S[a]+=list(csv.DictReader(fh))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.turns.csv")):
        with open(fn) as fh: T[a]+=list(csv.DictReader(fh))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.prompts.csv")):
        with open(fn) as fh:
            for r in csv.DictReader(fh): PH[(a,r["sample_id"])]=(int(f(r["step"])),r["prompt_hash"])
# B. hard-prompt successes paired by prompt
P("B. PAIRED BY PROMPT: group success count k per arm for every (step, prompt). How often do the bad arms have 1-4 successes where good arms have 0, and vice versa; and does long thinking accompany those rare successes?")
K={}  # (step,hash) -> arm -> (k, n_rew_with_long, n_rew)
for a,o in ARMS:
    lt=collections.Counter(r["sample_id"] for r in T[a] if 4000<=f(r["n_tok"])<16000)
    groups=collections.defaultdict(list)
    for r in S[a]: groups[r["group_id"]].append(r)
    for gid,rs in groups.items():
        key=PH.get((a,rs[0]["sample_id"]))
        if not key: continue
        k=sum(1 for r in rs if f(r["reward"])>0); rl=sum(1 for r in rs if f(r["reward"])>0 and lt[r["sample_id"]]>0)
        K.setdefault(key,{})[a]=(k,rl,len(rs))
full=[key for key,d in K.items() if len(d)==7]
P(f"  prompts with all 7 arms: {len(full)}")
P(f"  {'arm':<4}{'out':<5} {'k=0 prompts':>11} {'k 1-4':>6} {'k 5-11':>7} {'k>=12':>6} | {'mean k':>6} | {'rare successes (k<=4) w/ long think':>36} | {'prompts where this arm k in 1-4 and median of others = 0':>52}")
for a,o in ARMS:
    ks=[K[key][a][0] for key in full]; rare_long=sum(K[key][a][1] for key in full if 1<=K[key][a][0]<=4); rare=sum(K[key][a][0] for key in full if 1<=K[key][a][0]<=4)
    solo=sum(1 for key in full if 1<=K[key][a][0]<=4 and np.median([K[key][b][0] for b,_ in ARMS if b!=a])==0)
    P(f"  {a:<4}{o:<5} {sum(1 for k in ks if k==0):>11} {sum(1 for k in ks if 1<=k<=4):>6} {sum(1 for k in ks if 5<=k<=11):>7} {sum(1 for k in ks if k>=12):>6} | {np.mean(ks):>6.2f} | {rare_long:>14} of {rare:<5} = {100*rare_long/max(1,rare):>4.1f}%          | {solo:>52}")
P("")
# H. dramatic within-rollout changes
P("H. WITHIN-ROLLOUT REGIME CHANGES: rollouts containing a turn at least 8x longer than the mean of its previous turns (and >= 2k tokens); share of rollouts by advantage sign; and the exposure E of the tokens in those 'jump' turns")
P(f"{'arm':<4}{'out':<5} {'rollouts w/ jump':>16} {'rewarded':>8} {'punished':>8} {'zero':>5} | {'jump turns':>10} {'E(jump turns)':>13} {'mean jump ratio':>15} | {'first jump at rel pos':>21}")
for a,o in ARMS:
    by=collections.defaultdict(list)
    for r in T[a]: by[r["sample_id"]].append(r)
    tot=sum(f(r["n_gen"]) for r in S[a]); adv={r["sample_id"]:f(r["adv"]) for r in S[a]}
    nj=collections.Counter(); jt=0; Ej=0.0; ratios=[]; pos=[]
    for sid,rs in by.items():
        rs.sort(key=lambda r:f(r["turn"])); prev=[]; hit=False
        for r in rs:
            n=f(r["n_tok"])
            if prev and n>=2000 and n>=8*np.mean(prev):
                jt+=1; Ej+=adv[sid]*n; ratios.append(n/np.mean(prev))
                if not hit: pos.append(f(r["turn"])/max(1,f(r["n_turns"]))); hit=True
            prev.append(n)
        if hit: nj["rew" if adv[sid]>1e-9 else ("pun" if adv[sid]<-1e-9 else "zero")]+=1
    n_all=sum(nj.values())
    P(f"{a:<4}{o:<5} {n_all:>16} {nj['rew']:>8} {nj['pun']:>8} {nj['zero']:>5} | {jt:>10} {1e3*Ej/tot:>+13.2f} {np.mean(ratios):>15.1f} | {np.mean(pos):>21.2f}")
P("")
# I. tool-output size before long turns
P("I. CONTEXT BEFORE LONG THINKS: size of the tool output (non-generated gap) immediately preceding a turn, for long (4k-16k) vs other turns, by arm and advantage sign; and share of long turns preceded by a >2k-token tool output")
P(f"{'arm':<4}{'out':<5} {'gap before long, rew':>20} {'pun':>7} | {'gap before other turns':>22} | {'long turns after >2k output: rew%':>33} {'pun%':>5} | {'E(turns after >2k output)':>25}")
for a,o in ARMS:
    by=collections.defaultdict(list)
    for r in T[a]: by[r["sample_id"]].append(r)
    tot=sum(f(r["n_gen"]) for r in S[a])
    gl={"rew":[],"pun":[]}; go=[]; big={"rew":[0,0],"pun":[0,0]}; Eb=0.0
    for sid,rs in by.items():
        rs.sort(key=lambda r:f(r["turn"]))
        for i in range(1,len(rs)):
            gap=f(rs[i]["start"])-(f(rs[i-1]["start"])+f(rs[i-1]["n_tok"])); n=f(rs[i]["n_tok"]); adv=f(rs[i]["adv"])
            if gap>2000: Eb+=adv*n
            sgn="rew" if adv>1e-9 else ("pun" if adv<-1e-9 else None)
            if 4000<=n<16000:
                if sgn: gl[sgn].append(gap); big[sgn][0]+=int(gap>2000); big[sgn][1]+=1
            else: go.append(gap)
    P(f"{a:<4}{o:<5} {np.mean(gl['rew']):>20.0f} {np.mean(gl['pun']):>7.0f} | {np.mean(go):>22.0f} | {100*big['rew'][0]/max(1,big['rew'][1]):>33.1f} {100*big['pun'][0]/max(1,big['pun'][1]):>5.1f} | {1e3*Eb/tot:>+25.2f}")
open(f"{D}/dig2_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

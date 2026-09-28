import glob,csv,collections,math,os
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
def f(x,d=np.nan):
    try: return float(x)
    except: return d
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
G={}  # arm -> group_id -> dict(step, Eg_long, Eg_near, k_success, long_tokens, n_long_rollouts, prompt_hash)
PH={}
for a,o in ARMS:
    S=[]; T=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.samples.csv")):
        with open(fn) as fh: S+=list(csv.DictReader(fh))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.turns.csv")):
        with open(fn) as fh: T+=list(csv.DictReader(fh))
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s*_c*.prompts.csv")):
        with open(fn) as fh:
            for r in csv.DictReader(fh): PH[(a,r["sample_id"])]=r["prompt_hash"]
    tot=sum(f(r["n_gen"]) for r in S)
    groups=collections.defaultdict(lambda:dict(step=0,E_long=0.0,E_near=0.0,E_all=0.0,k=0,n=0,long_tok=0.0,near_tok=0.0,hash=None,rew_long_tok=0.0,pun_long_tok=0.0))
    for r in S:
        g=groups[r["group_id"]]; g["step"]=int(f(r["step"])); g["n"]+=1; g["k"]+=int(f(r["reward"])>0); g["E_all"]+=f(r["adv"])*f(r["n_gen"])
        if (a,r["sample_id"]) in PH: g["hash"]=PH[(a,r["sample_id"])]
    for r in T:
        gid=r["sample_id"].rsplit("_g",1)[0]; g=groups[gid]; nt=f(r["n_tok"]); adv=f(r["adv"])
        if 4000<=nt<16000:
            g["E_long"]+=adv*nt; g["long_tok"]+=nt
            if adv>1e-9: g["rew_long_tok"]+=nt
            elif adv<-1e-9: g["pun_long_tok"]+=nt
        if f(r["think_tok"])>=1000 and 0.10<=f(r["zlib_reason"],1)<0.25: g["E_near"]+=adv*nt; g["near_tok"]+=nt
    for g in groups.values():
        for k in ("E_long","E_near","E_all"): g[k]=1e3*g[k]/tot
    G[a]=groups
P("PER-GROUP VIEW. E_g(long) = 1000 x sum over the group's 4k-16k turns of adv x tokens / arm total tokens = 16 x within-group covariance of advantage with long-turn tokens (advantages sum to 0 within a group).")
P(f"{'arm':<4}{'out':<5} {'groups':>6} {'with long':>9} {'sum E':>7} {'median E_g':>10} {'mean E_g':>8} {'E_g<0':>6} {'E_g>0':>6} {'E_g=0':>6} {'top5 neg sum':>12} {'top5 pos sum':>12} {'sum excl top10 |E_g|':>20}")
for a,o in ARMS:
    E=np.array([g["E_long"] for g in G[a].values()]); wl=np.array([g["long_tok"]>0 for g in G[a].values()])
    srt=np.sort(E); top=np.argsort(-np.abs(E))[:10]; mask=np.ones(len(E),bool); mask[top]=False
    P(f"{a:<4}{o:<5} {len(E):>6} {int(wl.sum()):>9} {E.sum():>+7.2f} {np.median(E[wl]):>+10.3f} {E[wl].mean():>+8.3f} {int((E<-1e-12).sum()):>6} {int((E>1e-12).sum()):>6} {int((np.abs(E)<=1e-12).sum()):>6} {srt[:5].sum():>+12.2f} {srt[-5:].sum():>+12.2f} {E[mask].sum():>+20.2f}")
P("")
P("Rank test: pooled bad vs good on per-group E_g(long), groups with any long turn (Mann-Whitney U approx z)")
bad=np.array([g["E_long"] for a,o in ARMS if o=="BAD" for g in G[a].values() if g["long_tok"]>0]); good=np.array([g["E_long"] for a,o in ARMS if o!="BAD" for g in G[a].values() if g["long_tok"]>0])
allv=np.concatenate([bad,good]); ranks=np.argsort(np.argsort(allv))+1; R1=ranks[:len(bad)].sum(); n1,n2=len(bad),len(good); U=R1-n1*(n1+1)/2; mu=n1*n2/2; sd=math.sqrt(n1*n2*(n1+n2+1)/12); z=(U-mu)/sd
P(f"  bad groups {n1} (mean E_g {bad.mean():+.4f}, share negative {np.mean(bad<-1e-12):.2f}), good groups {n2} (mean {good.mean():+.4f}, share negative {np.mean(good<-1e-12):.2f}); z = {z:+.2f}, p = {math.erfc(abs(z)/math.sqrt(2)):.4f}")
P("")
if PH:
    P("PAIRED BY PROMPT (same prompt tokens across arms): per (step, prompt) E_g(long) per arm; bad-arm mean minus good-arm mean; sorted by |difference|")
    key=collections.defaultdict(dict)
    for a,o in ARMS:
        for g in G[a].values():
            if g["hash"]: key[(g["step"],g["hash"])][a]=g
    rows=[]
    for (s,h),d in key.items():
        if len(d)<7: continue
        b=np.mean([d[a]["E_long"] for a,o in ARMS if o=="BAD"]); gd=np.mean([d[a]["E_long"] for a,o in ARMS if o!="BAD"])
        rows.append((b-gd,s,h,d))
    rows.sort(key=lambda x:-abs(x[0]))
    P(f"  prompts present in all 7 arms: {len(rows)}; sum of (bad-good) over prompts = {sum(r[0] for r in rows):+.2f}; prompts with bad>good: {sum(1 for r in rows if r[0]>0)}, bad<good: {sum(1 for r in rows if r[0]<0)}")
    P(f"  {'bad-good':>9} {'step':>4} {'prompt':>8} | "+" ".join(f"{a+' E_g':>8} {'k':>2}" for a,o in ARMS))
    for dlt,s,h,d in rows[:20]:
        P(f"  {dlt:>+9.3f} {s:>4} {h[:8]:>8} | "+" ".join(f"{d[a]['E_long']:>+8.3f} {d[a]['k']:>2}" for a,o in ARMS))
    # how concentrated: cumulative share of the total bad-good gap explained by top prompts
    tot=sum(r[0] for r in rows); cum=0
    for i,r in enumerate(rows[:40]):
        cum+=r[0]
        if i+1 in (5,10,20,40): P(f"  top {i+1} prompts by |diff| explain {cum:+.2f} of {tot:+.2f}")
    # sign test per prompt
    P(f"  sign test over prompts: bad-good > 0 in {sum(1 for r in rows if r[0]>0)} of {len(rows)} (binomial p two-sided ~ {2*min(sum(1 for r in rows if r[0]>0),sum(1 for r in rows if r[0]<0))/len(rows):.2f} crude)")
    # per-prompt reward count comparison for long-turn rollouts: is the bad arms' advantage about k (group success count)?
    P("")
    P("  Where do long turns sit relative to group success count k? mean k of groups weighted by long-turn tokens, rewarded vs punished; adv magnitude = f(k)")
    for a,o in ARMS:
        gs=[g for g in G[a].values() if g["long_tok"]>0]
        wk_pun=sum(g["k"]*g["pun_long_tok"] for g in gs)/max(1,sum(g["pun_long_tok"] for g in gs)); wk_rew=sum(g["k"]*g["rew_long_tok"] for g in gs)/max(1,sum(g["rew_long_tok"] for g in gs))
        P(f"   {a} {o:<5} punished long tokens sit in groups with mean k = {wk_pun:.2f}; rewarded long tokens in groups with mean k = {wk_rew:.2f}; long tokens in zero-var groups: k=0 {sum(g['long_tok'] for g in gs if g['k']==0)/1e6:.2f}M, k=16 {sum(g['long_tok'] for g in gs if g['k']==16)/1e6:.2f}M")
open(f"{D}/groupcov_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

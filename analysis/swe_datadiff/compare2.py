import glob,csv,json,collections,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
C="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/dumpbrowse_cache"
EXP={"G":"nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920","M":"nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923"}
LBL={"G":"chain G (MINF)","M":"chain M (vLLM)"}
def loadcsv(pattern):
    rows=[]
    for fn in sorted(glob.glob(pattern)):
        with open(fn) as fh: rows+=list(csv.DictReader(fh))
    return rows
def col(rows,k,dtype=float):
    out=np.empty(len(rows),dtype=dtype)
    for i,r in enumerate(rows):
        v=r[k]
        try: out[i]=float(v) if v!="" else np.nan
        except: out[i]=np.nan
    return out
A={}
for a in EXP:
    rows=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv")
    gids=[r["sample_id"].rsplit("_g",1)[0] for r in rows]; gmap={g:i for i,g in enumerate(sorted(set(gids)))}
    A[a]=dict(rows=rows,adv=col(rows,"adv"),n=col(rows,"n_tok"),think=col(rows,"think_tok"),z=col(rows,"zlib_reason"),start=col(rows,"start"),step=col(rows,"step"),turn=col(rows,"turn"),nturns=col(rows,"n_turns"),g=np.array([gmap[x] for x in gids]),ng=len(gmap),gnames=sorted(set(gids)),sid=[r["sample_id"] for r in rows])
S={a:loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv") for a in EXP}
J={}
for a,e in EXP.items():
    for fn in glob.glob(f"{C}/{e}__target_step_*.idx.json"):
        try: d=json.load(open(fn))
        except Exception: continue
        if d.get("v")!=2: continue
        for s in d["summaries"]:
            if s: J[(a,s["sample_id"])]=s
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
rng=np.random.default_rng(1); NB=2000
W={a:rng.multinomial(A[a]["ng"],np.ones(A[a]["ng"])/A[a]["ng"],size=NB).astype(float) for a in EXP}
def expo(a,sel):
    d=A[a]; x=d["adv"]*d["n"]*sel; Ag=np.bincount(d["g"],weights=x,minlength=d["ng"]); Ng=np.bincount(d["g"],weights=d["n"],minlength=d["ng"])
    obs=1e3*Ag.sum()/Ng.sum(); bs=1e3*(W[a]@Ag)/(W[a]@Ng); return obs,bs
def line(label,selG,selM,extra=""):
    oG,bG=expo("G",selG); oM,bM=expo("M",selM); dd=bG-bM; se=dd.std(); d=oG-oM; p=math.erfc(abs(d/se)/math.sqrt(2)) if se>0 else 1
    P(f"{label:<18} {oG:>+8.3f} [{np.percentile(bG,2.5):>+7.3f},{np.percentile(bG,97.5):>+7.3f}] {oM:>+8.3f} [{np.percentile(bM,2.5):>+7.3f},{np.percentile(bM,97.5):>+7.3f}] {d:>+8.3f} {se:>6.3f} {p:>6.3f}{extra}")
P("TOKEN-WEIGHTED ADVANTAGE EXPOSURE per 1,000 generated tokens: E(class) = 1000 x sum over turns in the class of (advantage x turn tokens) / all generated tokens of the arm.")
P("Positive = the step gradient on net reinforces tokens of that class; negative = suppresses. 95% CI and the p for G-M from a group-level bootstrap (320 groups per arm, 2000 resamples).")
hdr=f"{'class':<18} {'G':>8} {'G 95% CI':>18} {'M':>8} {'M 95% CI':>18} {'G-M':>8} {'se':>6} {'p':>6}"
P(""); P("1. by turn length (tokens)"); P(hdr)
for lo,hi in [(0,500),(500,1000),(1000,2000),(2000,4000),(4000,8000),(8000,16000),(16000,10**9),(2000,10**9),(4000,10**9),(4000,16000)]:
    line(f"{lo}-{hi if hi<10**9 else 'inf'}",(A['G']['n']>=lo)&(A['G']['n']<hi),(A['M']['n']>=lo)&(A['M']['n']<hi))
P(""); P("2. by zlib ratio of the think part, turns with >=1,000 think tokens (last two columns: share of all generated tokens in the class, G / M)"); P(hdr+f" {'G tok%':>7} {'M tok%':>7}")
for lo,hi in [(0,0.10),(0.10,0.20),(0.20,0.25),(0.25,0.30),(0.30,0.35),(0.35,0.40),(0.40,9),(0,0.25),(0.10,0.25),(0.25,9)]:
    sels={a:(A[a]["think"]>=1000)&(A[a]["z"]>=lo)&(A[a]["z"]<hi) for a in EXP}
    line(f"{lo:.2f}-{hi:.2f}",sels["G"],sels["M"],f" {100*A['G']['n'][sels['G']].sum()/A['G']['n'].sum():>7.2f} {100*A['M']['n'][sels['M']].sum()/A['M']['n'].sum():>7.2f}")
P(""); P("3. by context position of the turn start"); P(hdr)
pb=[(0,16000),(16000,32000),(32000,64000),(64000,96000),(96000,128000),(128000,10**9)]
for lo,hi in pb: line(f"{lo//1000}k-{str(hi//1000)+'k' if hi<10**9 else 'inf'}",(A['G']['start']>=lo)&(A['G']['start']<hi),(A['M']['start']>=lo)&(A['M']['start']<hi))
P(""); P("4. long turns (>=4,000 tokens) by context position"); P(hdr)
for lo,hi in pb: line(f"{lo//1000}k-{str(hi//1000)+'k' if hi<10**9 else 'inf'}",(A['G']['start']>=lo)&(A['G']['start']<hi)&(A['G']['n']>=4000),(A['M']['start']>=lo)&(A['M']['start']<hi)&(A['M']['n']>=4000))
P(""); P("5. by relative position of the turn within its rollout (turn index / number of turns)"); P(hdr)
for lo,hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]:
    rel={a:A[a]["turn"]/np.maximum(1,A[a]["nturns"]) for a in EXP}
    line(f"{lo:.1f}-{hi:.1f}",(rel['G']>=lo)&(rel['G']<hi),(rel['M']>=lo)&(rel['M']<hi))
P(""); P("6. per step: exposure of long turns (>=4k tokens) and of near-repetitive think turns (zlib<0.25, >=1k think); tok% = share of the step's generated tokens in long turns; rewarded long turns = count with adv>0 of all long turns")
P(f"{'step':>4} | {'G long':>8} {'M long':>8} | {'G near':>8} {'M near':>8} | {'G tok%':>7} {'M tok%':>7} | {'G rewarded long':>17} {'M rewarded long':>17}")
for s in range(1,11):
    row=[]
    for a in EXP:
        d=A[a]; ms=d["step"]==s; tot=d["n"][ms].sum(); lg=ms&(d["n"]>=4000); nr=ms&(d["think"]>=1000)&(d["z"]<0.25)
        row.append((1e3*(d["adv"][lg]*d["n"][lg]).sum()/tot,1e3*(d["adv"][nr]*d["n"][nr]).sum()/tot,100*d["n"][lg].sum()/tot,int((lg&(d["adv"]>1e-9)).sum()),int(lg.sum())))
    P(f"{s:>4} | {row[0][0]:>+8.3f} {row[1][0]:>+8.3f} | {row[0][1]:>+8.3f} {row[1][1]:>+8.3f} | {row[0][2]:>7.2f} {row[1][2]:>7.2f} | {row[0][3]:>6} of {row[0][4]:<7} {row[1][3]:>6} of {row[1][4]:<7}")
P(""); P("7. which groups carry the long-turn (>=4k) exposure; rew = rewarded members of 16; long turns in rewarded / punished members")
for a in EXP:
    d=A[a]; lg=d["n"]>=4000; x=np.bincount(d["g"],weights=d["adv"]*d["n"]*lg,minlength=d["ng"]); order=np.argsort(-np.abs(x))
    P(f"{LBL[a]}: total sum(adv x long-turn tokens) = {x.sum():+.0f}; positive part {x[x>0].sum():+.0f}; negative part {x[x<0].sum():+.0f}; groups with any long turn {int((np.bincount(d['g'],weights=lg,minlength=d['ng'])>0).sum())}")
    for gi in order[:10]:
        m=d["g"]==gi; sids={d["sid"][i] for i in np.where(m)[0]}; rew=len({d["sid"][i] for i in np.where(m&(d["adv"]>1e-9))[0]}); npos=int((m&lg&(d["adv"]>1e-9)).sum()); nneg=int((m&lg&(d["adv"]<-1e-9)).sum()); step=int(d["step"][m][0])
        inst=J.get((a,next(iter(sids))),{}).get("instance_id","")
        P(f"   {x[gi]:>+9.0f}  step {step:>2}  group ..{d['gnames'][gi][-6:]}  rewarded {rew:>2}/16  long turns rewarded {npos:>2} punished {nneg:>2}  {inst}")
P(""); P("8. within-rollout profile: mean turn tokens by decile of turn index, rewarded vs punished rollouts")
P(f"{'decile':>6} | {'G rew':>7} {'G pun':>7} | {'M rew':>7} {'M pun':>7}")
for dec in range(10):
    vals=[]
    for a in EXP:
        d=A[a]; dd=np.minimum(9,(10*d["turn"]/np.maximum(1,d["nturns"])).astype(int))
        for sign in (1,-1):
            m=(dd==dec)&((d["adv"]>1e-9) if sign==1 else (d["adv"]<-1e-9)); vals.append(d["n"][m].mean() if m.any() else np.nan)
    P(f"{dec:>6} | {vals[0]:>7.0f} {vals[1]:>7.0f} | {vals[2]:>7.0f} {vals[3]:>7.0f}")
if J:
    P(""); P("9. jsonl flags (samples with summaries): agent_timed_out / eval_timed_out / oom_killed / eval_oom_killed and their rewards")
    for a in EXP:
        rs=[J[(a,r["sample_id"])] for r in S[a] if (a,r["sample_id"]) in J]
        P(f"{LBL[a]}: n {len(rs)}; "+"; ".join(f"{k} {sum(1 for s in rs if s.get(k))} (reward mean {np.mean([s['reward'] for s in rs if s.get(k)]) if any(s.get(k) for s in rs) else float('nan'):.2f})" for k in ("agent_timed_out","eval_timed_out","oom_killed","eval_oom_killed")))
    P(""); P("10. reward rate by rollout length class (generated tokens)")
    for lo,hi in [(0,10000),(10000,20000),(20000,40000),(40000,80000),(80000,10**9)]:
        ln=f"{lo:>6}-{(hi if hi<10**9 else 'inf'):<7}"
        for a in EXP:
            rs=[r for r in S[a] if lo<=float(r["n_gen"])<hi]; ln+=f" | {LBL[a]}: n {len(rs):>4} reward {np.mean([float(r['reward']) for r in rs]) if rs else float('nan'):.3f}"
        P(ln)
open(f"{D}/compare2_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

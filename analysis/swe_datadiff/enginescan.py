import glob,csv,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
ARMS=[("G","MINF"),("I","MINF"),("J","MINF"),("A","vLLM"),("K","vLLM"),("M","vLLM"),("N","vLLM")]
def loadcsv(pattern):
    rows=[]
    for fn in sorted(glob.glob(pattern)):
        with open(fn) as fh: rows+=list(csv.DictReader(fh))
    return rows
def col(rows,k):
    out=np.empty(len(rows))
    for i,r in enumerate(rows):
        try: out[i]=float(r[k]) if r[k]!="" else np.nan
        except: out[i]=np.nan
    return out
T={};S={}
for a,e in ARMS:
    T[a]=loadcsv(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv")+loadcsv(f"{D}/parts/{a}__s010_c*.turns.csv")
    S[a]=loadcsv(f"{D}/parts/{a}__s00[1-9]_c*.samples.csv")+loadcsv(f"{D}/parts/{a}__s010_c*.samples.csv")
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("ENGINE SCAN, steps 1-10: per-arm value of many generation-only features (no reward involved). Arm-level test: MINF arms (G, I, J) vs vLLM arms (A, K, M, N), Welch t on the 7 arm values; also whether all 3 MINF arms fall on one side of all 4 vLLM arms (perfect split, chance 2/35).")
feats=[]
def turnfeat(name,fn): feats.append((name,"turn",fn))
def sampfeat(name,fn): feats.append((name,"sample",fn))
turnfeat("mean turn tokens",lambda d:np.mean(d["n_tok"]))
turnfeat("p90 turn tokens",lambda d:np.percentile(d["n_tok"],90))
turnfeat("p99 turn tokens",lambda d:np.percentile(d["n_tok"],99))
turnfeat("share turns >=1k",lambda d:np.mean(d["n_tok"]>=1000))
turnfeat("share turns >=4k",lambda d:np.mean(d["n_tok"]>=4000))
turnfeat("share tokens in 4k-16k turns",lambda d:d["n_tok"][(d["n_tok"]>=4000)&(d["n_tok"]<16000)].sum()/d["n_tok"].sum())
turnfeat("think share of turn tokens",lambda d:d["think_tok"].sum()/d["n_tok"].sum())
turnfeat("mean answer part tokens (after </think>)",lambda d:np.mean(d["n_tok"]-d["think_tok"]))
turnfeat("share turns without </think>",lambda d:np.mean(d["has_close"]==0))
turnfeat("share turns with 2+ </think>",lambda d:np.mean(d["n_close"]>=2))
turnfeat("newline token share",lambda d:np.nanmean(d["nl_frac"]))
turnfeat("mean engine logprob (turn mean)",lambda d:np.nanmean(d["mean_gen_lp"]))
turnfeat("mean min engine logprob per turn",lambda d:np.nanmean(d["min_gen_lp"]))
turnfeat("engine logprob of </think>",lambda d:np.nanmean(d["g_close"]))
turnfeat("share </think> with logprob < -1",lambda d:np.nanmean(d["g_close"]<-1))
turnfeat("engine logprob of first token",lambda d:np.nanmean(d["g_first"]))
turnfeat("|d| trainer-engine (turn mean)",lambda d:np.nanmean(d["mean_abs_d"]))
turnfeat("d at </think>",lambda d:np.nanmean(d["d_close"]))
turnfeat("zlib of think (turns >=1k think)",lambda d:np.nanmean(d["zlib_reason"][d["think_tok"]>=1000]))
turnfeat("share zlib<0.25 among >=1k think",lambda d:np.nanmean(d["zlib_reason"][d["think_tok"]>=1000]<0.25))
turnfeat("share zlib<0.10 among >=1k think",lambda d:np.nanmean(d["zlib_reason"][d["think_tok"]>=1000]<0.10))
turnfeat("mean context position of turn start",lambda d:np.mean(d["start"]))
sampfeat("mean generated tokens",lambda d:np.mean(d["n_gen"]))
sampfeat("p90 generated tokens",lambda d:np.percentile(d["n_gen"],90))
sampfeat("mean assistant turns",lambda d:np.mean(d["n_turns"]))
sampfeat("share rollouts with 200 turns",lambda d:np.mean(d["n_turns"]>=200))
sampfeat("share truncated",lambda d:np.mean(d["truncated"]>0))
sampfeat("share ending in unclosed think",lambda d:np.mean(d["ends_seq_noclose"]>0))
sampfeat("mean longest turn",lambda d:np.mean(d["max_turn"]))
sampfeat("mean last turn",lambda d:np.mean(d["last_turn"]))
sampfeat("share with a loop turn",lambda d:np.mean(d["loop_turns"]>0))
sampfeat("share with a near-loop turn",lambda d:np.mean(d["near_loop_turns"]>0))
sampfeat("mean prompt length",lambda d:np.mean(d["prompt_len"]))
sampfeat("reward rate",lambda d:np.mean(d["reward"]))
sampfeat("openhands run time (s)",lambda d:np.nanmean(d["openhands_run_time"]))
sampfeat("model call time (s)",lambda d:np.nanmean(d["model_call_time"]))
sampfeat("seq_mult_prob_error median",lambda d:np.nanmedian(d["seq_mult_prob_error"]))
def arr(rows,keys): return {k:col(rows,k) for k in keys}
TK=["n_tok","think_tok","has_close","n_close","nl_frac","mean_gen_lp","min_gen_lp","g_close","g_first","mean_abs_d","d_close","zlib_reason","start"]
SK=["n_gen","n_turns","truncated","ends_seq_noclose","max_turn","last_turn","loop_turns","near_loop_turns","prompt_len","reward","openhands_run_time","model_call_time","seq_mult_prob_error"]
TA={a:arr(T[a],TK) for a,_ in ARMS}; SA={a:arr(S[a],SK) for a,_ in ARMS}
P(f"{'feature':<42} "+" ".join(f"{a:>9}" for a,_ in ARMS)+f" | {'MINF mean':>10} {'vLLM mean':>10} {'t':>6} {'p':>6} {'split':>5}")
for name,kind,fn in feats:
    vals={a:fn(TA[a] if kind=="turn" else SA[a]) for a,_ in ARMS}
    m=np.array([vals[a] for a,e in ARMS if e=="MINF"]); v=np.array([vals[a] for a,e in ARMS if e=="vLLM"])
    se=math.sqrt(m.var(ddof=1)/len(m)+v.var(ddof=1)/len(v)); t=(m.mean()-v.mean())/se if se>0 else 0
    df=(m.var(ddof=1)/3+v.var(ddof=1)/4)**2/((m.var(ddof=1)/3)**2/2+(v.var(ddof=1)/4)**2/3) if se>0 else 1
    from math import erfc,sqrt
    p=erfc(abs(t)/sqrt(2))  # normal approx
    split="YES" if (m.min()>v.max() or m.max()<v.min()) else ""
    fmt=(lambda x:f"{x:>9.4f}") if abs(vals["G"])<10 else (lambda x:f"{x:>9.1f}")
    P(f"{name:<42} "+" ".join(fmt(vals[a]) for a,_ in ARMS)+f" | {m.mean():>10.4f} {v.mean():>10.4f} {t:>+6.2f} {p:>6.3f} {split:>5}")
open(f"{D}/enginescan_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

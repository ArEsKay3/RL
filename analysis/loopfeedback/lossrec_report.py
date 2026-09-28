import glob,csv,collections
import numpy as np
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
NAMES={"G":"chain G MINF from0 no-pfx","I":"chain I MINF from0 pfx-on","J":"chain J MINF from0 pfx-on r2","A":"run A vLLM from0","K":"chain K vLLM from0 s1234","F":"chain F MINF fr vLLM10","L":"chain L MINF fr K10 pfx","B":"run B MINF fr MINF10","D":"chain D vLLM fr MINF10"}
GOOD=["A","K","J","F","L"]; BAD=["G","I","B","D"]
R=collections.defaultdict(list)
for fn in glob.glob(f"{L}/lossrec/*.lossrec.csv"):
    for r in csv.DictReader(open(fn)):
        R[(r["arm"],int(r["step"]))].append(r)
def agg(rows):
    sm=np.array([float(r["sample_mask"]) for r in rows]); adv=np.array([float(r["adv"]) for r in rows]); n=np.array([int(r["n_tok"]) for r in rows]); nl=np.array([int(r["n_loop_tok"]) for r in rows])
    wsn=np.array([float(r["w_sum_normal"]) for r in rows]); wsl=np.array([float(r["w_sum_loop"]) for r in rows])
    gn=np.array([float(r["w1mp_sum_normal"]) for r in rows]); gl=np.array([float(r["w1mp_sum_loop"]) for r in rows])
    on=np.array([float(r["onemp_sum_normal"]) for r in rows]); ol=np.array([float(r["onemp_sum_loop"]) for r in rows])
    pl=np.array([float(r["p_sum_loop"]) for r in rows]); pn=np.array([float(r["p_sum_normal"]) for r in rows]); oob=np.array([int(r["n_tis_oob"]) for r in rows])
    G=float((sm*n).sum()); active=sm*(np.abs(adv)>1e-9)
    out=dict(G=G,rollouts=len(rows),loop_tok_share=(sm*nl).sum()/G,zero_tok_share=(sm*(np.abs(adv)<=1e-9)*n).sum()/G,
             loss=(sm*(-adv)*(wsn+wsl)).sum()/G,loss_loop=(sm*(-adv)*wsl).sum()/G,
             absloss_loop_share=(sm*np.abs(adv)*wsl).sum()/max(1e-12,(sm*np.abs(adv)*(wsn+wsl)).sum()),
             grad=(sm*np.abs(adv)*(gn+gl)).sum()/G, grad_loop_share=(sm*np.abs(adv)*gl).sum()/max(1e-12,(sm*np.abs(adv)*(gn+gl)).sum()),
             grad_pos_share=(sm*np.clip(adv,0,None)*(gn+gl)).sum()/max(1e-12,(sm*np.abs(adv)*(gn+gl)).sum()),
             loop_push=(sm*adv*gl).sum()/G, loop_1mp=ol.sum()/max(1,nl.sum()), normal_1mp=on.sum()/max(1,(n-nl).sum()),
             loop_p=pl.sum()/max(1,nl.sum()), tis_oob_per_M=1e6*oob.sum()/max(1,n.sum()), w_mean_loop=wsl.sum()/max(1,nl.sum()), w_mean_normal=wsn.sum()/max(1,(n-nl).sum()))
    per=sm*np.abs(adv)*(gn+gl); per=np.sort(per)[::-1]; k=max(1,int(round(0.01*len(per)))); out["top1pct_grad_share"]=per[:k].sum()/max(1e-12,per.sum())
    out["mixed_tok_share"]=(active*n).sum()/G
    return out
A={k:agg(v) for k,v in R.items()}
steps=sorted({s for (_,s) in A})
o=[]; P=lambda *x:o.append(" ".join(str(v) for v in x))
P("LOSS RECONSTRUCTION FROM THE TOKEN DUMPS.  Loss as trained: force_on_policy_ratio (ratio = 1), token-level, TIS weights w = clamp(exp(lp_trainer - lp_engine), 0.2, 5), no KL, no dual clip;")
P("  per-token loss value = -adv_i * w_t ; reduction = sum over valid generated tokens / G, G = number of valid generated tokens in the 512-rollout batch (sample_mask applied).")
P("  gradient weight of a token (magnitude on its own logit) = |adv_i| * w_t * (1 - p_t), p_t = trainer probability of the sampled token.  Loop = repetitive reasoning block (>= 1,000 tok, zlib < 0.10).")
P("  effective update = sum of gradient weights / G  (what one optimizer step 'sees' per token of the batch, before the learning rate).  x1000 in the tables.")
P("")
P("="*150); P("(A) PER STEP, PER ARM"); P("="*150)
P(f"{'arm':<28} {'step':>4} {'G tokens':>10} | {'loop tok%':>9} {'zero-adv tok%':>13} | {'loss value':>10} {'loop share |loss|':>17} | {'eff update x1e3':>15} {'loop share grad':>15} {'pos grad share':>14} {'top1% roll':>10} | {'(1-p) normal':>12} {'(1-p) loop':>10} {'mean p loop':>11} | {'TIS oob/M':>9} {'w loop':>6} {'w norm':>6}")
for a in ["G","I","J","A","K","F","L","B","D"]:
    for s in steps:
        x=A.get((a,s))
        if not x: continue
        P(f"{NAMES[a]:<28} {s:>4} {x['G']:>10,.0f} | {100*x['loop_tok_share']:>8.1f}% {100*x['zero_tok_share']:>12.1f}% | {x['loss']:>+10.4f} {100*x['absloss_loop_share']:>16.1f}% | {1e3*x['grad']:>15.3f} {100*x['grad_loop_share']:>14.2f}% {100*x['grad_pos_share']:>13.1f}% {100*x['top1pct_grad_share']:>9.1f}% | {x['normal_1mp']:>12.4f} {x['loop_1mp']:>10.5f} {x['loop_p']:>11.4f} | {x['tis_oob_per_M']:>9.1f} {x['w_mean_loop']:>6.3f} {x['w_mean_normal']:>6.3f}")
    P("")
def band(a,lo,hi):
    rows=[r for s_ in range(lo,hi+1) for r in R.get((a,s_),[])]
    if not rows: return None
    x=agg(rows); x["n"]=len({int(r["step"]) for r in rows})
    x["grad_noloop"]=x["grad"]*x["G"]/max(1.0,x["G"]*(1-x["loop_tok_share"])); x["grad_active"]=x["grad"]*x["G"]/max(1.0,x["G"]*x["mixed_tok_share"])
    return x
P("="*150); P("(B) GOOD vs BAD ARMS, pooled over the steps in each band (token-weighted).  good = run A, chain K, chain J (healthy so far), chain F, chain L (too early to call) ; bad = chain G, chain I, run B, chain D"); P("  w/o loop tok = effective update if loop tokens were excluded from the denominator G; mixed-only = if only tokens of mixed (non-zero-advantage) groups were in G"); P("="*150)
for lo,hi in ((1,10),(11,15),(16,20),(21,25),(26,36)):
    P(f"\n--- steps {lo}-{hi} ---")
    P(f"{'arm':<28} {'n':>2} {'eff update x1e3':>15} {'w/o loop tok':>12} {'mixed-only':>10} {'loop tok%':>9} {'zero-adv tok%':>13} {'pos grad share':>14} {'top1% roll':>10} {'(1-p) normal':>12} {'(1-p) loop':>10} {'p loop':>7} {'loop grad%':>10} {'loop push x1e6':>14} {'TIS oob/M':>9}")
    for grp,arms in (("GOOD",GOOD),("BAD",BAD)):
        for a in arms:
            b=band(a,lo,hi)
            if not b: continue
            P(f"{grp+' '+NAMES[a]:<28} {b['n']:>2} {1e3*b['grad']:>15.3f} {1e3*b['grad_noloop']:>12.3f} {1e3*b['grad_active']:>10.3f} {100*b['loop_tok_share']:>8.1f}% {100*b['zero_tok_share']:>12.1f}% {100*b['grad_pos_share']:>13.1f}% {100*b['top1pct_grad_share']:>9.1f}% {b['normal_1mp']:>12.4f} {b['loop_1mp']:>10.5f} {b['loop_p']:>7.4f} {100*b['grad_loop_share']:>9.3f}% {1e6*b['loop_push']:>+14.2f} {b['tis_oob_per_M']:>9.1f}")
P(""); P("="*150); P("(C) PER STEP, from-scratch arms side by side: effective update x1e3 | loop tok% | zero-adv tok% | (1-p) normal | (1-p) loop"); P("="*150)
FS=["A","K","J","G","I"]
P(f"{'step':>4} | " + " | ".join(f"{a:>28}" for a in FS))
for s_ in steps:
    if s_>36: continue
    cells=[]
    for a in FS:
        x=A.get((a,s_))
        cells.append(f"{1e3*x['grad']:>6.2f} {100*x['loop_tok_share']:>4.1f}% {100*x['zero_tok_share']:>4.0f}% {x['normal_1mp']:.4f} {x['loop_1mp']:.4f}" if x else f"{'':>28}")
    P(f"{s_:>4} | "+" | ".join(cells))
open(f"{L}/lossrec_report.txt","w").write("\n".join(o)+"\n"); print("\n".join(o))

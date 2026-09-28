import glob,csv,json,os,collections,math,sys
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
C="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/dumpbrowse_cache"
EXP={"G":"nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920","M":"nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923"}
LBL={"G":"chain G (MINF)","M":"chain M (vLLM)"}
def num(x):
    try: return float(x)
    except: return np.nan
def loadcsv(pattern):
    rows=[]
    for fn in sorted(glob.glob(pattern)):
        with open(fn) as fh:
            for r in csv.DictReader(fh): rows.append(r)
    return rows
S={a:loadcsv(f"{D}/parts/{a}__s*_c*.samples.csv") for a in EXP}
T={a:loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv") for a in EXP}
for a in EXP:
    for r in S[a]:
        for k in r:
            if k not in ("arm","sample_id","group_id"): r[k]=num(r[k])
    for r in T[a]:
        for k in r:
            if k not in ("arm","sample_id"): r[k]=num(r[k])
# jsonl summaries (instance ids, error kinds) from the browser cache when available
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
def q(x,p): 
    x=np.asarray(x,dtype=float); x=x[~np.isnan(x)]; return np.percentile(x,p) if len(x) else np.nan
P("DATA DIFFERENCE REPORT: chain G (MINF from scratch, no prefix cache) vs chain M (vLLM from scratch, replica 1), training steps 1-10")
P(f"samples: G {len(S['G'])}  M {len(S['M'])};  turns: G {len(T['G'])}  M {len(T['M'])};  jsonl summaries available for {sum(1 for k in J if k[0]=='G')} G / {sum(1 for k in J if k[0]=='M')} M samples")
P("")
# ---------- A. per step
P("A. PER STEP (512 rollouts = 32 prompts x 16)")
P(f"{'arm':>3} {'step':>4} {'n':>4} {'reward':>6} {'zerovar':>7} {'mean_gen':>8} {'p90_gen':>7} {'max_gen':>7} {'turns':>5} {'trunc':>5} {'runaway':>7} {'loopS':>5} {'nearS':>5} {'loop_tok%':>9} {'|d|':>6} {'d>1':>5} {'clip':>4} {'sme_max':>7} {'masked':>6} {'adv*len':>8}")
for a in EXP:
    steps=sorted({int(r["step"]) for r in S[a]})
    for s in steps:
        rs=[r for r in S[a] if r["step"]==s]; groups=collections.defaultdict(list)
        for r in rs: groups[r["group_id"]].append(r["reward"])
        zv=sum(1 for g in groups.values() if len(set(g))==1)
        ng=np.array([r["n_gen"] for r in rs]); adv=np.array([r["adv"] for r in rs]); lt=sum(r["loop_tok"] for r in rs)
        P(f"{a:>3} {s:>4} {len(rs):>4} {np.mean([r['reward'] for r in rs]):>6.3f} {zv:>3}/{len(groups):<3} {ng.mean():>8.0f} {q(ng,90):>7.0f} {ng.max():>7.0f} {np.mean([r['n_turns'] for r in rs]):>5.1f} {int(sum(r['truncated'] for r in rs)):>5} {int(sum(r['ends_seq_noclose'] for r in rs)):>7} {int(sum(r['loop_turns']>0 for r in rs)):>5} {int(sum(r['near_loop_turns']>0 for r in rs)):>5} {100*lt/ng.sum():>9.2f} {np.mean([r['mean_abs_d'] for r in rs]):>6.4f} {int(sum(r['n_d_gt1'] for r in rs)):>5} {int(sum(r['n_clip_lo']+r['n_clip_hi'] for r in rs)):>4} {max(r['seq_mult_prob_error'] for r in rs):>7.3f} {int(sum(r['mask_after']==0 for r in rs)):>6} {float((adv*ng).sum()/ng.sum()):>8.4f}")
P("  zerovar = groups whose 16 rewards are all equal (no gradient); runaway = rollout ends in an unclosed think block; loopS/nearS = rollouts with a >=1000-token think block of zlib <0.10 / 0.10-0.25;")
P("  |d| = mean per-sample |prev_logprob - gen_logprob|; d>1 = tokens with |d|>1; clip = TIS-clipped tokens; sme_max = max seq_mult_prob_error; adv*len = sum(adv x n_gen)/sum(n_gen) = token-weighted advantage (length asymmetry).")
P("")
# ---------- B. reward-structure asymmetries
P("B. WHAT THE GRADIENT PUSHES TOWARD: rewarded (adv>0) vs punished (adv<0) rollouts, pooled over steps 1-10 (zero-advantage rollouts excluded)")
feats=[("n_gen","generated tokens"),("n_turns","assistant turns"),("max_turn","longest turn (tokens)"),("last_turn","last turn (tokens)"),("mean_turn","mean turn length"),("n_noclose","turns without </think>"),("loop_tok","loop tokens (zlib<0.10, >=1k)"),("near_loop_tok","near-loop tokens (zlib 0.10-0.25)"),("nl_frac","newline share of generated tokens"),("mean_gen_lp","mean engine logprob"),("mean_abs_d","mean |d|"),("n_d_gt1","tokens with |d|>1"),("openhands_run_time","openhands run time (s)"),("model_call_time","model call time (s)")]
P(f"{'feature':<36} | {'G adv>0':>10} {'G adv<0':>10} {'G diff':>10} | {'M adv>0':>10} {'M adv<0':>10} {'M diff':>10} | {'G-M diff':>9} {'boot p':>7}")
rng=np.random.default_rng(0)
def diff_stat(rs,f):
    pos=[r[f] for r in rs if r["adv"]>1e-9]; neg=[r[f] for r in rs if r["adv"]<-1e-9]
    return np.nanmean(pos),np.nanmean(neg),np.nanmean(pos)-np.nanmean(neg)
def boot_p(f):
    # cluster bootstrap over (arm,step) groups? use group-level resampling within arm, compare diff-of-diffs
    vals={}
    for a in EXP:
        groups=collections.defaultdict(list)
        for r in S[a]: groups[r["group_id"]].append(r)
        vals[a]=list(groups.values())
    obs=diff_stat(S["G"],f)[2]-diff_stat(S["M"],f)[2]
    bs=[]
    for _ in range(400):
        d=[]
        for a in EXP:
            idx=rng.integers(0,len(vals[a]),len(vals[a])); rs=[r for j in idx for r in vals[a][j]]; d.append(diff_stat(rs,f)[2])
        bs.append(d[0]-d[1])
    bs=np.array(bs); se=bs.std(); z=obs/se if se>0 else 0; p=math.erfc(abs(z)/math.sqrt(2))
    return p
for f,lab in feats:
    g=diff_stat(S["G"],f); m=diff_stat(S["M"],f); p=boot_p(f)
    P(f"{lab:<36} | {g[0]:>10.2f} {g[1]:>10.2f} {g[2]:>+10.2f} | {m[0]:>10.2f} {m[1]:>10.2f} {m[2]:>+10.2f} | {g[2]-m[2]:>+9.2f} {p:>7.3f}")
P("  diff = mean over rewarded minus mean over punished rollouts; G-M diff = difference of those diffs; boot p = two-sided p from a group-level bootstrap (400 resamples).")
P("")
# within-group correlation of reward with length
P("B2. within-group Spearman correlation of reward with rollout features (groups with reward variance only), mean over groups; and token-weighted advantage by turn-length class")
def spearman(x,y):
    x=np.argsort(np.argsort(x)); y=np.argsort(np.argsort(y)); 
    if x.std()==0 or y.std()==0: return np.nan
    return np.corrcoef(x,y)[0,1]
for f,lab in [("n_gen","generated tokens"),("n_turns","assistant turns"),("max_turn","longest turn"),("mean_gen_lp","mean engine logprob"),("nl_frac","newline share")]:
    line=f"{lab:<24}"
    for a in EXP:
        groups=collections.defaultdict(list)
        for r in S[a]: groups[r["group_id"]].append(r)
        cs=[spearman(np.array([r["reward"] for r in g]),np.array([r[f] for r in g])) for g in groups.values() if len(set(r["reward"] for r in g))>1]
        cs=np.array(cs); cs=cs[~np.isnan(cs)]
        line+=f" | {LBL[a]}: mean rho {cs.mean():+.3f} (n groups {len(cs)}, se {cs.std()/math.sqrt(len(cs)):.3f})"
    P(line)
P("")
P("B3. token-weighted advantage by turn length class: sum(adv x tokens in class)/sum(tokens in class); a positive number means the gradient rewards tokens of that class on net")
bins=[(0,500),(500,1000),(1000,2000),(2000,4000),(4000,8000),(8000,16000),(16000,10**9)]
P(f"{'turn tokens':<14} | " + " | ".join(f"{LBL[a]:>22} {'share%':>6}" for a in EXP))
for lo,hi in bins:
    line=f"{lo:>6}-{hi if hi<10**9 else 'inf':<6} |"
    for a in EXP:
        tot=sum(r["n_tok"] for r in T[a]); rs=[r for r in T[a] if lo<=r["n_tok"]<hi]; w=sum(r["n_tok"] for r in rs)
        line+=f" {(sum(r['adv']*r['n_tok'] for r in rs)/w if w else float('nan')):>+22.4f} {100*w/tot:>6.2f} |"
    P(line)
P("")
# ---------- C. turn-level distributions
P("C. TURN-LEVEL DISTRIBUTIONS pooled over steps 1-10")
for a in EXP:
    nt=np.array([r["n_tok"] for r in T[a]]); th=np.array([r["think_tok"] for r in T[a]]); hc=np.array([r["has_close"] for r in T[a]]); es=np.array([r["ends_seq"] for r in T[a]])
    P(f"{LBL[a]}: turns {len(nt)}; n_tok p50 {q(nt,50):.0f} p90 {q(nt,90):.0f} p99 {q(nt,99):.0f} p99.9 {q(nt,99.9):.0f} max {nt.max():.0f}; think share of turn tokens {th.sum()/nt.sum():.3f}; turns without </think> {int((hc==0).sum())} (of which sequence-final {int(((hc==0)&(es==1)).sum())}); turns with 2+ </think> {int(sum(r['n_close']>=2 for r in T[a]))}")
    for lo,hi in bins:
        rs=[r for r in T[a] if lo<=r["n_tok"]<hi and r["think_tok"]>=1000 and not math.isnan(r["zlib_reason"])]
        if not rs: continue
        z=np.array([r["zlib_reason"] for r in rs]); pos=np.array([r["adv"]>1e-9 for r in rs])
        P(f"   turns {lo}-{hi if hi<10**9 else 'inf'} tokens with >=1k think: n {len(rs)}, zlib<0.10 {int((z<0.10).sum())}, 0.10-0.25 {int(((z>=0.10)&(z<0.25)).sum())}, 0.25-0.40 {int(((z>=0.25)&(z<0.40)).sum())}, >=0.40 {int((z>=0.40).sum())}; rewarded share {pos.mean():.2f}; rewarded among zlib<0.25: {int((pos&(z<0.25)).sum())}")
P("")
# ---------- D. logprob channel details
P("D. LOGPROB CHANNEL (d = trainer - engine)")
for a in EXP:
    dc=np.array([r["d_close"] for r in T[a] if not math.isnan(r["d_close"])]); gc=np.array([r["g_close"] for r in T[a] if not math.isnan(r["g_close"])])
    di=np.array([r["d_imend"] for r in T[a] if not math.isnan(r["d_imend"])]); df=np.array([r["d_first"] for r in T[a]]); gf=np.array([r["g_first"] for r in T[a]])
    P(f"{LBL[a]}: d at </think>: mean {dc.mean():+.4f} sd {dc.std():.4f} p1 {q(dc,1):+.3f} p99 {q(dc,99):+.3f} | engine logprob of </think>: mean {gc.mean():.4f} p10 {q(gc,10):.3f} p1 {q(gc,1):.3f} share<-1 {(gc<-1).mean():.4f} | d at <|im_end|>: mean {di.mean():+.5f} sd {di.std():.4f} | first token: engine lp mean {gf.mean():.3f}, d mean {df.mean():+.4f} sd {df.std():.3f}")
    # by start position
    P("   mean |d| by context position of the turn start (tokens): "+"; ".join(f"{lo//1000}k-{hi//1000 if hi<10**9 else 'inf'}k: {np.mean([r['mean_abs_d'] for r in T[a] if lo<=r['start']<hi]):.4f} (n {sum(1 for r in T[a] if lo<=r['start']<hi)})" for lo,hi in [(0,16000),(16000,32000),(32000,64000),(64000,128000),(128000,10**9)]))
    P("   mean d (signed) by context position: "+"; ".join(f"{lo//1000}k-{hi//1000 if hi<10**9 else 'inf'}k: {np.mean([r['mean_d'] for r in T[a] if lo<=r['start']<hi]):+.5f}" for lo,hi in [(0,16000),(16000,32000),(32000,64000),(64000,128000),(128000,10**9)]))
    P("   mean |d| by turn length: "+"; ".join(f"{lo}-{hi if hi<10**9 else 'inf'}: {np.mean([r['mean_abs_d'] for r in T[a] if lo<=r['n_tok']<hi]):.4f}" for lo,hi in bins))
    P("   TIS-clipped tokens: "+f"low {int(sum(r['n_clip_lo'] for r in T[a]))}, high {int(sum(r['n_clip_hi'] for r in T[a]))}; tokens |d|>1: {int(sum(r['n_d_gt1'] for r in T[a]))} of {int(sum(r['n_tok'] for r in T[a]))}; in rewarded turns: {int(sum(r['n_d_gt1'] for r in T[a] if r['adv']>1e-9))}, punished: {int(sum(r['n_d_gt1'] for r in T[a] if r['adv']<-1e-9))}")
P("")
# ---------- E. outliers
P("E. OUTLIERS (top 8 per arm)")
for key,lab in [("n_gen","generated tokens"),("loop_tok","loop tokens"),("max_abs_d","max |d|"),("n_d_gt1","tokens with |d|>1"),("max_turn","longest turn")]:
    for a in EXP:
        rs=sorted(S[a],key=lambda r:-r[key])[:8]
        P(f"{LBL[a]} top by {lab}: "+"; ".join(f"s{int(r['step'])} {r['sample_id'][-8:]} {key}={r[key]:.0f} rew={r['reward']:.0f} adv={r['adv']:+.2f} gen={r['n_gen']:.0f} turns={r['n_turns']:.0f} loop={r['loop_tok']:.0f}"+(f" err={J[(a,r['sample_id'])].get('agent_error_kind')}" if (a,r['sample_id']) in J else "") for r in rs))
P("")
# ---------- F. jsonl-only fields
if J:
    P("F. JSONL FIELDS (from the browser index): agent_error_kind and resolved/patch by arm, steps with summaries")
    for a in EXP:
        rs=[J[(a,r["sample_id"])] for r in S[a] if (a,r["sample_id"]) in J]
        ek=collections.Counter(str(s.get("agent_error_kind")) for s in rs); P(f"{LBL[a]}: n {len(rs)}; error kinds {dict(ek)}; resolved {sum(1 for s in rs if s.get('resolved'))}; patch_exists false {sum(1 for s in rs if s.get('patch_exists') is False)}; truncated {sum(1 for s in rs if s.get('truncated'))}; runaway(jsonl) {sum(1 for s in rs if s.get('runaway'))}")
        # reward by error kind
        P("   reward mean by error kind: "+"; ".join(f"{k}: {np.mean([s['reward'] for s in rs if str(s.get('agent_error_kind'))==k]):.3f} (n {sum(1 for s in rs if str(s.get('agent_error_kind'))==k)})" for k in ek))
open(f"{D}/compare_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

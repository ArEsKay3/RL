import glob,re,json,csv,collections,math,sys
import numpy as np
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis"
EXP={}
for line in open(f"{L}/loopfeedback/jobs.txt"):
    a,e=line.split()[:2]; EXP[a]=e
ARMS=["G","I","A"]
NAMES={"G":"chain G (MINF from0, no pfx)","I":"chain I (MINF from0, pfx on)","A":"run A (vLLM from0)"}
rng=np.random.default_rng(0)
# per-rollout accumulators
R=collections.defaultdict(lambda:{"tok":0,"loop_tok":0,"n_blk":0,"n_long":0,"n_semi":0,"n_loop":0,"n_runaway":0,"max_blk":0,"lp_sum":0.0,"sent_norm":0,"blk_norm":0,"reward":0,"trunc":0,"lens":[]})
for arm in ARMS:
    for fn in glob.glob(f"{L}/loopfeedback/parts/{arm}__s*_c*.items.csv"):
        with open(fn) as fh:
            rd=csv.reader(fh); next(rd)
            for row in rd:
                _,step,sid,wv,adv,reward,trunc,ii,ni,ntok,nsent,zl,d8,mlp,hc,cls=row
                step=int(step); ntok=int(ntok); zl=float(zl)
                e=R[(arm,step,sid)]
                e["tok"]+=ntok; e["n_blk"]+=1; e["lp_sum"]+=float(mlp)*ntok; e["reward"]=int(float(reward)); e["trunc"]=int(trunc)
                e["max_blk"]=max(e["max_blk"],ntok); e["lens"].append(ntok)
                if cls!="normal": e["loop_tok"]+=ntok; e["n_loop"]+=1
                if cls=="runaway": e["n_runaway"]+=1
                if ntok>=1000:
                    e["n_long"]+=1
                    if zl<0.20: e["n_semi"]+=1
                if cls=="normal": e["sent_norm"]+=int(nsent); e["blk_norm"]+=1
# instance ids
inst={}
for arm in ARMS:
    for fn in glob.glob(f"{L}/rollouts/{EXP[arm]}__target_step_*.summary.jsonl"):
        ts=int(re.search(r"target_step_(\d+)",fn).group(1))
        for line in open(fn):
            d=json.loads(line); inst[(arm,ts+1,d["sample_id"])]=d["instance_id"]
steps=sorted({s for (_,s,_) in R})
def q(v,p):
    v=np.asarray(v,dtype=float); return float(np.percentile(v,p)) if len(v) else float('nan')
print("PER STEP, arms with identical base weights at step 1 and the same prompts at every step.")
print("loop blk = self-exited + runaway loop blocks; loop tok% = share of reasoning tokens in them; long blk = blocks >= 1000 tok; semi = long blocks with zlib < 0.20;")
print("blk p99 = 99th pct block length (tokens); sent/nblk = sentences per normal block; mean lp = token-weighted mean log p of generated reasoning tokens; trunc = context-truncated rollouts.")
hdr=f"{'step':>4} {'arm':<4} {'n':>4} | {'loop blk':>8} {'loop tok%':>9} | {'long blk':>8} {'semi':>5} | {'blk p50':>7} {'blk p99':>7} {'blk max':>8} | {'sent/nblk':>9} {'mean lp':>8} | {'reward':>6} {'trunc':>5}"
print(hdr)
for s in steps:
    for arm in ARMS:
        rows=[v for (a,st,_),v in R.items() if a==arm and st==s]
        if not rows: continue
        tok=sum(v["tok"] for v in rows); lt=sum(v["loop_tok"] for v in rows); lens=[x for v in rows for x in v["lens"]]
        sn=sum(v["sent_norm"] for v in rows); bn=sum(v["blk_norm"] for v in rows)
        print(f"{s:>4} {arm:<4} {len(rows):>4} | {sum(v['n_loop'] for v in rows):>8} {100*lt/max(1,tok):>8.1f}% | {sum(v['n_long'] for v in rows):>8} {sum(v['n_semi'] for v in rows):>5} | {q(lens,50):>7.0f} {q(lens,99):>7.0f} {max(lens):>8} | {sn/max(1,bn):>9.2f} {sum(v['lp_sum'] for v in rows)/max(1,tok):>8.4f} | {np.mean([v['reward'] for v in rows]):>6.3f} {sum(v['trunc'] for v in rows):>5}")
    print()
# paired by instance
def per_instance(arm,band):
    out=collections.defaultdict(lambda:collections.defaultdict(list))
    for (a,s,sid),v in R.items():
        if a!=arm or s not in band: continue
        key=(s,inst.get((a,s,sid)))
        if key[1] is None: continue
        d=out[key]
        d["loop_tok_share"].append(v["loop_tok"]/max(1,v["tok"]))
        d["any_loop"].append(1.0 if v["n_loop"]>0 else 0.0)
        d["runaway"].append(float(v["n_runaway"]))
        d["n_long"].append(float(v["n_long"]))
        d["n_semi"].append(float(v["n_semi"]))
        d["max_blk"].append(float(v["max_blk"]))
        d["reason_tok"].append(float(v["tok"]))
        d["sent_per_nblk"].append(v["sent_norm"]/max(1,v["blk_norm"]))
        d["mean_lp"].append(v["lp_sum"]/max(1,v["tok"]))
        d["reward"].append(float(v["reward"]))
        d["trunc"].append(float(v["trunc"]))
    return {k:{m:float(np.mean(x)) for m,x in d.items()} for k,d in out.items()}
METRICS=[("loop_tok_share","loop token share per rollout"),("any_loop","rollouts with >=1 loop block"),("runaway","runaway blocks per rollout"),("n_long","blocks >=1000 tok per rollout"),("n_semi","blocks >=1000 tok & zlib<0.20 per rollout"),("max_blk","longest block per rollout (tok)"),("reason_tok","reasoning tokens per rollout"),("sent_per_nblk","sentences per normal block"),("mean_lp","mean log p of reasoning tokens"),("reward","reward"),("trunc","context-truncated rollouts")]
def signflip(d,n=20000):
    d=np.asarray(d,dtype=float); obs=d.mean()
    if len(d)==0 or np.all(d==0): return float('nan')
    signs=rng.choice([-1.0,1.0],size=(n,len(d))); sims=(signs*d).mean(axis=1)
    return float((np.abs(sims)>=abs(obs)-1e-12).mean())
for band,label in [(set(range(1,11)),"steps 1-10"),(set(range(11,21)),"steps 11-20"),(set(range(21,31)),"steps 21-30")]:
    PI={arm:per_instance(arm,band) for arm in ARMS}
    keys=set(PI["G"])&set(PI["I"])&set(PI["A"])
    if not keys: continue
    print(f"\nPAIRED BY (step, SWE instance), {label}: {len(keys)} instances x 16 gens per arm. mean over instances of the per-instance mean; p = two-sided sign-flip permutation test on paired differences (20,000 draws).")
    print(f"{'metric':<42} {'G':>10} {'I':>10} {'A':>10} | {'G-A':>10} {'p':>7} | {'I-A':>10} {'p':>7} | {'G-I':>10} {'p':>7}")
    for m,lab in METRICS:
        g=np.array([PI["G"][k][m] for k in sorted(keys)]); i=np.array([PI["I"][k][m] for k in sorted(keys)]); a=np.array([PI["A"][k][m] for k in sorted(keys)])
        fmt=(lambda x:f"{x:>10.4f}") if m in ("loop_tok_share","any_loop","runaway","n_long","n_semi","sent_per_nblk","mean_lp","reward","trunc") else (lambda x:f"{x:>10.0f}")
        print(f"{lab:<42} {fmt(g.mean())} {fmt(i.mean())} {fmt(a.mean())} | {fmt((g-a).mean())} {signflip(g-a):>7.3f} | {fmt((i-a).mean())} {signflip(i-a):>7.3f} | {fmt((g-i).mean())} {signflip(g-i):>7.3f}")

import glob,csv,json,re,collections,sys,os
import numpy as np
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
R="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
ARMS=json.load(open(f"{L}/splice_arms.json"))
def RS(c): return set(c.get("replay_steps") or range(1,c["replay_end"]+1))
def rdesc(c):
    r=sorted(RS(c)); return f"steps {r[0]}-{r[-1]}" if r==list(range(r[0],r[-1]+1)) else "steps "+",".join(map(str,r))
COMP=["M","N","G","I","J","A","K","D"]
cache_fn=f"{L}/splice_gen_cache.json"; cache=json.load(open(cache_fn)) if os.path.exists(cache_fn) else {}
S=collections.defaultdict(lambda:{"roll":set(),"tok":0,"blk":0,"rep_tok":0,"rep_n":0,"roll_rep":set(),"chunks":set()})
for a in ARMS:
    for fn in glob.glob(f"{L}/parts/{a}__s*_c*.items.csv"):
        m=re.search(r"__s(\d+)_c(\d+)",fn); step,ck=int(m.group(1)),int(m.group(2))
        S[(a,step)]["chunks"].add(ck)
        with open(fn) as fh:
            rd=csv.reader(fh); next(rd)
            for row in rd:
                _,st,sid,wv,adv,reward,trunc,ii,ni,ntok,nsent,zl,d8,mlp,hc,cls=row
                ntok=int(ntok); zl=float(zl); e=S[(a,step)]
                e["roll"].add(sid); e["tok"]+=ntok; e["blk"]+=1
                if ntok>=1000 and zl<0.10: e["rep_tok"]+=ntok; e["rep_n"]+=1; e["roll_rep"].add(sid)
gen={}
for (a,s),e in S.items():
    tot=0
    for ck in sorted(e["chunks"]):
        key=f"{a}:{s}:{ck}"
        if key not in cache:
            fn=f"{R}/{ARMS[a]['exp']}/dumps/token_level/step_{s:05d}_chunk_{ck:03d}.pt"
            cache[key]=int(np.asarray(load(fn)["token_mask"]).sum())
        tot+=cache[key]
    gen[(a,s)]=tot
json.dump(cache,open(cache_fn,"w"),indent=0)
joint={}
for r in csv.DictReader(open(f"{L}/joint_loopshare.csv")): joint[(r["arm"],int(r["step"]))]=float(r["share_all_gen_pct"])
partial=set()
for r in csv.DictReader(open(f"{L}/loopshare.csv")):
    if int(r["rollouts"])<512: partial.add((r["arm"],int(r["step"])))
with open(f"{L}/splice_loopshare.csv","w") as fh:
    w=csv.writer(fh); w.writerow(["arm","step","mode","rollouts","chunks","reasoning_tokens","repetitive_blocks","repetitive_tokens","gen_tokens","share_all_gen_pct","rollouts_with_repetitive_block"])
    for (a,s),e in sorted(S.items()):
        mode="replay" if (s in RS(ARMS[a]) and ARMS[a].get("kind","splice")=="splice") else "live"
        w.writerow([a,s,mode,len(e["roll"]),len(e["chunks"]),e["tok"],e["rep_n"],e["rep_tok"],gen[(a,s)],round(100*e["rep_tok"]/max(1,gen[(a,s)]),3),len(e["roll_rep"])])
    for a in COMP:
        for (aa,s),v in sorted(joint.items()):
            if aa==a: w.writerow([a,s,"live","","","","","","",round(v,3),""])
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("LOOP-TOKEN SHARE, CROSS-TRAIN (SPLICE) RUNS WITH THEIR ASSOCIATED ARMS")
P("share = tokens in repetitive reasoning blocks (block >= 1,000 tokens, zlib ratio < 0.10) / ALL generated tokens of the step's 512 trained rollouts.")
for a,c in ARMS.items():
    if c.get("kind")=="resume": desc=c["design"]+f"; live from step {c['replay_end']+1}, same prompts per step as every arm"
    elif c.get("kind","splice")=="lr0": desc=f"{c['engine']} engine, weights frozen at chain {c['source']} step {c['replay_end']} (lr = 0, optimizer state not loaded); live {c['engine']} generation from step {c['replay_end']+1} on, same prompts per step as every arm"
    else: desc=f"{c['engine']} engine, fresh from the base model; {rdesc(c)} replay chain {c['source']}'s recorded rollouts (no generation by this engine), live {c['engine']} generation on the other steps"
    P(f"{c.get('label','chain '+a)} = {desc}"+(f"; {c['note']}" if c.get("note") else "")+f".  EXP {c['exp']}"+("" if any(aa==a for (aa,_) in S) else "  [no dumps yet]"))
P("comparators: M = chain M vLLM from0 r1 (source of the replayed rollouts) | N = chain N vLLM from0 r2 | G = chain G MINF from0 no-pfx | I = chain I MINF from0 pfx r1 | J = chain J MINF from0 pfx r2 | A = run A vLLM from0 | K = chain K vLLM from0 seed 1234 | D = chain D vLLM from MINF step 10 (comparator for R, R′)")
P("'r' after a value = replayed step (equals the source arm by construction; tooling check).  '*' = partial step (< 512 rollouts).")
P("")
steps=sorted({s for (_,s) in S}|{s for (a,s) in joint if a in COMP})
hdr=f"{'step':>4} | "+" ".join(f"{a+' %':>8}" for a in ARMS)+" | "+" ".join(f"{a+' %':>6}" for a in COMP); P(hdr)
for s in steps:
    cells=[]
    for a in ARMS:
        e=S.get((a,s))
        if not e or not gen.get((a,s)): cells.append(f"{'':>8}"); continue
        v=100*e["rep_tok"]/gen[(a,s)]; suf="r" if (s in RS(ARMS[a]) and ARMS[a].get("kind","splice")=="splice") else ("*" if len(e["roll"])<512 else " ")
        cells.append(f"{v:>7.1f}{suf}")
    comp=[f"{joint[(a,s)]:>5.1f}{'*' if (a,s) in partial else ' '}" if (a,s) in joint else f"{'':>6}" for a in COMP]
    P(f"{s:>4} | "+" ".join(cells)+" | "+" ".join(comp))
P("")
for a,c in ARMS.items():
    diffs=[(s,100*S[(a,s)]["rep_tok"]/gen[(a,s)],joint.get((c["source"],s))) for s in sorted({s for (aa,s) in S if aa==a}) if c.get("kind","splice")=="splice" and s in RS(c) and gen.get((a,s)) and (c["source"],s) in joint]
    if diffs:
        md=max(abs(v-w) for _,v,w in diffs); P(f"replay check {c.get('label','chain '+a)} vs chain {c['source']} on steps {diffs[0][0]}-{diffs[-1][0]}: max |difference| = {md:.2f} percentage points over {len(diffs)} steps")
    live=[(s,100*S[(a,s)]["rep_tok"]/gen[(a,s)],len(S[(a,s)]["roll"])) for s in sorted({s for (aa,s) in S if aa==a}) if (s not in RS(c) or c.get("kind","splice")!="splice") and gen.get((a,s))]
    P(f"live steps {c.get('label','chain '+a)}: "+(", ".join(f"{s}: {v:.1f}% ({n} rollouts)" for s,v,n in live) if live else "none yet"))
open(f"{L}/splice_loopshare.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

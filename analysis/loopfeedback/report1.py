import csv,collections,statistics as st,sys
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
ARMS={"G":"chain G  MINF from scratch, no prefix cache","A":"run A    vLLM from scratch","F":"chain F  MINF from vLLM step 10","B":"run B    MINF from MINF step 10","D":"chain D  vLLM from MINF step 10","I":"chain I  MINF from scratch, prefix cache on"}
rows=list(csv.DictReader(open(f"{L}/items.csv")))
for r in rows:
    r["step"]=int(r["step"]); r["n_tok"]=int(r["n_tok"]); r["zlib"]=float(r["zlib"]); r["has_close"]=r["has_close"]=="1"
by=collections.defaultdict(list)
for r in rows: by[(r["arm"],r["step"])].append(r)
print("STEP 1 - REASONING-BLOCK LABELING (token dumps; one row per (arm, train step); every available step)")
print("Legend: blocks = reasoning (thinking) blocks; loop-like = block >= 1000 tokens whose text compresses to < 10 % with zlib (heavy repetition), and which DID close with </think>;")
print("        runaway = last block of a context-window-truncated rollout with no </think>; tok share = share of all reasoning tokens in loop-like + runaway blocks;")
print("        exited = loop-like blocks that ended with </think> (partial loops that recovered); blk p95 = 95th percentile block length in tokens.")
for arm,name in ARMS.items():
    steps=sorted(s for (a,s) in by if a==arm)
    if not steps: continue
    print(f"\n{name}")
    print(f"{'step':>4} {'rollouts':>8} {'blocks':>7} {'loop-like':>9} {'runaway':>7} {'tok share':>9} {'blk p50':>7} {'blk p95':>7} {'blk max':>8} {'loop tok p50':>12}")
    for s in steps:
        R=by[(arm,s)]; n_roll=len(set(r["sample_id"] for r in R))
        ll=[r for r in R if r["cls"]=="looplike"]; ra=[r for r in R if r["cls"]=="runaway"]
        tot=sum(r["n_tok"] for r in R); lt=sum(r["n_tok"] for r in ll+ra)
        toks=sorted(r["n_tok"] for r in R); p50=toks[len(toks)//2]; p95=toks[int(0.95*(len(toks)-1))]
        lp50=(sorted(r["n_tok"] for r in ll+ra)[len(ll+ra)//2]) if ll+ra else 0
        print(f"{s:>4} {n_roll:>8} {len(R):>7} {len(ll):>9} {len(ra):>7} {lt/tot:>9.1%} {p50:>7} {p95:>7} {max(toks):>8} {lp50:>12}")
print("\nCALIBRATION - zlib compression ratio of reasoning blocks >= 1000 tokens, all arms pooled (lower = more repetitive)")
big=[r for r in rows if r["n_tok"]>=1000]
bins=[(0,0.03),(0.03,0.06),(0.06,0.10),(0.10,0.15),(0.15,0.20),(0.20,0.25),(0.25,0.30),(0.30,0.40),(0.40,1.01)]
print(f"{'zlib ratio bin':>16} {'blocks':>8} {'share':>7} {'median tokens':>13} {'share w/o </think>':>18}")
for a,b in bins:
    sel=[r for r in big if a<=r["zlib"]<b]
    if sel: print(f"{f'[{a:.2f},{b:.2f})':>16} {len(sel):>8} {len(sel)/len(big):>7.1%} {sorted(r['n_tok'] for r in sel)[len(sel)//2]:>13} {sum(1 for r in sel if not r['has_close'])/len(sel):>18.1%}")
print("(a clean gap between the low-ratio mass and the bulk around 0.3 validates the 0.10 cut)")

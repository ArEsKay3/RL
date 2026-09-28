import glob,re,csv,collections,json
import numpy as np
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
ARMS=["G","I","J","A","K","M","N","S","T","U","V","V2","V3","P4","P4b","Q4","Q4b","F","L","B","D"]
NAMES={"G":"chain G  MINF from0, no prefix cache","I":"chain I  MINF from0, prefix cache on","J":"chain J  MINF from0, prefix cache on, replica 2","A":"run A    vLLM from0","K":"chain K  vLLM from0, seed 1234","M":"chain M  vLLM from0, replica 1 (J args)","N":"chain N  vLLM from0, replica 2 (J args)","S":"chain S  vLLM from0, 9-15 stack, masking off","T":"chain T  MINF from0, 9-15 stack, masking off","P4":"chain P⁗ MINF from0, prefix cache kept across refits","Q4":"chain Q⁗ vLLM from0, vLLM prefix caching disabled","P4b":"chain P⁗2 MINF from0, prefix cache kept, seed 1234","Q4b":"chain Q⁗2 vLLM from0, prefix caching disabled, seed 1234","U":"chain U  MINF from0, main915 stack","V":"chain V  MINF from0, vLLM-parity adapter, prefix cache kept","V2":"chain V2 MINF from0, vLLM-parity adapter, seed 1234","V3":"chain V3 MINF from0, vLLM-parity adapter, seed 4321","F":"chain F  MINF from vLLM step 10","L":"chain L  MINF from chain K step 10, prefix cache on","B":"run B    MINF from MINF step 10","D":"chain D  vLLM from MINF step 10"}
S=collections.defaultdict(lambda:{"roll":set(),"tok":0,"blk":0,"cls_tok":0,"cls_n":0,"rep_tok":0,"rep_n":0,"run_tok":0,"run_n":0,"run_rep_n":0,"roll_loop":set(),"roll_rep":set(),
    "pos_rep_n":0,"pos_rep_tok":0,"pos_rep_advtok":0.0,"neg_rep_n":0,"neg_rep_tok":0,"neg_rep_advtok":0.0,"zero_rep_n":0,"zero_rep_tok":0,
    "pos_exit_n":0,"pos_run_n":0,"pos_roll":set(),"neg_roll":set()})
for arm in ARMS:
    for fn in glob.glob(f"{L}/parts/{arm}__s*_c*.items.csv"):
        with open(fn) as fh:
            rd=csv.reader(fh); next(rd)
            for row in rd:
                _,step,sid,wv,adv,reward,trunc,ii,ni,ntok,nsent,zl,d8,mlp,hc,cls=row
                step=int(step); ntok=int(ntok); zl=float(zl); adv=float(adv); hc=hc=="1"
                e=S[(arm,step)]; e["roll"].add(sid); e["tok"]+=ntok; e["blk"]+=1
                rep=(ntok>=1000 and zl<0.10)
                if cls!="normal": e["cls_tok"]+=ntok; e["cls_n"]+=1; e["roll_loop"].add(sid)
                if cls=="runaway":
                    e["run_tok"]+=ntok; e["run_n"]+=1
                    if rep: e["run_rep_n"]+=1
                if rep:
                    e["rep_tok"]+=ntok; e["rep_n"]+=1; e["roll_rep"].add(sid)
                    if adv>1e-9:
                        e["pos_rep_n"]+=1; e["pos_rep_tok"]+=ntok; e["pos_rep_advtok"]+=adv*ntok; e["pos_roll"].add(sid)
                        if hc: e["pos_exit_n"]+=1
                        else: e["pos_run_n"]+=1
                    elif adv<-1e-9: e["neg_rep_n"]+=1; e["neg_rep_tok"]+=ntok; e["neg_rep_advtok"]+=adv*ntok; e["neg_roll"].add(sid)
                    else: e["zero_rep_n"]+=1; e["zero_rep_tok"]+=ntok
steps=sorted({s for (_,s) in S})
out=open(f"{L}/loopshare.csv","w"); w=csv.writer(out)
w.writerow(["arm","step","rollouts","reasoning_tokens","blocks","loop_tokens_classbased","loop_blocks_classbased","repetitive_tokens","repetitive_blocks","runaway_tokens","runaway_blocks","runaway_blocks_repetitive","rollouts_with_loop_classbased","rollouts_with_repetitive_block",
            "rep_blocks_adv_pos","rep_tokens_adv_pos","rep_advtok_pos","rep_blocks_adv_neg","rep_tokens_adv_neg","rep_advtok_neg","rep_blocks_adv_zero","rep_tokens_adv_zero","rep_pos_selfexited","rep_pos_runaway"])
for arm in ARMS:
    for s in steps:
        e=S.get((arm,s))
        if not e: continue
        w.writerow([arm,s,len(e["roll"]),e["tok"],e["blk"],e["cls_tok"],e["cls_n"],e["rep_tok"],e["rep_n"],e["run_tok"],e["run_n"],e["run_rep_n"],len(e["roll_loop"]),len(e["roll_rep"]),
                    e["pos_rep_n"],e["pos_rep_tok"],round(e["pos_rep_advtok"]),e["neg_rep_n"],e["neg_rep_tok"],round(e["neg_rep_advtok"]),e["zero_rep_n"],e["zero_rep_tok"],e["pos_exit_n"],e["pos_run_n"]])
out.close()
print("DEFINITIONS")
print("  reasoning block   = generated tokens from the start of an assistant turn up to </think> (or to the end of the turn if </think> never came).")
print("  repetitive block  = block >= 1,000 tokens whose UTF-8 text compresses with zlib (level 6) to < 10 % of its size.  (normal long blocks: 25-40 %)")
print("  runaway block     = last block of a rollout that hit the context window, with no </think>.")
print("  class-based share = tokens in (repetitive OR runaway) blocks / all reasoning tokens of the step's 512 trained rollouts.  [what the earlier tables showed]")
print("  strict share      = tokens in repetitive blocks only / all reasoning tokens.  (runaways count only if they are themselves repetitive)")
print()
hdr=f"{'step':>4} " + " ".join(f"{a+' strict%':>9} {'cls%':>5} {'nrep':>4}" for a in ARMS)
print("STRICT loop-token share (%), class-based share (%), and number of repetitive blocks, per step and arm")
print(hdr)
for s in steps:
    cells=[]
    for a in ARMS:
        e=S.get((a,s))
        if not e: cells.append(f"{'':>9} {'':>5} {'':>4}"); continue
        cells.append(f"{100*e['rep_tok']/max(1,e['tok']):>9.1f} {100*e['cls_tok']/max(1,e['tok']):>5.1f} {e['rep_n']:>4}")
    print(f"{s:>4} "+" ".join(cells))
print()
print("CASE-CONTROL, steps 1-10 (arms that went bad: G, I; stayed healthy: A). Repetitive blocks by the sign of the rollout advantage they were trained with.")
print("  adv>0 = the loop sat in a REWARDED rollout (reinforced); adv<0 = punished; adv=0 = zero-variance group (no gradient). advtok = sum(adv x block tokens).")
print(f"{'arm':<40} {'rep blocks':>10} {'adv>0':>6} {'tokens':>9} {'advtok':>9} {'self-exited':>11} {'runaway':>8} | {'adv<0':>6} {'tokens':>9} {'advtok':>9} | {'adv=0':>6} {'tokens':>9} | {'rewarded rollouts w/ loop':>26}")
for band,label in [(range(1,11),"steps 1-10"),(range(11,21),"steps 11-20"),(range(21,31),"steps 21-30")]:
    print(f"--- {label}")
    for a in ARMS:
        es=[S[(a,s)] for s in band if (a,s) in S]
        if not es: continue
        n=sum(e["rep_n"] for e in es)
        pn=sum(e["pos_rep_n"] for e in es); pt=sum(e["pos_rep_tok"] for e in es); pa=sum(e["pos_rep_advtok"] for e in es); pe=sum(e["pos_exit_n"] for e in es); pr=sum(e["pos_run_n"] for e in es)
        nn=sum(e["neg_rep_n"] for e in es); nt=sum(e["neg_rep_tok"] for e in es); na=sum(e["neg_rep_advtok"] for e in es)
        zn=sum(e["zero_rep_n"] for e in es); zt=sum(e["zero_rep_tok"] for e in es)
        prl=sum(len(e["pos_roll"]) for e in es)
        print(f"{NAMES[a]:<40} {n:>10} {pn:>6} {pt:>9,} {pa:>9,.0f} {pe:>11} {pr:>8} | {nn:>6} {nt:>9,} {na:>9,.0f} | {zn:>6} {zt:>9,} | {prl:>26}")
print()
print("PER STEP, steps 1-10: repetitive blocks in rewarded rollouts (adv>0) -> count / tokens, and in punished rollouts (adv<0) -> count / tokens")
print(f"{'step':>4} " + " ".join(f"{a+': +n':>6} {'+tok':>7} {'-n':>4} {'-tok':>7}" for a in ["G","I","A"]))
for s in range(1,11):
    cells=[]
    for a in ["G","I","A"]:
        e=S.get((a,s))
        cells.append(f"{e['pos_rep_n']:>6} {e['pos_rep_tok']:>7,} {e['neg_rep_n']:>4} {e['neg_rep_tok']:>7,}")
    print(f"{s:>4} "+" ".join(cells))

import glob,csv,json,os,re,sys,collections
import numpy as np
L=os.environ["LF_L"]; R=os.environ["LF_R"]
sys.path.insert(0,os.path.join(L,"..","..","tools"))
from pt_numpy import load as ptload
EXP={}; DESC={}
for line in open(f"{L}/jobs_hsg.txt"):
    f=line.split(); EXP[f[0]]=f[1]; DESC[f[0]]=" ".join(f[2:])
S=collections.defaultdict(lambda:{"roll":set(),"tok":0,"blk":0,"rep_tok":0,"rep_n":0,"run_tok":0,"run_n":0,"roll_rep":set(),"pos_rep_n":0,"pos_rep_tok":0,"neg_rep_n":0,"neg_rep_tok":0,"zero_rep_n":0,"long_n":0,"blk_max":0,"trunc":set()})
for arm in EXP:
    for fn in glob.glob(f"{L}/parts/{arm}__s*_c*.items.csv"):
        with open(fn) as fh:
            rd=csv.reader(fh); next(rd)
            for row in rd:
                _,step,sid,wv,adv,reward,trunc,ii,ni,ntok,nsent,zl,d8,mlp,hc,cls=row
                step=int(step); ntok=int(ntok); zl=float(zl); adv=float(adv)
                e=S[(arm,step)]; e["roll"].add(sid); e["tok"]+=ntok; e["blk"]+=1; e["blk_max"]=max(e["blk_max"],ntok)
                if trunc=="1": e["trunc"].add(sid)
                if ntok>=1000: e["long_n"]+=1
                if cls=="runaway": e["run_tok"]+=ntok; e["run_n"]+=1
                if ntok>=1000 and zl<0.10:
                    e["rep_tok"]+=ntok; e["rep_n"]+=1; e["roll_rep"].add(sid)
                    if adv>1e-9: e["pos_rep_n"]+=1; e["pos_rep_tok"]+=ntok
                    elif adv<-1e-9: e["neg_rep_n"]+=1; e["neg_rep_tok"]+=ntok
                    else: e["zero_rep_n"]+=1
cache_fn=f"{L}/joint_gen_cache.json"; cache=json.load(open(cache_fn)) if os.path.exists(cache_fn) else {}
gen=collections.Counter()
for (a,s) in S:
    for fn in glob.glob(f"{R}/{EXP[a]}/dumps/token_level/step_{s:05d}_chunk_*.pt"):
        key=f"{a}:{os.path.basename(fn)}:{os.path.getsize(fn)}"
        if key not in cache: cache[key]=int(np.asarray(ptload(fn)["token_mask"]).sum())
        gen[(a,s)]+=cache[key]
json.dump(cache,open(cache_fn,"w"))
cmp=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{L}/../loopfeedback/joint_loopshare.csv")): cmp[r["arm"]][int(r["step"])]=float(r["share_all_gen_pct"])
CMP=["V","V2","V3","P4","P4b","J","G","A","K","Q4"]
out=[]
P=lambda *x: out.append(" ".join(str(v) for v in x))
P("HSG vLLM-parity arms: loop-token share per training step (repetitive reasoning blocks >= 1,000 tok, zlib < 10 %).")
P("joint % = loop tokens / all generated tokens of the step (the column used in the CMH main table); strict % = loop tokens / reasoning tokens.")
P("CMH comparators (joint %) at the same step: V/V2/V3 = chain V and seed replicas (vLLM-parity MINF), P4/P4b = chain P'''' and seed replica (MINF keep-prefix), J = chain J (MINF clean), G = chain G (MINF poisoned), A = run A (vLLM), K = chain K (vLLM seed 1234), Q4 = chain Q'''' (vLLM, prefix caching off).")
P("")
P(f"{'arm':>4} {'step':>4} {'rows':>4} {'gen tok':>10} {'reas tok':>9} {'joint%':>6} {'strict%':>7} {'rep blk':>7} {'+adv':>4} {'-adv':>4} {'0adv':>4} {'runaway':>7} {'trunc':>5} {'blk max':>7} | "+" ".join(f"{a:>5}" for a in CMP))
for a in EXP:
    for s in sorted(st for (aa,st) in S if aa==a):
        e=S[(a,s)]; g=gen.get((a,s),0)
        P(f"{a:>4} {s:>4} {len(e['roll']):>4} {g:>10,} {e['tok']:>9,} {100*e['rep_tok']/max(1,g):>6.1f} {100*e['rep_tok']/max(1,e['tok']):>7.1f} {e['rep_n']:>7} {e['pos_rep_n']:>4} {e['neg_rep_n']:>4} {e['zero_rep_n']:>4} {e['run_n']:>7} {len(e['trunc']):>5} {e['blk_max']:>7,} | "+" ".join(f"{cmp[c][s]:>5.1f}" if s in cmp[c] else f"{'':>5}" for c in CMP))
P("")
P("Complete steps only (512 rows). Steps 1-10 do not separate arms on CMH; the verdict window is steps 20-30.")
txt="\n".join(out); open(f"{L}/hsg_loopshare.txt","w").write(txt+"\n"); print(txt)
with open(f"{L}/hsg_loopshare.csv","w") as fh:
    w=csv.writer(fh); w.writerow(["arm","step","rows","gen_tokens","reasoning_tokens","joint_pct","strict_pct","repetitive_blocks","rep_adv_pos","rep_adv_neg","rep_adv_zero","runaway_blocks","truncated_rollouts","block_max_tokens"])
    for (a,s),e in sorted(S.items()):
        g=gen.get((a,s),0); w.writerow([a,s,len(e["roll"]),g,e["tok"],round(100*e["rep_tok"]/max(1,g),3),round(100*e["rep_tok"]/max(1,e["tok"]),3),e["rep_n"],e["pos_rep_n"],e["neg_rep_n"],e["zero_rep_n"],e["run_n"],len(e["trunc"]),e["blk_max"]])

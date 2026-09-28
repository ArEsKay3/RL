import glob,csv,re,collections
import numpy as np
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
T="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/tokens"
KEXP="nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-seed1234-20260922"
blk=collections.defaultdict(lambda:{"blocks":0,"reason":0,"rep":[],"roll":set(),"roll_loop":set()})
for fn in glob.glob(f"{L}/parts/K__s*_c*.items.csv"):
    with open(fn) as fh:
        rd=csv.reader(fh); next(rd)
        for row in rd:
            s=int(row[1]); ntok=int(row[9]); zl=float(row[11]); e=blk[s]; e["blocks"]+=1; e["reason"]+=ntok; e["roll"].add(row[2])
            if ntok>=1000 and zl<0.10: e["rep"].append(ntok); e["roll_loop"].add(row[2])
gen=collections.Counter(); nroll=collections.Counter()
for fn in glob.glob(f"{T}/{KEXP}__step_*_chunk_*.rows.csv"):
    s=int(re.search(r"step_(\d+)_chunk",fn).group(1))
    with open(fn) as fh:
        for r in csv.DictReader(fh): gen[s]+=int(float(r["n_gen_tokens"])); nroll[s]+=1
J={}
for r in csv.DictReader(open(f"{L}/joint_loopshare.csv")): J[(r["arm"],int(r["step"]))]=float(r["share_all_gen_pct"])
S={}
for r in csv.DictReader(open(f"{L}/loopshare.csv")): S[(r["arm"],int(r["step"]))]=100*int(r["repetitive_tokens"])/max(1,int(r["reasoning_tokens"]))
steps=sorted(s for s in blk if len(blk[s]["roll"])>=512 or s==max(blk))
print("CHAIN K (vLLM from scratch, seed 1234) so far, next to the other arms at the same steps.  share = loop tokens / all generated tokens (%), strict = loop tokens / reasoning tokens (%).")
print(f"{'step':>4} {'K rollouts':>10} | {'K loops':>7} {'P(loop|roll)':>12} {'E[size|loop]':>12} | {'K share':>7} {'A share':>7} {'G share':>7} {'I share':>7} {'main share':>10} | {'K strict':>8} {'A strict':>8} {'G strict':>8} {'I strict':>8} | K rank (1=lowest)")
for s in steps:
    e=blk[s]; rep=np.array(e["rep"]) if e["rep"] else np.array([0.0]); n=len(e["roll"])
    ksh=100*sum(e["rep"])/max(1,gen[s]) if gen[s] else float('nan'); kst=100*sum(e["rep"])/max(1,e["reason"])
    others={a:J.get((a,s)) for a in ("A","G","I","main")}
    pres={a:v for a,v in others.items() if v is not None}
    rank=1+sum(1 for v in pres.values() if v<ksh) if ksh==ksh else None
    f=lambda v:f"{v:>7.1f}" if v is not None else f"{'':>7}"
    print(f"{s:>4} {n:>10} | {len(e['rep']):>7} {100*len(e['roll_loop'])/max(1,n):>11.2f}% {rep.mean() if e['rep'] else 0:>12,.0f} | {ksh:>7.1f} {f(others['A'])} {f(others['G'])} {f(others['I'])} {f(others['main']) if others['main'] is not None else '':>10} | {kst:>8.1f} {S.get(('A',s),float('nan')):>8.1f} {S.get(('G',s),float('nan')):>8.1f} {S.get(('I',s),float('nan')):>8.1f} | {rank} of {len(pres)+1}{'  (partial step)' if n<512 else ''}")
full=[s for s in steps if len(blk[s]["roll"])>=512]
if full:
    print(f"\nPOOLED over K's complete steps {full[0]}-{full[-1]}:")
    def pooled(arm):
        if arm=="K":
            rep=[x for s in full for x in blk[s]["rep"]]; g=sum(gen[s] for s in full); rt=sum(blk[s]["reason"] for s in full); b=sum(blk[s]["blocks"] for s in full); rl=sum(len(blk[s]["roll_loop"]) for s in full); n=sum(len(blk[s]["roll"]) for s in full)
            return len(rep),b,rl,n,np.mean(rep) if rep else 0,np.median(rep) if rep else 0,100*sum(rep)/g,100*sum(rep)/rt
        return None
    k=pooled("K")
    print(f"  K: loops {k[0]}, blocks {k[1]:,}, P(loop|block) {k[0]/k[1]:.2e}, P(loop|rollout) {100*k[2]/k[3]:.2f}%, E[size|loop] {k[4]:,.0f}, median {k[5]:,.0f}, loop/gen {k[6]:.2f}%, loop/reason {k[7]:.2f}%")
    for a in ("A","G","I","main"):
        vals=[J[(a,s)] for s in full if (a,s) in J]
        if vals: print(f"  {a}: step-mean loop/gen over the same steps = {np.mean(vals):.2f}%  (per step: {' '.join(f'{v:.1f}' for v in vals)})")
    print(f"  K: step-mean loop/gen = {np.mean([100*sum(blk[s]['rep'])/max(1,gen[s]) for s in full]):.2f}%")

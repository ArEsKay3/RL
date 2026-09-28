import glob,csv,json,re,collections
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
T="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/tokens"
EXP={}
for line in open(f"{L}/jobs.txt"):
    a,e=line.split()[:2]; EXP[a]=e
rowmap=json.load(open(f"{L}/dataset_row_map.json"))
# ---- main chain: all rollouts (metrics) + text scan for long-turn rollouts
main=collections.defaultdict(lambda:{"n":0,"gen":0,"rep_tok":0,"rep_turns":0,"rep_unclosed":0,"scanned":0,"ge30k":0,"reason_est":0,"rep_reason_est":0})
text={}
for fn in glob.glob(f"{L}/mainchain_text/*.csv"):
    for r in csv.DictReader(open(fn)): text[(r["results_dir"],r["instance_id"],r["epoch_ms"])]=r
n_all=n_mapped=0
for fn in glob.glob(f"{L}/mainchain/*.csv"):
    for r in csv.DictReader(open(fn)):
        n_all+=1
        rm=rowmap.get(r["instance_id"])
        if not rm: continue
        step=rm["steps"][0]; n_mapped+=1
        e=main[step]; e["n"]+=1; e["gen"]+=int(r["completion_tokens"]); e["ge30k"]+=int(int(r["turns_ge30k"])>0)
        t=text.get((r["results_dir"],r["instance_id"],r["epoch_ms"]))
        if t:
            e["scanned"]+=1; e["rep_tok"]+=int(t["rep_tokens"]); e["rep_turns"]+=int(t["rep_turns"]); e["rep_unclosed"]+=int(t["rep_unclosed"])
            e["reason_est"]+=int(t["reason_tokens_est"]); e["rep_reason_est"]+=int(t["rep_reason_tokens_est"])
# ---- dump arms: strict loop tokens (loopshare.csv) and generated tokens (rows.csv)
ls={}
for r in csv.DictReader(open(f"{L}/loopshare.csv")): ls[(r["arm"],int(r["step"]))]=r
gen=collections.Counter(); genrows=collections.Counter()
for a,exp in EXP.items():
    for fn in glob.glob(f"{T}/{exp}__step_*_chunk_*.rows.csv"):
        s=int(re.search(r"step_(\d+)_chunk",fn).group(1))
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                try: gen[(a,s)]+=int(float(r["n_gen_tokens"])); genrows[(a,s)]+=1
                except: pass
import sys
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load as _ptload
import numpy as _np
import os as _os
_cache_fn=f"{L}/joint_gen_cache.json"; _cache=json.load(open(_cache_fn)) if _os.path.exists(_cache_fn) else {}
for (a,s),r in list(ls.items()):
    if gen.get((a,s)) and genrows.get((a,s),0)>=int(r["rollouts"]): continue
    gen[(a,s)]=0
    for fn in glob.glob(f"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/{EXP[a]}/dumps/token_level/step_{s:05d}_chunk_*.pt"):
        key=f"{a}:{_os.path.basename(fn)}:{_os.path.getsize(fn)}"
        if key not in _cache: _cache[key]=int(_np.asarray(_ptload(fn)["token_mask"]).sum())
        gen[(a,s)]+=_cache[key]
json.dump(_cache,open(_cache_fn,"w"))
ARMS=["G","I","J","A","K","M","N","S","T","U","V","V2","V3","P4","P4b","Q4","Q4b","F","L","B","D"]
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P(f"LOOP-TOKEN SHARE WITH THE MAIN MINF CHAIN.  main chain: {n_all} Gym rollout results, {n_mapped} mapped to a step via the dataset row order (step = row // 32); {len(text)} rollouts with a turn >= 8k tokens scanned for repetitive turns.")
P("Definition used in EVERY column here: share = tokens in repetitive reasoning blocks / ALL generated tokens of the step's rollouts (denominator includes tool calls and answers).")
P("  repetitive block = reasoning block >= 1,000 tokens whose text zlib-compresses to < 10 %.  Dump arms: exact from token dumps (512 trained rollouts/step).")
P("  main chain: from OpenHands trajectories; a turn counts as repetitive if its completion tokens >= 1,000 and its <think> text compresses to < 10 %; only rollouts with a turn >= 8k tokens were scanned (>= 98 % of loop tokens in every dump arm sit in such blocks).")
P("  main-chain steps with ~1,024 rollouts contain rollouts regenerated after a segment restart (both attempts kept).")
P("  For reference the last two columns give the dump-arm share over REASONING tokens only (the strict share plotted earlier) for chain G and run A.")
P("")
hdr=f"{'step':>4} | {'main n':>6} {'main %':>7} {'rep turns':>9} {'unclosed':>8} | " + " ".join(f"{a+' %':>6}" for a in ARMS) + f" | {'G reas%':>7} {'A reas%':>7}"
P("arms: main = MINF from scratch (original chain) | G = chain G MINF from0 no-pfx | I = chain I MINF from0 pfx-on | J = chain J MINF from0 pfx-on replica 2 | A = run A vLLM from0 | K = chain K vLLM from0 seed 1234 | M = chain M vLLM from0 r1 (J args) | N = chain N vLLM from0 r2 (J args) | S = chain S vLLM from0, 9-15 stack, flagged-sample masking off | T = chain T MINF from0, 9-15 stack, masking off | U = chain U MINF from0, main915 stack (nemo_rl main@9-15 rebuild; BROKEN dumps, withheld) | V = chain V MINF from0 on the Megatron vLLM-numerical-parity adapter, prefix cache kept | V2 / V3 = seed 1234 / 4321 replicas of chain V | P4 = chain P⁗ MINF from0, prefix cache kept across weight updates (keep-prefix knob), nvshmem | P4b = chain P⁗2 second seed (1234) of P⁗ | Q4b = chain Q⁗2 second seed (1234) of Q⁗ | Q4 = chain Q⁗ vLLM from0 with vLLM prefix caching DISABLED (all earlier vLLM arms ran with it on) | F = chain F MINF from vLLM10 | L = chain L MINF from chain K step 10 (pfx on, seed 1234) | B = run B MINF from MINF10 | D = chain D vLLM from MINF10")
P(hdr)
steps=sorted(set(main)|{s for (_,s) in ls})
for s in steps:
    if s<1: continue
    m=main.get(s)
    mc=f"{m['n']:>6} {100*m['rep_tok']/max(1,m['gen']):>6.1f}% {m['rep_turns']:>9} {m['rep_unclosed']:>8}" if m else f"{'':>6} {'':>7} {'':>9} {'':>8}"
    cells=[]
    for a in ARMS:
        r=ls.get((a,s)); g=gen.get((a,s),0)
        cells.append(f"{100*int(r['repetitive_tokens'])/g:>5.1f}%" if (r and g) else f"{'':>6}")
    def reas(a):
        r=ls.get((a,s)); return f"{100*int(r['repetitive_tokens'])/max(1,int(r['reasoning_tokens'])):>6.1f}%" if r else f"{'':>7}"
    P(f"{s:>4} | {mc} | "+" ".join(cells)+f" | {reas('G')} {reas('A')}")
P("")
P("REPETITIVE-BLOCK COUNTS per step (main chain: repetitive turns among scanned rollouts; dump arms: repetitive blocks):")
P(f"{'step':>4} {'main':>5} " + " ".join(f"{a:>5}" for a in ARMS))
for s in steps:
    if s<1: continue
    m=main.get(s); cells=[f"{int(ls[(a,s)]['repetitive_blocks']):>5}" if (a,s) in ls else f"{'':>5}" for a in ARMS]
    P(f"{s:>4} {m['rep_turns'] if m else '':>5} "+" ".join(cells))
open(f"{L}/joint_loopshare.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))
# csv for plotting
with open(f"{L}/joint_loopshare.csv","w") as fh:
    w=csv.writer(fh); w.writerow(["arm","step","share_all_gen_pct"])
    for s in sorted(main):
        if s>=1 and main[s]["gen"]: w.writerow(["main",s,100*main[s]["rep_tok"]/main[s]["gen"]])
    for (a,s),r in sorted(ls.items()):
        if gen.get((a,s)): w.writerow([a,s,100*int(r["repetitive_tokens"])/gen[(a,s)]])

import json,glob,re,math,collections,sys
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
NAMES={"G":"chain G (MINF from0, no pfx)","I":"chain I (MINF from0, pfx on)","A":"run A (vLLM from0)","F":"chain F (MINF from vLLM10)","B":"run B (MINF from MINF10)","D":"chain D (vLLM from MINF10)"}
ORDER=["G","I","A","F","B","D"]
EDGES=[0.99,0.9,0.7,0.5,0.3,0.1,0.03,0.01]; NB=9
BINLAB=[">=.99",".9-.99",".7-.9",".5-.7",".3-.5",".1-.3",".03-.1",".01-.03","<.01"]
ZBINS=[0.03,0.06,0.10,0.15,0.20,0.25,0.30,0.40]
VLAB={"dot":"'.'","dnn":"'.\\n\\n'","dn":"'.\\n'","bqdot":"'`.'","pdot":"').'","q":"'?'","ex":"'!'","other":"other"}
agg=collections.defaultdict(dict)   # (arm,step) -> cls -> entry
def new():
    return {"n_blocks":0,"n_tokens":0,"n_sent_final":0,"hist":{},"lp":{},"next_close":{},"next_cont":{},"n_close_after":{},"close_hist":[0]*NB,"close_n":0,
            "long_blocks_zbin":[0]*(len(ZBINS)+1),"long_tokens_zbin":[0]*(len(ZBINS)+1)}
def addh(dst,src):
    for k,v in src.items():
        if k in dst: dst[k]=[a+b for a,b in zip(dst[k],v)]
        else: dst[k]=list(v)
def merge(e,v):
    for f in ("n_blocks","n_tokens","n_sent_final","close_n"): e[f]+=v[f]
    e["close_hist"]=[a+b for a,b in zip(e["close_hist"],v["close_hist"])]
    e["long_blocks_zbin"]=[a+b for a,b in zip(e["long_blocks_zbin"],v["long_blocks_zbin"])]
    e["long_tokens_zbin"]=[a+b for a,b in zip(e["long_tokens_zbin"],v["long_tokens_zbin"])]
    for k in ("hist","next_close","next_cont"): addh(e[k],v[k])
    for k,(n,s) in v["lp"].items():
        if k in e["lp"]: e["lp"][k][0]+=n; e["lp"][k][1]+=s
        else: e["lp"][k]=[n,s]
    for k,n in v["n_close_after"].items(): e["n_close_after"][k]=e["n_close_after"].get(k,0)+n
nfiles=0
for fn in glob.glob(f"{L}/sfp/*.sfp.json"):
    m=re.search(r"([A-Z])__s(\d+)_c(\d+)\.sfp\.json",fn); arm,step=m.group(1),int(m.group(2)); nfiles+=1
    P=json.load(open(fn))
    for cls,v in P.items():
        e=agg[(arm,step)].get(cls)
        if e is None: e=agg[(arm,step)][cls]=new()
        merge(e,v)
def pooled(arm,classes,steps=None):
    e=new()
    for (a,s),D in agg.items():
        if a!=arm or (steps is not None and s not in steps): continue
        for cls,v in D.items():
            if cls in classes: merge(e,v)
    return e
def sh(h,lo=None,hi=None):
    n=sum(h)
    return (sum(h[lo:hi])/n) if n else float('nan')
def gm(e,v):
    n,s=e["lp"].get(v,[0,0.0]); return math.exp(s/n) if n else float('nan')
def median_bin(h):
    n=sum(h); c=0
    for i,x in enumerate(h):
        c+=x
        if c>=n/2: return BINLAB[i]
    return "-"
out=[]
def P(*a): out.append(" ".join(str(x) for x in a))
P(f"SENTENCE-FINAL TOKEN PROBABILITIES ({nfiles} chunk files). p = probability the sampling engine assigned to the token it actually emitted (from generation_logprobs).")
P("Bins of p: >=.99 | .9-.99 | .7-.9 | .5-.7 | .3-.5 | .1-.3 | .03-.1 | .01-.03 | <.01.  'geo-mean p' = exp(mean log p).")
P("Block classes: normal = ordinary reasoning block; looplike = >=1000 tokens, zlib ratio < 0.10, closed with </think>; runaway = unclosed last block of a context-truncated rollout.")
P("")
P("="*100); P("A. NORMAL blocks, all steps pooled, per arm: how sure is the model when it emits each sentence-final variant?"); P("="*100)
for v in ("dot","dnn","dn","bqdot","pdot","q","ex"):
    P(f"\n--- token {VLAB[v]} ---")
    P(f"{'arm':<30} {'count':>10} {'share of sent-finals':>20} {'geo-mean p':>10} | {'p>=.99':>7} {'p>=.9':>7} {'p<.5':>7} {'p<.1':>7} {'p<.01':>7} | {'median bin':>10}")
    for arm in ORDER:
        e=pooled(arm,{"normal"}); h=e["hist"].get(v)
        if not h: continue
        P(f"{NAMES[arm]:<30} {sum(h):>10,} {sum(h)/max(1,e['n_sent_final']):>20.1%} {gm(e,v):>10.3f} | {sh(h,0,1):>7.1%} {sh(h,0,2):>7.1%} {sh(h,4,None):>7.1%} {sh(h,6,None):>7.1%} {sh(h,8,None):>7.1%} | {median_bin(h):>10}")
P(""); P("="*100); P("B. Same, by block class (all steps pooled): '.' and '.\\n\\n' inside normal vs loop-like vs runaway blocks"); P("="*100)
P(f"{'arm':<30} {'class':<9} {'n blocks':>9} | {'.: count':>10} {'geo p':>6} {'p>=.99':>7} {'p<.5':>6} {'p<.1':>6} | {'.nn: count':>10} {'geo p':>6} {'p>=.99':>7} {'p<.5':>6} {'p<.1':>6} | {'share .nn of (.+.nn)':>20}")
for arm in ORDER:
    for cls in ("normal","looplike","runaway"):
        e=pooled(arm,{cls}); hd=e["hist"].get("dot",[0]*NB); hn=e["hist"].get("dnn",[0]*NB)
        if sum(hd)==0: continue
        P(f"{NAMES[arm]:<30} {cls:<9} {e['n_blocks']:>9,} | {sum(hd):>10,} {gm(e,'dot'):>6.3f} {sh(hd,0,1):>7.1%} {sh(hd,4,None):>6.1%} {sh(hd,6,None):>6.1%} | {sum(hn):>10,} {gm(e,'dnn'):>6.3f} {sh(hn,0,1):>7.1%} {sh(hn,4,None):>6.1%} {sh(hn,6,None):>6.1%} | {sum(hn)/max(1,sum(hd)+sum(hn)):>20.1%}")
P(""); P("="*100); P("C. The token AFTER a bare '.' (normal blocks, all steps pooled): p of </think> when the model closed, p of the continuation token when it did not"); P("="*100)
dotcount="'.' count"
P(f"{'arm':<30} {dotcount:>10} {'-> </think>':>11} | {'p(</think>) geo':>15} {'p>=.99':>7} {'p<.5':>6} | {'p(next word) geo':>16} {'p>=.99':>7} {'p<.5':>6}")
for arm in ORDER:
    e=pooled(arm,{"normal"}); hc=e["next_close"].get("dot",[0]*NB); hk=e["next_cont"].get("dot",[0]*NB); hd=e["hist"].get("dot",[0]*NB)
    if sum(hd)==0: continue
    def gmh(h):
        # approximate geo-mean from bin centers
        cents=[0.995,0.945,0.8,0.6,0.4,0.2,0.06,0.02,0.005]; n=sum(h)
        return math.exp(sum(c*math.log(p) for c,p in zip(h,cents))/n) if n else float('nan')
    P(f"{NAMES[arm]:<30} {sum(hd):>10,} {sum(hc)/max(1,sum(hd)):>11.1%} | {gmh(hc):>15.3f} {sh(hc,0,1):>7.1%} {sh(hc,4,None):>6.1%} | {gmh(hk):>16.3f} {sh(hk,0,1):>7.1%} {sh(hk,4,None):>6.1%}")
P(""); P("="*100); P("D. PER STEP, NORMAL blocks: bare '.'  -> geo-mean p and share of '.' emitted with p < 0.5 (an uncertain sentence end)"); P("="*100)
steps=sorted({s for (_,s) in agg})
P(f"{'step':>4} " + " ".join(f"{'  '+a+': geo p  %p<.5':>19}" for a in ORDER))
for s in steps:
    cells=[]
    for a in ORDER:
        D=agg.get((a,s),{}).get("normal")
        if not D or not D["hist"].get("dot"): cells.append(f"{'':>19}"); continue
        h=D["hist"]["dot"]; cells.append(f"{gm(D,'dot'):>12.3f} {sh(h,4,None):>6.1%}")
    P(f"{s:>4} "+" ".join(cells))
P(""); P("="*100); P("E. PER STEP, NORMAL blocks: '.\\n\\n' -> geo-mean p and share emitted with p < 0.5"); P("="*100)
P(f"{'step':>4} " + " ".join(f"{'  '+a+': geo p  %p<.5':>19}" for a in ORDER))
for s in steps:
    cells=[]
    for a in ORDER:
        D=agg.get((a,s),{}).get("normal")
        if not D or not D["hist"].get("dnn"): cells.append(f"{'':>19}"); continue
        h=D["hist"]["dnn"]; cells.append(f"{gm(D,'dnn'):>12.3f} {sh(h,4,None):>6.1%}")
    P(f"{s:>4} "+" ".join(cells))
P(""); P("="*100); P("F. PER STEP, NORMAL blocks: the </think> token -> geo-mean p (approx. from bins) and share of closes sampled at p < 0.5"); P("="*100)
P(f"{'step':>4} " + " ".join(f"{'  '+a+': geo p  %p<.5':>19}" for a in ORDER))
cents=[0.995,0.945,0.8,0.6,0.4,0.2,0.06,0.02,0.005]
for s in steps:
    cells=[]
    for a in ORDER:
        D=agg.get((a,s),{}).get("normal")
        if not D or not D["close_n"]: cells.append(f"{'':>19}"); continue
        h=D["close_hist"]; g=math.exp(sum(c*math.log(p) for c,p in zip(h,cents))/sum(h))
        cells.append(f"{g:>12.3f} {sh(h,4,None):>6.1%}")
    P(f"{s:>4} "+" ".join(cells))
P(""); P("="*100); P("G. PER STEP: reasoning blocks >= 1000 tokens by zlib compression ratio (all classes). loops < 0.10 | semi-repetitive 0.10-0.20 | ordinary >= 0.20.  tok% = share of ALL reasoning tokens of the step."); P("="*100)
for arm in ORDER:
    P(f"\n{NAMES[arm]}")
    P(f"{'step':>4} {'blocks':>7} {'reason tok':>11} | {'long blks':>9} {'z<.10':>6} {'.10-.20':>7} {'>=.20':>6} | {'tok% z<.10':>10} {'tok% .10-.20':>12} {'tok% >=.20':>10} |")
    for s in steps:
        D=agg.get((arm,s))
        if not D: continue
        nb=sum(v["n_blocks"] for v in D.values()); nt=sum(v["n_tokens"] for v in D.values())
        lb=[0]*(len(ZBINS)+1); lt=[0]*(len(ZBINS)+1)
        for v in D.values():
            lb=[a+b for a,b in zip(lb,v["long_blocks_zbin"])]; lt=[a+b for a,b in zip(lt,v["long_tokens_zbin"])]
        loops=sum(lb[:3]); semi=sum(lb[3:5]); ordn=sum(lb[5:]); tl=sum(lt[:3]); ts=sum(lt[3:5]); to=sum(lt[5:])
        P(f"{s:>4} {nb:>7,} {nt:>11,} | {sum(lb):>9,} {loops:>6} {semi:>7} {ordn:>6} | {tl/max(1,nt):>10.1%} {ts/max(1,nt):>12.1%} {to/max(1,nt):>10.1%} |")
open(f"{L}/sfp_report.txt","w").write("\n".join(out)+"\n")
print("\n".join(out))

import sys,os,json,re,math,zlib
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
HF="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf"
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}"
if os.path.exists(base+".sfp.json"): sys.exit(0)
t=json.load(open(f"{HF}/tokenizer.json")); inv={v:k for k,v in t["model"]["vocab"].items()}; special={}
for a in t.get("added_tokens",[]): inv[a["id"]]=a["content"]; special[a["id"]]=a["content"]
def b2u():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1)); cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return dict(zip([chr(c) for c in cs],bs))
u2b=b2u(); cache={}
def dec(i):
    i=int(i); s=cache.get(i)
    if s is None:
        raw=inv.get(i,"")
        if i in special: s=raw
        else:
            try: s=bytes(u2b[c] for c in raw).decode("utf-8","replace")
            except KeyError: s=raw
        cache[i]=s
    return s
IM_END,THINK_END=11,13
PUNCT=set(".!?")
def is_sent_final(s):
    st=s.strip()
    return bool(st) and all(ch in ".!?)\"'*`" for ch in st) and any(ch in PUNCT for ch in st)
VARS={"'.'":"dot","'.\\n\\n'":"dnn","'.\\n'":"dn","'`.'":"bqdot","').'":"pdot","'?'":"q","'!'":"ex"}
def var_of(s): return VARS.get(repr(s),"other")
EDGES=[0.99,0.9,0.7,0.5,0.3,0.1,0.03,0.01]
NB=len(EDGES)+1
def pbin(lp):
    p=math.exp(lp)
    for i,e in enumerate(EDGES):
        if p>=e: return i
    return NB-1
ZBINS=[0.03,0.06,0.10,0.15,0.20,0.25,0.30,0.40]
def zbin(z):
    for i,e in enumerate(ZBINS):
        if z<e: return i
    return len(ZBINS)
agg={}
def A_(cls):
    e=agg.get(cls)
    if e is None:
        e=agg[cls]={"n_blocks":0,"n_tokens":0,"n_sent_final":0,"hist":{},"lp":{},"next_close":{},"next_cont":{},"n_close_after":{},"close_hist":[0]*NB,"close_n":0,
                    "long_blocks_zbin":[0]*(len(ZBINS)+1),"long_tokens_zbin":[0]*(len(ZBINS)+1)}
    return e
def H(d,k):
    h=d.get(k)
    if h is None: h=d[k]=[0]*NB
    return h
obj=load(chunk)
lengths=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(lengths)]); lp_off=np.concatenate([[0],np.cumsum(lengths-1)])
mask=np.asarray(obj["token_mask"]).astype(bool); gen=np.asarray(obj["generation_logprobs"]).astype(np.float32)
ids=np.asarray(obj["input_ids"]); sids=list(obj["sample_ids"]); tags=obj.get("tags") or [{}]*len(sids)
for r in range(len(sids)):
    mk=mask[ids_off[r]+1:ids_off[r+1]]; g=gen[lp_off[r]:lp_off[r+1]][mk]; tk=ids[ids_off[r]+1:ids_off[r+1]][mk]
    if len(tk)==0: continue
    tg=tags[r] or {}; trunc=bool(tg.get("rollout_truncated"))
    ends=np.where(tk==IM_END)[0]; starts=np.concatenate([[0],ends+1]); stops=np.concatenate([ends,[len(tk)]])
    blocks=[]
    for a,b in zip(starts,stops):
        if b<=a: continue
        seg=tk[a:b]; ce=np.where(seg==THINK_END)[0]; r_end=a+int(ce[0]) if len(ce) else b
        if r_end>a: blocks.append((a,r_end,len(ce)>0))
    for bi,(a,b,closed) in enumerate(blocks):
        a=int(a); b=int(b); toks=tk[a:b]; lps=g[a:b]; n=b-a
        strs=[dec(x) for x in toks]
        tb="".join(strs).encode("utf-8","replace"); zr=len(zlib.compress(tb,6))/max(1,len(tb))
        runaway=(not closed) and trunc and bi==len(blocks)-1
        cls="runaway" if runaway else ("looplike" if (n>=1000 and zr<0.10) else "normal")
        A=A_(cls); A["n_blocks"]+=1; A["n_tokens"]+=n
        if n>=1000:
            zb=zbin(zr); A["long_blocks_zbin"][zb]+=1; A["long_tokens_zbin"][zb]+=n
        for i,s in enumerate(strs):
            if not is_sent_final(s): continue
            v=var_of(s); A["n_sent_final"]+=1
            H(A["hist"],v)[pbin(float(lps[i]))]+=1
            e=A["lp"].get(v)
            if e is None: e=A["lp"][v]=[0,0.0]
            e[0]+=1; e[1]+=float(lps[i])
            if i==n-1:
                if closed and b<len(g):
                    H(A["next_close"],v)[pbin(float(g[b]))]+=1; A["n_close_after"][v]=A["n_close_after"].get(v,0)+1
            else:
                H(A["next_cont"],v)[pbin(float(lps[i+1]))]+=1
        if closed and b<len(g):
            A["close_n"]+=1; A["close_hist"][pbin(float(g[b]))]+=1
json.dump(agg,open(base+".sfp.json","w"))
print("done",arm,step,ck,flush=True)

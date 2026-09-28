import sys,os,json,re,math,collections
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
HF="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf"
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}"
if os.path.exists(base+".punct.json"): sys.exit(0)
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
obj=load(chunk)
lengths=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(lengths)]); lp_off=np.concatenate([[0],np.cumsum(lengths-1)])
mask=np.asarray(obj["token_mask"]).astype(bool); gen=np.asarray(obj["generation_logprobs"]).astype(np.float32); advs=np.asarray(obj["advantages"]).astype(np.float32)
ids=np.asarray(obj["input_ids"]); sids=list(obj["sample_ids"]); tags=obj.get("tags") or [{}]*len(sids)
agg={}
def A_(cls,wv):
    k=f"{cls}|{wv}"
    if k not in agg: agg[k]={"n_blocks":0,"n_blocks_closed":0,"n_sent_final":0,"var":{},"close_prev":{},"close_lp_sum":0.0,"close_n":0,"close_lp_hist":[0]*8}
    return agg[k]
def bump(d,key,adv,lp,nxt):
    e=d.get(key)
    p=math.exp(lp)
    if e is None: e=d[key]=[0,0.0,0.0,0.0,0.0,0,0,0]   # count, sum adv, sum adv(1-p), sum |adv|(1-p), sum lp, next=</think>, next=newline-ish, next=other
    e[0]+=1; e[1]+=adv; e[2]+=adv*(1-p); e[3]+=abs(adv)*(1-p); e[4]+=lp; e[5+nxt]+=1
for r in range(len(sids)):
    mk=mask[ids_off[r]+1:ids_off[r+1]]; g=gen[lp_off[r]:lp_off[r+1]][mk]; tk=ids[ids_off[r]+1:ids_off[r+1]][mk]
    if len(tk)==0: continue
    adv=float(advs[lp_off[r]:lp_off[r+1]][mk][0]); tg=tags[r] or {}; wv=tg.get("weight_version"); trunc=bool(tg.get("rollout_truncated"))
    ends=np.where(tk==IM_END)[0]; starts=np.concatenate([[0],ends+1]); stops=np.concatenate([ends,[len(tk)]])
    blocks=[]
    for a,b in zip(starts,stops):
        if b<=a: continue
        seg=tk[a:b]; ce=np.where(seg==THINK_END)[0]; r_end=a+int(ce[0]) if len(ce) else b
        if r_end>a: blocks.append((a,r_end,len(ce)>0))
    for bi,(a,b,closed) in enumerate(blocks):
        toks=tk[a:b]; lps=g[a:b]; n=b-a
        # class: reuse the zlib rule cheaply via decoded text
        strs=[dec(x) for x in toks]
        import zlib
        tb="".join(strs).encode("utf-8","replace"); zr=len(zlib.compress(tb,6))/max(1,len(tb))
        runaway=(not closed) and trunc and bi==len(blocks)-1
        cls="runaway" if runaway else ("looplike" if (n>=1000 and zr<0.10) else "normal")
        A=A_(cls,wv); A["n_blocks"]+=1; A["n_blocks_closed"]+=int(closed)
        for i,s in enumerate(strs):
            if not is_sent_final(s): continue
            # next token: </think> if this is the last token of a closed block
            if i==n-1: nxt=0 if closed else 2
            else:
                ns=strs[i+1]; nxt=1 if ns.strip()=="" and "\n" in ns else 2
            A["n_sent_final"]+=1
            bump(A["var"],repr(s),adv,float(lps[i]),nxt)
        if closed:
            # the </think> token itself: previous token variant and its logprob
            ci=b  # index of </think> in tk
            lpc=float(g[ci]) if ci<len(g) else 0.0
            prev=repr(strs[-1]) if strs else "''"
            A["close_n"]+=1; A["close_lp_sum"]+=lpc
            hb=min(7,int(-lpc/0.5)) if lpc<0 else 0; A["close_lp_hist"][hb]+=1
            e=A["close_prev"].get(prev)
            if e is None: A["close_prev"][prev]=[0,0.0,0.0]
            A["close_prev"][prev][0]+=1; A["close_prev"][prev][1]+=lpc; A["close_prev"][prev][2]+=adv*(1-math.exp(lpc))
json.dump(agg,open(base+".punct.json","w"))
print("done",arm,step,ck,flush=True)

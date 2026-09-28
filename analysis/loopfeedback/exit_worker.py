import sys,os,json,re,math,zlib,collections
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2)); stem=f"{arm}__s{step:03d}_c{ck:03d}"
tf=f"{L}/targets/{stem}.txt"
if not os.path.exists(tf): sys.exit(0)
out=f"{L}/exits/{stem}.csv"
if os.path.exists(out): sys.exit(0)
targets={}
for line in open(tf):
    sid,ii,cls=line.split(); targets.setdefault(sid,{})[int(ii)]=cls
HF="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf"
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
def is_sf(s):
    st=s.strip(); return bool(st) and all(ch in ".!?)\"'*`" for ch in st) and any(ch in ".!?" for ch in st)
SENT=re.compile(r"(?<=[.!?])\s+|\n+")
obj=load(chunk)
lengths=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(lengths)]); lp_off=np.concatenate([[0],np.cumsum(lengths-1)])
mask=np.asarray(obj["token_mask"]).astype(bool); gen=np.asarray(obj["generation_logprobs"]).astype(np.float32); advs=np.asarray(obj["advantages"]).astype(np.float32)
ids=np.asarray(obj["input_ids"]); rewards=np.asarray(obj["rewards"]).astype(np.float32); sids=list(obj["sample_ids"]); tags=obj.get("tags") or [{}]*len(sids)
w=open(out+".tmp","w"); w.write("arm,step,sample_id,item_idx,cls,n_tok,zlib,adv,reward,has_close,p_close,exit_tok,p_exit_tok,exit_tok2,p_exit_tok2,n_dot,p_dot_mean,n_dnn,p_dnn_mean,share_dnn_sf,minp_tok,minp,minp_posfrac,n_low_p_lt01,cycle_len,cycle_match,top_sent,top_n\n")
for r in range(len(sids)):
    sid=sids[r]
    if sid not in targets: continue
    mk=mask[ids_off[r]+1:ids_off[r+1]]; g=gen[lp_off[r]:lp_off[r+1]][mk]; tk=ids[ids_off[r]+1:ids_off[r+1]][mk]
    adv=float(advs[lp_off[r]:lp_off[r+1]][mk][0]); rew=float(rewards[r])
    ends=np.where(tk==IM_END)[0]; starts=np.concatenate([[0],ends+1]); stops=np.concatenate([ends,[len(tk)]])
    blocks=[]
    for a,b in zip(starts,stops):
        if b<=a: continue
        seg=tk[a:b]; ce=np.where(seg==THINK_END)[0]; r_end=a+int(ce[0]) if len(ce) else b
        if r_end>a: blocks.append((a,r_end,len(ce)>0))
    for ii,cls in targets[sid].items():
        if ii>=len(blocks): continue
        a,b,closed=blocks[ii]; toks=tk[a:b]; lps=g[a:b]; n=b-a; strs=[dec(x) for x in toks]; text="".join(strs)
        tb=text.encode("utf-8","replace"); zr=len(zlib.compress(tb,6))/max(1,len(tb))
        p=np.exp(lps)
        sf=[i for i,s in enumerate(strs) if is_sf(s)]
        dots=[i for i in sf if strs[i]=="."]; dnn=[i for i in sf if strs[i]==".\n\n"]
        p_close=float(np.exp(g[b])) if closed and b<len(g) else float('nan')
        exit_tok=repr(strs[-1]) if strs else ""; p_exit=float(p[-1]) if n else float('nan')
        exit_tok2=repr(strs[-2]) if n>1 else ""; p_exit2=float(p[-2]) if n>1 else float('nan')
        mi=int(np.argmin(p)) if n else 0
        idsl=toks.astype(np.int64); best=(0,0.0)
        if n>800:
            for lag in range(60,min(3000,n//3),1):
                mm=(idsl[lag:]==idsl[:-lag]).mean()
                if mm>best[1]: best=(lag,mm)
        sents=[s.strip() for s in SENT.split(text) if len(s.strip())>20]; top=collections.Counter(sents).most_common(1); top_s,top_n=(top[0] if top else ("",0))
        w.write(",".join(map(str,[arm,step,sid,ii,cls,n,f"{zr:.4f}",f"{adv:.3f}",f"{rew:.0f}",int(closed),f"{p_close:.5f}",json.dumps(exit_tok),f"{p_exit:.5f}",json.dumps(exit_tok2),f"{p_exit2:.5f}",len(dots),f"{float(p[dots].mean()) if dots else float('nan'):.4f}",len(dnn),f"{float(p[dnn].mean()) if dnn else float('nan'):.4f}",f"{len(dnn)/max(1,len(sf)):.3f}",json.dumps(repr(strs[mi])),f"{float(p[mi]):.6f}",f"{mi/max(1,n):.3f}",int((p<0.1).sum()),best[0],f"{best[1]:.3f}",json.dumps(top_s[:120]),top_n]))+"\n")
w.close(); os.replace(out+".tmp",out); print("done",stem,flush=True)

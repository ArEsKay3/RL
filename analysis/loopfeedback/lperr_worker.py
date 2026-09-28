import sys,os,json,re,zlib
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
HF="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf"
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}"
if os.path.exists(base+".lperr.npz"): sys.exit(0)
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
EDGES=np.array([1e-3,1e-2,0.1,0.5,0.9,0.99]); NB=len(EDGES)+1; NS=11
obj=load(chunk)
lengths=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(lengths)]); lp_off=np.concatenate([[0],np.cumsum(lengths-1)])
mask=np.asarray(obj["token_mask"]).astype(bool); gen=np.asarray(obj["generation_logprobs"]).astype(np.float64); prev=np.asarray(obj["prev_logprobs"]).astype(np.float64)
adv_all=np.asarray(obj["advantages"]).astype(np.float64); ids=np.asarray(obj["input_ids"]); sids=list(obj["sample_ids"])
smask=np.asarray(obj["sample_mask_after"]).astype(np.float64)
T=np.zeros((3,2,NB,NS)); E=np.zeros((3,2,NB,NS)); masked=np.zeros(2)
def accum(out,keyprob,sel,g,p,d,w_raw,w,pt,pe):
    if not sel.any(): return
    b=np.digitize(keyprob[sel],EDGES)
    cols=[np.ones(sel.sum()),d[sel],np.abs(d[sel]),pe[sel],pt[sel],w[sel],w[sel]*(1-pt[sel]),1-pt[sel],(w_raw[sel]<0.2).astype(float),(w_raw[sel]>5).astype(float),(d[sel]>0).astype(float)]
    for k,c in enumerate(cols): out[:,k]+=np.bincount(b,weights=c,minlength=NB)
for r in range(len(sids)):
    mk=mask[ids_off[r]+1:ids_off[r+1]]; tk=ids[ids_off[r]+1:ids_off[r+1]][mk]; g=gen[lp_off[r]:lp_off[r+1]][mk]; p=prev[lp_off[r]:lp_off[r+1]][mk]
    n=len(tk)
    if n==0: continue
    adv=float(adv_all[lp_off[r]:lp_off[r+1]][mk][0]); sgn=0 if adv>1e-9 else (1 if adv<-1e-9 else 2)
    ends=np.where(tk==IM_END)[0]; starts=np.concatenate([[0],ends+1]); stops=np.concatenate([ends,[n]])
    loop=np.zeros(n,dtype=bool)
    for a,b_ in zip(starts,stops):
        if b_<=a: continue
        seg=tk[a:b_]; ce=np.where(seg==THINK_END)[0]; r_end=a+int(ce[0]) if len(ce) else b_
        if r_end-a<1000: continue
        tb="".join(dec(x) for x in tk[a:r_end]).encode("utf-8","replace")
        if len(zlib.compress(tb,6))/max(1,len(tb))<0.10: loop[a:r_end]=True
    if smask[r]<0.5: masked[0]+=1; masked[1]+=n; continue
    d=g-p; w_raw=np.exp(np.clip(-d,-50,50)); w=np.clip(w_raw,0.2,5.0); pt=np.exp(p); pe=np.exp(g)
    for c,sel in ((1,loop),(0,~loop)):
        accum(T[sgn,c],pt,sel,g,p,d,w_raw,w,pt,pe); accum(E[sgn,c],pe,sel,g,p,d,w_raw,w,pt,pe)
np.savez(base+".lperr.npz",T=T,E=E,masked=masked,edges=EDGES)
print("done",arm,step,ck,flush=True)

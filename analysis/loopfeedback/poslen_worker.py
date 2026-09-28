import sys,os,json,re,zlib,csv
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
HF="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf"
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}"
if os.path.exists(base+".poslen.json"): sys.exit(0)
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
EDGES=np.array([8000,16000,32000,64000,96000,128000,160000],dtype=np.int64)
NB=len(EDGES)+1
CTX=196608
obj=load(chunk)
lengths=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(lengths)]); lp_off=np.concatenate([[0],np.cumsum(lengths-1)])
mask=np.asarray(obj["token_mask"]).astype(bool); gen=np.asarray(obj["generation_logprobs"]).astype(np.float64); prev=np.asarray(obj["prev_logprobs"]).astype(np.float64)
ids=np.asarray(obj["input_ids"]); sids=list(obj["sample_ids"]); tags=obj.get("tags") or [{}]*len(sids)
Z=lambda:np.zeros(NB)
tok={c:{"n":Z(),"abs_err":Z(),"err":Z(),"gen":Z(),"prev":Z(),"err_gt1":Z()} for c in ("normal","rep")}
blk={"n":Z(),"n_long":Z(),"n_rep":Z(),"rep_tok":Z(),"rep_closed":Z(),"norm_tok":Z()}
close={"n":Z(),"lp":Z()}
loops=[]
for r in range(len(sids)):
    L=int(lengths[r]); mk=mask[ids_off[r]+1:ids_off[r+1]]; pos=np.nonzero(mk)[0]+1
    if len(pos)==0: continue
    tk=ids[ids_off[r]+1:ids_off[r+1]][mk]; g=gen[lp_off[r]:lp_off[r+1]][mk]; p=prev[lp_off[r]:lp_off[r+1]][mk]
    trunc=bool((tags[r] or {}).get("rollout_truncated"))
    ends=np.where(tk==IM_END)[0]; starts=np.concatenate([[0],ends+1]); stops=np.concatenate([ends,[len(tk)]])
    cls=np.zeros(len(tk),dtype=np.int8)
    blocks=[]
    for a,b in zip(starts,stops):
        if b<=a: continue
        seg=tk[a:b]; ce=np.where(seg==THINK_END)[0]; r_end=a+int(ce[0]) if len(ce) else b
        if r_end>a: blocks.append((int(a),int(r_end),len(ce)>0))
    for bi,(a,b,closed) in enumerate(blocks):
        n=b-a; strs=[dec(x) for x in tk[a:b]]; tb="".join(strs).encode("utf-8","replace"); zr=len(zlib.compress(tb,6))/max(1,len(tb))
        rep=(n>=1000 and zr<0.10); ab=int(pos[a]); bn=int(np.searchsorted(EDGES,ab,side="right"))
        blk["n"][bn]+=1
        if n>=1000: blk["n_long"][bn]+=1
        if rep:
            blk["n_rep"][bn]+=1; blk["rep_tok"][bn]+=n; blk["rep_closed"][bn]+=int(closed); cls[a:b]=1
            loops.append([arm,step,sids[r],ab,n,int(closed),f"{zr:.4f}",CTX-ab,int(trunc),L])
        else:
            blk["norm_tok"][bn]+=n
        if closed and b<len(tk) and tk[b]==THINK_END and not rep:
            cb=int(np.searchsorted(EDGES,int(pos[b]),side="right")); close["n"][cb]+=1; close["lp"][cb]+=float(g[b])
    bins=np.searchsorted(EDGES,pos,side="right")
    err=g-p
    for c,sel in (("normal",cls==0),("rep",cls==1)):
        if not sel.any(): continue
        bb=bins[sel]; e=err[sel]
        tok[c]["n"]+=np.bincount(bb,minlength=NB); tok[c]["abs_err"]+=np.bincount(bb,weights=np.abs(e),minlength=NB); tok[c]["err"]+=np.bincount(bb,weights=e,minlength=NB)
        tok[c]["gen"]+=np.bincount(bb,weights=g[sel],minlength=NB); tok[c]["prev"]+=np.bincount(bb,weights=p[sel],minlength=NB); tok[c]["err_gt1"]+=np.bincount(bb,weights=(np.abs(e)>1).astype(float),minlength=NB)
out={"tok":{c:{k:v.tolist() for k,v in d.items()} for c,d in tok.items()},"blk":{k:v.tolist() for k,v in blk.items()},"close":{k:v.tolist() for k,v in close.items()}}
json.dump(out,open(base+".poslen.json","w"))
with open(base+".loops.csv","w") as fh:
    w=csv.writer(fh); w.writerow(["arm","step","sample_id","abs_start","n_tok","closed","zlib","room","rollout_truncated","seq_len"]); w.writerows(loops)
print("done",arm,step,ck,flush=True)

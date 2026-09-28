import sys,os,re,glob
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}.tokexp.npz"
if os.path.exists(base): sys.exit(0)
V=131072
obj=load(chunk)
L=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(L)]); lp_off=np.concatenate([[0],np.cumsum(L-1)])
ids=np.asarray(obj["input_ids"]).astype(np.int64); tm=np.asarray(obj["token_mask"]).astype(bool); advs=np.asarray(obj["advantages"]).astype(np.float64); gen=np.asarray(obj["generation_logprobs"]).astype(np.float64)
cnt=np.zeros(V); ea=np.zeros(V); cpos=np.zeros(V); cneg=np.zeros(V); cnt_long=np.zeros(V); ea_long=np.zeros(V); lpsum=np.zeros(V)
for i in range(len(L)):
    row=ids[ids_off[i]:ids_off[i+1]]; mrow=tm[ids_off[i]:ids_off[i+1]]; a=advs[lp_off[i]:lp_off[i+1]]; g=gen[lp_off[i]:lp_off[i+1]]
    tok=row[1:][mrow[1:]]; av=a[mrow[1:]]; lp=g[mrow[1:]]
    cnt+=np.bincount(tok,minlength=V); ea+=np.bincount(tok,weights=av,minlength=V); lpsum+=np.bincount(tok,weights=lp,minlength=V)
    cpos+=np.bincount(tok[av>1e-9],minlength=V); cneg+=np.bincount(tok[av<-1e-9],minlength=V)
    edges=np.diff(np.concatenate([[0],mrow.astype(np.int8),[0]])); starts=np.where(edges==1)[0]; ends=np.where(edges==-1)[0]
    for s,e in zip(starts,ends):
        if 4000<=e-s<16000:
            seg=row[s:e]; sa=a[s-1:e-1]; cnt_long+=np.bincount(seg,minlength=V); ea_long+=np.bincount(seg,weights=sa,minlength=V)
np.savez_compressed(base+".tmp.npz",cnt=cnt,ea=ea,cpos=cpos,cneg=cneg,cnt_long=cnt_long,ea_long=ea_long,lpsum=lpsum); os.replace(base+".tmp.npz",base); print("done",arm,step,ck)

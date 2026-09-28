import sys,os,re,glob,csv,json
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}"
if os.path.exists(base+".eff.csv"): sys.exit(0)
zl={}
with open(base+".turns.csv") as fh:
    for r in csv.DictReader(fh): zl[(r["sample_id"],int(r["turn"]))]=(float(r["zlib_reason"]) if r["zlib_reason"] else 1.0)
obj=load(chunk)
L=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(L)]); lp_off=np.concatenate([[0],np.cumsum(L-1)])
ids=np.asarray(obj["input_ids"]).astype(np.int64); tm=np.asarray(obj["token_mask"]).astype(bool)
gen=np.asarray(obj["generation_logprobs"]).astype(np.float64); prev=np.asarray(obj["prev_logprobs"]).astype(np.float64); advs=np.asarray(obj["advantages"]).astype(np.float64); rewards=np.asarray(obj["rewards"]).astype(np.float64); sids=list(obj["sample_ids"])
BUCK=[(0,256),(256,1024),(1024,4096),(4096,16384),(16384,10**9)]
agg={}
def A(cls,b,key,val):
    k=f"{cls}|{b}|{key}"; agg[k]=agg.get(k,0.0)+float(val)
out=open(base+".eff.csv.tmp","w"); out.write("sample_id,turn,adv,reward,n_tok,think_tok,zlib,cls,s1mp_think_tr,s1mp_think_eng,s1mp_entry_tr,s1mp_deep_tr,s1mp_answer_tr,close_p_tr,close_p_eng,n_lowp_think,mean_ptr_think\n")
for i,sid in enumerate(sids):
    row=ids[ids_off[i]:ids_off[i+1]]; mrow=tm[ids_off[i]:ids_off[i+1]]; n=len(row)
    p=prev[lp_off[i]:lp_off[i+1]]; g=gen[lp_off[i]:lp_off[i+1]]; a=advs[lp_off[i]:lp_off[i+1]]
    edges=np.diff(np.concatenate([[0],mrow.astype(np.int8),[0]])); starts=np.where(edges==1)[0]; ends=np.where(edges==-1)[0]
    adv=float(a[starts[0]-1]) if len(starts) else 0.0; rew=float(rewards[i])
    for k,(s,e) in enumerate(zip(starts,ends)):
        seg=row[s:e]; ptr=np.exp(p[s-1:e-1]); peng=np.exp(g[s-1:e-1]); d=np.abs(p[s-1:e-1]-g[s-1:e-1]); lp_tr=p[s-1:e-1]
        w13=np.where(seg==13)[0]; tt=int(w13[0]) if len(w13) else len(seg)
        z=zl.get((sid,k),1.0); cls="loop" if (tt>=1000 and z<0.10) else ("near" if (tt>=1000 and z<0.25) else ("normal_long" if tt>=1000 else "short"))
        th=ptr[:tt]; s1=(1-th); s1e=(1-peng[:tt])
        ans=ptr[tt+1:] if len(w13) else ptr[0:0]
        close_tr=float(ptr[tt]) if len(w13) else float("nan"); close_eng=float(peng[tt]) if len(w13) else float("nan")
        out.write(f"{sid},{k},{adv:.5f},{rew:.0f},{len(seg)},{tt},{z:.4f},{cls},{s1.sum():.4f},{s1e.sum():.4f},{s1[:256].sum():.4f},{s1[4096:].sum():.4f},{(1-ans).sum():.4f},{close_tr:.5f},{close_eng:.5f},{int((th<0.5).sum())},{th.mean() if tt else float('nan'):.5f}\n")
        # position aggregates over think tokens, by class
        for lo,hi in BUCK:
            if tt<=lo: break
            sl=slice(lo,min(hi,tt)); c=sl.stop-sl.start
            A(cls,f"{lo}",'n',c); A(cls,f"{lo}",'absd',d[sl].sum()); A(cls,f"{lo}",'lptr',lp_tr[sl].sum()); A(cls,f"{lo}",'lpeng',g[s-1:e-1][sl].sum()); A(cls,f"{lo}",'s1mp',s1[sl].sum()); A(cls,f"{lo}",'adv_s1mp',adv*s1[sl].sum()); A(cls,f"{lo}",'lowp',(th[sl]<0.5).sum())
out.close(); os.replace(base+".eff.csv.tmp",base+".eff.csv"); json.dump(agg,open(base+".effagg.json","w")); print("done",arm,step,ck)

import sys,os,json,re,zlib,csv
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
HF="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf"
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}"
if os.path.exists(base+".lossrec.csv"): sys.exit(0)
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
TIS_MIN,TIS_MAX=0.2,5.0
obj=load(chunk)
lengths=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(lengths)]); lp_off=np.concatenate([[0],np.cumsum(lengths-1)])
mask=np.asarray(obj["token_mask"]).astype(bool); gen=np.asarray(obj["generation_logprobs"]).astype(np.float64); prev=np.asarray(obj["prev_logprobs"]).astype(np.float64)
adv_all=np.asarray(obj["advantages"]).astype(np.float64); ids=np.asarray(obj["input_ids"]); sids=list(obj["sample_ids"]); tags=obj.get("tags") or [{}]*len(sids)
smask=np.asarray(obj["sample_mask_after"]).astype(np.float64); rewards=np.asarray(obj["rewards"]).astype(np.float64)
rows=[]
for r in range(len(sids)):
    mk=mask[ids_off[r]+1:ids_off[r+1]]; tk=ids[ids_off[r]+1:ids_off[r+1]][mk]; g=gen[lp_off[r]:lp_off[r+1]][mk]; p=prev[lp_off[r]:lp_off[r+1]][mk]
    n=len(tk)
    if n==0: continue
    adv=float(adv_all[lp_off[r]:lp_off[r+1]][mk][0]); trunc=bool((tags[r] or {}).get("rollout_truncated"))
    ends=np.where(tk==IM_END)[0]; starts=np.concatenate([[0],ends+1]); stops=np.concatenate([ends,[n]])
    loop=np.zeros(n,dtype=bool); nloopblk=0
    blocks=[]
    for a,b in zip(starts,stops):
        if b<=a: continue
        seg=tk[a:b]; ce=np.where(seg==THINK_END)[0]; r_end=a+int(ce[0]) if len(ce) else b
        if r_end>a: blocks.append((int(a),int(r_end)))
    for a,b in blocks:
        if b-a<1000: continue
        tb="".join(dec(x) for x in tk[a:b]).encode("utf-8","replace")
        if len(zlib.compress(tb,6))/max(1,len(tb))<0.10: loop[a:b]=True; nloopblk+=1
    w=np.exp(np.clip(p-g,-50,50)); oob=int(((w>TIS_MAX)|(w<TIS_MIN)).sum()); w=np.clip(w,TIS_MIN,TIS_MAX)
    pt=np.exp(p); one_m_p=1.0-pt
    def sums(sel):
        if not sel.any(): return (0,0.0,0.0,0.0,0.0,0.0)
        return (int(sel.sum()),float(w[sel].sum()),float((w[sel]*one_m_p[sel]).sum()),float(one_m_p[sel].sum()),float(np.abs(g[sel]-p[sel]).sum()),float(pt[sel].sum()))
    L=sums(loop); N=sums(~loop)
    rows.append([arm,step,sids[r],f"{adv:.6f}",f"{rewards[r]:g}",int(smask[r]),int(trunc),n,L[0],nloopblk,
                 f"{N[1]:.4f}",f"{L[1]:.4f}",f"{N[2]:.5f}",f"{L[2]:.5f}",f"{N[3]:.5f}",f"{L[3]:.5f}",f"{N[4]:.5f}",f"{L[4]:.5f}",f"{N[5]:.3f}",f"{L[5]:.3f}",oob])
with open(base+".lossrec.csv","w") as fh:
    wr=csv.writer(fh); wr.writerow(["arm","step","sample_id","adv","reward","sample_mask","truncated","n_tok","n_loop_tok","n_loop_blocks",
        "w_sum_normal","w_sum_loop","w1mp_sum_normal","w1mp_sum_loop","onemp_sum_normal","onemp_sum_loop","absdiff_sum_normal","absdiff_sum_loop","p_sum_normal","p_sum_loop","n_tis_oob"]); wr.writerows(rows)
print("done",arm,step,ck,flush=True)

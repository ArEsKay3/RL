import sys,os,re,json,zlib,glob
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}"
if os.path.exists(base+".samples.csv"): sys.exit(0)
HF="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf"
t=json.load(open(f"{HF}/tokenizer.json")); inv={v:k for k,v in t["model"]["vocab"].items()}; special={}
for a in t.get("added_tokens",[]): inv[a["id"]]=a["content"]; special[a["id"]]=a["content"]
def b2u():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1)); cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return dict(zip([chr(c) for c in cs],bs))
u2b=b2u(); cache={}
def tok_bytes(i):
    i=int(i); b=cache.get(i)
    if b is None:
        if i in special: b=special[i].encode()
        else:
            raw=inv.get(i,"")
            try: b=bytes(u2b[c] for c in raw)
            except KeyError: b=raw.encode()
        cache[i]=b
    return b
def decode_bytes(ids): return b"".join(tok_bytes(i) for i in ids)
def zr(b): return round(len(zlib.compress(b,6))/len(b),4) if len(b)>=64 else ""
NL={1010,1267,1011}
obj=load(chunk)
L=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(L)]); lp_off=np.concatenate([[0],np.cumsum(L-1)])
ids=np.asarray(obj["input_ids"]).astype(np.int64); tm=np.asarray(obj["token_mask"]).astype(bool)
gen=np.asarray(obj["generation_logprobs"]).astype(np.float64); prev=np.asarray(obj["prev_logprobs"]).astype(np.float64); advs=np.asarray(obj["advantages"]).astype(np.float64)
rewards=np.asarray(obj["rewards"]).astype(np.float64); sids=list(obj["sample_ids"]); tags=obj.get("tags") or [{}]*len(sids)
smb=np.asarray(obj["sample_mask_before"]); sma=np.asarray(obj["sample_mask_after"]); sme=np.asarray(obj["seq_mult_prob_error"]).astype(np.float64)
T=open(base+".turns.csv.tmp","w"); T.write("arm,step,chunk,sample_id,turn,n_turns,start,n_tok,think_tok,has_close,n_close,ends_seq,last_is_imend,zlib_reason,zlib_all,nl_frac,mean_gen_lp,min_gen_lp,mean_d,mean_abs_d,max_abs_d,n_d_gt1,n_clip_lo,n_clip_hi,d_close,g_close,d_imend,g_imend,g_first,d_first,reward,adv\n")
S=open(base+".samples.csv.tmp","w"); S.write("arm,step,chunk,sample_id,group_id,n_total,prompt_len,n_gen,n_turns,max_turn,last_turn,mean_turn,n_close,n_noclose,ends_seq_noclose,loop_turns,loop_tok,near_loop_turns,near_loop_tok,reward,adv,truncated,gen_len_tag,weight_version,openhands_run_time,model_call_time,mean_gen_lp,mean_d,mean_abs_d,max_abs_d,n_d_gt1,n_clip_lo,n_clip_hi,seq_mult_prob_error,mask_before,mask_after,nl_frac\n")
for i,sid in enumerate(sids):
    row=ids[ids_off[i]:ids_off[i+1]]; mrow=tm[ids_off[i]:ids_off[i+1]]; n=len(row)
    g=gen[lp_off[i]:lp_off[i+1]]; p=prev[lp_off[i]:lp_off[i+1]]; a=advs[lp_off[i]:lp_off[i+1]]
    edges=np.diff(np.concatenate([[0],mrow.astype(np.int8),[0]])); starts=np.where(edges==1)[0]; ends=np.where(edges==-1)[0]
    nturn=len(starts); tg=tags[i] or {}
    adv_s=float(a[starts[0]-1]) if nturn else 0.0
    agg=dict(n_gen=int(mrow.sum()),max_turn=0,last_turn=0,n_close=0,n_noclose=0,ends_noclose=0,loop_turns=0,loop_tok=0,near_turns=0,near_tok=0,sum_g=0.0,sum_d=0.0,sum_ad=0.0,max_ad=0.0,n_gt1=0,n_lo=0,n_hi=0,nl=0)
    for k,(s,e) in enumerate(zip(starts,ends)):
        seg=row[s:e]; nt=e-s; gg=g[s-1:e-1]; pp=p[s-1:e-1]; d=pp-gg
        w13=np.where(seg==13)[0]; has_close=len(w13)>0; think_tok=int(w13[0]) if has_close else nt; ncl=len(w13)
        bts=decode_bytes(seg); rb=decode_bytes(seg[:think_tok]) if has_close else bts
        zrs=zr(rb); za=zr(bts); nlf=float(np.isin(seg,list(NL)).mean())
        ends_seq=int(e==n); last_imend=int(seg[-1]==11)
        ad=np.abs(d); r=np.exp(d)
        d_close=d[think_tok] if has_close else ""; g_close=gg[think_tok] if has_close else ""
        d_im=d[-1] if last_imend else ""; g_im=gg[-1] if last_imend else ""
        T.write(f"{arm},{step},{ck},{sid},{k},{nturn},{s},{nt},{think_tok},{int(has_close)},{ncl},{ends_seq},{last_imend},{zrs},{za},{nlf:.4f},{gg.mean():.5f},{gg.min():.4f},{d.mean():.6f},{ad.mean():.6f},{ad.max():.4f},{int((ad>1).sum())},{int((r<0.2).sum())},{int((r>5).sum())},{d_close if d_close=='' else f'{d_close:.5f}'},{g_close if g_close=='' else f'{g_close:.5f}'},{d_im if d_im=='' else f'{d_im:.5f}'},{g_im if g_im=='' else f'{g_im:.5f}'},{gg[0]:.5f},{d[0]:.5f},{rewards[i]:.0f},{adv_s:.5f}\n")
        agg["max_turn"]=max(agg["max_turn"],nt); agg["last_turn"]=nt
        if has_close: agg["n_close"]+=1
        else:
            agg["n_noclose"]+=1
            if ends_seq: agg["ends_noclose"]+=1
        if zrs!="" and think_tok>=1000:
            if zrs<0.10: agg["loop_turns"]+=1; agg["loop_tok"]+=think_tok
            elif zrs<0.25: agg["near_turns"]+=1; agg["near_tok"]+=think_tok
        agg["sum_g"]+=gg.sum(); agg["sum_d"]+=d.sum(); agg["sum_ad"]+=ad.sum(); agg["max_ad"]=max(agg["max_ad"],float(ad.max())); agg["n_gt1"]+=int((ad>1).sum()); agg["n_lo"]+=int((r<0.2).sum()); agg["n_hi"]+=int((r>5).sum()); agg["nl"]+=int(np.isin(seg,list(NL)).sum())
    ng=max(1,agg["n_gen"]); pl=int(starts[0]) if nturn else n
    S.write(f"{arm},{step},{ck},{sid},{sid.rsplit('_g',1)[0]},{n},{pl},{agg['n_gen']},{nturn},{agg['max_turn']},{agg['last_turn']},{agg['n_gen']/max(1,nturn):.1f},{agg['n_close']},{agg['n_noclose']},{agg['ends_noclose']},{agg['loop_turns']},{agg['loop_tok']},{agg['near_turns']},{agg['near_tok']},{rewards[i]:.0f},{adv_s:.5f},{int(bool(tg.get('rollout_truncated',0)))},{tg.get('rollout_generation_length','')},{tg.get('weight_version','')},{tg.get('rollout_env_extra:openhands_run_time','')},{tg.get('rollout_env_extra:total_model_call_time','')},{agg['sum_g']/ng:.5f},{agg['sum_d']/ng:.6f},{agg['sum_ad']/ng:.6f},{agg['max_ad']:.4f},{agg['n_gt1']},{agg['n_lo']},{agg['n_hi']},{sme[i]:.5f},{smb[i]:.0f},{sma[i]:.0f},{agg['nl']/ng:.4f}\n")
T.close(); S.close(); os.replace(base+".turns.csv.tmp",base+".turns.csv"); os.replace(base+".samples.csv.tmp",base+".samples.csv"); print("done",arm,step,ck,len(sids))

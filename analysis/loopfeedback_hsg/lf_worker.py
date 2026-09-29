import sys,os,json,re,zlib,collections,math
import numpy as np
sys.path.insert(0,os.environ.get("LF_TOOLS",os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","tools")))
from pt_numpy import load
arm,exp,chunk,outdir=sys.argv[1:5]
HF=os.environ.get("LF_HF","/lustre/fsw/portfolios/llmservice/users/pjin/devel/nemo-rl-ultra-v3-nano-opd-dev-20260513/results/mopd_ultrav3_to_nanov3_5_repro_v5_kd_opt_full-hsg-20260524-r1/step_18/hf")
m=re.search(r"step_(\d+)_chunk_(\d+)",chunk); step,ck=int(m.group(1)),int(m.group(2))
base=f"{outdir}/{arm}__s{step:03d}_c{ck:03d}"
if os.path.exists(base+".agg.json"): sys.exit(0)
t=json.load(open(f"{HF}/tokenizer.json")); inv={v:k for k,v in t["model"]["vocab"].items()}
special={}
for a in t.get("added_tokens",[]): inv[a["id"]]=a["content"]; special[a["id"]]=a["content"]
def b2u():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1)); cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return dict(zip([chr(c) for c in cs],bs))
u2b=b2u()
cache={}
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
SENT=re.compile(r"(?<=[.!?])\s+|\n+")
W1=re.compile(r"[A-Za-z][A-Za-z']*")
obj=load(chunk)
lengths=np.asarray(obj["input_lengths"]).astype(np.int64); ids_off=np.concatenate([[0],np.cumsum(lengths)]); lp_off=np.concatenate([[0],np.cumsum(lengths-1)])
mask=np.asarray(obj["token_mask"]).astype(bool); gen=np.asarray(obj["generation_logprobs"]).astype(np.float32); advs=np.asarray(obj["advantages"]).astype(np.float32)
ids=np.asarray(obj["input_ids"]); rewards=np.asarray(obj["rewards"]).astype(np.float32); sids=list(obj["sample_ids"]); tags=obj.get("tags") or [{}]*len(sids)
agg={}   # (cls,wv) -> {"n_sent":..,"n_items":..,"n_tok":..,"u":{},"b":{},"t":{}}
items_out=open(base+".items.csv","w"); items_out.write("arm,step,sample_id,wv,adv,reward,truncated,item_idx,n_items,n_tok,n_sent,zlib,dist8,mean_lp,has_close,cls\n")
cyc_out=open(base+".cycles.jsonl","w")
def bump(d,key,adv,lp):
    e=d.get(key)
    p=math.exp(lp)
    if e is None: d[key]=[1,adv,adv*(1-p),abs(adv)*(1-p),lp]
    else: e[0]+=1; e[1]+=adv; e[2]+=adv*(1-p); e[3]+=abs(adv)*(1-p); e[4]+=lp
for r in range(len(sids)):
    mk=mask[ids_off[r]+1:ids_off[r+1]]; g=gen[lp_off[r]:lp_off[r+1]][mk]; tk=ids[ids_off[r]+1:ids_off[r+1]][mk]
    if len(tk)==0: continue
    adv=float(advs[lp_off[r]:lp_off[r+1]][mk][0]); rew=float(rewards[r]); tg=tags[r] or {}
    wv=tg.get("weight_version"); trunc=bool(tg.get("rollout_truncated")); sid=sids[r]
    # split into turns at <|im_end|>
    ends=np.where(tk==IM_END)[0]; starts=np.concatenate([[0],ends+1]); stops=np.concatenate([ends,[len(tk)]])
    items=[]
    for a,b in zip(starts,stops):
        if b<=a: continue
        seg=tk[a:b]; ce=np.where(seg==THINK_END)[0]
        r_end=a+int(ce[0]) if len(ce) else b   # reasoning = tokens before </think>
        if r_end<=a: continue
        items.append((a,r_end,len(ce)>0))
    n_items=len(items)
    for ii,(a,b,has_close) in enumerate(items):
        toks=tk[a:b]; lps=g[a:b]; n_tok=b-a
        strs=[dec(x) for x in toks]; text="".join(strs)
        if not text.strip(): continue
        offs=np.cumsum([0]+[len(s) for s in strs[:-1]])
        tb=text.encode("utf-8","replace"); zr=len(zlib.compress(tb,6))/max(1,len(tb))
        words=text.split(); st=max(1,len(words)//20000)
        grams=[" ".join(words[i:i+8]) for i in range(0,max(0,len(words)-8),st)]
        d8=len(set(grams))/len(grams) if grams else 1.0
        runaway=(not has_close) and trunc and ii==n_items-1
        cls="runaway" if runaway else ("looplike" if (n_tok>=1000 and zr<0.10) else "normal")
        # sentence starts
        pos=0; sent_pos=[0]
        for mm in SENT.finditer(text): sent_pos.append(mm.end())
        n_sent=0
        key=(cls,wv); A=agg.setdefault(key,{"n_sent":0,"n_items":0,"n_tok":0,"u":{},"b":{},"t":{}})
        A["n_items"]+=1; A["n_tok"]+=int(n_tok)
        for sp in sent_pos:
            if sp>=len(text): continue
            mw=W1.search(text,sp)
            if not mw or mw.start()-sp>3: continue   # sentence must start with a word within 3 chars
            ti=int(np.searchsorted(offs,mw.start(),side="right")-1); lp=float(lps[ti]) if 0<=ti<len(lps) else 0.0
            ws=W1.findall(text[mw.start():mw.start()+80])[:3]; ws=[w.lower() for w in ws]
            if not ws: continue
            n_sent+=1
            bump(A["u"],ws[0],adv,lp)
            if len(ws)>=2: bump(A["b"]," ".join(ws[:2]),adv,lp)
            if len(ws)>=3: bump(A["t"]," ".join(ws[:3]),adv,lp)
        A["n_sent"]+=n_sent
        items_out.write(f"{arm},{step},{sid},{wv},{adv:.4f},{rew:.0f},{int(trunc)},{ii},{n_items},{n_tok},{n_sent},{zr:.4f},{d8:.4f},{float(lps.mean()):.5f},{int(has_close)},{cls}\n")
        if cls!="normal":
            sents=[s.strip() for s in SENT.split(text) if len(s.strip())>20]
            top=collections.Counter(sents).most_common(3)
            cyc_out.write(json.dumps({"arm":arm,"step":step,"sample_id":sid,"wv":wv,"adv":adv,"reward":rew,"truncated":trunc,"item_idx":ii,"n_items":n_items,"n_tok":int(n_tok),"zlib":round(zr,4),"dist8":round(d8,4),"mean_lp":round(float(lps.mean()),5),"cls":cls,"top":[(s[:160],c) for s,c in top]})+"\n")
items_out.close(); cyc_out.close()
out={}
for (cls,wv),A in agg.items():
    k=f"{cls}|{wv}"
    out[k]={"n_sent":A["n_sent"],"n_items":A["n_items"],"n_tok":A["n_tok"],"u":A["u"],"b":A["b"],"t":{kk:v for kk,v in A["t"].items() if v[0]>=2}}
json.dump(out,open(base+".agg.json","w"))
print("done",arm,step,ck,len(sids),flush=True)

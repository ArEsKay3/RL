import glob,json,math
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
HF="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/swe_e2e_corrected/base_model/step_18/hf"
t=json.load(open(f"{HF}/tokenizer.json")); inv={v:k for k,v in t["model"]["vocab"].items()}
for a in t.get("added_tokens",[]): inv[a["id"]]=a["content"]
def b2u():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1)); cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return dict(zip([chr(c) for c in cs],bs))
u2b=b2u()
def name(i):
    raw=inv.get(int(i),"?")
    if int(i)<1000: return raw
    try: return bytes(u2b[c] for c in raw).decode("utf-8","replace")
    except KeyError: return raw
ARMS=[("G","BAD"),("I","BAD"),("J","good"),("A","good"),("K","good"),("M","good"),("N","good")]
agg={}
for a,o in ARMS:
    acc=None
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.tokexp.npz"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.tokexp.npz")):
        d=np.load(fn)
        if acc is None: acc={k:d[k].copy() for k in d.files}
        else:
            for k in d.files: acc[k]+=d[k]
    agg[a]=acc
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
tot={a:agg[a]["cnt"].sum() for a,_ in ARMS}
P("TOKEN-LEVEL EXPOSURE, steps 1-10. For each vocabulary item: E_tok = 1000 x sum of advantages over its generated occurrences / all generated tokens of the arm; freq = occurrences per 1k generated tokens.")
P("Bad-vs-good difference in E_tok, averaged over arms, for tokens with >= 2,000 occurrences in every arm. Positive difference = the bad arms' data reinforces this token more (or suppresses it less).")
V=len(agg["G"]["cnt"]); ok=np.all([agg[a]["cnt"]>=2000 for a,_ in ARMS],axis=0)
E={a:1e3*agg[a]["ea"]/tot[a] for a,_ in ARMS}; F={a:1e3*agg[a]["cnt"]/tot[a] for a,_ in ARMS}
Eb=np.mean([E[a] for a,o in ARMS if o=="BAD"],axis=0); Eg=np.mean([E[a] for a,o in ARMS if o!="BAD"],axis=0); Fb=np.mean([F[a] for a,o in ARMS if o=="BAD"],axis=0); Fg=np.mean([F[a] for a,o in ARMS if o!="BAD"],axis=0)
sdg=np.std([E[a] for a,o in ARMS if o!="BAD"],axis=0,ddof=1)
diff=Eb-Eg; z=np.where(sdg>0,diff/np.maximum(sdg,1e-9),0)
P(f"tokens considered: {int(ok.sum())}; total generated tokens per arm ~{np.mean(list(tot.values()))/1e6:.1f}M")
P(""); P("A. top 30 tokens by bad-minus-good exposure difference (most reinforced in bad arms relative to good)")
P(f"{'token':<22} {'diff E':>8} {'z vs good sd':>12} {'E bad':>7} {'E good':>7} | {'freq bad':>8} {'freq good':>9} {'freq ratio':>10} | per-arm E: "+" ".join(f"{a:>7}" for a,_ in ARMS))
idx=np.argsort(-np.where(ok,diff,-1e9))[:30]
for i in idx: P(f"{repr(name(i)):<22} {diff[i]:>+8.3f} {z[i]:>+12.1f} {Eb[i]:>+7.3f} {Eg[i]:>+7.3f} | {Fb[i]:>8.2f} {Fg[i]:>9.2f} {Fb[i]/max(Fg[i],1e-9):>10.3f} | "+" ".join(f"{E[a][i]:>+7.3f}" for a,_ in ARMS))
P(""); P("B. top 20 tokens by good-minus-bad (more reinforced in good arms)")
idx=np.argsort(np.where(ok,diff,1e9))[:20]
for i in idx: P(f"{repr(name(i)):<22} {diff[i]:>+8.3f} {z[i]:>+12.1f} {Eb[i]:>+7.3f} {Eg[i]:>+7.3f} | {Fb[i]:>8.2f} {Fg[i]:>9.2f} {Fb[i]/max(Fg[i],1e-9):>10.3f} | "+" ".join(f"{E[a][i]:>+7.3f}" for a,_ in ARMS))
P(""); P("C. deliberation / hedging markers explicitly: E per arm and frequency ratio bad/good")
words=["Wait","ĠWait","Actually","ĠActually","Hmm","ĠHmm","But","ĠBut","ĠLet","Let","Ġre","Ġagain","Ġdouble","Ġcheck","Ġverify","Ġmaybe","ĠMaybe","Ġperhaps","ĠAlternatively","Alternatively","ĠHowever","However","ĠSo","So","ĠOkay","Okay","ĠOK","ĠNo","No","ĠYes","Ġactually","Ġwait","Ġhmm","ĠI","I","Ġthink","Ġthe","Ċ","ĊĊ","</think>","<|im_end|>","<tool_call>"]
vocab=t["model"]["vocab"]; added={a["content"]:a["id"] for a in t.get("added_tokens",[])}
P(f"{'token':<18} "+" ".join(f"{a+' E':>8}" for a,_ in ARMS)+f" | {'bad-good':>8} {'freq b/g':>8} {'freq/1k':>7}")
for w in words:
    i=added.get(w,vocab.get(w))
    if i is None: continue
    P(f"{repr(name(i)):<18} "+" ".join(f"{E[a][i]:>+8.3f}" for a,_ in ARMS)+f" | {diff[i]:>+8.3f} {Fb[i]/max(Fg[i],1e-9):>8.3f} {Fg[i]:>7.2f}")
P(""); P("D. reward-free frequency fingerprint: tokens whose frequency differs most between MINF arms (G,I,J) and vLLM arms (A,K,M,N) with a perfect split, >= 2,000 occurrences")
Fm=np.mean([F[a] for a in ("G","I","J")],axis=0); Fv=np.mean([F[a] for a in ("A","K","M","N")],axis=0)
mn=np.min([F[a] for a in ("G","I","J")],axis=0); mx=np.max([F[a] for a in ("G","I","J")],axis=0); vmn=np.min([F[a] for a in ("A","K","M","N")],axis=0); vmx=np.max([F[a] for a in ("A","K","M","N")],axis=0)
split=ok&((mn>vmx)|(mx<vmn)); ratio=np.log(np.maximum(Fm,1e-9)/np.maximum(Fv,1e-9))
P(f"perfect-split tokens: {int(split.sum())} of {int(ok.sum())} (chance ~ {2/35*ok.sum():.0f})")
idx=np.argsort(-np.where(split,np.abs(ratio),-1))[:25]
for i in idx:
    if not split[i]: break
    P(f"  {repr(name(i)):<22} MINF {Fm[i]:>8.3f}/1k  vLLM {Fv[i]:>8.3f}/1k  log-ratio {ratio[i]:>+6.3f}   per-arm: "+" ".join(f"{a} {F[a][i]:.3f}" for a,_ in ARMS))
P(""); P("E. inside long turns (4k-16k) only: tokens with the largest bad-minus-good exposure difference (>= 500 occurrences in every arm's long turns)")
El={a:1e3*agg[a]["ea_long"]/tot[a] for a,_ in ARMS}; okl=np.all([agg[a]["cnt_long"]>=500 for a,_ in ARMS],axis=0)
dl=np.mean([El[a] for a,o in ARMS if o=="BAD"],axis=0)-np.mean([El[a] for a,o in ARMS if o!="BAD"],axis=0)
idx=np.argsort(-np.where(okl,dl,-1e9))[:20]
for i in idx: P(f"  {repr(name(i)):<22} diff {dl[i]:>+7.3f}   per-arm: "+" ".join(f"{El[a][i]:>+7.3f}" for a,_ in ARMS))
open(f"{D}/tokexp_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

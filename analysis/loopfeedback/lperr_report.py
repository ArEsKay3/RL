import glob,re,collections
import numpy as np
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
NAMES={"G":"chain G MINF from0 no-pfx","I":"chain I MINF from0 pfx-on","J":"chain J MINF from0 pfx-on r2","A":"run A vLLM from0","K":"chain K vLLM from0 s1234","F":"chain F MINF fr vLLM10","L":"chain L MINF fr K10 pfx","B":"run B MINF fr MINF10","D":"chain D vLLM fr MINF10"}
ORDER=["A","K","J","F","L","G","I","B","D"]
T=collections.defaultdict(lambda:np.zeros((3,2,7,11))); E=collections.defaultdict(lambda:np.zeros((3,2,7,11)))
for fn in glob.glob(f"{L}/lperr/*.lperr.npz"):
    m=re.search(r"([A-Z])__s(\d+)_c",fn); a,s=m.group(1),int(m.group(2)); z=np.load(fn); T[(a,s)]+=z["T"]; E[(a,s)]+=z["E"]
BINS=["<1e-3","1e-3..1e-2","1e-2..0.1","0.1..0.5","0.5..0.9","0.9..0.99",">0.99"]
# stats: 0 n,1 sum d,2 sum|d|,3 sum pe,4 sum pt,5 sum w,6 sum w(1-pt),7 sum(1-pt),8 n w_raw<0.2,9 n w_raw>5,10 n d>0
o=[]; P=lambda *x:o.append(" ".join(str(v) for v in x))
def pool(D,a,lo,hi): 
    xs=[D[(a,s)] for s in range(lo,hi+1) if (a,s) in D]
    return (sum(xs),len(xs)) if xs else (None,0)
P("ENGINE-vs-TRAINER LOGPROB MISMATCH ON LOOP TOKENS vs NORMAL TOKENS.  d = log p_engine - log p_trainer of the sampled token (d>0: the engine that sampled it was more confident than the trainer).")
P("  Token classes: loop = inside a repetitive reasoning block (>=1,000 tok, zlib<0.10); normal = everything else.  Sample-masked rollouts excluded (they are not in the loss).")
P("  Because the engine chose the token, d is biased positive for every engine (tokens the engine over-estimated are more likely to be drawn); compare ARMS, not the absolute sign.")
P("")
P("="*160); P("(1) OVERALL, per arm and step band"); P("="*160)
P(f"{'arm':<28} {'steps':>5} {'class':>6} {'tokens':>13} | {'mean d':>8} {'mean |d|':>8} {'frac d>0':>8} | {'mean p_eng':>10} {'mean p_trn':>10} {'(1-p_eng)':>9} {'(1-p_trn)':>9} {'tail ratio':>10} | {'mean w':>7} {'grad-wtd w':>10} {'oob/M':>6}")
for lo,hi in ((1,10),(11,20),(21,36)):
    P(f"--- steps {lo}-{hi}")
    for a in ORDER:
        X,ns=pool(T,a,lo,hi)
        if X is None: continue
        for c,lab in ((0,"normal"),(1,"loop")):
            v=X[:,c].sum(axis=(0,1)); n=v[0]
            if n<1000: continue
            P(f"{NAMES[a]:<28} {ns:>5} {lab:>6} {n:>13,.0f} | {v[1]/n:>+8.5f} {v[2]/n:>8.5f} {v[10]/n:>8.3f} | {v[3]/n:>10.5f} {v[4]/n:>10.5f} {1-v[3]/n:>9.5f} {1-v[4]/n:>9.5f} {(1-v[3]/n)/(1-v[4]/n):>10.3f} | {v[5]/n:>7.4f} {v[6]/v[7]:>10.4f} {1e6*(v[8]+v[9])/n:>6.1f}")
P("")
P("="*160); P("(2) MISMATCH PROFILE BY TRAINER PROBABILITY OF THE SAMPLED TOKEN, steps 1-20 pooled (the window before the arms separate), from-scratch arms + chain F."); P("="*160)
P("  For each bin: share of the class's tokens in the bin, mean d, mean w (TIS weight applied), share of the class's gradient weight (1-p_trn) that sits in the bin.")
for c,lab in ((0,"NORMAL tokens"),(1,"LOOP tokens")):
    P(f"\n----- {lab} -----")
    P(f"{'trainer p bin':<12} | "+" | ".join(f"{a:^28}" for a in ["A","K","J","G","I","F"]))
    P(f"{'':<12} | "+" | ".join(f"{'tok%':>6} {'mean d':>7} {'w':>5} {'grad%':>6}" for _ in range(6)))
    pooled={a:pool(T,a,1,20)[0] for a in ["A","K","J","G","I","F"]}
    for b in range(7):
        cells=[]
        for a in ["A","K","J","G","I","F"]:
            X=pooled[a]
            if X is None: cells.append(f"{'':>28}"); continue
            v=X[:,c,b].sum(axis=0); tot=X[:,c].sum(axis=(0,1))
            if v[0]<50: cells.append(f"{100*v[0]/tot[0]:>5.2f}% {'':>7} {'':>5} {'':>6}"); continue
            cells.append(f"{100*v[0]/tot[0]:>5.2f}% {v[1]/v[0]:>+7.3f} {v[5]/v[0]:>5.3f} {100*v[7]/tot[7]:>5.1f}%")
        P(f"{BINS[b]:<12} | "+" | ".join(cells))
P("")
P("="*160); P("(3) SAME PROFILE BINNED BY THE ENGINE'S PROBABILITY (what the sampler saw), steps 1-20 pooled: mean d and mean trainer p per engine-p bin"); P("="*160)
for c,lab in ((0,"NORMAL tokens"),(1,"LOOP tokens")):
    P(f"\n----- {lab} -----")
    P(f"{'engine p bin':<12} | "+" | ".join(f"{a:^24}" for a in ["A","K","J","G","I","F"]))
    P(f"{'':<12} | "+" | ".join(f"{'tok%':>6} {'mean d':>7} {'p_trn':>8}" for _ in range(6)))
    pooled={a:pool(E,a,1,20)[0] for a in ["A","K","J","G","I","F"]}
    for b in range(7):
        cells=[]
        for a in ["A","K","J","G","I","F"]:
            X=pooled[a]
            if X is None: cells.append(f"{'':>24}"); continue
            v=X[:,c,b].sum(axis=0); tot=X[:,c].sum(axis=(0,1))
            if v[0]<50: cells.append(f"{100*v[0]/tot[0]:>5.2f}% {'':>7} {'':>8}"); continue
            cells.append(f"{100*v[0]/tot[0]:>5.2f}% {v[1]/v[0]:>+7.3f} {v[4]/v[0]:>8.4f}")
        P(f"{BINS[b]:<12} | "+" | ".join(cells))
P("")
P("="*160); P("(4) PER STEP: mean d on loop tokens and on normal tokens, from-scratch arms (blank = fewer than 1,000 loop tokens)"); P("="*160)
steps=sorted({s for (_,s) in T})
P(f"{'step':>4} | "+" | ".join(f"{a+' d_loop':>9} {'d_norm':>8} {'tail':>6}" for a in ["A","K","J","G","I"]))
for s in steps:
    cells=[]
    for a in ["A","K","J","G","I"]:
        X=T.get((a,s))
        if X is None: cells.append(f"{'':>25}"); continue
        vl=X[:,1].sum(axis=(0,1)); vn=X[:,0].sum(axis=(0,1))
        dl=f"{vl[1]/vl[0]:>+9.4f}" if vl[0]>=1000 else f"{'':>9}"; tail=f"{(1-vl[3]/vl[0])/(1-vl[4]/vl[0]):>6.2f}" if vl[0]>=1000 else f"{'':>6}"
        cells.append(f"{dl} {vn[1]/vn[0]:>+8.4f} {tail}")
    P(f"{s:>4} | "+" | ".join(cells))
P("")
P("="*160); P("(5) LOOP TOKENS BY ADVANTAGE SIGN of their rollout, steps 1-20 pooled: is the mismatch different in rewarded vs punished loop rollouts?"); P("="*160)
P(f"{'arm':<28} | {'adv>0 tokens':>12} {'mean d':>8} {'(1-p_eng)':>9} {'(1-p_trn)':>9} | {'adv<0 tokens':>12} {'mean d':>8} {'(1-p_eng)':>9} {'(1-p_trn)':>9} | {'adv=0 tokens':>12} {'mean d':>8}")
for a in ORDER:
    X,_=pool(T,a,1,20)
    if X is None: continue
    cells=[]
    for sg in range(3):
        v=X[sg,1].sum(axis=0)
        if v[0]<1000: cells.append(f"{v[0]:>12,.0f} {'':>8} {'':>9} {'':>9}" if sg<2 else f"{v[0]:>12,.0f} {'':>8}"); continue
        cells.append(f"{v[0]:>12,.0f} {v[1]/v[0]:>+8.5f} {1-v[3]/v[0]:>9.5f} {1-v[4]/v[0]:>9.5f}" if sg<2 else f"{v[0]:>12,.0f} {v[1]/v[0]:>+8.5f}")
    P(f"{NAMES[a]:<28} | "+" | ".join(cells))
open(f"{L}/lperr_report.txt","w").write("\n".join(o)+"\n"); print("\n".join(o))

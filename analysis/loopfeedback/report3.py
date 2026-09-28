import pickle,collections,math,sys
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
agg=pickle.load(open(f"{L}/agg.pkl","rb"))
CANDS=[x for x in sys.argv[1:]] or ["given","actually","wait","but","however","although","hmm","let","i","so","now","this","maybe","alternatively","given the time","let me try","but the test","i think","let me check","but that","let me"]
NAMES={"G":"chain G","A":"run A","F":"chain F","B":"run B","D":"chain D","I":"chain I"}
def gram_of(u): return {1:"u",2:"b",3:"t"}[len(u.split())]
def series(arm,u):
    """per step: usage/1k in normal blocks, push share (all classes), push share (normal only), modal wv"""
    g=gram_of(u); out={}
    for (a,s),A in agg.items():
        if a!=arm: continue
        n_sent=0; cnt=0; push=0.0; push_n=0.0; den=0.0; wvc=collections.Counter()
        for k,v in A.items():
            cls,wv=k.split("|"); vals=v[g].get(u)
            den+=sum(x[3] for x in v[g].values())
            if cls=="normal": n_sent+=v["n_sent"]; wvc[wv]+=v["n_sent"]
            if vals:
                push+=vals[2]
                if cls=="normal": cnt+=vals[0]; push_n+=vals[2]
        out[s]=dict(usage=1000*cnt/max(1,n_sent),push=push/den if den else 0.0,push_n=push_n/den if den else 0.0,wv=wvc.most_common(1)[0][0] if wvc else None)
    return out
def corr(xs,ys):
    n=len(xs)
    if n<4: return float('nan')
    mx=sum(xs)/n; my=sum(ys)/n; sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(sx*sy) if sx and sy else float('nan')
print("STEP 2+3 - PUSH vs RESPONSE per candidate unit.")
print("push(s) = sum over sentence starts of the unit in training batch s of adv*(1-p) [p = sampling prob of the unit's first token], divided by the same sum over ALL")
print("          sentence starts at step s: a signed share in [-1,1]; positive = update s pushes the model to start sentences with this unit more.")
print("usage(s) = occurrences per 1,000 sentence starts in NORMAL blocks of batch s. dUsage(s+k) = usage(s+k)-usage(s+k-1).")
print("corr = Pearson correlation over all steps of push(s) with dUsage(s+1) and dUsage(s+2); 'G-A' uses chain G minus run A on the common steps (same prompts).")
print("baseline = corr(dUsage(s), dUsage(s+1)) of the arm itself (autocorrelation of the response).")
print(f"\n{'unit':<16} {'arm':>6} {'steps':>5} {'mean usage':>10} {'usage 1st':>9} {'usage last':>10} {'mean push':>9} {'push>0 steps':>12} | {'corr k=1':>8} {'corr k=2':>8} {'baseline':>8} | {'G-A k=1':>8} {'G-A k=2':>8}")
for u in CANDS:
    SG=series("G",u); SA=series("A",u)
    for arm in ("G","A","F","B","D"):
        S=series(arm,u); steps=sorted(S)
        if len(steps)<6: continue
        us=[S[s]["usage"] for s in steps]; ps=[S[s]["push"] for s in steps]
        d={steps[i]:us[i]-us[i-1] for i in range(1,len(steps))}
        c1=corr([S[s]["push"] for s in steps if s+1 in d],[d[s+1] for s in steps if s+1 in d])
        c2=corr([S[s]["push"] for s in steps if s+2 in d],[d[s+2] for s in steps if s+2 in d])
        ds=[d[s] for s in steps if s in d]; base=corr(ds[:-1],ds[1:])
        ga1=ga2=float('nan')
        if arm=="G":
            common=[s for s in steps if s in SA]
            if len(common)>6:
                dg={s:SG[s]["usage"]-SG[s-1]["usage"] for s in common if s-1 in SG}; da={s:SA[s]["usage"]-SA[s-1]["usage"] for s in common if s-1 in SA}
                dd={s:dg[s]-da[s] for s in dg if s in da}
                ga1=corr([SG[s]["push"]-SA[s]["push"] for s in common if s+1 in dd],[dd[s+1] for s in common if s+1 in dd])
                ga2=corr([SG[s]["push"]-SA[s]["push"] for s in common if s+2 in dd],[dd[s+2] for s in common if s+2 in dd])
        print(f"{u:<16} {NAMES[arm]:>6} {len(steps):>5} {sum(us)/len(us):>10.2f} {us[0]:>9.2f} {us[-1]:>10.2f} {sum(ps)/len(ps):>+9.4f} {sum(1 for p in ps if p>0):>12} | {c1:>+8.2f} {c2:>+8.2f} {base:>+8.2f} | {ga1:>+8.2f} {ga2:>+8.2f}")
    print()

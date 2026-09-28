import pickle,collections,statistics as st,json,math,sys
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
agg=pickle.load(open(f"{L}/agg.pkl","rb"))
ARMS=["G","A","F","B","D","I"]
NAMES={"G":"chain G (MINF from0)","A":"run A (vLLM from0)","F":"chain F (MINF from vLLM10)","B":"run B (MINF from MINF10)","D":"chain D (vLLM from MINF10)","I":"chain I (MINF from0, prefix)"}
def usage(arm,gram,cls_set=("normal",)):
    """per step: unit -> count, and total sentence starts, restricted to classes"""
    out={}
    for (a,s),A in agg.items():
        if a!=arm: continue
        cnt=collections.Counter(); tot=0
        for k,v in A.items():
            cls=k.split("|")[0]
            if cls not in cls_set: continue
            tot+=v["n_sent"]
            for u,vals in v[gram].items(): cnt[u]+=vals[0]
        out[s]=(cnt,tot)
    return out
def slope(xs,ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    b=sxy/sxx if sxx else 0.0
    res=[y-(my+b*(x-mx)) for x,y in zip(xs,ys)]; se=math.sqrt(sum(r*r for r in res)/max(1,n-2)/sxx) if sxx and n>2 else float('inf')
    return b, (b/se if se else 0.0)
G=usage("G","u"); Gb=usage("G","b"); Gt=usage("G","t")
print("STEP 1b - CANDIDATE DISCOVERY. Unit = first 1/2/3 words of a sentence inside a NORMAL reasoning block (loops excluded). usage = per 1,000 sentence starts.")
print("Growth is a least-squares slope over ALL steps of the arm (per 10 steps), with its t-statistic; A = run A (vLLM, same prompts) as control, F = chain F.")
for gram,U,minc,label in (("u",G,600,"single first words"),("b",Gb,300,"first two words"),("t",Gt,150,"first three words")):
    steps=sorted(U); tot={s:U[s][1] for s in steps}
    allu=collections.Counter()
    for s in steps: allu.update(U[s][0])
    cands=[u for u,c in allu.items() if c>=minc]
    A=usage("A",gram); F=usage("F",gram)
    res=[]
    for u in cands:
        ys=[1000*U[s][0][u]/tot[s] for s in steps]; b,t=slope(steps,ys)
        ya=[1000*A[s][0][u]/A[s][1] for s in sorted(A)] if A else []; ba,_=slope(sorted(A),ya) if len(ya)>2 else (float('nan'),0)
        yf=[1000*F[s][0][u]/F[s][1] for s in sorted(F)] if F else []; bf,_=slope(sorted(F),yf) if len(yf)>2 else (float('nan'),0)
        res.append((u,st.mean(ys),ys[0],ys[-1],b*10,t,ba*10 if ya else float('nan'),bf*10 if yf else float('nan'),allu[u]))
    res.sort(key=lambda x:-x[5])
    print(f"\n--- {label}: top 20 GROWING in chain G by t-stat (min total count {minc}) ---")
    print(f"{'unit':<28} {'mean/1k':>8} {'step1':>7} {'last':>7} {'G slope/10st':>12} {'t':>6} {'A slope/10st':>12} {'F slope/10st':>12} {'G count':>8}")
    for u,m,y0,y1,b,t,ba,bf,c in res[:20]: print(f"{u:<28} {m:>8.2f} {y0:>7.2f} {y1:>7.2f} {b:>+12.3f} {t:>+6.1f} {ba:>+12.3f} {bf:>+12.3f} {c:>8}")
    print(f"--- {label}: top 10 SHRINKING in chain G ---")
    for u,m,y0,y1,b,t,ba,bf,c in res[-10:][::-1]: print(f"{u:<28} {m:>8.2f} {y0:>7.2f} {y1:>7.2f} {b:>+12.3f} {t:>+6.1f} {ba:>+12.3f} {bf:>+12.3f} {c:>8}")
print("\n--- LOOP-ENRICHED sentence starts (chain G, all steps pooled): share inside loop-like+runaway blocks vs inside normal blocks ---")
for gram,minc in (("u",300),("b",200)):
    N=usage("G",gram,("normal",)); Lp=usage("G",gram,("looplike","runaway"))
    cn=collections.Counter(); cl=collections.Counter(); tn=tl=0
    for s in N: cn.update(N[s][0]); tn+=N[s][1]
    for s in Lp: cl.update(Lp[s][0]); tl+=Lp[s][1]
    rows=[(u,1000*cl[u]/tl,1000*cn[u]/tn,cl[u]) for u in cl if cl[u]>=minc]
    rows.sort(key=lambda x:-(x[1]+0.1)/(x[2]+0.1))
    print(f"{'unit':<28} {'in loops /1k':>12} {'in normal /1k':>13} {'ratio':>6} {'loop count':>10}")
    for u,a,b,c in rows[:15]: print(f"{u:<28} {a:>12.1f} {b:>13.1f} {(a+0.1)/(b+0.1):>6.1f} {c:>10}")
    print()
print("--- CYCLE SENTENCES: top repeated sentence of each loop-like/runaway block; generic ones = appear as a top sentence in >= 3 distinct rollouts (chain G) ---")
cyc=[json.loads(l) for l in open(f"{L}/cycles.jsonl")]
c=collections.defaultdict(set); cnt=collections.Counter()
for r in cyc:
    if r["arm"]!="G": continue
    for s,n in r["top"][:1]:
        key=s.lower()[:90]; c[key].add(r["sample_id"]); cnt[key]+=n
gen=[(k,len(v),cnt[k]) for k,v in c.items() if len(v)>=3]; gen.sort(key=lambda x:-x[1])
print(f"{'rollouts':>8} {'repeats':>8}  sentence (first 90 chars)")
for k,n,m in gen[:15]: print(f"{n:>8} {m:>8}  {k}")
print(f"(distinct loop blocks in chain G: {sum(1 for r in cyc if r['arm']=='G')}; with a generic top sentence: {sum(len(v) for k,v in c.items() if len(v)>=3)})")

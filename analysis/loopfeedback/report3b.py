import pickle,collections,math,statistics as st
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
agg=pickle.load(open(f"{L}/agg.pkl","rb"))
NAMES={"G":"chain G (MINF from0)","A":"run A (vLLM from0)","F":"chain F (MINF from vLLM10)","B":"run B (MINF from MINF10)","D":"chain D (vLLM from MINF10)"}
def slope(xs,ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n; sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    b=sxy/sxx if sxx else 0.0; res=[y-(my+b*(x-mx)) for x,y in zip(xs,ys)]
    se=math.sqrt(sum(r*r for r in res)/max(1,n-2)/sxx) if sxx and n>2 else float('inf'); return b,(b/se if se else 0.0)
def corr(xs,ys):
    n=len(xs)
    if n<5: return float('nan')
    mx=sum(xs)/n; my=sum(ys)/n; sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(sx*sy) if sx and sy else float('nan')
def rankcorr(xs,ys):
    rx=[sorted(xs).index(x) for x in xs]; ry=[sorted(ys).index(y) for y in ys]; return corr(rx,ry)
print("STEP 3 (refined) - Does the direction of the gradient on a sentence-start unit predict how its usage drifts?")
print("For unit u at training step s: excess push(u,s) = [sum adv*(1-p) over u's sentence starts] / [sum |adv|*(1-p) over u's sentence starts]  minus the same ratio over ALL sentence starts.")
print("  (signed fraction of u's gradient weight that is positive, relative to the batch; +0.10 = u's starts sit in noticeably better-rewarded rollouts than average). All classes of block.")
print("usage(s) = u per 1,000 sentence starts in NORMAL blocks of batch s. slope = least-squares slope of usage over ALL steps (per 10 steps), t = its t-stat.")
print("Cross-unit test per arm: rank correlation between mean excess push and usage slope over all units with enough counts. If reinforcement of a unit drives its drift, this should be clearly positive.")
for gram,minc,label in (("u",600,"single first words"),("b",300,"first two words")):
    print(f"\n==================== {label} ====================")
    for arm in ("G","A","F","B","D"):
        steps=sorted(s for (a,s) in agg if a==arm)
        if not steps: continue
        tot_push={}; per={}
        for s in steps:
            A=agg[(arm,s)]; pos_all=neg_all=0.0; cnt=collections.Counter(); posu=collections.defaultdict(float); absu=collections.defaultdict(float); nsent=0
            for k,v in A.items():
                cls=k.split("|")[0]
                for u,vals in v[gram].items():
                    posu[u]+=vals[2]; absu[u]+=vals[3]
                    if cls=="normal": cnt[u]+=vals[0]
                if cls=="normal": nsent+=v["n_sent"]
            allpos=sum(posu.values()); allabs=sum(absu.values()); base=allpos/allabs if allabs else 0.0
            per[s]=(cnt,nsent,posu,absu,base)
        units=collections.Counter()
        for s in steps: units.update(per[s][0])
        rows=[]
        for u,c in units.items():
            if c<minc: continue
            ex=[]; us=[]
            for s in steps:
                cnt,nsent,posu,absu,base=per[s]
                us.append(1000*cnt[u]/max(1,nsent))
                ex.append((posu[u]/absu[u]-base) if absu[u]>0 else None)
            exv=[e for e in ex if e is not None]
            b,t=slope(steps,us)
            # lagged corr: excess push at s vs smoothed usage change (mean of s+1,s+2 minus mean of s-1,s)
            xs=[];ys=[]
            for i,s in enumerate(steps):
                if ex[i] is None or i<1 or i+2>=len(steps): continue
                xs.append(ex[i]); ys.append((us[i+1]+us[i+2])/2-(us[i-1]+us[i])/2)
            rows.append((u,c,st.mean(us),st.mean(exv) if exv else float('nan'),b*10,t,corr(xs,ys)))
        rows=[r for r in rows if not math.isnan(r[3])]
        rc=rankcorr([r[3] for r in rows],[r[4] for r in rows]) if len(rows)>5 else float('nan')
        pc=corr([r[3] for r in rows],[r[4]/max(1e-9,r[2]) for r in rows]) if len(rows)>5 else float('nan')
        print(f"\n--- {NAMES[arm]}: {len(rows)} units; RANK CORR(mean excess push, usage slope) = {rc:+.2f}; corr(mean excess push, relative slope) = {pc:+.2f}; median over units of per-step corr(push, later usage change) = {st.median([r[6] for r in rows if not math.isnan(r[6])]):+.2f}")
        rows.sort(key=lambda r:-r[3])
        print(f"{'unit':<24} {'count':>8} {'mean use/1k':>11} {'mean excess push':>16} {'slope/10st':>10} {'t':>6} {'corr(push->dUse)':>16}")
        for r in rows[:10]: print(f"{r[0]:<24} {r[1]:>8} {r[2]:>11.2f} {r[3]:>+16.4f} {r[4]:>+10.3f} {r[5]:>+6.1f} {r[6]:>+16.2f}")
        print("   ...")
        for r in rows[-8:]: print(f"{r[0]:<24} {r[1]:>8} {r[2]:>11.2f} {r[3]:>+16.4f} {r[4]:>+10.3f} {r[5]:>+6.1f} {r[6]:>+16.2f}")
        if arm=="G":
            print("   named candidates:")
            for name in ["given","actually","wait","but","however","alternatively","i","i'll","i'm","so","let","this","now","maybe","given the","let me","i think","but that","i need","i'll try","i'm going","let me try","let me check","so we","alternatively i"]:
                m=[r for r in rows if r[0]==name]
                if m: r=m[0]; print(f"   {r[0]:<21} {r[1]:>8} {r[2]:>11.2f} {r[3]:>+16.4f} {r[4]:>+10.3f} {r[5]:>+6.1f} {r[6]:>+16.2f}")

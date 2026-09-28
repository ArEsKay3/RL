import glob,csv,json,urllib.request,collections,time
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
EXP={"G":"nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920","M":"nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923"}
B="http://127.0.0.1:8790"
def get(p):
    with urllib.request.urlopen(B+p,timeout=900) as r: return json.loads(r.read())
def f(x,d=0.0):
    try: return float(x)
    except: return d
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P("ACTION AFTER A LONG THINK (4k-16k tokens), steps 1-10, chain G (MINF) vs chain M (vLLM): tool called at the end of the long think, by outcome of the rollout; and the tool called in the NEXT turn.")
for a,e in EXP.items():
    T=[]
    for fn in sorted(glob.glob(f"{D}/parts/{a}__s00[1-9]_c*.turns.csv"))+sorted(glob.glob(f"{D}/parts/{a}__s010_c*.turns.csv")):
        with open(fn) as fh: T+=list(csv.DictReader(fh))
    lg=[r for r in T if 4000<=f(r["n_tok"])<16000]
    byfile=collections.defaultdict(list)
    for r in lg: byfile[int(r["step"])-1].append(r)
    tool=collections.Counter(); nxt=collections.Counter(); n_ev=collections.Counter(); t0=time.time(); miss=0
    for ts,rs in sorted(byfile.items()):
        fl=f"target_step_{ts:05d}.jsonl"; rows=get(f"/api/records?run={e}&file={fl}&limit=5000")["rows"]; byid={x["sample_id"]:x["i"] for x in rows}
        bysid=collections.defaultdict(list)
        for r in rs: bysid[r["sample_id"]].append(r)
        for sid,rr in bysid.items():
            if sid not in byid: miss+=len(rr); continue
            turns=get(f"/api/turns?run={e}&file={fl}&i={byid[sid]}")["turns"]
            for r in rr:
                k=int(f(r["turn"])); outc="rewarded" if f(r["adv"])>1e-9 else ("punished" if f(r["adv"])<-1e-9 else "zero")
                if k>=len(turns): miss+=1; continue
                names=[c["name"] for c in turns[k]["calls"]] or ["(no tool call)"]; tool[(outc,names[0])]+=1; n_ev[outc]+=1
                if k+1<len(turns): nn=[c["name"] for c in turns[k+1]["calls"]] or ["(no tool call)"]; nxt[(outc,nn[0])]+=1
    P(f"\n== chain {a}: long-think events {sum(n_ev.values())} (rewarded {n_ev['rewarded']}, punished {n_ev['punished']}, zero-adv {n_ev['zero']}); missing {miss}; {time.time()-t0:.0f}s")
    tools=sorted({t for (_,t) in tool},key=lambda t:-sum(v for (o,tt),v in tool.items() if tt==t))
    P(f"  {'tool at end of long think':<26} "+" ".join(f"{o:>16}" for o in ("rewarded","punished","zero")))
    for t in tools: P(f"  {t:<26} "+" ".join(f"{tool[(o,t)]:>6} ({100*tool[(o,t)]/max(1,n_ev[o]):>5.1f}%)" for o in ("rewarded","punished","zero")))
    P(f"  {'tool in the NEXT turn':<26} "+" ".join(f"{o:>16}" for o in ("rewarded","punished","zero")))
    for t in sorted({t for (_,t) in nxt},key=lambda t:-sum(v for (o,tt),v in nxt.items() if tt==t))[:8]: P(f"  {t:<26} "+" ".join(f"{nxt[(o,t)]:>6} ({100*nxt[(o,t)]/max(1,n_ev[o]):>5.1f}%)" for o in ("rewarded","punished","zero")))
open(f"{D}/tool_after_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

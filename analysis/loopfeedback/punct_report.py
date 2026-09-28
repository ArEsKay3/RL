import json,glob,re,collections,math,sys
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
NAMES={"G":"chain G (MINF from0)","A":"run A (vLLM from0)","F":"chain F (MINF from vLLM10)","B":"run B (MINF from MINF10)","D":"chain D (vLLM from MINF10)","I":"chain I (MINF from0 prefix)"}
agg={}
for fn in glob.glob(f"{L}/parts/*.punct.json")+glob.glob(f"{L}/punct/*.punct.json"):
    m=re.search(r"([A-Z])__s(\d+)_c(\d+)\.punct\.json",fn); arm,step=m.group(1),int(m.group(2))
    P=json.load(open(fn)); A=agg.setdefault((arm,step),{})
    for k,v in P.items():
        e=A.setdefault(k,{"n_blocks":0,"n_blocks_closed":0,"n_sent_final":0,"var":{},"close_prev":{},"close_lp_sum":0.0,"close_n":0,"close_lp_hist":[0]*8})
        for f in ("n_blocks","n_blocks_closed","n_sent_final","close_lp_sum","close_n"): e[f]+=v[f]
        e["close_lp_hist"]=[a+b for a,b in zip(e["close_lp_hist"],v["close_lp_hist"])]
        for u,vals in v["var"].items():
            if u in e["var"]:
                for i in range(8): e["var"][u][i]+=vals[i]
            else: e["var"][u]=list(vals)
        for u,vals in v["close_prev"].items():
            if u in e["close_prev"]:
                for i in range(3): e["close_prev"][u][i]+=vals[i]
            else: e["close_prev"][u]=list(vals)
print("PUNCTUATION PASS: sentence-final punctuation tokens inside reasoning blocks, and the token right before </think>.")
print("Pooled over chain G, all steps, NORMAL blocks. 'next' = what the model generated right after the punctuation token.")
pool={}
for (arm,step),A in agg.items():
    if arm!="G": continue
    for k,v in A.items():
        if not k.startswith("normal"): continue
        for u,vals in v["var"].items():
            e=pool.setdefault(u,[0]*8)
            for i in range(8): e[i]+=vals[i]
tot=sum(e[0] for e in pool.values())
print(f"\n{'token (repr)':<16} {'count':>9} {'share':>6} | {'next=</think>':>13} {'next=newline tok':>16} {'next=other':>10} | {'mean p of token':>15}")
for u,e in sorted(pool.items(),key=lambda x:-x[1][0])[:14]:
    print(f"{u:<16} {e[0]:>9,} {e[0]/tot:>6.1%} | {e[5]/e[0]:>13.1%} {e[6]/e[0]:>16.1%} {e[7]/e[0]:>10.1%} | {math.exp(e[4]/e[0]):>15.3f}")
print("\nToken immediately before </think> (chain G, normal blocks, all steps pooled): what does a close look like?")
cp=collections.defaultdict(lambda:[0,0.0,0.0])
for (arm,step),A in agg.items():
    if arm!="G": continue
    for k,v in A.items():
        if not k.startswith("normal"): continue
        for u,vals in v["close_prev"].items():
            cp[u][0]+=vals[0]; cp[u][1]+=vals[1]; cp[u][2]+=vals[2]
tc=sum(v[0] for v in cp.values())
print(f"{'prev token (repr)':<18} {'closes':>8} {'share':>6} {'mean p(</think>)':>16}")
for u,v in sorted(cp.items(),key=lambda x:-x[1][0])[:10]: print(f"{u:<18} {v[0]:>8,} {v[0]/tc:>6.1%} {math.exp(v[1]/v[0]):>16.3f}")
def per_step(arm):
    rows={}
    for (a,step),A in agg.items():
        if a!=arm: continue
        nb=nbc=ns=cl=0; lpc=0.0; var=collections.defaultdict(lambda:[0]*8)
        for k,v in A.items():
            if not k.startswith("normal"): continue
            nb+=v["n_blocks"]; nbc+=v["n_blocks_closed"]; ns+=v["n_sent_final"]; cl+=v["close_n"]; lpc+=v["close_lp_sum"]
            for u,vals in v["var"].items():
                for i in range(8): var[u][i]+=vals[i]
        den=sum(e[3] for e in var.values())
        def g(u): return var.get(u,[0]*8)
        dot=g("'.'"); dnn=g("'.\\n\\n'"); dn=g("'.\\n'")
        rows[step]=dict(blocks=nb,sent=ns,spb=ns/max(1,nb),close_rate=1000*cl/max(1,ns),p_close=math.exp(lpc/cl) if cl else float('nan'),
            sh_dot=dot[0]/max(1,ns),sh_dnn=dnn[0]/max(1,ns),sh_dn=dn[0]/max(1,ns),
            p_dot=math.exp(dot[4]/dot[0]) if dot[0] else float('nan'),p_dnn=math.exp(dnn[4]/dnn[0]) if dnn[0] else float('nan'),
            push_dot=dot[2]/den if den else 0.0,push_dnn=dnn[2]/den if den else 0.0,push_dn=dn[2]/den if den else 0.0,
            dot_close=dot[5]/max(1,dot[0]),dnn_close=dnn[5]/max(1,dnn[0]))
    return rows
print("\nPER STEP, NORMAL blocks. spb = sentence-final tokens per block; close/1k = </think> per 1,000 sentence-final tokens (the empirical 'stop deliberating' rate);")
print("share '.' / '.\\n\\n' / '.\\n' = share of sentence-final tokens that are that variant; p = mean sampling prob of that token when sampled; p(</think>) = mean prob of the sampled close;")
print("push = sum adv*(1-p) over that token's occurrences in the training batch / sum |adv|*(1-p) over all sentence-final tokens (signed share; + = pushed up by update s).")
for arm in ("G","A","F","B","D","I"):
    R=per_step(arm)
    if not R: continue
    print(f"\n{NAMES[arm]}")
    print(f"{'step':>4} {'blocks':>7} {'spb':>6} {'close/1k':>8} {'p(</think>)':>11} | {'share .':>7} {'share .nn':>9} {'share .n':>8} | {'p(.)':>6} {'p(.nn)':>6} | {'push .':>8} {'push .nn':>8} {'push .n':>8} | {'. ->close':>9} {'.nn->close':>10}")
    for s in sorted(R):
        r=R[s]; print(f"{s:>4} {r['blocks']:>7} {r['spb']:>6.2f} {r['close_rate']:>8.1f} {r['p_close']:>11.3f} | {r['sh_dot']:>7.1%} {r['sh_dnn']:>9.1%} {r['sh_dn']:>8.1%} | {r['p_dot']:>6.3f} {r['p_dnn']:>6.3f} | {r['push_dot']:>+8.4f} {r['push_dnn']:>+8.4f} {r['push_dn']:>+8.4f} | {r['dot_close']:>9.1%} {r['dnn_close']:>10.1%}")

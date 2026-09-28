import sys,os,json,re,csv,glob,zlib
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
listfile=sys.argv[1]; tag=os.path.splitext(os.path.basename(listfile))[0]
outfn=f"{L}/mainchain_text/{tag}.csv"
if os.path.exists(outfn): sys.exit(0)
out=open(outfn+".tmp","w"); w=csv.writer(out)
w.writerow(["results_dir","instance_id","epoch_ms","turns","turns_matched","completion_tokens","reason_tokens_est","rep_turns","rep_tokens","rep_reason_tokens_est","rep_unclosed","max_turn_tokens","max_turn_zlib","max_turn_closed","last_turn_tokens","last_turn_zlib","last_turn_closed"])
for line in open(listfile):
    d=line.strip()
    if not d: continue
    base=os.path.basename(d); m=re.match(r"(.+)_(\d{13})_([0-9a-f]+)$",base)
    inst,ep=(m.group(1),int(m.group(2))) if m else (base,0)
    fs=glob.glob(os.path.join(d,"trajectories","*","output.jsonl"))
    if not fs: continue
    try:
        with open(fs[0]) as fh: rec=json.loads(fh.readline())
    except Exception: continue
    hist=rec.get("history") or []
    turns=[]
    for ev in hist:
        if ev.get("source")!="agent": continue
        t=ev.get("tool_call_metadata") or {}; mr=t.get("model_response") or {}
        if not mr: continue
        ch=(mr.get("choices") or [{}])[0].get("message") or {}
        content=ch.get("content") or ""
        if "<think>" not in content and "</think>" not in content:
            thought=""; closed=True
        else:
            s=content.find("<think>"); s=s+7 if s>=0 else 0
            e=content.find("</think>",s)
            if e>=0: thought=content[s:e]; closed=True
            else: thought=content[s:]; closed=False
        ct=int(((mr.get("usage") or {}).get("completion_tokens")) or 0)
        tb=thought.encode("utf-8","replace"); zr=len(zlib.compress(tb,6))/max(1,len(tb)) if tb else 1.0
        frac=len(thought)/max(1,len(content)) if content else 0.0
        turns.append((ct,zr,closed,frac))
    if not turns: continue
    mf=os.path.join(d,"nemo_gym_metrics.json")
    try: acc=int(((json.load(open(mf)).get("per_turn_metrics") or {}).get("accumulated_token_usage") or {}).get("completion_tokens") or 0)
    except Exception: acc=sum(t[0] for t in turns)
    rep=[t for t in turns if t[0]>=1000 and t[1]<0.10]
    mx=max(turns,key=lambda t:t[0]); last=turns[-1]
    w.writerow([os.path.basename(os.path.dirname(d)),inst,ep,len(hist),len(turns),acc,round(sum(t[0]*t[3] for t in turns)),len(rep),sum(t[0] for t in rep),round(sum(t[0]*t[3] for t in rep)),sum(1 for t in rep if not t[2]),mx[0],f"{mx[1]:.3f}",int(mx[2]),last[0],f"{last[1]:.3f}",int(last[2])])
out.close(); os.rename(outfn+".tmp",outfn); print("done",tag,flush=True)

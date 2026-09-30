import glob,json,os,re,sys,collections,statistics
L=os.environ["LF_L"]; R=os.environ["LF_R"]
EXP={}
for line in open(f"{L}/jobs_hsg.txt"):
    f=line.split(); EXP[f[0]]=f[1]
def head(raw):
    i=raw.find(b', "messages": ')
    if i<0: i=raw.find(b',"messages":')
    r=json.loads((raw[:i]+b"}").decode("utf-8","replace")) if i>0 else json.loads(raw.decode("utf-8","replace"))
    r["_harness_fail"]=(b'"agent_timed_out": false' in raw and b'"agent_error_kind": null' in raw)
    return r
def pct(v,q):
    v=sorted(v); return v[min(len(v)-1,int(q*len(v)))] if v else 0
cache_fn=f"{L}/rollout_stats_cache.json"; cache=json.load(open(cache_fn)) if os.path.exists(cache_fn) else {}
rows_out=[]
print(f"{'arm':>4} {'step':>4} {'rows':>5} {'groups':>6} {'wv mix':>12} {'reward':>6} {'hfail':>5} {'trunc':>5} {'turns p50':>9} {'gen mean':>8} {'gen p50':>7} {'gen p90':>7} {'gen max':>7} {'tot p50':>7} {'malformed':>9} {'invalid tc':>10} {'file MB':>8}")
for a,e in EXP.items():
    for fn in sorted(glob.glob(f"{R}/{e}/dumps/rollouts/target_step_*.jsonl")):
        ts=int(re.search(r"target_step_(\d+)",fn).group(1)); step=ts+1
        ck=f"{a}:{os.path.basename(fn)}:{os.path.getsize(fn)}"
        if ck in cache:
            c=cache[ck]; print(c["line"]); rows_out.append(c["row"]); continue
        rs=[]
        with open(fn,"rb") as fh:
            for raw in fh:
                if not raw.endswith(b"\n"): break
                try: rs.append(head(raw))
                except Exception: continue
        if not rs: continue
        wv=collections.Counter(f"{r.get('start_weight_version')}->{r.get('end_weight_version')}" for r in rs)
        gl=[r.get("generation_length",0) for r in rs]; tt=[r.get("total_tokens",0) for r in rs]; tu=[r.get("num_assistant_turns",0) for r in rs]
        rew=[float(r.get("reward",0)) for r in rs]; tr=sum(1 for r in rs if r.get("truncated")); hf=sum(1 for r in rs if r.get("_harness_fail"))
        mal=-1
        groups=len({r.get("group_id") for r in rs})
        line=(f"{a:>4} {step:>4} {len(rs):>5} {groups:>6} {','.join(f'{k}:{v}' for k,v in sorted(wv.items())):>12} {statistics.mean(rew):>6.3f} {hf:>5} {tr:>5} {pct(tu,0.5):>9} {statistics.mean(gl):>8.0f} {pct(gl,0.5):>7} {pct(gl,0.9):>7} {max(gl):>7} {pct(tt,0.5):>7} {mal:>9} {'-':>10} {os.path.getsize(fn)/1e6:>8.0f}")
        print(line)
        row=dict(arm=a,step=step,rows=len(rs),groups=groups,reward=round(statistics.mean(rew),4),harness_fail=hf,truncated=tr,turns_p50=pct(tu,0.5),gen_mean=round(statistics.mean(gl)),gen_p50=pct(gl,0.5),gen_p90=pct(gl,0.9),gen_max=max(gl),total_p50=pct(tt,0.5))
        rows_out.append(row)
        if len(rs)==512: cache[ck]={"line":line,"row":row}; json.dump(cache,open(cache_fn,"w"))
json.dump(cache,open(cache_fn,"w"))
json.dump(rows_out,open(f"{L}/hsg_rollout_stats.json","w"),indent=0)

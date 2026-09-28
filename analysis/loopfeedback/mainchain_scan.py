import sys,os,json,re,csv,glob
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
resdir=sys.argv[1]; tag=os.path.basename(resdir)
smap=json.load(open(f"{L}/instance_step_map.json"))
out=open(f"{L}/mainchain/{tag}.csv","w"); w=csv.writer(out)
w.writerow(["results_dir","instance_id","epoch_ms","step","resolved","n_turns","completion_tokens","max_turn","turns_ge8k","turns_ge30k","turns_ge100k","gen_start","run_time"])
n=0
for d in sorted(os.listdir(resdir)):
    m=re.match(r"(.+)_(\d{13})_([0-9a-f]+)$",d)
    if not m: continue
    inst,ep=m.group(1),int(m.group(2))
    mf=os.path.join(resdir,d,"nemo_gym_metrics.json")
    if not os.path.exists(mf): continue
    try: M=json.load(open(mf))
    except Exception: continue
    pt=M.get("per_turn_metrics") or {}; tu=pt.get("token_usages") or []
    ct=[int(t.get("completion_tokens") or 0) for t in tu]
    acc=(pt.get("accumulated_token_usage") or {}).get("completion_tokens")
    step=smap.get(inst,{}).get("step","")
    w.writerow([tag,inst,ep,step,int(bool(M.get("resolved"))),len(ct),acc if acc is not None else sum(ct),max(ct) if ct else 0,
                sum(1 for x in ct if x>=8000),sum(1 for x in ct if x>=30000),sum(1 for x in ct if x>=100000),M.get("generation_start_timestamp",""),M.get("openhands_run_time","")])
    n+=1
out.close(); print("done",tag,n,flush=True)

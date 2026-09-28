import sys,os,re,json
arm,exp,step,outdir=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
R="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
out=f"{outdir}/{arm}__s{step:03d}.refit_ts.csv"
if os.path.exists(out): sys.exit(0)
fn=f"{R}/{exp}/dumps/rollouts/target_step_{step-1:05d}.jsonl"
dec=json.JSONDecoder(); rows=[]
rx=dict(sid=re.compile(rb'"sample_id":\s*"([^"]+)"'),sv=re.compile(rb'"start_weight_version":\s*(\d+)'),ev=re.compile(rb'"end_weight_version":\s*(\d+)'),ca=re.compile(rb'"committed_at":\s*("[^"]*"|[\d.]+)'))
with open(fn,"rb") as fh:
    for line in fh:
        head=line[:4000]; m={k:r.search(head) for k,r in rx.items()}
        if not m["sid"]: continue
        sid=m["sid"].group(1).decode(); sv=int(m["sv"].group(1)) if m["sv"] else -1; ev=int(m["ev"].group(1)) if m["ev"] else -1; ca=m["ca"].group(1).decode().strip('"') if m["ca"] else ""
        i=line.find(b'"response_latencies"')
        if i<0: continue
        j=line.find(b"[",i); s=line[j:j+2_000_000].decode("utf-8","replace")
        try: lst,_=dec.raw_decode(s)
        except Exception: continue
        for k,x in enumerate(lst):
            rows.append(f"{arm},{step},{sid},{sv},{ev},{ca},{k},{x.get('timestamp','')},{x.get('latency','')}")
with open(out+".tmp","w") as fh:
    fh.write("arm,step,sample_id,start_v,end_v,committed_at,turn,ts,latency\n"); fh.write("\n".join(rows)+"\n")
os.replace(out+".tmp",out); print("done",arm,step,len(rows))

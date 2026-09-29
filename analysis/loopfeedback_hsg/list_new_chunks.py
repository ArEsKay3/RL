import glob,os,re,sys
L=os.environ["LF_L"]; R=os.environ["LF_R"]
for line in open(f"{L}/jobs_hsg.txt"):
    a,e=line.split()[:2]
    for fn in sorted(glob.glob(f"{R}/{e}/dumps/token_level/step_*_chunk_*.pt")):
        m=re.search(r"step_(\d+)_chunk_(\d+)",fn); s,k=int(m.group(1)),int(m.group(2))
        if os.path.exists(f"{L}/parts/{a}__s{s:03d}_c{k:03d}.agg.json"): continue
        if os.path.getmtime(fn) < __import__("time").time()-120: print(a,e,fn,f"{L}/parts")

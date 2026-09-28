import json,glob,re,os,pickle,collections
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
agg={}   # (arm,step) -> {cls|wv: {n_sent,n_items,n_tok,u,b,t}}
for fn in sorted(glob.glob(f"{L}/parts/*.agg.json")):
    m=re.search(r"([A-Z])__s(\d+)_c(\d+)\.agg\.json",fn); arm,step=m.group(1),int(m.group(2))
    part=json.load(open(fn)); A=agg.setdefault((arm,step),{})
    for k,v in part.items():
        e=A.setdefault(k,{"n_sent":0,"n_items":0,"n_tok":0,"u":{},"b":{},"t":{}})
        for f in ("n_sent","n_items","n_tok"): e[f]+=v[f]
        for g in ("u","b","t"):
            d=e[g]
            for unit,vals in v[g].items():
                if unit in d:
                    for i in range(5): d[unit][i]+=vals[i]
                else: d[unit]=list(vals)
pickle.dump(agg,open(f"{L}/agg.pkl","wb"))
n=sum(1 for _ in glob.glob(f"{L}/parts/*.items.csv"))
with open(f"{L}/items.csv","w") as w:
    first=True
    for fn in sorted(glob.glob(f"{L}/parts/*.items.csv")):
        lines=open(fn).read().splitlines()
        if first: w.write(lines[0]+"\n"); first=False
        w.write("\n".join(lines[1:])+("\n" if len(lines)>1 else ""))
with open(f"{L}/cycles.jsonl","w") as w:
    for fn in sorted(glob.glob(f"{L}/parts/*.cycles.jsonl")): w.write(open(fn).read())
print(f"merged {len(agg)} (arm,step) aggregates from {n} parts; items.csv {os.path.getsize(L+'/items.csv')/1e6:.1f} MB; cycles {sum(1 for _ in open(L+'/cycles.jsonl'))} loop-like/runaway items")

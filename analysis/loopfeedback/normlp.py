import glob,re,json,csv,collections
import numpy as np
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis"
EXP={}
for line in open("jobs.txt"):
    a,e=line.split()[:2]; EXP[a]=e
ARMS=["G","I","A","F","B","D"]; rng=np.random.default_rng(2)
R=collections.defaultdict(lambda:[0.0,0,0.0,0])  # (arm,step,sid): lp_sum normal, tok normal, lp_sum long normal >=300, tok
for arm in ARMS:
    for fn in glob.glob(f"parts/{arm}__s*_c*.items.csv"):
        with open(fn) as fh:
            rd=csv.reader(fh); next(rd)
            for row in rd:
                step=int(row[1]); cls=row[15]
                if cls!="normal": continue
                ntok=int(row[9]); mlp=float(row[13]); e=R[(arm,step,row[2])]; e[0]+=mlp*ntok; e[1]+=ntok
                if ntok>=300: e[2]+=mlp*ntok; e[3]+=ntok
inst={}
for arm in ARMS:
    for fn in glob.glob(f"{L}/rollouts/{EXP[arm]}__target_step_*.summary.jsonl"):
        ts=int(re.search(r"target_step_(\d+)",fn).group(1))
        for line in open(fn):
            d=json.loads(line); inst[(arm,ts+1,d["sample_id"])]=d["instance_id"]
def signflip(d,n=20000):
    d=np.asarray(d,float)
    if len(d)<2: return float('nan')
    obs=d.mean(); sims=(rng.choice([-1.0,1.0],size=(n,len(d)))*d).mean(axis=1); return float((np.abs(sims)>=abs(obs)-1e-12).mean())
steps=sorted({s for (_,s,_) in R})
print("SHARPNESS OF NORMAL REASONING: token-weighted mean log p of sampled tokens in NORMAL blocks (loops excluded). Higher (closer to 0) = model more certain of what it samples.")
print("Per step: arm means; G-A and I-A = paired difference over the step's 32 SWE instances (instance means of 16 gens), p = sign-flip test on those 32 pairs.")
print(f"{'step':>4} {'G':>8} {'I':>8} {'A':>8} {'F':>8} {'B':>8} {'D':>8} | {'G-A':>8} {'p':>5} | {'I-A':>8} {'p':>5} | {'cum G-A':>8}")
cum=[]
for s in steps:
    m={}; PI={a:collections.defaultdict(list) for a in ARMS}
    for a in ARMS:
        lp=tk=0
        for (aa,st,sid),v in R.items():
            if aa!=a or st!=s or v[1]==0: continue
            lp+=v[0]; tk+=v[1]; ii=inst.get((a,s,sid))
            if ii is not None: PI[a][ii].append(v[0]/v[1])
        m[a]=lp/tk if tk else float('nan')
    def pair(x,y):
        keys=sorted(set(PI[x])&set(PI[y]))
        if len(keys)<2: return float('nan'),float('nan'),[]
        d=np.array([np.mean(PI[x][k])-np.mean(PI[y][k]) for k in keys]); return d.mean(),signflip(d),list(d)
    ga,pga,dga=pair("G","A"); ia,pia,_=pair("I","A")
    cum+=dga
    cell=lambda v:f"{v:>8.4f}" if v==v else f"{'':>8}"
    print(f"{s:>4} {cell(m['G'])} {cell(m['I'])} {cell(m['A'])} {cell(m['F'])} {cell(m['B'])} {cell(m['D'])} | {cell(ga)} {pga:>5.2f} | {cell(ia)} {pia:>5.2f} | {np.mean(cum) if cum else float('nan'):>+8.4f}")

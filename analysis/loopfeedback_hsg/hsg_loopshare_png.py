import csv,json,os,collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
L=os.environ["LF_L"]
cmp=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{L}/../loopfeedback/joint_loopshare.csv")): cmp[r["arm"]][int(r["step"])]=float(r["share_all_gen_pct"])
hsg=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{L}/hsg_loopshare.csv")):
    if int(r["rows"])==512: hsg[r["arm"]][int(r["step"])]=float(r["joint_pct"])
hf={(r["arm"],r["step"]):r.get("harness_fail",0) for r in json.load(open(f"{L}/hsg_rollout_stats.json")) if r["rows"]==512}
SER=[("VHnC","V-HSG-noprefix-r2 (parity MINF, seed 42)","#d62728",2.6,hsg),("VHnC2","V-HSG2-noprefix-r2 (seed 1234)","#ff7f0e",2.6,hsg),("VHnC3","V-HSG3-noprefix-r2 (seed 4321)","#9467bd",2.6,hsg),
     ("V","chain V (CMH parity MINF)","#e377c2",1.3,cmp),("P4","chain P'''' (CMH MINF keep-prefix)","#8c564b",1.3,cmp),("J","chain J (CMH MINF clean)","#2ca02c",1.3,cmp),
     ("G","chain G (CMH MINF poisoned)","#000000",1.3,cmp),("A","run A (CMH vLLM)","#1f77b4",1.3,cmp),("Q4","chain Q'''' (CMH vLLM, no prefix cache)","#17becf",1.3,cmp)]
fig,ax=plt.subplots(figsize=(13,6.2),dpi=150)
ax.axvspan(20,30,color="#f0f0f0",zorder=0); ax.text(25,29,"verdict window 20-30",ha="center",color="#666")
for k,name,col,w,src in SER:
    d=src.get(k,{}); xs=[s for s in sorted(d) if s<=40]
    if len(xs)<2: continue
    ax.plot(xs,[d[s] for s in xs],color=col,lw=w,label=name,alpha=1 if src is hsg else 0.85,zorder=3 if src is hsg else 2)
    if src is hsg:
        bad=[s for s in xs if hf.get((k,s),0)>50]
        if bad: ax.scatter(bad,[d[s] for s in bad],facecolors="white",edgecolors=col,s=30,zorder=4)
ax.set_xlim(1,40); ax.set_ylim(0,30); ax.set_xlabel("training step"); ax.set_ylabel("loop-token share, % of all generated tokens")
ax.set_title("Loop-token share per training step: HSG vLLM-parity arms (round 2, Lustre workspace) vs CMH comparators")
ax.grid(alpha=0.3); ax.legend(loc="upper left",fontsize=8,ncol=2,framealpha=0.9)
fig.tight_layout(); fig.savefig(f"{L}/hsg_loopshare.png"); print(f"{L}/hsg_loopshare.png")

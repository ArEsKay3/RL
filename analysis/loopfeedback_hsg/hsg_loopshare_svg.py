import csv,os,collections
L=os.environ["LF_L"]
cmp=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{L}/../loopfeedback/joint_loopshare.csv")): cmp[r["arm"]][int(r["step"])]=float(r["share_all_gen_pct"])
hsg=collections.defaultdict(dict); hf=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{L}/hsg_loopshare.csv")):
    if int(r["rows"])==512: hsg[r["arm"]][int(r["step"])]=float(r["joint_pct"])
import json
for r in json.load(open(f"{L}/hsg_rollout_stats.json")):
    if r["rows"]==512: hf[r["arm"]][r["step"]]=r.get("harness_fail",0)
SER=[("VH","V-HSG (MINF, vLLM-parity, HSG, seed 42)","#d62728",3,hsg),("VH2","V-HSG2 (seed 1234)","#ff7f0e",3,hsg),("VH3","V-HSG3 (seed 4321)","#9467bd",3,hsg),("VHn","V-HSG-noprefix (MINF prefix cache off, seed 42)","#bcbd22",3,hsg),("VHn2","V-HSG2-noprefix (seed 1234)","#7f7f7f",3,hsg),("VHn3","V-HSG3-noprefix (seed 4321)","#e6550d",3,hsg),
     ("V","chain V (CMH, vLLM-parity MINF)","#e377c2",1.5,cmp),("P4","chain P'''' (CMH MINF keep-prefix)","#8c564b",1.5,cmp),("J","chain J (CMH MINF clean)","#2ca02c",1.5,cmp),
     ("G","chain G (CMH MINF poisoned)","#000000",1.5,cmp),("A","run A (CMH vLLM)","#1f77b4",1.5,cmp),("Q4","chain Q'''' (CMH vLLM no prefix)","#17becf",1.5,cmp)]
W,H,ml,mr,mt,mb=1100,520,60,330,40,50; xmax=40; ymax=30
def X(s): return ml+(W-ml-mr)*(s-1)/(xmax-1)
def Y(v): return mt+(H-mt-mb)*(1-min(v,ymax)/ymax)
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica,Arial" font-size="12"><rect width="{W}" height="{H}" fill="white"/>']
o.append(f'<text x="{ml}" y="22" font-size="15" font-weight="bold">Loop-token share per training step: HSG vLLM-parity arms vs CMH comparators (joint % = repetitive-block tokens / all generated tokens)</text>')
for v in range(0,ymax+1,5): o.append(f'<line x1="{ml}" y1="{Y(v)}" x2="{W-mr}" y2="{Y(v)}" stroke="#ddd"/><text x="{ml-8}" y="{Y(v)+4}" text-anchor="end">{v}%</text>')
for s in range(1,xmax+1,5): o.append(f'<text x="{X(s)}" y="{H-mb+18}" text-anchor="middle">{s}</text>')
o.append(f'<text x="{(ml+W-mr)/2}" y="{H-12}" text-anchor="middle">training step</text>')
o.append(f'<rect x="{X(20)}" y="{mt}" width="{X(30)-X(20)}" height="{H-mt-mb}" fill="#f4f4f4"/><text x="{X(25)}" y="{mt+14}" text-anchor="middle" fill="#666">verdict window 20-30</text>')
ly=mt+10
for k,name,col,w,src in SER:
    d=src.get(k,{}); pts=[(X(s),Y(v)) for s,v in sorted(d.items()) if s<=xmax]
    if len(pts)>1: o.append('<polyline fill="none" stroke="%s" stroke-width="%s" points="%s"/>'%(col,w," ".join(f"{x:.1f},{y:.1f}" for x,y in pts)))
    for s,v in sorted(d.items()):
        if s<=xmax and src is hsg:
            bad=hf.get(k,{}).get(s,0)
            o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="{"white" if bad>50 else col}" stroke="{col}" stroke-width="2"/>')
    o.append(f'<line x1="{W-mr+15}" y1="{ly}" x2="{W-mr+45}" y2="{ly}" stroke="{col}" stroke-width="{w}"/><text x="{W-mr+52}" y="{ly+4}">{name}</text>'); ly+=18
o.append(f'<text x="{W-mr+15}" y="{ly+10}" fill="#444">open circle = step with &gt;50 harness-failed rows (reward forced 0)</text>')
o.append('</svg>'); open(f"{L}/hsg_loopshare.svg","w").write("\n".join(o)); print(f"{L}/hsg_loopshare.svg")

import csv,collections
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
NAMES={"main":"main chain: MINF from scratch (original, no dumps; from trajectories)","G":"chain G: MINF from scratch, no prefix cache","I":"chain I: MINF from scratch, prefix cache on","J":"chain J: MINF from scratch, prefix cache on, replica 2","A":"run A: vLLM from scratch","K":"chain K: vLLM from scratch, seed 1234","M":"chain M: vLLM from scratch, replica 1 (chain J argument set)","N":"chain N: vLLM from scratch, replica 2 (chain J argument set)","S":"chain S: vLLM from scratch, 9-15 stack, masking off","T":"chain T: MINF from scratch, 9-15 stack, masking off","U":"chain U: MINF from scratch, main915 stack","V":"chain V: MINF from scratch, vLLM-parity adapter, prefix cache kept","V2":"chain V2: seed 1234 replica of chain V","V3":"chain V3: seed 4321 replica of chain V","P4":"chain P⁗: MINF from scratch, prefix cache kept across refits","Q4":"chain Q⁗: vLLM from scratch, prefix caching disabled","P4b":"chain P⁗2: MINF from scratch, prefix cache kept, seed 1234","Q4b":"chain Q⁗2: vLLM from scratch, prefix caching disabled, seed 1234","F":"chain F: MINF from vLLM step 10","L":"chain L: MINF from chain K step 10, prefix cache on, seed 1234","B":"run B: MINF from MINF step 10","D":"chain D: vLLM from MINF step 10"}
COL={"main":"#000000","G":"#c0392b","I":"#e67e22","J":"#7f6000","A":"#2980b9","K":"#1abc9c","M":"#5dade2","N":"#154360","S":"#1f618d","T":"#a04000","U":"#873600","V":"#922b21","V2":"#cb4335","V3":"#f1948a","P4":"#6c3483","Q4":"#1a5276","P4b":"#7d3c98","Q4b":"#2e86c1","F":"#27ae60","L":"#145a32","B":"#8e44ad","D":"#d35400"}
DASH={"main":"","G":"","I":"6,3","J":"1,3","A":"","K":"6,3","M":"10,4","N":"2,4","S":"4,3","T":"4,3","U":"1,3","V":"","V2":"6,3","V3":"2,3","P4":"","Q4":"6,3","P4b":"2,3","Q4b":"8,3,2,3","F":"6,3","L":"1,3","B":"2,3","D":"8,3,2,3"}
data=collections.defaultdict(dict); maxstep=36
for r in csv.DictReader(open(f"{L}/joint_loopshare.csv")):
    s=int(r["step"]); data[r["arm"]][s]=float(r["share_all_gen_pct"]); maxstep=max(maxstep,s)
W,H=1250,600; ml,mr,mt,mb=70,420,50,60
ymax=max(5,5*int(max(v for d in data.values() for v in d.values())/5)+5)
def X(s): return ml+(s-1)/(maxstep-1)*(W-ml-mr)
def Y(v): return mt+(1-v/ymax)*(H-mt-mb)
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="13">','<rect width="100%" height="100%" fill="white"/>']
o.append(f'<text x="{ml}" y="28" font-size="16" font-weight="bold">Loop-token share per training step, all arms including the main MINF chain</text>')
o.append(f'<text x="{ml}" y="44" fill="#444">tokens in repetitive reasoning blocks (&#8805; 1,000 tokens, zlib ratio &lt; 0.10) / ALL generated tokens of the step\'s rollouts</text>')
for v in range(0,ymax+1,5):
    o.append(f'<line x1="{ml}" y1="{Y(v)}" x2="{W-mr}" y2="{Y(v)}" stroke="#e5e5e5"/>'); o.append(f'<text x="{ml-8}" y="{Y(v)+4}" text-anchor="end">{v}%</text>')
for s in range(1,maxstep+1):
    if s%5==0 or s==1: o.append(f'<text x="{X(s)}" y="{H-mb+18}" text-anchor="middle">{s}</text>')
    o.append(f'<line x1="{X(s)}" y1="{Y(0)}" x2="{X(s)}" y2="{Y(0)+4}" stroke="#888"/>')
o.append(f'<line x1="{ml}" y1="{Y(0)}" x2="{W-mr}" y2="{Y(0)}" stroke="#333"/><line x1="{ml}" y1="{Y(0)}" x2="{ml}" y2="{mt}" stroke="#333"/>')
o.append(f'<text x="{(ml+W-mr)/2}" y="{H-12}" text-anchor="middle">training step (same 32 prompts per step in every arm)</text>')
o.append(f'<text transform="translate(16,{(mt+H-mb)/2}) rotate(-90)" text-anchor="middle">loop-token share of all generated tokens (%)</text>')
ly=mt+10
for a in ["main","G","I","J","A","K","M","N","S","T","U","V","V2","V3","P4","P4b","Q4","Q4b","F","L","B","D"]:
    pts=[(X(s),Y(v)) for s,v in sorted(data[a].items()) if s>=1]
    if not pts: continue
    d=" ".join(f"{x:.1f},{y:.1f}" for x,y in pts); dash=f' stroke-dasharray="{DASH[a]}"' if DASH[a] else ""
    sw="3" if a=="main" else "2.2"
    o.append(f'<polyline points="{d}" fill="none" stroke="{COL[a]}" stroke-width="{sw}"{dash}/>')
    for x,y in pts: o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{COL[a]}"/>')
    o.append(f'<line x1="{W-mr+15}" y1="{ly}" x2="{W-mr+55}" y2="{ly}" stroke="{COL[a]}" stroke-width="{sw}"{dash}/><text x="{W-mr+62}" y="{ly+4}">{NAMES[a]}</text>'); ly+=22
o.append('</svg>')
open(f"{L}/loopshare_all.svg","w").write("\n".join(o)); print("wrote",f"{L}/loopshare_all.svg")

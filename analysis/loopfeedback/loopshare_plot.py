import csv,collections
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
NAMES={"G":"chain G: MINF from scratch, no prefix cache","I":"chain I: MINF from scratch, prefix cache on","A":"run A: vLLM from scratch","F":"chain F: MINF from vLLM step 10","B":"run B: MINF from MINF step 10","D":"chain D: vLLM from MINF step 10"}
COL={"G":"#c0392b","I":"#e67e22","A":"#2980b9","F":"#27ae60","B":"#8e44ad","D":"#d35400"}
DASH={"G":"","I":"6,3","A":"","F":"6,3","B":"2,3","D":"8,3,2,3"}
data=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{L}/loopshare.csv")):
    data[r["arm"]][int(r["step"])]=100*int(r["repetitive_tokens"])/max(1,int(r["reasoning_tokens"]))
W,H=1100,560; ml,mr,mt,mb=70,330,50,60
xs=list(range(1,37)); ymax=45
def X(s): return ml+(s-1)/(36-1)*(W-ml-mr)
def Y(v): return mt+(1-v/ymax)*(H-mt-mb)
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="13">','<rect width="100%" height="100%" fill="white"/>']
o.append(f'<text x="{ml}" y="28" font-size="16" font-weight="bold">Loop-token share per training step</text>')
o.append(f'<text x="{ml}" y="44" fill="#444">tokens inside repetitive reasoning blocks (block &#8805; 1,000 tokens, zlib ratio &lt; 0.10) / all reasoning tokens of the step\'s 512 trained rollouts</text>')
for v in range(0,ymax+1,5):
    o.append(f'<line x1="{ml}" y1="{Y(v)}" x2="{W-mr}" y2="{Y(v)}" stroke="#e5e5e5"/>'); o.append(f'<text x="{ml-8}" y="{Y(v)+4}" text-anchor="end">{v}%</text>')
for s in xs:
    if s%5==0 or s==1: o.append(f'<text x="{X(s)}" y="{H-mb+18}" text-anchor="middle">{s}</text>')
    o.append(f'<line x1="{X(s)}" y1="{Y(0)}" x2="{X(s)}" y2="{Y(0)+4}" stroke="#888"/>')
o.append(f'<line x1="{ml}" y1="{Y(0)}" x2="{W-mr}" y2="{Y(0)}" stroke="#333"/><line x1="{ml}" y1="{Y(0)}" x2="{ml}" y2="{mt}" stroke="#333"/>')
o.append(f'<text x="{(ml+W-mr)/2}" y="{H-12}" text-anchor="middle">training step (batch trained at that step; same 32 prompts in every arm)</text>')
o.append(f'<text transform="translate(16,{(mt+H-mb)/2}) rotate(-90)" text-anchor="middle">loop-token share (%)</text>')
ly=mt+10
for a in ["G","I","A","F","B","D"]:
    pts=[(X(s),Y(v)) for s,v in sorted(data[a].items())]
    if not pts: continue
    d=" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    dash=f' stroke-dasharray="{DASH[a]}"' if DASH[a] else ""
    o.append(f'<polyline points="{d}" fill="none" stroke="{COL[a]}" stroke-width="2.2"{dash}/>')
    for x,y in pts: o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{COL[a]}"/>')
    o.append(f'<line x1="{W-mr+15}" y1="{ly}" x2="{W-mr+55}" y2="{ly}" stroke="{COL[a]}" stroke-width="2.2"{dash}/><text x="{W-mr+62}" y="{ly+4}">{NAMES[a]}</text>')
    ly+=22
o.append('</svg>')
open(f"{L}/loopshare.svg","w").write("\n".join(o))
print("wrote",f"{L}/loopshare.svg")

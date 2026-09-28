import csv,json,glob,collections,datetime
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
SERIES=[("A","run A","vLLM from scratch","#2a78d6","7,4"),("G","chain G","MINF from scratch, no prefix cache","#eb6834",""),("K","chain K","vLLM from scratch, seed 1234","#1baf7a","7,4"),("M","chain M","vLLM from scratch, replica 1, chain J argument set (blue squares)","#2a78d6","10,4"),("N","chain N","vLLM from scratch, replica 2, chain J argument set (blue diamonds)","#2a78d6","2,4"),("S","chain S","vLLM from scratch, clean 2026-09-15 stack, flagged-sample masking off (blue triangles)","#2a78d6","4,3"),("Q4","chain Q⁗","vLLM from scratch with vLLM prefix caching DISABLED; engine control for chain P⁗ (blue triangles down). Every earlier vLLM arm ran with prefix caching on.","#2a78d6","7,4"),("Q4b","chain Q⁗2","second seed (grpo.seed 1234) of chain Q⁗: vLLM from scratch, prefix caching disabled (aqua triangles down)","#1baf7a","2,4"),
        ("I","chain I","MINF from scratch, prefix cache on","#eda100",""),("J","chain J","MINF from scratch, prefix cache on, replica 2 (same hue as chain I, square markers)","#eda100","10,4"),("main","main chain","MINF from scratch (original; from trajectories)","#e87ba4",""),("T","chain T","MINF from scratch, clean 2026-09-15 stack, flagged-sample masking off (orange triangles)","#eb6834",""),("V","chain V","MINF from scratch on the Megatron vLLM-numerical-parity inference adapter (vLLM kernels and rounding order inside MINF), prefix cache kept across refits (red diamonds)","#e34948",""),("V2","chain V2","seed 1234 replica of chain V: MINF from scratch, vLLM-parity adapter, prefix cache kept (red squares)","#e34948","10,4"),("V3","chain V3","seed 4321 replica of chain V (red triangles up)","#e34948","2,4"),("P4","chain P⁗","MINF from scratch, prefix cache kept across weight updates (invalidate_prefix_cache_on_weight_update=false), nvshmem refit (violet triangles)","#4a3aa7",""),("P4b","chain P⁗2","second seed (grpo.seed 1234) of chain P⁗: MINF from scratch, prefix cache kept across weight updates (violet squares)","#4a3aa7","10,4"),
        ("F","chain F","MINF from vLLM step 10","#008300","2,4"),("L","chain L","MINF from chain K step 10, prefix cache on, seed 1234 (black squares)","#0b0b0b","10,4"),("B","run B","MINF from MINF step 10","#4a3aa7","2,4"),("D","chain D","vLLM from MINF step 10","#e34948","2,4")]
SURF,INK,INK2,GRID,ZEBRA="#fcfcfb","#0b0b0b","#52514e","#e6e5e1","#f3f2ef"
EVAL={"A":"good","F":"good","G":"bad","main":"bad","I":"presumed","D":"presumed","B":"presumed","L":"presumed","J":"good","K":"presumed_good","M":"presumed_good"}; STATUS={"good":"#0ca30c","bad":"#d03b3b","presumed":"#fab219","presumed_good":"#3987e5"}; EVLAB={"good":"eval good","bad":"eval bad","presumed":"presumed bad","presumed_good":"presumed good"}; EVSYM={"good":"\u2713","bad":"\u2717","presumed":"?","presumed_good":"?"}; EVOP={"good":0.10,"bad":0.10,"presumed":0.16,"presumed_good":0.12}
data=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{L}/joint_loopshare.csv")): data[r["arm"]][int(r["step"])]=float(r["share_all_gen_pct"])
partial=set()
for r in csv.DictReader(open(f"{L}/loopshare.csv")):
    if int(r["rollouts"])<512: partial.add((r["arm"],int(r["step"])))
rowmap=json.load(open(f"{L}/dataset_row_map.json")); mn=collections.Counter()
for fn in glob.glob(f"{L}/mainchain/*.csv"):
    for r in csv.DictReader(open(fn)):
        rm=rowmap.get(r["instance_id"])
        if rm: mn[rm["steps"][0]]+=1
for s,n in mn.items():
    if n>=1000: partial.add(("main",s))
partial.add(("J",13))
maxstep=max(s for d in data.values() for s in d); ymax=5*int(max(v for d in data.values() for v in d.values())/5)+5
TABLES=[("Table 1: arms trained from scratch. Loop-token share (%) by step and arm",["A","K","M","N","S","Q4","Q4b","G","I","J","main","T","V","V2","V3","P4","P4b"]),
        ("Table 2: transplants from a step-10 checkpoint",["F","L","B","D"])]
W=max(1600,52+122*max(len(k) for _,k in TABLES)+140); ml,mr,mt=64,372,84; PH=560; pb=mt+PH
def X(s): return ml+(s-1)/(maxstep-1)*(W-ml-mr)
def Y(v): return mt+(1-v/ymax)*PH
o=[]; A=o.append
esc=lambda t:t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
# ---- plot
A(f'<text x="{ml}" y="30" font-size="17" font-weight="600" fill="{INK}">Loop-token share per training step, all arms</text>')
import textwrap as _tw
for _k,_line in enumerate(_tw.wrap(f"share = tokens inside repetitive reasoning blocks (block &#8805; 1,000 tokens, zlib ratio &lt; 0.10) / all generated tokens of the step\'s rollouts. Same 32 prompts per step in every arm. Solid = MINF from scratch, dashed = vLLM from scratch, dotted = transplants from a step-10 checkpoint; triangles = arms started 2026-09-25 (chains S, T on the clean 2026-09-15 stack with flagged-sample masking off; chain P⁗ = MINF from scratch with the prefix cache kept across refits; chain Q⁗ = vLLM from scratch with prefix caching disabled).", 175)):
    A(f'<text x="{ml}" y="{50+14*_k}" font-size="12.5" fill="{INK2}">{_line}</text>')
for v in range(0,ymax+1,5):
    A(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{W-mr}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>'); A(f'<text x="{ml-8}" y="{Y(v)+4:.1f}" font-size="11.5" fill="{INK2}" text-anchor="end">{v}%</text>')
for s in range(1,maxstep+1):
    if s%5==0 or s==1: A(f'<text x="{X(s):.1f}" y="{pb+18}" font-size="11.5" fill="{INK2}" text-anchor="middle">{s}</text>')
A(f'<line x1="{ml}" y1="{pb}" x2="{W-mr}" y2="{pb}" stroke="{INK2}" stroke-width="1"/>')
A(f'<text x="{(ml+W-mr)/2:.0f}" y="{pb+38}" font-size="12" fill="{INK2}" text-anchor="middle">training step</text>')
A(f'<text transform="translate(18,{mt+PH/2:.0f}) rotate(-90)" font-size="12" fill="{INK2}" text-anchor="middle">loop tokens / all generated tokens (%)</text>')
def marker(key,x,y,hs,col,title=""):
    if key in ("J","L","M","P4b","V2"): shape,tag=f'<rect x="{x-hs:.1f}" y="{y-hs:.1f}" width="{2*hs}" height="{2*hs}"',"rect"
    elif key in ("N","U","V"): shape,tag=f'<rect x="{x-hs:.1f}" y="{y-hs:.1f}" width="{2*hs}" height="{2*hs}" transform="rotate(45 {x:.1f} {y:.1f})"',"rect"
    elif key in ("S","T","P4","V3"): shape,tag=f'<polygon points="{x:.1f},{y-hs-1:.1f} {x+hs+1:.1f},{y+hs:.1f} {x-hs-1:.1f},{y+hs:.1f}"',"polygon"
    elif key in ("Q4","Q4b"): shape,tag=f'<polygon points="{x:.1f},{y+hs+1:.1f} {x+hs+1:.1f},{y-hs:.1f} {x-hs-1:.1f},{y-hs:.1f}"',"polygon"
    else: shape,tag=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{hs}"',"circle"
    return f'{shape} fill="{col}" stroke="{SURF}" stroke-width="2">'+(f'<title>{esc(title)}</title>' if title else '')+f'</{tag}>'
for key,short,desc,col,dash in SERIES:
    pts=sorted(data[key].items())
    if not pts: continue
    d=" ".join(f"{X(s):.1f},{Y(v):.1f}" for s,v in pts); da=f' stroke-dasharray="{dash}"' if dash else ""
    sw="3" if key=="L" else "2"
    A(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round"{da}/>')
    for s,v in pts:
        hs=5 if key in ("L","S","T","P4","Q4","Q4b","V3") else 4
        A(marker(key,X(s),Y(v),hs,col,short+" ("+desc+") - step "+str(s)+": "+f"{v:.1f}%"+(" (partial step)" if (key,s) in partial else "")))
    ls,lv=pts[-1]; A(f'<text x="{X(ls)+9:.1f}" y="{Y(lv)+4:.1f}" font-size="11.5" fill="{INK}"{" font-weight=\"700\"" if key=="L" else ""}>{short}</text>')
# legend
import textwrap
lx=W-mr+30; ly=mt+6
A(f'<text x="{lx}" y="{ly}" font-size="12" font-weight="600" fill="{INK}">arms</text>'); ly+=16
for key,short,desc,col,dash in SERIES:
    da=f' stroke-dasharray="{dash}"' if dash else ""
    A(f'<line x1="{lx}" y1="{ly}" x2="{lx+34}" y2="{ly}" stroke="{col}" stroke-width="2"{da}/>'+marker(key,lx+17,ly,4,col))
    A(f'<text x="{lx+42}" y="{ly+4}" font-size="11.5" fill="{INK}">{short}</text>')
    dl=textwrap.wrap(desc, 46)[:3]
    for k,line in enumerate(dl): A(f'<text x="{lx+42}" y="{ly+17+12*k}" font-size="10.5" fill="{INK2}">{esc(line)}</text>')
    ly+=20+12*len(dl)
# ---- tables
HDR={"step":("step",""),"A":("A: vLLM","from scratch"),"G":("G: MINF","from scratch, no pfx"),"K":("K: vLLM","from scratch, seed 1234"),"M":("M: vLLM","from scratch, r1 (J args)"),"N":("N: vLLM","from scratch, r2 (J args)"),"S":("S: vLLM","9-15 stack, mask off"),"T":("T: MINF","9-15 stack, mask off"),"U":("U: MINF","main915 stack"),"V":("V: MINF","vLLM-parity, keep pfx"),"V2":("V2: MINF","vLLM-parity, seed 1234"),"V3":("V3: MINF","vLLM-parity, seed 4321"),"P4":("P⁗: MINF","from scratch, keep pfx"),"P4b":("P⁗2: MINF","keep pfx, seed 1234"),"Q4":("Q⁗: vLLM","from scratch, no pfx"),"Q4b":("Q⁗2: vLLM","no pfx, seed 1234"),"I":("I: MINF","from scratch, pfx r1"),"J":("J: MINF","from scratch, pfx r2"),"main":("main: MINF","from scratch (orig.)"),"F":("F: MINF","<- vLLM step 10"),"L":("L: MINF","<- K step 10, pfx"),"B":("B: MINF","<- MINF step 10"),"D":("D: vLLM","<- MINF step 10")}
hh=30; rh=15
cw1=[52]+[122]*len(TABLES[0][1]); tx0=(W-sum(cw1))//2
def NOTE(text, y, size=11, width=205):
    lines=textwrap.wrap(text, width)
    for k,line in enumerate(lines): A(f'<text x="{tx0}" y="{y+3+13*k}" font-size="{size}" fill="{INK2}">{esc(line)}</text>')
    return y+13*max(1,len(lines))
def draw_table(keys, ty):
    cw=[52]+[122]*len(keys); tx=(W-sum(cw))//2
    present=[s for s in range(1,maxstep+1) if any(s in data[k] for k in keys)]
    if not present: return ty
    s0=present[0]; nrows=maxstep-s0+1
    x=tx+cw[0]
    for key,wd in zip(keys,cw[1:]):
        ev=EVAL.get(key)
        if ev:
            A(f'<rect x="{x}" y="{ty-16}" width="{wd}" height="{16+hh+2+nrows*rh+6}" fill="{STATUS[ev]}" fill-opacity="{EVOP[ev]}"/>')
            A(f'<circle cx="{x+6}" cy="{ty-8}" r="4" fill="{STATUS[ev]}"/><text x="{x+14}" y="{ty-4}" font-size="10" fill="{INK2}">{EVSYM[ev]} {EVLAB[ev]}</text>')
        x+=wd
    A(f'<rect x="{tx}" y="{ty}" width="{sum(cw)}" height="{hh}" fill="{ZEBRA}" fill-opacity="0.7"/>')
    x=tx
    for c,wd in zip(["step"]+keys,cw):
        l1,l2=HDR[c]
        A(f'<text x="{x+wd-4}" y="{ty+12}" font-size="10.5" font-weight="600" fill="{INK}" text-anchor="end" font-family="monospace">{esc(l1)}</text>')
        if l2: A(f'<text x="{x+wd-4}" y="{ty+25}" font-size="9.5" fill="{INK2}" text-anchor="end" font-family="monospace">{esc(l2)}</text>')
        x+=wd
    ty+=hh+rh
    for s in range(s0,maxstep+1):
        if s%2==0: A(f'<rect x="{tx}" y="{ty-11}" width="{sum(cw)}" height="{rh}" fill="{ZEBRA}"/>')
        x=tx; A(f'<text x="{x+cw[0]-4}" y="{ty}" font-size="11" fill="{INK2}" text-anchor="end" font-family="monospace">{s}</text>'); x+=cw[0]
        for key,wd in zip(keys,cw[1:]):
            v=data[key].get(s); txt="" if v is None else f"{v:.1f}"+("*" if (key,s) in partial else "")
            A(f'<text x="{x+wd-4}" y="{ty}" font-size="11" fill="{INK}" text-anchor="end" font-family="monospace">{txt}</text>'); x+=wd
        ty+=rh
    return ty
ty=max(pb+70,ly+40)
A(f'<text x="{tx0}" y="{ty}" font-size="13" font-weight="600" fill="{INK}">{TABLES[0][0]}</text>'); ty+=14
ty=NOTE(f'* = partial step (fewer than 512 rollouts), main-chain step with rollouts regenerated after a restart, or a step never checkpointed (chain J step 13). Blank = no data. Main-chain step 1 is missing from its Gym results. Two tables because 14 arms do not fit one row; both share the step axis.', ty)
ty=NOTE(f'Column shading = eval outcome of the arm: green {EVSYM["good"]} eval good (SWE-Bench Verified pass@1 held or improved), red {EVSYM["bad"]} eval bad (declined); yellow {EVSYM["presumed"]} presumed bad and light blue {EVSYM["presumed_good"]} presumed good = not yet evaluated, called from the loop trajectory; unshaded = no call.', ty)
ty+=36
ty=draw_table(TABLES[0][1], ty)
ty+=34
tx2=(W-sum([52]+[122]*len(TABLES[1][1])))//2
A(f'<text x="{tx2}" y="{ty}" font-size="13" font-weight="600" fill="{INK}">{TABLES[1][0]}</text>'); ty+=36
ty=draw_table(TABLES[1][1], ty)
ty+=14
ty=NOTE(f'Seeds: grpo.seed 42 everywhere except chain K and chain L (1234); chain M and chain N = vLLM from scratch with the chain J argument set (overlap_param_gather off), seed 42, replicas 1 and 2. Chains S and T = the clean 2026-09-15 stack (nemo_rl 952eaf85b + dumps, MLM 880de0fce + guard) with Gym flagged-sample masking OFF, no prefix cache, overlap off, seed 42. Chain P⁗ = MINF from scratch (chain G/J recipe, nvshmem, overlap off) with invalidate_prefix_cache_on_weight_update=false, so prefix-cache blocks survive each refit; same knob as chain P‴ in the splice table; chain P⁗2 = its second seed (grpo.seed 1234). Chain Q⁗2 = second seed (1234) of chain Q⁗. Chain V = MINF from scratch with the vLLM numerical-parity adapter (inference_vllm_parity=true: vLLM kernels and rounding order, TP4 generation) and the prefix cache kept, started 2026-09-27 14:30 (segment timed out 2026-09-28 07:33 at step 35); chains V2 and V3 = its seed 1234 and 4321 replicas started 2026-09-28. Chain U (main915 stack, MINF from scratch, 2026-09-27) is registered but withheld from this table: its dumps show single-message episodes of 50-520 generated tokens with reward 0 on every rollout, so its 0.0 % loop share is an artifact of a broken run, not a result. Chain Q⁗ = vLLM from scratch (run A / chain M recipe) with vLLM prefix caching disabled; note that every earlier vLLM arm (A, K, M, N, S, Q, R, D) ran with vLLM prefix caching ON (the default in this tree on this hardware). Dynamic-engine guard (applied 2026-09-22 07:47): chain J and chain L fully guarded; chain F and chain G unguarded; chain I unguarded for steps 1-15, guarded from 16; main chain unguarded. chain F ran overlap_param_gather on and its prefix caching turned off after step 18; chain L is internally consistent (prefix on, overlap off).', ty-3, size=10.5, width=215)
ty=NOTE(f'Rendered {datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z")} from joint_loopshare.csv / loopshare.csv. Dump arms exact from token dumps; main chain from OpenHands trajectories (rollouts with an 8k+ turn scanned). Colors: reference categorical palette, fixed slot order; identity also carried by dash pattern, marker shape, direct labels and the tables.', ty-3, size=10.5, width=215)
H=ty+16
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',f'<rect width="100%" height="100%" fill="{SURF}"/>']+o+['</svg>']
open(f"{L}/loopshare_all_table.svg","w").write("\n".join(svg)); print("wrote",f"{L}/loopshare_all_table.svg",f"{W}x{H}")

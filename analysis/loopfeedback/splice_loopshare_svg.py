import csv,json,collections,datetime,textwrap
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
ARMS=json.load(open(f"{L}/splice_arms.json"))
def RS(c): return set(c.get("replay_steps") or range(1,c["replay_end"]+1))
def is_replay(key,s): return key in ARMS and ARMS[key].get("kind","splice")=="splice" and s in RS(ARMS[key])
def contiguous_from_1(c): r=sorted(RS(c)); return r==list(range(1,len(r)+1))
SERIES=[("P","chain P","MINF engine; steps 1-20 replay chain M's recorded rollouts (no generation), live MINF generation from step 21 (triangles up)","#4a3aa7",""),
        ("Q","chain Q","vLLM engine; same replay of chain M steps 1-20, live vLLM generation from step 21 (triangles down)","#4a3aa7","7,4"),
        ("R","chain R","vLLM engine; steps 1-10 replay chain G's recorded rollouts, live vLLM generation from step 11 (green triangles down)","#008300","7,4"),
        ("Rp","chain R′","vLLM engine; steps 1-16 replay chain G's recorded rollouts, live vLLM generation from step 17 (green triangles down, dotted)","#008300","2,4"),
        ("X","chain X","vLLM engine; steps 1-10 replay chain G with its 127 rewarded deep-think rollouts masked (test), live vLLM from step 11 (red triangles down)","#e34948","7,4"),
        ("Y","chain Y","vLLM engine; steps 1-10 replay chain G with 127 random rewarded rollouts masked (control), live vLLM from step 11 (red triangles down, dotted)","#e34948","2,4"),
        ("Z","chain Z","vLLM engine; steps 1-10 replay chain M with its 213 punished deep-think rollouts masked (mirror), live vLLM from step 11 (black triangles down)","#0b0b0b","7,4"),
        ("AA","chain AA","vLLM engine; step-range ablation: only chain G steps 8-10 replayed, steps 1-7 and 11-25 live vLLM (aqua triangles down)","#1baf7a","7,4"),
        ("AB","chain AB","vLLM engine; step-range ablation: chain G steps 1-7 replayed, live vLLM from step 8 (aqua triangles down, dotted)","#1baf7a","2,4"),
        ("Pp","chain P′","MINF engine on FROZEN weights (lr = 0) taken from chain P step 20; live MINF generation from step 21 (magenta triangles up)","#e87ba4",""),
        ("Qp","chain Q′","vLLM engine on the same frozen chain P step-20 weights; live vLLM generation from step 21 (magenta triangles down)","#e87ba4","7,4"),
        ("Pt","chain P‴","MINF engine, chain P step 20 resumed with normal training; prefix cache KEPT across weight updates (invalidate_prefix_cache_on_weight_update=false), nvshmem refit (green triangles up)","#008300",""),
        ("M","chain M","vLLM from scratch, replica 1, chain J argument set; source of the rollouts replayed by P and Q (blue squares)","#2a78d6","10,4"),
        ("N","chain N","vLLM from scratch, replica 2, chain J argument set (blue diamonds)","#2a78d6","2,4"),
        ("G","chain G","MINF from scratch, no prefix cache; source of the rollouts replayed by R and R′","#eb6834",""),
        ("I","chain I","MINF from scratch, prefix cache on","#eda100",""),
        ("J","chain J","MINF from scratch, prefix cache on, replica 2 (same hue as chain I, square markers)","#eda100","10,4"),
        ("A","run A","vLLM from scratch","#2a78d6","7,4"),
        ("K","chain K","vLLM from scratch, seed 1234","#1baf7a","7,4"),
        ("D","chain D","vLLM from MINF step 10 (transplant; red circles, dotted); comparator for chains R and R′","#e34948","2,4")]
SURF,INK,INK2,GRID,ZEBRA,REPLAY="#fcfcfb","#0b0b0b","#52514e","#e6e5e1","#f3f2ef","#a9a7a2"
EVAL={"A":"good","G":"bad","I":"presumed","J":"good","K":"presumed_good","M":"presumed_good","D":"presumed"}; STATUS={"good":"#0ca30c","bad":"#d03b3b","presumed":"#fab219","presumed_good":"#3987e5"}; EVLAB={"good":"eval good","bad":"eval bad","presumed":"presumed bad","presumed_good":"presumed good"}; EVSYM={"good":"✓","bad":"✗","presumed":"?","presumed_good":"?"}; EVOP={"good":0.10,"bad":0.10,"presumed":0.16,"presumed_good":0.12}
data=collections.defaultdict(dict); partial=set()
for r in csv.DictReader(open(f"{L}/splice_loopshare.csv")):
    a,s=r["arm"],int(r["step"]); data[a][s]=float(r["share_all_gen_pct"])
    if r["rollouts"] and int(r["rollouts"])<512: partial.add((a,s))
for r in csv.DictReader(open(f"{L}/loopshare.csv")):
    if int(r["rollouts"])<512: partial.add((r["arm"],int(r["step"])))
partial.add(("J",13))
ACTIVE={a:c for a,c in ARMS.items() if data.get(a)}
SERIES=[t for t in SERIES if t[0] not in ARMS or t[0] in ACTIVE]
focus=[s for a in list(ACTIVE)+["M","N"] for s in data.get(a,{})]
maxstep=max(30,max(focus) if focus else 30)
for a in list(data): data[a]={s:v for s,v in data[a].items() if s<=maxstep}
ymax=5*int(max(v for d in data.values() for v in d.values())/5)+5
W=max(1600,52+122*len(SERIES)+140); ml,mr,mt=64,372,84; PH=560; pb=mt+PH
def X(s): return ml+(s-1)/(maxstep-1)*(W-ml-mr)
def Y(v): return mt+(1-v/ymax)*PH
o=[]; A=o.append
esc=lambda t:t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def lab(a): return ARMS[a].get("label","chain "+a)
SPL={a:c for a,c in ACTIVE.items() if c.get("kind","splice")=="splice"}; LR0={a:c for a,c in ACTIVE.items() if c.get("kind")=="lr0"}; RES={a:c for a,c in ACTIVE.items() if c.get("kind")=="resume"}
groups=collections.OrderedDict()
for a,c in SPL.items():
    if contiguous_from_1(c): groups.setdefault((c["source"],max(RS(c))),[]).append(a)
design="; ".join(f"{'/'.join(lab(a) for a in arms)}: steps 1-{e} train on chain {src}\'s recorded rollouts instead of generating, live generation with their own engine from step {e+1}" for (src,e),arms in groups.items())
odd=[(a,c) for a,c in SPL.items() if not contiguous_from_1(c)]
if odd: design+="; "+"; ".join(f"{lab(a)}: only chain {c['source']} steps {min(RS(c))}-{max(RS(c))} replayed, live on the other steps" for a,c in odd)
if LR0: design+="; "+"; ".join(f"{lab(a)}: weights frozen at chain {c['source']} step {c['replay_end']} (lr = 0), {c['engine']} generation from step {c['replay_end']+1}" for a,c in LR0.items())
if RES: design+="; "+"; ".join(f"{lab(a)}: {c['design']}, live from step {c['replay_end']+1}" for a,c in RES.items())
engines=", ".join(f"{lab(a)}: {c['engine']}" for a,c in ACTIVE.items())
A(f'<text x="{ml}" y="30" font-size="17" font-weight="600" fill="{INK}">Loop-token share per training step: cross-train (splice) runs with their associated arms</text>')
for _k,_line in enumerate(textwrap.wrap(f"share = tokens inside repetitive reasoning blocks (block &#8805; 1,000 tokens, zlib ratio &lt; 0.10) / all generated tokens of the step\'s rollouts. Splice runs start from the base model. {design} ({engines}). Solid = MINF-generated, dashed = vLLM-generated; grey = replayed source data. Comparator arms are clipped at step {maxstep}; their full history is in loopshare_all_table.svg.", 175)):
    A(f'<text x="{ml}" y="{50+14*_k}" font-size="12.5" fill="{INK2}">{_line}</text>')
ends=sorted({max(RS(c)) for c in SPL.values() if contiguous_from_1(c) and max(RS(c))<maxstep})
if ends:
    A(f'<rect x="{ml}" y="{mt}" width="{X(ends[0]+0.5)-ml:.1f}" height="{PH}" fill="{ZEBRA}" fill-opacity="0.8"/>')
    for i,e in enumerate(ends):
        arms=", ".join(lab(a).replace("chain ","") for a,c in SPL.items() if contiguous_from_1(c) and max(RS(c))==e)
        A(f'<line x1="{X(e+0.5):.1f}" y1="{mt}" x2="{X(e+0.5):.1f}" y2="{pb}" stroke="{INK2}" stroke-width="1" stroke-dasharray="4,3"/>')
        A(f'<text x="{X(e+0.5)-6:.1f}" y="{mt+14+13*i}" font-size="11" fill="{INK2}" text-anchor="end">{arms}: replay ends (step {e})</text>')
    A(f'<text x="{X(ends[-1]+0.5)+6:.1f}" y="{mt+14}" font-size="11" fill="{INK2}">live generation</text>')
for v in range(0,ymax+1,5):
    A(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{W-mr}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>'); A(f'<text x="{ml-8}" y="{Y(v)+4:.1f}" font-size="11.5" fill="{INK2}" text-anchor="end">{v}%</text>')
for s in range(1,maxstep+1):
    if s%5==0 or s==1: A(f'<text x="{X(s):.1f}" y="{pb+18}" font-size="11.5" fill="{INK2}" text-anchor="middle">{s}</text>')
A(f'<line x1="{ml}" y1="{pb}" x2="{W-mr}" y2="{pb}" stroke="{INK2}" stroke-width="1"/>')
A(f'<text x="{(ml+W-mr)/2:.0f}" y="{pb+38}" font-size="12" fill="{INK2}" text-anchor="middle">training step</text>')
A(f'<text transform="translate(18,{mt+PH/2:.0f}) rotate(-90)" font-size="12" fill="{INK2}" text-anchor="middle">loop tokens / all generated tokens (%)</text>')
def marker(key,x,y,hs,col,title,replay=False):
    fill=SURF if replay else col; stroke=REPLAY if replay else SURF
    if key in ("P","Pp","Pt"): shape=f'<polygon points="{x:.1f},{y-hs-1:.1f} {x+hs+1:.1f},{y+hs:.1f} {x-hs-1:.1f},{y+hs:.1f}"'
    elif key in ("Q","R","Rp","Qp","X","Y","Z","AA","AB"): shape=f'<polygon points="{x:.1f},{y+hs+1:.1f} {x+hs+1:.1f},{y-hs:.1f} {x-hs-1:.1f},{y-hs:.1f}"'
    elif key in ("J","M"): shape=f'<rect x="{x-hs:.1f}" y="{y-hs:.1f}" width="{2*hs}" height="{2*hs}"'
    elif key=="N": shape=f'<rect x="{x-hs:.1f}" y="{y-hs:.1f}" width="{2*hs}" height="{2*hs}" transform="rotate(45 {x:.1f} {y:.1f})"'
    else: shape=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{hs}"'
    tag="polygon" if key in ("P","Q","R","Rp","Pp","Qp","Pt","X","Y","Z","AA","AB") else ("rect" if key in ("J","M","N") else "circle")
    return f'{shape} fill="{fill}" stroke="{stroke}" stroke-width="2"><title>{esc(title)}</title></{tag}>'
for key,short,desc,col,dash in SERIES:
    pts=sorted(data[key].items())
    if not pts: continue
    da=f' stroke-dasharray="{dash}"' if dash else ""; sw="3" if key in ARMS else "2"; hs=5 if key in ARMS else 4
    if key in ARMS:
        runs=[]
        for s,v in pts:
            m=is_replay(key,s)
            if runs and runs[-1][0]==m: runs[-1][1].append((s,v))
            else:
                if runs: runs[-1][1].append((s,v))
                runs.append([m,[(s,v)]])
        for m,seg in runs:
            if len(seg)<2: continue
            if m: A(f'<polyline points="{" ".join(f"{X(s):.1f},{Y(v):.1f}" for s,v in seg)}" fill="none" stroke="{REPLAY}" stroke-width="1.5" stroke-linejoin="round"{da}/>')
            else: A(f'<polyline points="{" ".join(f"{X(s):.1f},{Y(v):.1f}" for s,v in seg)}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round"{da}/>')
    else:
        A(f'<polyline points="{" ".join(f"{X(s):.1f},{Y(v):.1f}" for s,v in pts)}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round"{da}/>')
    for s,v in pts:
        replay=is_replay(key,s)
        A(marker(key,X(s),Y(v),hs,col,short+" ("+desc+") - step "+str(s)+": "+f"{v:.1f}%"+(f" (replayed chain {ARMS[key]['source']} data)" if replay else "")+(" (partial step)" if (key,s) in partial else ""),replay))
    ls,lv=pts[-1]; A(f'<text x="{X(ls)+9:.1f}" y="{Y(lv)+4:.1f}" font-size="11.5" fill="{INK}"{" font-weight=\"700\"" if key in ARMS else ""}>{short}</text>')
lx=W-mr+30; ly=mt+6
A(f'<text x="{lx}" y="{ly}" font-size="12" font-weight="600" fill="{INK}">arms</text>'); ly+=16
for key,short,desc,col,dash in SERIES:
    da=f' stroke-dasharray="{dash}"' if dash else ""
    A(f'<line x1="{lx}" y1="{ly}" x2="{lx+34}" y2="{ly}" stroke="{col}" stroke-width="2"{da}/>'+marker(key,lx+17,ly,4,col,short))
    A(f'<text x="{lx+42}" y="{ly+4}" font-size="11.5" fill="{INK}">{short}</text>')
    dl=textwrap.wrap(desc, 46)[:4]
    for k,line in enumerate(dl): A(f'<text x="{lx+42}" y="{ly+17+12*k}" font-size="10.5" fill="{INK2}">{esc(line)}</text>')
    ly+=20+12*len(dl)
A(f'<line x1="{lx}" y1="{ly}" x2="{lx+34}" y2="{ly}" stroke="{REPLAY}" stroke-width="1.5"/>'+marker("P",lx+17,ly,4,"#4a3aa7","replayed",True)+f'<text x="{lx+42}" y="{ly+4}" font-size="11.5" fill="{INK}">grey = replayed step</text>')
for k,line in enumerate(textwrap.wrap("value equals the source arm's by construction (same rollouts); this engine generated nothing", 46)): A(f'<text x="{lx+42}" y="{ly+17+12*k}" font-size="10.5" fill="{INK2}">{esc(line)}</text>')
ty=max(pb+70,ly+44); rh=15; cols=["step"]+[k for k,*_ in SERIES]; cw=[52]+[122]*len(SERIES); tx0=(W-sum(cw))//2
A(f'<text x="{tx0}" y="{ty}" font-size="13" font-weight="600" fill="{INK}">Table: loop-token share (%) by step and arm</text>')
def NOTE(text, y, size=11, width=205):
    lines=textwrap.wrap(text, width)
    for k,line in enumerate(lines): A(f'<text x="{tx0}" y="{y+3+13*k}" font-size="{size}" fill="{INK2}">{esc(line)}</text>')
    return y+13*max(1,len(lines))
ty+=14
ty=NOTE(f'r = replayed step (grey): the splice run trained on its source arm\'s recorded rollouts for this step and generated nothing, so the value equals the source arm\'s by construction and only checks the tooling. Live steps are the experiment. * = partial step (fewer than 512 rollouts) or a step never checkpointed (chain J step 13). Blank = no data.', ty)
ty=NOTE(f'Column shading = eval outcome of the arm: green {EVSYM["good"]} eval good, red {EVSYM["bad"]} eval bad (SWE-Bench Verified pass@1); yellow {EVSYM["presumed"]} presumed bad and light blue {EVSYM["presumed_good"]} presumed good = not yet evaluated, called from the loop trajectory; unshaded = no call (chains M, N and the splice / frozen-weight runs).', ty)
ty+=36
nrows=maxstep
HDR={"step":("step",""),"P":("P: MINF","M replay 1-20, live 21+"),"Q":("Q: vLLM","M replay 1-20, live 21+"),"X":("X: vLLM","G replay 1-10 masked"),"AA":("AA: vLLM","G replay 8-10 only"),"AB":("AB: vLLM","G replay 1-7, live 8+"),"Y":("Y: vLLM","G replay 1-10 control"),"Z":("Z: vLLM","M replay 1-10 masked"),"Pp":("P′: MINF lr0","frozen P step-20 wts"),"Qp":("Q′: vLLM lr0","frozen P step-20 wts"),"Pt":("P‴: MINF","<- P step 20, keep pfx"),"R":("R: vLLM","G replay 1-10, live 11+"),"Rp":("R′: vLLM","G replay 1-16, live 17+"),"M":("M: vLLM","from scratch, r1 (source)"),"N":("N: vLLM","from scratch, r2 (J args)"),"G":("G: MINF","from scratch, no pfx"),"I":("I: MINF","from scratch, pfx r1"),"J":("J: MINF","from scratch, pfx r2"),"A":("A: vLLM","from scratch"),"K":("K: vLLM","from scratch, seed 1234"),"D":("D: vLLM","<- MINF step 10")}
hh=30; x=tx0+cw[0]
for (key,*_),wd in zip(SERIES,cw[1:]):
    ev=EVAL.get(key)
    if ev:
        A(f'<rect x="{x}" y="{ty-16}" width="{wd}" height="{16+hh+2+nrows*rh+6}" fill="{STATUS[ev]}" fill-opacity="{EVOP[ev]}"/>')
        A(f'<circle cx="{x+6}" cy="{ty-8}" r="4" fill="{STATUS[ev]}"/><text x="{x+14}" y="{ty-4}" font-size="10" fill="{INK2}">{EVSYM[ev]} {EVLAB[ev]}</text>')
    x+=wd
x=tx0
A(f'<rect x="{tx0}" y="{ty}" width="{sum(cw)}" height="{hh}" fill="{ZEBRA}" fill-opacity="0.7"/>')
for c,wd in zip(cols,cw):
    l1,l2=HDR[c]
    A(f'<text x="{x+wd-4}" y="{ty+12}" font-size="10.5" font-weight="600" fill="{INK}" text-anchor="end" font-family="monospace">{esc(l1)}</text>')
    if l2: A(f'<text x="{x+wd-4}" y="{ty+25}" font-size="9.5" fill="{INK2}" text-anchor="end" font-family="monospace">{esc(l2)}</text>')
    x+=wd
ty+=hh+rh
for s in range(1,maxstep+1):
    if s%2==0: A(f'<rect x="{tx0}" y="{ty-11}" width="{sum(cw)}" height="{rh}" fill="{ZEBRA}"/>')
    x=tx0; A(f'<text x="{x+cw[0]-4}" y="{ty}" font-size="11" fill="{INK2}" text-anchor="end" font-family="monospace">{s}</text>'); x+=cw[0]
    for (key,*_),wd in zip(SERIES,cw[1:]):
        v=data[key].get(s); replay=is_replay(key,s)
        txt="" if v is None else f"{v:.1f}"+("r" if replay else ("*" if (key,s) in partial else ""))
        A(f'<text x="{x+wd-4}" y="{ty}" font-size="11" fill="{REPLAY if replay else INK}" text-anchor="end" font-family="monospace">{txt}</text>'); x+=wd
    ty+=rh
ty+=14
runs="; ".join(f"{lab(a)} = {c['exp']} ({c['engine']}, "+(f"frozen chain {c['source']} step-{c['replay_end']} weights, lr = 0" if c.get('kind')=="lr0" else (c['design'] if c.get('kind')=="resume" else f"chain {c['source']} steps {min(RS(c))}-{max(RS(c))} replayed"+(f"; {c['note']}" if c.get('note') else "")))+")" for a,c in ARMS.items())
pending=[lab(a) for a in ARMS if a not in ACTIVE]
ty=NOTE(f'Design (from the Cross Train Experiment session): {runs}. All start from the base model with grpo.seed 42 and overlap_param_gather off; the MINF splice runs prefix caching on (inert during the replay). Step k of a splice run trains on step-k rollouts of its source arm, exact 1:1. Questions: does a training history generated by the other engine change what an engine does once it generates for itself (P vs chains G/I/J; Q vs chains M/N and run A / chain K; R and R′ vs chain G\'s own continuation); and on identical frozen weights, do the two engines loop at the same rate (P′ vs Q′; both vs P/Q whose weights keep training); and does keeping the MINF prefix cache across refits change the MINF continuation (P‴ vs chain P steps 21-31, same prompts).'+(f' Registered but without dumps yet (held): {", ".join(pending)}.' if pending else ''), ty-3, size=10.5, width=215)
if LR0: ty=NOTE('Frozen-weight arms P′ and Q′ (15 live steps each, 21-35, both ended at their 8 h walltime): all real weights verified bit-identical to chain P step 20 at steps 25/30/35; the only drift is the MoE router expert_bias (24 buffers, about 0.001 per step, 0.015 max at step 35), same mechanism on both arms. Chain Q′ step 21 stalled ~20 min while the Gym code was missing from the bind mount and a few rollouts were retried.', ty-3, size=10.5, width=215)
ty=NOTE(f'Rendered {datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z")} from splice_loopshare.csv (splice runs exact from token dumps) and joint_loopshare.csv (comparators). Colors: reference categorical palette, fixed slot order; identity also carried by dash pattern, marker shape, direct labels and this table.', ty-3, size=10.5, width=215)
H=ty+16
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',f'<rect width="100%" height="100%" fill="{SURF}"/>']+o+['</svg>']
open(f"{L}/loopshare_splice_table.svg","w").write("\n".join(svg)); print("wrote",f"{L}/loopshare_splice_table.svg",f"{W}x{H}")

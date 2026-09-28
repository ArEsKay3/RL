import csv,json,glob,collections,datetime,textwrap
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
SURF,INK,INK2,GRID,ZEBRA="#fcfcfb","#0b0b0b","#52514e","#e6e5e1","#f3f2ef"
STATUS={"good":"#0ca30c","bad":"#d03b3b","presumed":"#fab219","presumed_good":"#3987e5"}; EVLAB={"good":"eval good","bad":"eval bad","presumed":"presumed bad","presumed_good":"presumed good"}; EVSYM={"good":"✓","bad":"✗","presumed":"?","presumed_good":"?"}; EVOP={"good":0.10,"bad":0.10,"presumed":0.16,"presumed_good":0.12}
BASE=[("A","vLLM","run 1","good"),("K","vLLM","run 2","presumed_good"),("M","vLLM","run 3","presumed_good"),
      ("main","MINF","run 1","bad"),("G","MINF","run 2","bad"),("I","MINF","run 3","presumed"),("J","Good MINF","run 4","good")]
EXTRA=[("P4","MINF","no cache invalidation",None),("Q4","vLLM","no prefix caching",None)]
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
esc=lambda t:t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def render(cols, fn, title):
    keys=[c[0] for c in cols]; maxstep=max(s for k in keys for s in data[k]) if any(data[k] for k in keys) else 1
    cw=[52]+[122]*len(cols); W=sum(cw)+120; tx=(W-sum(cw))//2
    o=[]; A=o.append
    def NOTE(text,y,size=11,width=int((W-2*tx)/6.3)):
        lines=textwrap.wrap(text,width)
        for k,line in enumerate(lines): A(f'<text x="{tx}" y="{y+3+13*k}" font-size="{size}" fill="{INK2}">{esc(line)}</text>')
        return y+13*max(1,len(lines))
    ty=34
    for line in textwrap.wrap(title, int((W-2*tx)/8.8)):
        A(f'<text x="{tx}" y="{ty}" font-size="16" font-weight="600" fill="{INK}">{esc(line)}</text>'); ty+=20
    ty-=2
    ty=NOTE("Each cell: tokens inside repetitive reasoning loops / all tokens generated in that step's 512 training rollouts, in percent. A repetitive loop is a reasoning segment of at least 1,000 tokens whose text zlib-compresses to under 10 % of its size (the model repeating itself). Every run uses the same recipe, the same base checkpoint and the same 32 prompts at each step; only the rollout-generation engine (vLLM, or MINF = Megatron in-engine inference) and the replica differ.", ty)
    ty=NOTE(f"Column shading = SWE-Bench Verified outcome of the run: green {EVSYM['good']} eval good (pass@1 held or improved), red {EVSYM['bad']} eval bad (declined); yellow {EVSYM['presumed']} presumed bad and light blue {EVSYM['presumed_good']} presumed good = not evaluated yet, called from the loop trajectory; unshaded = no call yet.", ty)
    ty=NOTE("* = step with fewer than the full 512 rollouts, or a step re-run after a restart. Blank = no data. MINF run 1 was reconstructed from its agent trajectories (its step 1 is missing); all other runs come from exact token dumps.", ty)
    STYLE={"A":("#2a78d6","7,4","circle"),"K":("#2a78d6","7,4","square"),"M":("#2a78d6","7,4","diamond"),"main":("#eb6834","","circle"),"G":("#eb6834","","square"),"I":("#eb6834","","diamond"),"J":("#008300","","circle"),"P4":("#4a3aa7","","tri_up"),"Q4":("#1baf7a","7,4","tri_down")}
    def marker(kind,x,y,hs,col,title=""):
        if kind=="square": shape,tag=f'<rect x="{x-hs:.1f}" y="{y-hs:.1f}" width="{2*hs}" height="{2*hs}"',"rect"
        elif kind=="diamond": shape,tag=f'<rect x="{x-hs:.1f}" y="{y-hs:.1f}" width="{2*hs}" height="{2*hs}" transform="rotate(45 {x:.1f} {y:.1f})"',"rect"
        elif kind=="tri_up": shape,tag=f'<polygon points="{x:.1f},{y-hs-1:.1f} {x+hs+1:.1f},{y+hs:.1f} {x-hs-1:.1f},{y+hs:.1f}"',"polygon"
        elif kind=="tri_down": shape,tag=f'<polygon points="{x:.1f},{y+hs+1:.1f} {x+hs+1:.1f},{y-hs:.1f} {x-hs-1:.1f},{y-hs:.1f}"',"polygon"
        else: shape,tag=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{hs}"',"circle"
        return f'{shape} fill="{col}" stroke="{SURF}" stroke-width="1.5">'+(f'<title>{esc(title)}</title>' if title else '')+f'</{tag}>'
    ty+=24
    ml=tx+40; pw=W-tx-ml-10; PH=340; ptop=ty; pb=ptop+PH
    ymax=5*int(max(v for k in keys for v in data[k].values())/5)+5
    X=lambda s: ml+(s-1)/(max(2,maxstep)-1)*pw
    Y=lambda v: ptop+(1-v/ymax)*PH
    for v in range(0,ymax+1,5):
        A(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>'); A(f'<text x="{ml-8}" y="{Y(v)+4:.1f}" font-size="11" fill="{INK2}" text-anchor="end">{v}%</text>')
    for s in range(1,maxstep+1):
        if s%5==0 or s==1: A(f'<text x="{X(s):.1f}" y="{pb+16}" font-size="11" fill="{INK2}" text-anchor="middle">{s}</text>')
    A(f'<line x1="{ml}" y1="{pb}" x2="{ml+pw}" y2="{pb}" stroke="{INK2}" stroke-width="1"/>')
    A(f'<text x="{ml+pw/2:.0f}" y="{pb+34}" font-size="11.5" fill="{INK2}" text-anchor="middle">training step</text>')
    A(f'<text transform="translate({tx-4},{ptop+PH/2:.0f}) rotate(-90)" font-size="11.5" fill="{INK2}" text-anchor="middle">loop tokens / all generated tokens (%)</text>')
    for key,g,sub,ev in cols:
        pts=sorted(data[key].items())
        if not pts: continue
        col,dash,kind=STYLE[key]; da=f' stroke-dasharray="{dash}"' if dash else ""
        A(f'<polyline points="{" ".join(f"{X(s):.1f},{Y(v):.1f}" for s,v in pts)}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"{da}/>')
        for s,v in pts: A(marker(kind,X(s),Y(v),3.5,col,f"{g}, {sub} - step {s}: {v:.1f}%"+(" (partial step)" if (key,s) in partial else "")))
    ly=pb+58; lx=ml; rowh=18
    for key,g,sub,ev in cols:
        col,dash,kind=STYLE[key]; label=f"{g}, {sub}"; wdt=int(len(label)*6.6)+70
        if lx+wdt>ml+pw: lx=ml; ly+=rowh
        da=f' stroke-dasharray="{dash}"' if dash else ""
        A(f'<line x1="{lx}" y1="{ly}" x2="{lx+30}" y2="{ly}" stroke="{col}" stroke-width="2"{da}/>'+marker(kind,lx+15,ly,3.5,col))
        A(f'<text x="{lx+38}" y="{ly+4}" font-size="11" fill="{INK}">{esc(label)}</text>'); lx+=wdt
    ty=ly+44
    hh=30; rh=15; nrows=maxstep+2
    x=tx+cw[0]
    for (key,g,sub,ev),wd in zip(cols,cw[1:]):
        if ev:
            A(f'<rect x="{x}" y="{ty-16}" width="{wd}" height="{16+hh+2+nrows*rh+6}" fill="{STATUS[ev]}" fill-opacity="{EVOP[ev]}"/>')
            A(f'<circle cx="{x+6}" cy="{ty-8}" r="4" fill="{STATUS[ev]}"/><text x="{x+14}" y="{ty-4}" font-size="10" fill="{INK2}">{EVSYM[ev]} {EVLAB[ev]}</text>')
        x+=wd
    A(f'<rect x="{tx}" y="{ty}" width="{sum(cw)}" height="{hh}" fill="{ZEBRA}" fill-opacity="0.7"/>')
    x=tx
    A(f'<text x="{x+cw[0]-4}" y="{ty+12}" font-size="10.5" font-weight="600" fill="{INK}" text-anchor="end" font-family="monospace">step</text>'); x+=cw[0]
    for (key,g,sub,ev),wd in zip(cols,cw[1:]):
        A(f'<text x="{x+wd-4}" y="{ty+12}" font-size="10.5" font-weight="600" fill="{INK}" text-anchor="end" font-family="monospace">{esc(g)}</text>')
        A(f'<text x="{x+wd-4}" y="{ty+25}" font-size="9.5" fill="{INK2}" text-anchor="end" font-family="monospace">{esc(sub)}</text>'); x+=wd
    ty+=hh+rh
    for s in range(1,maxstep+1):
        if s%2==0: A(f'<rect x="{tx}" y="{ty-11}" width="{sum(cw)}" height="{rh}" fill="{ZEBRA}"/>')
        x=tx; A(f'<text x="{x+cw[0]-4}" y="{ty}" font-size="11" fill="{INK2}" text-anchor="end" font-family="monospace">{s}</text>'); x+=cw[0]
        for (key,*_),wd in zip(cols,cw[1:]):
            v=data[key].get(s); txt="" if v is None else f"{v:.1f}"+("*" if (key,s) in partial else "")
            A(f'<text x="{x+wd-4}" y="{ty}" font-size="11" fill="{INK}" text-anchor="end" font-family="monospace">{txt}</text>'); x+=wd
        ty+=rh
    A(f'<line x1="{tx}" y1="{ty-11}" x2="{tx+sum(cw)}" y2="{ty-11}" stroke="{INK2}" stroke-width="1"/>')
    for lo,hi in ((21,30),(31,40)):
        x=tx; A(f'<text x="{x+cw[0]-4}" y="{ty+2}" font-size="10" font-weight="600" fill="{INK}" text-anchor="end" font-family="monospace">mean {lo}-{hi}</text>'); x+=cw[0]
        for (key,*_),wd in zip(cols,cw[1:]):
            vs=[data[key][s] for s in range(lo,hi+1) if s in data[key] and (key,s) not in partial]
            txt=f"{sum(vs)/len(vs):.1f}"+("" if len(vs)>=hi-lo+1 else f" ({len(vs)})") if vs else ""
            A(f'<text x="{x+wd-4}" y="{ty+2}" font-size="10.5" font-weight="600" fill="{INK}" text-anchor="end" font-family="monospace">{txt}</text>'); x+=wd
        ty+=rh
    ty+=16
    ty=NOTE("Mean rows average the complete steps in the window; a count in parentheses means fewer than ten steps were available.", ty-3, size=10.5)
    if any(k in ("P4","Q4") for k in keys):
        ty=NOTE("MINF (no cache invalidation) = MINF with its prefix cache kept across weight updates instead of being cleared after every refit. vLLM (no prefix caching) = vLLM with its prefix cache turned off; every other vLLM run here had it on. Both started 2026-09-25/26 and are still training.", ty-3, size=10.5)
    ty=NOTE(f"Rendered {datetime.datetime.now().strftime('%Y-%m-%d %H:%M %Z')}.", ty-3, size=10.5)
    H=ty+16
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',f'<rect width="100%" height="100%" fill="{SURF}"/>']+o+['</svg>']
    open(fn,"w").write("\n".join(svg)); print("wrote",fn,f"{W}x{H}")
render(BASE, f"{L}/loopshare_share_vllm_vs_minf.svg", "Percent of generated tokens spent in repetitive reasoning loops, per training step: vLLM vs MINF rollout generation")
render(BASE+EXTRA, f"{L}/loopshare_share_vllm_vs_minf_with_cache_variants.svg", "Percent of generated tokens spent in repetitive reasoning loops, per training step: vLLM vs MINF, plus two prefix-cache variants")

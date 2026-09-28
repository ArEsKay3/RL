import csv,glob,collections,statistics as st,json,math
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
NAMES={"G":"chain G (MINF from0)","A":"run A (vLLM from0)","F":"chain F (MINF fr vLLM10)","B":"run B (MINF fr MINF10)","D":"chain D (vLLM fr MINF10)","I":"chain I (MINF from0 pfx)"}
def parse_line(line):
    fields=[]; i=0; n=len(line)
    while i<n:
        if line[i]=='"':
            j=i+1
            while j<n:
                if line[j]=='\\': j+=2; continue
                if line[j]=='"': break
                j+=1
            fields.append(json.loads(line[i:j+1])); i=j+1
            if i<n and line[i]==',': i+=1
        else:
            j=line.find(',',i)
            if j<0: j=n
            fields.append(line[i:j]); i=j+1
    return fields
rows=[]; bad=0
for fn in glob.glob(f"{L}/exits/*.csv"):
    lines=open(fn).read().splitlines()
    if not lines: continue
    hdr=lines[0].split(",")
    for ln in lines[1:]:
        f=parse_line(ln)
        if len(f)!=len(hdr): bad+=1; continue
        rows.append(dict(zip(hdr,f)))
print(f"parsed {len(rows)} blocks ({bad} malformed lines skipped)")
for r in rows:
    for k in ("n_tok","step","n_dot","n_dnn","n_low_p_lt01","cycle_len","top_n"): r[k]=int(r[k])
    for k in ("zlib","adv","p_close","p_exit_tok","p_exit_tok2","p_dot_mean","p_dnn_mean","share_dnn_sf","minp","minp_posfrac","cycle_match"):
        try: r[k]=float(r[k])
        except: r[k]=float('nan')
    for k in ("exit_tok","exit_tok2","minp_tok"):
        v=r[k]
        if isinstance(v,str) and len(v)>=2 and v[0] in "'\"" and v[-1]==v[0]:
            try: r[k]=eval(v)
            except Exception: pass
def q(xs,p):
    xs=sorted(x for x in xs if not math.isnan(x)); 
    return xs[int(p*(len(xs)-1))] if xs else float('nan')
print("HOW LOOPS END. Per arm, all steps pooled. 'self-exited loop' = repetitive block (zlib<0.10, >=1000 tok) that closed with </think>; 'runaway' = cut by the context window;")
print("'baseline' = 3 % random sample of normal blocks >= 300 tokens. exit tok = last token before </think>; p = its sampling probability; 'fork exit' = exit tok sampled at p < 0.05.")
print(f"\n{'arm':<26} {'class':<9} {'n':>5} {'tok p50':>7} {'zlib p50':>8} {'cycle p50':>9} {'match':>6} | {'p(</think>) p50':>15} {'p25':>6} {'<0.5':>6} | {'exit tok = .':>12} {'= .nn':>6} {'other':>6} | {'p(exit tok) p50':>15} {'fork exit':>9} {'p(prev tok) p50':>15} | {'p(.) in blk':>11} {'p(.nn) in blk':>13} {'share .nn':>9}")
for arm in ("G","A","F","B","D","I"):
    for cls in ("looplike","runaway","baseline"):
        R=[r for r in rows if r["arm"]==arm and r["cls"]==cls]
        if not R: continue
        closed=[r for r in R if r["has_close"]=="1"]
        et=collections.Counter(r["exit_tok"] for r in closed); n=len(closed)
        print(f"{NAMES[arm]:<26} {cls:<9} {len(R):>5} {q([r['n_tok'] for r in R],.5):>7.0f} {q([r['zlib'] for r in R],.5):>8.3f} {q([r['cycle_len'] for r in R if r['cycle_len']>0],.5):>9.0f} {q([r['cycle_match'] for r in R if r['cycle_len']>0],.5):>6.2f} | "
              f"{q([r['p_close'] for r in closed],.5):>15.3f} {q([r['p_close'] for r in closed],.25):>6.3f} {sum(1 for r in closed if r['p_close']<0.5)/max(1,n):>6.1%} | "
              f"{et.get('.',0)/max(1,n):>12.1%} {et.get('.\\n\\n',0)/max(1,n):>6.1%} {1-(et.get('.',0)+et.get('.\\n\\n',0))/max(1,n):>6.1%} | "
              f"{q([r['p_exit_tok'] for r in closed],.5):>15.3f} {sum(1 for r in closed if r['p_exit_tok']<0.05)/max(1,n):>9.1%} {q([r['p_exit_tok2'] for r in closed],.5):>15.3f} | "
              f"{q([r['p_dot_mean'] for r in R],.5):>11.3f} {q([r['p_dnn_mean'] for r in R],.5):>13.3f} {q([r['share_dnn_sf'] for r in R],.5):>9.2f}")
    print()
print("SELF-EXITED LOOPS, chain G, split by how the exit happened: 'fork' = last token before </think> had p < 0.05 (the loop's usual continuation was overridden by a rare sample); 'smooth' = p >= 0.05.")
G=[r for r in rows if r["arm"]=="G" and r["cls"]=="looplike" and r["has_close"]=="1"]
for name,sel in (("fork exits",[r for r in G if r["p_exit_tok"]<0.05]),("smooth exits",[r for r in G if r["p_exit_tok"]>=0.05])):
    print(f"  {name:<13} n={len(sel):>4}  tokens p50 {q([r['n_tok'] for r in sel],.5):>7.0f}  p(</think>) p50 {q([r['p_close'] for r in sel],.5):.3f}  exit tok: "+", ".join(f"{k!r} {v/len(sel):.0%}" for k,v in collections.Counter(r['exit_tok'] for r in sel).most_common(4)) if sel else f"  {name}: none")
print("\nSELF-EXITED LOOPS in chain G by step (all steps): n, median tokens, median p(</think>) at the exit, share of fork exits, and share of loops whose cycle ends in '.\\n\\n' (share_dnn_sf > 0.5).")
print(f"{'step':>4} {'n':>3} {'tok p50':>7} {'p(</think>) p50':>15} {'fork exit':>9} {'.nn-cycles':>10} | {'runaways':>8} {'runaway tok p50':>15}")
byst=collections.defaultdict(list); byr=collections.defaultdict(list)
for r in rows:
    if r["arm"]!="G": continue
    if r["cls"]=="looplike" and r["has_close"]=="1": byst[r["step"]].append(r)
    if r["cls"]=="runaway": byr[r["step"]].append(r)
for s in sorted(set(byst)|set(byr)):
    R=byst.get(s,[]); RR=byr.get(s,[])
    print(f"{s:>4} {len(R):>3} {q([r['n_tok'] for r in R],.5) if R else 0:>7.0f} {q([r['p_close'] for r in R],.5) if R else float('nan'):>15.3f} {(sum(1 for r in R if r['p_exit_tok']<0.05)/len(R)) if R else 0:>9.0%} {(sum(1 for r in R if r['share_dnn_sf']>0.5)/len(R)) if R else 0:>10.0%} | {len(RR):>8} {q([r['n_tok'] for r in RR],.5) if RR else 0:>15.0f}")

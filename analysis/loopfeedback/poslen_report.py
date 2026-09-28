import glob,json,re,csv,collections
import numpy as np
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
NAMES={"G":"chain G MINF from0 no-pfx","I":"chain I MINF from0 pfx-on","A":"run A vLLM from0","F":"chain F MINF fr vLLM10","B":"run B MINF fr MINF10","D":"chain D vLLM fr MINF10"}
LAB=["0-8k","8-16k","16-32k","32-64k","64-96k","96-128k","128-160k","160k+"]; NB=8
def band(s): return "1-10" if s<=10 else ("11-20" if s<=20 else "21+")
agg={}
def A_(arm,bd):
    k=(arm,bd)
    if k not in agg: agg[k]={"tok":{c:{f:np.zeros(NB) for f in ("n","abs_err","err","gen","prev","err_gt1")} for c in ("normal","rep")},"blk":{f:np.zeros(NB) for f in ("n","n_long","n_rep","rep_tok","rep_closed","norm_tok")},"close":{f:np.zeros(NB) for f in ("n","lp")}}
    return agg[k]
nf=0
for fn in glob.glob(f"{L}/poslen/*.poslen.json"):
    m=re.search(r"([A-Z])__s(\d+)_c(\d+)\.poslen\.json",fn); arm,step=m.group(1),int(m.group(2)); nf+=1
    P=json.load(open(fn))
    for bd in (band(step),"all"):
        e=A_(arm,bd)
        for c in ("normal","rep"):
            for f,v in P["tok"][c].items(): e["tok"][c][f]+=np.array(v)
        for f,v in P["blk"].items(): e["blk"][f]+=np.array(v)
        for f,v in P["close"].items(): e["close"][f]+=np.array(v)
loops=[]
for fn in glob.glob(f"{L}/poslen/*.loops.csv"): loops+=list(csv.DictReader(open(fn)))
out=[]
def P(*x): out.append(" ".join(str(v) for v in x))
P(f"CONTEXT-LENGTH ANALYSIS ({nf} chunk files). Position = absolute token index in the sequence (prompt + all turns + tool outputs) at which a generated token or a reasoning block sits.")
P("Bins of position: 0-8k | 8-16k | 16-32k | 32-64k | 64-96k | 96-128k | 128-160k | 160k+ (context window 196,608).")
P("")
for bd in ("1-10","11-20","21+","all"):
    P("="*110); P(f"TRAINING STEPS {bd}"); P("="*110)
    P("\n(a) LOOP RATE BY POSITION OF THE BLOCK START: repetitive blocks per 10,000 blocks | share of generated tokens at that position inside repetitive blocks")
    P(f"{'position':>9} | " + " | ".join(f"{a+': blocks':>12} {'rep/1e4':>7} {'loop tok%':>9}" for a in ("G","I","A")))
    for i in range(NB):
        cells=[]
        for a in ("G","I","A"):
            e=agg.get((a,bd))
            if not e or e["blk"]["n"][i]==0: cells.append(f"{'':>12} {'':>7} {'':>9}"); continue
            nt=e["tok"]["normal"]["n"][i]+e["tok"]["rep"]["n"][i]
            cells.append(f"{int(e['blk']['n'][i]):>12,} {1e4*e['blk']['n_rep'][i]/e['blk']['n'][i]:>7.2f} {100*e['tok']['rep']['n'][i]/max(1,nt):>8.1f}%")
        P(f"{LAB[i]:>9} | "+" | ".join(cells))
    P("\n(b) LOOP LENGTH vs ROOM: repetitive blocks by start position -> count, median length, median length / remaining room (196,608 - start), share self-exited")
    P(f"{'position':>9} | " + " | ".join(f"{a+': n':>6} {'med len':>8} {'len/room':>8} {'exited':>6}" for a in ("G","I","A")))
    for i in range(NB):
        cells=[]
        for a in ("G","I","A"):
            Ls=[r for r in loops if r["arm"]==a and (bd=="all" or band(int(r["step"]))==bd) and int(np.searchsorted([8000,16000,32000,64000,96000,128000,160000],int(r["abs_start"]),side="right"))==i]
            if not Ls: cells.append(f"{'':>6} {'':>8} {'':>8} {'':>6}"); continue
            n=[int(r["n_tok"]) for r in Ls]; fr=[int(r["n_tok"])/int(r["room"]) for r in Ls]; ex=np.mean([int(r["closed"]) for r in Ls])
            cells.append(f"{len(Ls):>6} {np.median(n):>8,.0f} {np.median(fr):>8.2f} {ex:>6.0%}")
        P(f"{LAB[i]:>9} | "+" | ".join(cells))
    P("\n(c) ENGINE vs TRAINER LOGPROB MISMATCH BY POSITION, NORMAL-block tokens: mean |gen lp - trainer lp| (nats/token) | mean signed (gen - trainer) | tokens with |diff| > 1 per 10k | mean gen lp (sharpness)")
    P(f"{'position':>9} | " + " | ".join(f"{a+': |d|':>9} {'signed d':>9} {'>1/1e4':>7} {'gen lp':>7}" for a in ("G","I","A")))
    for i in range(NB):
        cells=[]
        for a in ("G","I","A"):
            e=agg.get((a,bd))
            if not e: cells.append(f"{'':>9} {'':>9} {'':>7} {'':>7}"); continue
            t=e["tok"]["normal"]; n=t["n"][i]
            if n<1000: cells.append(f"{'':>9} {'':>9} {'':>7} {'':>7}"); continue
            cells.append(f"{t['abs_err'][i]/n:>9.4f} {t['err'][i]/n:>+9.4f} {1e4*t['err_gt1'][i]/n:>7.1f} {t['gen'][i]/n:>7.3f}")
        P(f"{LAB[i]:>9} | "+" | ".join(cells))
    P("\n(d) same mismatch inside REPETITIVE blocks (loop tokens)")
    P(f"{'position':>9} | " + " | ".join(f"{a+': |d|':>9} {'signed d':>9} {'>1/1e4':>7} {'gen lp':>7}" for a in ("G","I","A")))
    for i in range(NB):
        cells=[]
        for a in ("G","I","A"):
            e=agg.get((a,bd))
            if not e: cells.append(f"{'':>9} {'':>9} {'':>7} {'':>7}"); continue
            t=e["tok"]["rep"]; n=t["n"][i]
            if n<1000: cells.append(f"{'':>9} {'':>9} {'':>7} {'':>7}"); continue
            cells.append(f"{t['abs_err'][i]/n:>9.4f} {t['err'][i]/n:>+9.4f} {1e4*t['err_gt1'][i]/n:>7.1f} {t['gen'][i]/n:>7.3f}")
        P(f"{LAB[i]:>9} | "+" | ".join(cells))
    P("\n(e) </think> closes of normal blocks by position: closes per 1,000 normal blocks starting in the bin | geo-mean p(</think>)")
    P(f"{'position':>9} | " + " | ".join(f"{a+': closes/1k':>13} {'p(close)':>8}" for a in ("G","I","A")))
    for i in range(NB):
        cells=[]
        for a in ("G","I","A"):
            e=agg.get((a,bd))
            if not e or e["blk"]["n"][i]==0: cells.append(f"{'':>13} {'':>8}"); continue
            nb=e["blk"]["n"][i]-e["blk"]["n_rep"][i]; c=e["close"]["n"][i]
            cells.append(f"{1000*c/max(1,nb):>13.1f} {np.exp(e['close']['lp'][i]/c) if c else float('nan'):>8.3f}")
        P(f"{LAB[i]:>9} | "+" | ".join(cells))
    P("")
open(f"{L}/poslen_report.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))

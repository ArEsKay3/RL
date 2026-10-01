import csv,json,os,collections,itertools,datetime
L=os.environ["LF_L"]
cmp=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{L}/../loopfeedback/joint_loopshare.csv")): cmp[r["arm"]][int(r["step"])]=float(r["share_all_gen_pct"])
hsg=collections.defaultdict(dict); rows={}
for r in csv.DictReader(open(f"{L}/hsg_loopshare.csv")):
    if int(r["rows"])==512: hsg[r["arm"]][int(r["step"])]=float(r["joint_pct"]); rows[(r["arm"],int(r["step"]))]=r
stats={(r["arm"],r["step"]):r for r in json.load(open(f"{L}/hsg_rollout_stats.json")) if r["rows"]==512}
ARMS=[("VHnC","V-HSG-noprefix-r2 (MINF from scratch, vLLM-parity build, MINF prefix cache off, seed 42, round 2 on Lustre)"),("VHnC2","V-HSG2-noprefix-r2 (same, seed 1234)"),("VHnC3","V-HSG3-noprefix-r2 (same, seed 4321)")]
CMP=[("V","chain V (CMH vLLM-parity MINF)"),("A","run A (CMH vLLM)"),("Q4","chain Q'''' (CMH vLLM, prefix caching off)"),("J","chain J (CMH MINF clean)"),("P4","chain P'''' (CMH MINF keep-prefix)"),("G","chain G (CMH MINF poisoned)")]
def signflip(d):
    n=len(d); obs=abs(sum(d)); cnt=0
    for signs in itertools.product((1,-1),repeat=n):
        if abs(sum(s*x for s,x in zip(signs,d)))>=obs-1e-12: cnt+=1
    return cnt/2**n
o=[f"# HSG vLLM-parity arms: loop-share verdict ({datetime.datetime.now():%Y-%m-%d %H:%M} CDT)\n",
"Joint loop share = tokens in repetitive reasoning blocks (>= 1,000 tokens, zlib < 10 %) / all generated tokens of the step's 512 trained rollouts. Verdict window = steps 20-30. Test = paired sign-flip permutation on per-step differences (same prompts per step on every arm). Only clean steps are used (0 harness-failed rows; restart duplicates excluded).\n",
"## Verdict window\n","| arm | steps | mean % | "+" | ".join(f"vs {n} (mean %)" for _,n in CMP)+" |","|---|---|---|"+"---|"*len(CMP)]
for a,name in ARMS:
    steps=[s for s in range(20,31) if s in hsg[a]]
    if not steps: continue
    cells=[]
    for c,_ in CMP:
        ss=[s for s in steps if s in cmp[c]]
        if len(ss)<4: cells.append("n/a"); continue
        d=[hsg[a][s]-cmp[c][s] for s in ss]
        cells.append(f"{sum(cmp[c][s] for s in ss)/len(ss):.2f}: diff {sum(d)/len(d):+.2f}, higher {sum(1 for x in d if x>0)}/{len(d)}, p={signflip(d):.3f}")
    o.append(f"| {name} | {steps[0]}-{steps[-1]} | {sum(hsg[a][s] for s in steps)/len(steps):.2f} | "+" | ".join(cells)+" |")
o+=["\n## Per step\n","| arm | step | joint % | strict % | rep blocks (+adv/-adv/0) | runaway | truncated | reward | harness-fail rows | gen mean / p50 / p90 | "+" | ".join(n.split(" (")[0] for _,n in CMP)+" |","|---|---|---|---|---|---|---|---|---|---|"+"---|"*len(CMP)]
for a,name in ARMS:
    for s in sorted(hsg[a]):
        r=rows[(a,s)]; st=stats.get((a,s),{})
        o.append(f"| {name.split(' (')[0]} | {s} | {r['joint_pct']} | {r['strict_pct']} | {r['repetitive_blocks']} ({r['rep_adv_pos']}/{r['rep_adv_neg']}/{r['rep_adv_zero']}) | {r['runaway_blocks']} | {r['truncated_rollouts']} | {st.get('reward','')} | {st.get('harness_fail','')} | {st.get('gen_mean','')} / {st.get('gen_p50','')} / {st.get('gen_p90','')} | "+" | ".join(f"{cmp[c][s]:.1f}" if s in cmp[c] else "" for c,_ in CMP)+" |")
open(f"{L}/hsg_verdict.md","w").write("\n".join(o)+"\n"); print(f"{L}/hsg_verdict.md")

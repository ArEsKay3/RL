import sys,glob,csv,collections,hashlib
import numpy as np
sys.path.insert(0,"/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools")
from pt_numpy import load
arm_run,src_run,mask_csv=sys.argv[1],sys.argv[2],sys.argv[3]; lo=int(sys.argv[4]) if len(sys.argv)>4 else 1; hi=int(sys.argv[5]) if len(sys.argv)>5 else 10
masked=collections.defaultdict(set)
with open(mask_csv) as fh:
    for r in csv.DictReader(fh): masked[int(r["train_step"])].add(r["sample_id"])
def rows(run,step):
    out=[]
    for fn in sorted(glob.glob(f"{run}/dumps/token_level/step_{step:05d}_chunk_*.pt")):
        o=load(fn); L=np.asarray(o["input_lengths"]).astype(np.int64); off=np.concatenate([[0],np.cumsum(L)]); ids=np.asarray(o["input_ids"]).astype(np.int32); tm=np.asarray(o["token_mask"]).astype(np.int8); rw=np.asarray(o["rewards"]); smb=np.asarray(o["sample_mask_before"]); sma=np.asarray(o["sample_mask_after"])
        for i,sid in enumerate(o["sample_ids"]):
            a=ids[off[i]:off[i+1]]; m=tm[off[i]:off[i+1]]; out.append(dict(sid=sid,h=hashlib.sha1(a.tobytes()).hexdigest(),hm=hashlib.sha1(m.tobytes()).hexdigest(),rw=float(rw[i]),smb=float(smb[i]),sma=float(sma[i]),n=int(L[i])))
    return out
print(f"REPLAY VERIFICATION (content join on input_ids): {arm_run.split('/')[-1]} vs source {src_run.split('/')[-1]}, mask {mask_csv.split('/')[-1]}")
print(f"{'step':>4} {'rows':>5} {'same sids':>9} {'content in src':>14} {'mask identical':>14} {'reward identical':>16} | {'listed':>6} {'listed found':>12} {'before=0 on listed':>18} {'before=0 elsewhere':>18} {'after=0 total':>13}")
for s in range(lo,hi+1):
    A=rows(arm_run,s)
    if not A: print(f"{s:>4}  no dump yet"); continue
    S=rows(src_run,s); byh=collections.defaultdict(list)
    for r in S: byh[r["h"]].append(r)
    same=sum(1 for r in A if r["sid"] in {x["sid"] for x in S}); found=0; mask_ok=0; rw_ok=0; listed=masked.get(s,set())
    exp=collections.Counter(x["h"] for x in S if x["sid"] in listed); obs=collections.Counter(r["h"] for r in A if r["smb"]==0)
    for r in A:
        cands=byh.get(r["h"],[])
        if not cands: continue
        found+=1; src=cands[0]; mask_ok+=int(src["hm"]==r["hm"]); rw_ok+=int(src["rw"]==r["rw"])
    lf=sum(min(exp[h],sum(1 for r in A if r["h"]==h)) for h in exp); b0l=sum(min(exp[h],obs[h]) for h in exp); b0o=sum(obs[h]-min(exp[h],obs[h]) for h in obs)
    a0=sum(1 for r in A if r["sma"]==0)
    print(f"{s:>4} {len(A):>5} {same:>9} {found:>14} {mask_ok:>14} {rw_ok:>16} | {len(listed):>6} {lf:>12} {b0l:>18} {b0o:>18} {a0:>13}")
print("duplicate input_ids within a group are matched by count (per content hash: masked arm rows vs listed source rows). expected for a masked replay arm: content in src = rows = 512 (bit-identical input_ids), mask/reward identical = 512, listed found = listed, before=0 on listed = listed, before=0 elsewhere = 0; 'same sids' is expected to be 0 (ids are re-minted by the replaying run).")

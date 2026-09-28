#!/bin/bash
# Score the masked-replay arms X/Y/Z: verify replayed steps 1-10, extract live steps, report loop share by window vs chain R.
D=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff
R=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs
M=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay/masks
declare -A RUN SRC MASK
RUN[X]=nano35-swe-v2-splice-vllm-runGdata-to10-maskX-rewarded-deep-20260927; SRC[X]=nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920; MASK[X]=$D/poison_G_rewarded_deep_steps1-10.csv
RUN[Y]=nano35-swe-v2-splice-vllm-runGdata-to10-maskY-random-rewarded-20260927; SRC[Y]=nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920; MASK[Y]=$D/control_G_random_rewarded_steps1-10.csv
RUN[Z]=nano35-swe-v2-splice-vllm-runMdata-to10-maskZ-punished-deep-20260927; SRC[Z]=nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923; MASK[Z]=$D/poison_M_punished_deep_steps1-10.csv
declare -A LO HI
LO[X]=1; HI[X]=10; LO[Y]=1; HI[Y]=10; LO[Z]=1; HI[Z]=10
for pair in "AA nano35-swe-v2-splice-vllm-runGdata-8to10 8 10 minf_dump-from0-nopg-noprefix-20260920" "AB nano35-swe-v2-splice-vllm-runGdata-1to7 1 7 minf_dump-from0-nopg-noprefix-20260920" "AC nano35-swe-v2-splice-vllm-runMdata-8to10 8 10 vllm_dump-from0-nopg-r1-20260923"; do set -- $pair; d=$(ls -d $R/$2-* 2>/dev/null | head -1); [ -n "$d" ] && { RUN[$1]=$(basename $d); LO[$1]=$3; HI[$1]=$4; SRC[$1]=nano35-swe-v2-stream128-inorder1-cmh-64n-$5; MASK[$1]=$D/nomask.csv; }; done
ARMS="X Y Z ${RUN[AA]:+AA} ${RUN[AB]:+AB} ${RUN[AC]:+AC}"
date "+%F %T %Z"
TS=$(date +%H%M); : > $D/jobs_xyz_$TS.txt
for a in $ARMS; do
  d=$R/${RUN[$a]}/dumps/token_level; [ -d $d ] || { echo "$a: no dumps yet"; continue; }
  steps=$(ls $d | sed 's/step_0*\([0-9]*\)_chunk.*/\1/' | sort -un | tr '\n' ' '); echo "$a: dump steps $steps"
  for fn in $d/step_*_chunk_*.pt; do b=$(basename $fn .pt); s=$(echo $b | sed 's/step_0*\([0-9]*\)_chunk_0*\([0-9]*\)/\1/'); c=$(echo $b | sed 's/step_0*\([0-9]*\)_chunk_0*\([0-9]*\)/\2/'); { [ $s -lt ${LO[$a]} ] || [ $s -gt ${HI[$a]} ]; } && [ ! -f $D/parts/${a}__s$(printf %03d $s)_c$(printf %03d $c).samples.csv ] && echo "$a ${RUN[$a]} $fn $D/parts" >> $D/jobs_xyz_$TS.txt; done
done
if [ -s $D/jobs_xyz_$TS.txt ]; then sed "s/jobs_features.txt/jobs_xyz_$TS.txt/; s/features_%j/featxyz_${TS}_%j/g" $D/features.sbatch > $D/featxyz_$TS.sbatch; J=$(sbatch --parsable $D/featxyz_$TS.sbatch | tail -1); end=$((SECONDS+420)); while squeue -j $J -h 2>/dev/null | grep -q .; do [ $SECONDS -ge $end ] && break; sleep 15; done; echo "features job $J: $(tail -1 $D/featxyz_${TS}_$J.out 2>/dev/null)"; fi
for a in $ARMS; do
  d=$R/${RUN[$a]}/dumps/token_level; [ -d $d ] || continue
  hi=$(ls $d 2>/dev/null | sed 's/step_0*\([0-9]*\)_chunk.*/\1/' | sort -un | awk -v h=${HI[$a]} '$1<=h' | tail -1); [ -n "$hi" ] && [ $hi -ge ${LO[$a]} ] && { echo "== $a replay verification, steps ${LO[$a]}-$hi (no mask expected for AA/AB/AC)"; timeout 500 python3 $D/verify_replay.py $R/${RUN[$a]} $R/${SRC[$a]} ${MASK[$a]} ${LO[$a]} $hi | tail -n +3 | grep -v "^duplicate"; }
done
python3 - <<'PY'
import glob,csv
import numpy as np
D="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff"
def loadcsv(p):
    rows=[]
    for fn in sorted(glob.glob(p)):
        with open(fn) as fh: rows+=list(csv.DictReader(fh))
    return rows
def col(rows,k):
    o=np.empty(len(rows))
    for i,r in enumerate(rows):
        try: o[i]=float(r[k]) if r[k]!="" else np.nan
        except: o[i]=np.nan
    return o
print("\nLIVE STEPS (11+): loop share % and E(4k-16k) per step; windows vs chain R (unmasked G replay: 4.2 / 6.5 / 4.7 %, stopped at 25), the looping arms G/I (21-25: 7.7 / 5.3 %, 26-30: 13.1 / 11.1 %) and the healthy band (16-20: 0.3-3.7 %, 21-25: 0.2-0.8 %, 26-30: 0.2-0.7 %)")
for a,lab in (("X","X test (G data, rewarded deep-think masked)"),("Y","Y control (G data, random rewarded masked)"),("Z","Z mirror (M data, punished deep-think masked)"),("AA","AA ablation (G data at steps 8-10 only, live elsewhere)"),("AB","AB ablation (G data at steps 1-7 only, live elsewhere)"),("AC","AC control (M data at steps 8-10 only)"),("R","chain R (G data, unmasked)")):
    T=loadcsv(f"{D}/parts/{a}__s*_c*.turns.csv")
    if not T: print(f"{lab}: no live features yet"); continue
    step=col(T,"step").astype(int); n=col(T,"n_tok"); adv=col(T,"adv"); th=col(T,"think_tok"); z=col(T,"zlib_reason")
    cells=[]
    for s in sorted(set(step)):
        if a in ('X','Y','Z','R') and s<11: continue
        if a in ('X','Y','Z') and s>30: continue
        m=step==s; lp=m&(th>=1000)&(z<0.10); lg=m&(n>=4000)&(n<16000); cells.append(f"s{s}: {100*n[lp].sum()/n[m].sum():.1f}%/{1e3*(adv[lg]*n[lg]).sum()/n[m].sum():+.1f}")
    wins=[]
    for lo,hi in ((11,15),(16,20),(21,25),(26,30)):
        m=(step>=lo)&(step<=hi)
        if m.any(): lp=m&(th>=1000)&(z<0.10); wins.append(f"{lo}-{hi}: {100*n[lp].sum()/n[m].sum():.2f}% ({len(set(step[m]))} steps)")
    print(f"{lab}: "+" | ".join(wins)); print("   "+"  ".join(cells))
PY

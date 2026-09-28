#!/bin/bash
#SBATCH --job-name=swe-dump-refresh-splice
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/refresh_splice_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/refresh_splice_%j.err
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback
date
python3 - > jobs_splice_new.txt <<'PY'
import glob,os,re,json
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
R="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
for a,c in json.load(open(f"{L}/splice_arms.json")).items():
    for fn in sorted(glob.glob(f"{R}/{c['exp']}/dumps/token_level/step_*_chunk_*.pt")):
        m=re.search(r"step_(\d+)_chunk_(\d+)",fn); s,k=int(m.group(1)),int(m.group(2))
        if not os.path.exists(f"{L}/parts/{a}__s{s:03d}_c{k:03d}.agg.json"): print(a,c["exp"],fn,f"{L}/parts")
PY
wc -l jobs_splice_new.txt
[ -s jobs_splice_new.txt ] && xargs -P 20 -L 1 python3 lf_worker.py < jobs_splice_new.txt; echo "PHASE1_DONE $(date)"
python3 splice_loopshare.py > /dev/null && python3 splice_loopshare_svg.py; echo "ALL_DONE $(date)"

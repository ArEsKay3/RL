#!/bin/bash
#SBATCH --job-name=swe-dump-refresh2
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/refresh2_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/refresh2_%j.err
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback
date
python3 - > jobs_new.txt <<'PY'
import glob,os,re
L="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback"
R="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs"
EXP={}
for line in open(f"{L}/jobs.txt"):
    a,e=line.split()[:2]; EXP[a]=e
for a,e in EXP.items():
    for fn in sorted(glob.glob(f"{R}/{e}/dumps/token_level/step_*_chunk_*.pt")):
        m=re.search(r"step_(\d+)_chunk_(\d+)",fn); s,k=int(m.group(1)),int(m.group(2))
        if not os.path.exists(f"{L}/parts/{a}__s{s:03d}_c{k:03d}.agg.json"): print(a,e,fn,f"{L}/parts")
PY
wc -l jobs_new.txt; [ -s jobs_new.txt ] && xargs -P 40 -L 1 python3 lf_worker.py < jobs_new.txt; echo "PHASE1_DONE $(date)"
python3 loopshare.py > loopshare.txt && python3 joint_loopshare.py > /dev/null && python3 joint_loopshare_plot.py && python3 loopshare_plot.py; python3 loopshare_svg.py; python3 share_tables.py; echo "ALL_DONE $(date)"

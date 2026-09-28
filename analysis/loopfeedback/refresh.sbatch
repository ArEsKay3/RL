#!/bin/bash
#SBATCH --job-name=swe-dump-refresh
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/refresh_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/refresh_%j.err
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback
date; echo "phase 1: loop labels + punct on $(wc -l < jobs_new.txt) new chunks"
xargs -P 40 -L 1 python3 lf_worker.py < jobs_new.txt
sed 's#/parts$#/punct#' jobs_new.txt | xargs -P 40 -L 1 python3 punct_worker.py
echo "PHASE1_DONE $(date)"
echo "phase 2: regenerate reports"
python3 merge.py && python3 report1.py > report1.txt && python3 punct_report.py > punct_report.txt
python3 loopshare.py > loopshare.txt && python3 loopshare_plot.py && python3 early_compare.py > early_compare.txt && python3 normlp.py > normlp.txt
echo "PHASE2_DONE $(date)"
echo "phase 3: sentence-final probability pass on $(wc -l < jobs_sfp.txt) chunks"
xargs -P 40 -L 1 python3 sfp_worker.py < jobs_sfp.txt
python3 sfp_report.py > /dev/null
echo "ALL_DONE $(date) $(ls sfp/*.sfp.json | wc -l) sfp parts"

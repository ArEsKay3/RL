#!/bin/bash
#SBATCH --job-name=swe-dump-sfp
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/sfp_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/sfp_%j.err
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback
date; echo "phase 1: loop labels + punct on $(wc -l < jobs_new.txt) new chunks"
xargs -P 40 -L 1 python3 lf_worker.py < jobs_new.txt
sed 's#/parts$#/punct#' jobs_new.txt | xargs -P 40 -L 1 python3 punct_worker.py
echo "PHASE1_DONE $(date)"
echo "phase 2: sentence-final probability pass on $(wc -l < jobs_sfp.txt) chunks"
xargs -P 40 -L 1 python3 sfp_worker.py < jobs_sfp.txt
echo "ALL_DONE $(date) $(ls sfp/*.sfp.json | wc -l) parts"

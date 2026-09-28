#!/bin/bash
#SBATCH --job-name=swe-dump-K
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/k_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/k_%j.err
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback
date; xargs -P 15 -L 1 python3 lf_worker.py < jobs_K.txt; python3 k_compare.py > k_compare.txt; echo "ALL_DONE $(date)"

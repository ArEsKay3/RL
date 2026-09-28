#!/bin/bash
#SBATCH --job-name=swe-dump-mainchain
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/mainchain_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback/mainchain_%j.err
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback
date; xargs -P 8 -L 1 python3 mainchain_scan.py < jobs_mainchain.txt; echo "SCAN_DONE $(date)"
python3 mainchain_report.py > mainchain_report.txt; echo "ALL_DONE $(date)"

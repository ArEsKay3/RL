#!/bin/bash
#SBATCH --job-name=dumpbrowse-prewarm
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools/dumpbrowse/logs/prewarm_%j.out
# usage: sbatch prewarm.sbatch <list file with lines "<run> <target_step file>">
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/tools/dumpbrowse
date; wc -l "$1"
xargs -P 12 -L 1 python3 build_index.py < "$1"
echo "DONE $(date)"

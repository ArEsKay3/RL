#!/bin/bash
#SBATCH --job-name=swe-token-bias
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=220G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.err
set -u
W=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump
U=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs
FILES=()
for exp in nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-nopg-noprefix-20260919 nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-20260920; do
  for f in $U/$exp/dumps/token_level/step_*_chunk_*.pt; do
    [[ -f "$f" ]] || continue
    out="$W/analysis/token_bias/${exp}__$(basename "$f" .pt).bias.npz"
    [[ -f "$out" && ! "$f" -nt "$out" ]] || FILES+=("$f")
  done
done
echo "chunks to audit: ${#FILES[@]}"; date
cd $W/tools && NPROC=16 python3 token_bias_by_id.py $W/analysis/token_bias "${FILES[@]}"; echo "exit $?"; date

#!/bin/bash
#SBATCH --job-name=swe-dump-summaries
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=220G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.err
set -u
W=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump
U=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs
EXPS="nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-20260920 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-nopg-noprefix-20260919 nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919"
hostname; date
ROLL=(); TOK=()
for exp in $EXPS; do
  for f in $U/$exp/dumps/rollouts/target_step_*.jsonl; do
    [[ -f "$f" ]] || continue
    name=$(basename "$f" .jsonl)
    out="$W/analysis/rollouts/${exp}__${name}.summary.jsonl"
    [[ -f "$out" && ! "$f" -nt "$out" ]] || ROLL+=("$f")
  done
  for f in $U/$exp/dumps/token_level/step_*_chunk_*.pt; do
    [[ -f "$f" ]] || continue
    name=$(basename "$f" .pt)
    out="$W/analysis/tokens/${exp}__${name}.rows.csv"
    [[ -f "$out" && ! "$f" -nt "$out" ]] || TOK+=("$f")
  done
done
echo "rollout files to summarise: ${#ROLL[@]}"; echo "token chunks to summarise: ${#TOK[@]}"
cd $W/tools
if (( ${#TOK[@]} )); then NPROC=12 python3 token_level_summary.py $W/analysis/tokens "${TOK[@]}"; echo "token_level_summary exit $?"; fi
if (( ${#ROLL[@]} )); then NPROC=16 python3 rollout_summary.py $W/analysis/rollouts "${ROLL[@]}"; echo "rollout_summary exit $?"; fi
date; echo "DONE rollouts=$(ls $W/analysis/rollouts/*.summary.jsonl | wc -l) token_rows=$(ls $W/analysis/tokens/*.rows.csv | wc -l)"

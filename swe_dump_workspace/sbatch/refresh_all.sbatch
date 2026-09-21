#!/bin/bash
#SBATCH --job-name=swe-dump-refresh
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
EXPS="nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-nopg-noprefix-20260919 nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-20260920 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-20260921 nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921"
stale() { [[ -f "$2" && ! "$1" -nt "$2" ]] && return 1 || return 0; }
cd $W/tools; date
ROLL=(); TOK=()
for exp in $EXPS; do
  for f in $U/$exp/dumps/rollouts/target_step_*.jsonl; do [[ -f "$f" ]] || continue; stale "$f" "$W/analysis/rollouts/${exp}__$(basename "$f" .jsonl).summary.jsonl" && ROLL+=("$f"); done
  for f in $U/$exp/dumps/token_level/step_*_chunk_*.pt; do [[ -f "$f" ]] || continue; stale "$f" "$W/analysis/tokens/${exp}__$(basename "$f" .pt).rows.csv" && TOK+=("$f"); done
done
echo "summaries: ${#ROLL[@]} rollout files, ${#TOK[@]} token chunks"
(( ${#TOK[@]} )) && NPROC=12 python3 token_level_summary.py $W/analysis/tokens "${TOK[@]}"
(( ${#ROLL[@]} )) && NPROC=16 python3 rollout_summary.py $W/analysis/rollouts "${ROLL[@]}"
python3 build_wv_map.py
for spec in "think_stop:thinkaudit.json:think_stop_audit.py" "token_bias:bias.npz:token_bias_by_id.py" "think_close:tc.npz:think_close_by_step.py"; do
  IFS=: read dir ext script <<<"$spec"; FILES=()
  for exp in $EXPS; do for f in $U/$exp/dumps/token_level/step_*_chunk_*.pt; do [[ -f "$f" ]] || continue; stale "$f" "$W/analysis/$dir/${exp}__$(basename "$f" .pt).$ext" && FILES+=("$f"); done; done
  echo "$script: ${#FILES[@]} chunks"
  (( ${#FILES[@]} )) && NPROC=16 python3 $script $W/analysis/$dir "${FILES[@]}" > /dev/null
done
python3 aggregate_think_audit.py > /dev/null && python3 aggregate_token_bias.py > /dev/null && python3 aggregate_think_close.py > /dev/null
for exp in $EXPS nano35-swe-v2-stream128-inorder1-cmh-64n-minf; do [[ -d $U/$exp/checkpoints ]] && chmod -R g+rX,o+rX $U/$exp/checkpoints 2>/dev/null; done; echo "perms refreshed"
echo "REFRESH DONE"; date

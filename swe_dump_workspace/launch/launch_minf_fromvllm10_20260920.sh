#!/bin/bash
# Chain F: MINF continuing from run A (vLLM lineage) step_10. Verifies the seed copy, then submits 5 singleton segments one per minute at Nice=2000.
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump
U=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs
EXP=nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920
SEED=$U/$EXP/checkpoints/step_10
LOG=analysis/logs/launch_minf_fromvllm10_20260920.log
nf=$(find $SEED -type f | wc -l); nw=$(find $SEED/policy/weights -type f | wc -l); nb=$(find $SEED -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
echo "seed check $(date '+%H:%M:%S'): files=$nf weights=$nw bytes=$nb" | tee -a $LOG
if [[ "$nf" != "140" || "$nw" != "134" || "$nb" != "395124903238" ]]; then echo "ABORT: seed copy incomplete" | tee -a $LOG; exit 1; fi
for i in 1 2 3 4 5; do
  [[ $i -gt 1 ]] && sleep 65
  echo "== submitting segment $i $(date '+%H:%M:%S')" >> $LOG
  out=$(ENGINE=minf RUN_DATE=20260920 EXP_NAME=$EXP SEED_CHECKPOINT=$SEED bash ./launch_swe_dump.sh +checkpointing.load_replay_buffer=false 2>&1)
  echo "$out" | grep -E "Submitted batch job|engine:|Seed checkpoint|ERROR|error" >> $LOG
  jid=$(echo "$out" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$")
  [[ -n "$jid" ]] && scontrol update JobId=$jid Nice=2000 && echo "nice2000 $jid" >> $LOG
done
echo "ALL_SUBMITTED $(date '+%H:%M:%S')" >> $LOG
grep -oE "Submitted batch job [0-9]+" $LOG | grep -oE "[0-9]+$" | tr '\n' ' '

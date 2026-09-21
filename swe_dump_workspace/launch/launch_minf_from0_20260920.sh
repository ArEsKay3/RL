#!/bin/bash
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump
LOG=analysis/logs/launch_minf_from0_20260920.log
for i in 2 3 4 5; do
  sleep 65
  echo "== submitting segment $i $(date '+%H:%M:%S')" >> $LOG
  out=$(ENGINE=minf RUN_DATE=20260920 EXP_NAME=nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-20260920 bash ./launch_swe_dump.sh +checkpointing.load_replay_buffer=false 2>&1)
  echo "$out" | grep -E "Submitted batch job|ERROR|error" >> $LOG
  jid=$(echo "$out" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$")
  [ -n "$jid" ] && scontrol update JobId=$jid Nice=2000 && echo "nice2000 $jid" >> $LOG
done
echo "ALL_SUBMITTED $(date '+%H:%M:%S')" >> $LOG
grep -oE "Submitted batch job [0-9]+" $LOG | grep -oE "[0-9]+$" | tr '\n' ' '

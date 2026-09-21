#!/bin/bash
# chain H: MINF from scratch with max_tokens=204800 and chunked prefill disabled
# (plus the launcher's MINF defaults: no prefix caching, no overlap_param_gather).
# 5 singleton segments at Nice=30. Run ONCE, only after smoke 3886521 passed.
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump
LOG=analysis/logs/launch_minf_from0_maxtok_nochunk_20260920.log
if [[ -s $LOG ]] && grep -q ALL_SUBMITTED $LOG; then echo "already launched: $(grep -oE 'Submitted batch job [0-9]+' $LOG | grep -oE '[0-9]+$' | tr '\n' ' ')"; exit 0; fi
for i in 1 2 3 4 5; do
  [[ $i -gt 1 ]] && sleep 65
  echo "== submitting segment $i $(date '+%H:%M:%S')" >> $LOG
  out=$(ENGINE=minf RUN_DATE=20260920 EXP_NAME=nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-maxtok-nochunk-20260920 bash ./launch_swe_dump.sh +checkpointing.load_replay_buffer=false policy.generation.mcore_generation_config.max_tokens=204800 policy.generation.mcore_generation_config.enable_chunked_prefill=false 2>&1)
  echo "$out" | grep -E "Submitted batch job|engine:|ERROR|error" >> $LOG
  jid=$(echo "$out" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$")
  [[ -n "$jid" ]] && scontrol update JobId=$jid Nice=30 && echo "nice30 $jid" >> $LOG
done
echo "ALL_SUBMITTED $(date '+%H:%M:%S')" >> $LOG
grep -oE "Submitted batch job [0-9]+" $LOG | grep -oE "[0-9]+$" | tr '\n' ' '

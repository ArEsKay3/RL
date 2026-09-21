#!/bin/bash
# Chain F continuation (user, 2026-09-21 10:0x): 5 more singleton segments from the latest checkpoint (step_35 at submission), target step_70.
# Keeps chain F's steps 19-35 engine settings explicitly (prefix caching off, overlap_param_gather on); Nice left at 0 (user: no nice on active chains).
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump
U=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs
EXP=nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920
SEED=$U/$EXP/checkpoints/step_10
LOG=analysis/logs/launch_minf_fromvllm10_cont_20260921.log
grep -q ALL_SUBMITTED $LOG 2>/dev/null && { echo "already submitted; see $LOG"; exit 1; }
echo "latest checkpoint at submission: $(cat $U/$EXP/checkpoints/latest_checkpoint_status.json)" | tee -a $LOG
for i in 1 2 3 4 5; do
  [[ $i -gt 1 ]] && sleep 65
  echo "== submitting segment $i $(date '+%H:%M:%S')" >> $LOG
  out=$(ENGINE=minf RUN_DATE=20260920 EXP_NAME=$EXP SEED_CHECKPOINT=$SEED bash ./launch_swe_dump.sh +checkpointing.load_replay_buffer=false policy.generation.mcore_generation_config.enable_prefix_caching=false policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=true 2>&1)
  echo "$out" | grep -E "Submitted batch job|engine:|Seed checkpoint|ERROR|error" >> $LOG
done
echo "ALL_SUBMITTED $(date '+%H:%M:%S')" >> $LOG
grep -oE "Submitted batch job [0-9]+" $LOG | grep -oE "[0-9]+$" | tr '\n' ' '

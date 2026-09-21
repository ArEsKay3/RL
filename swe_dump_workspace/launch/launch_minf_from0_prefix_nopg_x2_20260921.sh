#!/bin/bash
# Two MINF-from-scratch replicas with prefix caching ON and overlap_param_gather OFF (user, 2026-09-21 10:5x),
# on partition batch_long with 8 h segments; 3 singleton segments per replica, Nice 0.
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump
LOG=analysis/logs/launch_minf_from0_prefix_nopg_x2_20260921.log
grep -q ALL_SUBMITTED $LOG 2>/dev/null && { echo "already submitted; see $LOG"; exit 1; }
for r in r1 r2; do
  EXP=nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-${r}-20260921
  for i in 1 2 3; do
    sleep 65
    echo "== submitting $r segment $i $(date '+%H:%M:%S')" >> $LOG
    out=$(ENGINE=minf RUN_DATE=20260921 EXP_NAME=$EXP SLURM_PARTITION=batch_long WALLTIME=8:00:00 SLURM_QOS=normal bash ./launch_swe_dump.sh +checkpointing.load_replay_buffer=false policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false policy.generation.mcore_generation_config.enable_prefix_caching=true 2>&1)
    echo "$out" | grep -E "Submitted batch job|engine:|ERROR|error" >> $LOG
  done
done
echo "ALL_SUBMITTED $(date '+%H:%M:%S')" >> $LOG
grep -oE "Submitted batch job [0-9]+" $LOG | grep -oE "[0-9]+$" | tr '\n' ' '

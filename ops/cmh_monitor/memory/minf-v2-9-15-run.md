---
name: minf-v2-9-15-run
description: "minf_v2_9_15 = first 86-node 4 h run of the MR52 v2 (SingleController) + MINF arm from pipeline_B, job 3766344 submitted 2026-09-15 11:01 on QOS normal, W&B nvidia/nano35-rlvr-main-tot; fork nemo_rl 7961ce437 pushed"
metadata:
  type: project
---

Launched at the user's request right after the SC+MINF smoke 3764675 passed. Command (from the /scratch pipeline_B
checkout, U=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users):
`CONTAINER=$U/rkirby/containers/nemo-rl_fd45cb8-67127697-gym.sqsh PERSISTENT_CACHE=$U/rkirby/persistent_cache
HF_HOME=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/rkirby/hf_home USE_IMAGE_GYM=0
GYM_VENV_DIR=/opt/gym_venvs MINF=1 EXP_NAME=minf_v2_9_15 bash RLVR/nemotron-3.5-nano/scripts/launch_nano35_rlvr_v2_86n.sh`.
Shape: 32 train + 32 gen + 2 gym + 20 judges, rlvr_sc_minf.yaml, 100 steps max, checkpoint every 10 steps
(backend simple + save_data_plane), 4:00:00 walltime; sauramishra's equivalent SC run managed ~28 steps in 4 h.
Stack: pipeline fc708e1; nemo_rl fork 7961ce437 (pushed 2026-09-15: 3 cherry-picks on 6e20063f2); Bridge 6b43a2c9e;
Megatron-LM 38bb3fd9f; Gym 3615823e mounted over the image venvs. Priority 94134 under nemotron_sw_post; at submit
time kpuvvada's 6x48-node jobs (105202) and simengs' 2x32 (104776) were ahead and batch had 0 idle nodes.
Lever if the queue is slow: `scontrol update job=3766344 account=nemotron_rl_systems` (fairshare 0.962 -> ~106k
priority), age weight is 0 so nothing is lost by re-prioritising.
Relaunch rule: reuse EXP_NAME=minf_v2_9_15 to resume from its checkpoints; 86-node relaunches need the user's OK.

**How to apply:** monitor with the cron; when it ends, report steps/checkpoints and delete the cron.
Related: [[pipeline-b-mr52-branch]], [[minf-v2-launch-plan]].

**Outcome (2026-09-15):** job 3766344 queued 11:01 -> started 15:30 (4.5 h in queue under normal QOS, estimate swung
11:39..21:20) -> TIMEOUT 19:30:23 after 4:00:11, ray.sub exit_code=0, ENDED marker written. Judges healthy at 15:38,
Ray head 15:41, driver 15:42, first rollouts ~15:49, first optimizer step done 16:04. 25 optimizer steps closed
(0..24), 25 nvshmem refits, 0 errors/tracebacks/actor deaths, 0 Gym venv builds (mounted Gym 3615823e on image venvs
at 86 nodes). Cadence ~8.5 min/step after pipeline fill (step_1 16:05 -> step_25 19:29). Checkpoints kept: step_10
(17:25), step_20 (18:49) permanent rungs + step_25 (19:29, latest/resume point), 371 GB each, all with
policy/weights/data_plane/rollout_recovery/train_dataloader/config. Step metrics never reached ray-driver.log (the
controller's stdout is block-buffered and lost on SIGKILL) -> metrics only in W&B
nvidia/nano35-rlvr-main-tot/runs/mirzxvk5. Resume = same command with EXP_NAME=minf_v2_9_15 (starts from step_25).
Data: curriculum_amplified_dolphin_v41 (199,680 rows, byte-identical to akamehra's original), 25*512 = 12,800 prompts
consumed. The user parked further RLVR follow-ups in favour of the SWE-E2E Option A plan.

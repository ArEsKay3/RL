---
name: hsg-cluster-facts
description: "HSG (oci-hsg-cs-001) site facts for the SWE-E2E v2 campaign after the 2026-09-29 move from CMH: run root, partitions, QOS, account, node shape, launcher, reaper, monitor location"
metadata:
  type: project
---

Run manager moved from CMH to HSG on 2026-09-29 (login node oci-hsg-cs-001-vscode-01, CDT clock). Verified by probing, not taken from CMH notes:

- Run root `R=/lustre/fsw/portfolios/llmservice/users/rkirby/runs` (NOT the nemotron portfolio: `/lustre/fs*/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby` exists but is empty). `/lustre/fsw` and `/lustre/fs1` are separate mounts here, not a symlink. User quota on fsw 117 T used of 250 T.
- Slurm: partitions `batch` (4 h max) and `batch_long` (7 d); account `nemotron_sw_post`; QOS `normal` for runs, `short` for smokes; no rkirby reservation; MaxJobCount 40000. Nodes: 144 CPUs, 4 GPUs (GB200 NVL72), block topology (`--segment`, SEGMENT_SIZE=16 in the launcher).
- Existing HSG launcher: `/home/rkirby/swe_dump/launch_swe_dump.sh` (trees rkirby/swe-v2-dump 7f8a2b9d + MLM 880de0fce) with HSG paths (model under pjin, data under sdevare, container rl-gym.64248325.sqsh under coreai_dlalgo_ci, sandbox under igitman), GPUS_PER_NODE=4, CPUS_PER_WORKER=144, W&B project ultra-v3-swe-e2e-convergence. Its 16-node MINF smoke `nano35-swe-v2-minf_dump-smoke-hsg-16n` completed 2026-09-21 (job 7334051, 1 h 17 m); run dir layout identical to CMH (ray_logs/<jid>-logs, dumps/rollouts, dumps/token_level).
- GPU-idle reaper here is `OccupiedIdleGPUsJobReaper`, exempted via SLURM_COMMENT JSON (exemptIdleTimeMins 240) in the launcher.
- Monitor: clone `/home/rkirby/workspaces/nemo-rl-workspace/ops-monitor`, branch `rkirby/hsg-ops-monitor-20260929`, scripts `ops/hsg_monitor/tick.sh` and `sclog.sh`. Slurm broker MCP (slurm_my_jobs, slurm_query, slurm_job_status, slurm_cancel_job) works on HSG. `gh` is installed and logged in as ArEsKay3.

**Why:** every CMH path, job id, reservation and the hero-res QOS in the older notes is history; nothing from CMH Lustre (checkpoints, dumps, gym_results) exists on HSG.

**How to apply:** an arm resumes on HSG only after rkirby copies its `checkpoints/step_N/` into `R/<run>/checkpoints/`; otherwise it is a fresh start. Launch nothing until rkirby names the arms. Related: [[hsg-move-handoff]], [[feedback-no-start-with-replay-buffer-restore]], [[rlvr-minf-path-overrides]] (older HSG defaults for the RLVR line, different roots).

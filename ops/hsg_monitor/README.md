# HSG ops monitor (oci-hsg-cs-001)

Adapted 2026-09-29 from `ops/cmh_monitor/` for the HSG cluster. Same self-discovering design; only site facts changed.

| item | CMH | HSG |
|---|---|---|
| cluster | aws-cmh-slurm-1 | oci-hsg-cs-001 (login `oci-hsg-cs-001-vscode-01`) |
| run root `R` | /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs | /lustre/fsw/portfolios/llmservice/users/rkirby/runs |
| partition / max wall | batch_long 8 h segments (hero-res) | batch 4 h; batch_long 7 d |
| QOS | hero-res (reservation sla_res_nemotron_sw_post) | normal (short for smokes); no reservation |
| account | nemotron_sw_post | nemotron_sw_post |
| squeue user | rkirby | rkirby |
| node | 140 CPUs, 8 GPUs | 144 CPUs, 4 GPUs (GB200 NVL72), `--segment` block topology |
| launcher | swe_dump/launch_swe_dump.sh (CMH paths) | /home/rkirby/swe_dump/launch_swe_dump.sh (HSG paths, GPUS_PER_NODE=4, SEGMENT_SIZE=16, smoke verified 09-21 job 7334051) |
| GPU-idle reaper | uid 146504, 90 min | OccupiedIdleGPUsJobReaper via SLURM_COMMENT (exemptIdleTimeMins 240) |
| SC log cache | ~/.claude/jobs/e247f0cd/tmp | the script's own directory |

`tick.sh` prints the queue header (running / pending / held), one `queued:` line per non-running job (qos, partition, nodes, reason, start estimate), then the per-arm block for each RUNNING job. `sclog.sh <jobid>` resolves the SingleController log.

Every launch and resume carries `+checkpointing.load_replay_buffer=false`. Nothing from CMH Lustre is present here; an arm resumes on HSG only after its `checkpoints/step_N/` is copied into `R/<run>/checkpoints/`.

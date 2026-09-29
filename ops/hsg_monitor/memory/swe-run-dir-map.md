---
name: swe-run-dir-map
description: "Full mapping of SWE-E2E v2 run directories to arm letters, job ids, rungs and summaries; lives at swe_dump/analysis/run_dir_map.md"
metadata:
  type: reference
---

Canonical run-dir -> arm-letter map:
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/run_dir_map.md`

Written 2026-09-25 on the user's request. Covers all 54 directories under
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/`, grouped as: running now,
queued, lettered dump arms (run A - chain N + main chain, workspace swe_dump), the splice/cross-train family
(chain P/Q/R/R-prime, workspace swe_replay_splice), the frozen-weights pair (chain P-prime/Q-prime, workspace
swe_lr0 overlay), smokes, pre-created-never-ran dirs, and the pre-SWE RLVR legacy dirs.

Each row carries the run dir, the job ids (which are exactly the `ray_logs/<jobid>[-N]-logs/` subdir names, `-N`
= Slurm requeue), the rungs on disk, and a one-line summary. Letters S (swe_915 vLLM from scratch, masking off)
and T (swe_915 MINF from scratch, masking off) were assigned 2026-09-25 by me and adopted by both peer sessions.
Letters H and O have no run dir - never launched.

**How to apply:** regenerate or amend when a new arm launches rather than re-deriving from `ls`; the letter
legend here is the same one the cron monitor and [[feedback-arm-labels]] use. Related: [[swe-splice-experiments]],
[[swe-dump-runs-nopg]], [[swe-915-workspace]].

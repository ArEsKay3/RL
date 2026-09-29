---
name: minf-refit-engine-pause-verified
description: "Verified on minf-v2 (job 3688117): the MINF dynamic engine is PAUSED+SUSPENDED on all 128 generation ranks during every nvshmem refit; apparent overlap in ray-driver.log is a stdout-vs-stderr ordering artifact plus ~4-9 s clock skew on 10 CMH training nodes"
metadata:
  type: project
---

User (2026-09-12) suspected the nvshmem refit ran while the engine was stepping. Checked all 14 refits of job
3688117 from `users/rkirby/runs/minf-v2/3688117-logs/ray-driver.log`:

- Code path: `MegatronWeightSynchronizer.sync_weights` -> `suspend_for_refit` (worker `_sleep`: coordinator PAUSE ->
  PAUSED -> SUSPEND -> SUSPENDED -> dist barrier) BEFORE dispatching either side's copy, `resume_after_refit` after.
  While PAUSED/SUSPENDED `run_engine_with_coordinator` only sleeps. The collector-level
  `pause_generation_for_refit` is unimplemented for MegatronGeneration (the "no native pause/resume" warning); the
  engine pause happens inside the synchronizer instead.
- Log proof: per generation worker stderr stream the order is always step prints -> "dynamic engine suspended" ->
  its `[PE n]` copy-service lines -> "dynamic engine resumed" -> step prints. 0 engine step prints while suspended,
  0 gen-side PE lines while not suspended. Engine quiet gap == driver `idle/refit_bubble` (18-21 s).
- Artifact 1: `print()` lines (driver banners, "[Rank N] paused/resumed inference engine",
  "prepare_for_generation START/END") land in ray-driver.log AFTER the whole stderr trail of the refit (~33k lines
  later), next to post-refit engine prints. Engine/logging output (stderr) is timely. Never trust cross-stream order
  in that file; use the engine step timestamps and `[PE n] [HH:MM:SS.mmm]` stamps.
- Artifact 2: 10 of 32 training nodes have wall clocks 4-9 s behind (one +1.2 s) the rest, stable across the run,
  so their PE timestamps fall inside the engine-active period. Cluster NTP issue, not overlap.
- Semantics: kv_cache_management_mode=persist, in_flight_weight_updates=true, recompute_kv_cache=false -> active
  requests stay resident (suspend deallocates 0 GB), weights swap in place, requests continue with new weights.

**How to apply:** to re-verify on a new run, use awk over ray-driver.log keyed on `MegatronPolicyWorker[rank=N]`
+ ip: track suspended/resumed state per rank and count step prints (`dynamic_engine.py:... | step N | HH:MM:SS`)
while suspended. A smoke run (11 nodes) is enough; do not launch large jobs for this.
Related: [[minf-v2-launch-plan]].

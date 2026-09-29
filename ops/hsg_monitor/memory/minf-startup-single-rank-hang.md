---
name: minf-startup-single-rank-hang
description: "2026-09-27 - chain P⁗2 segment 2 (4033270) hung at startup for 61 min because one MINF generation rank never returned from prepare_for_generation (coordinator/ZMQ start) on nvl72d180-T11; how to diagnose a silent startup hang in minutes with the Ray state API on the head node"
metadata:
  type: project
---

**Incident.** chain P⁗2 (MINF from scratch, prefix cache kept, seed 1234) segment 2, job 4033270, started 08:33:05 on 2026-09-27 (resume from step_16). All 256 MegatronPolicyWorker actors came up, loaded the checkpoint and built CUDA graphs; 255 ranks finished `prepare_for_generation` by 08:39:50 ("Coordinator started"), but rank 23 (generation actor `megatron_generation-5-3`, GPU 3 / uuid GPU-8701e854… on nvl72d180-T11, 10.67.18.167) stopped after "Initialized persistent inference engine", i.e. inside `_start_inference_coordinator` → `start_listening_to_data_parallel_coordinator`. The driver waits on all 256 calls, so no SingleControllerActor was ever created, the driver log froze at 08:39:51, and 64 nodes sat idle. No traceback, no NVSHMEMX_ERROR, no NCCL watchdog, node not drained. Not the nvl72d074-T14 nvshmem failure (nvshmem is only initialised later by the controller). rkirby authorised the manager to cancel on suspected hangs; cancelled 09:33 after 61 min; 4033271 resumes from step_16.

**How it was diagnosed (repeatable in ~2 min):**
1. Symptoms: job RUNNING, `ray-driver.log` mtime frozen just after "Inference co-ordinator is ready", no worker `.out` with `:actor_name:SingleControllerActor`, no session log file written after the freeze.
2. Head-node dashboard is reachable from the login node: IP in `session_*/logs/dashboard.log` ("http server initialized at IP:8265"). `curl http://IP:8265/api/v0/tasks?limit=2000&detail=1` → count tasks by state; a startup hang shows exactly one RUNNING task with an old `start_time_ms`. `/api/v0/actors` gives actor_id/node_id/pid (names look like `lm_policy-<rank>-<k>` / `megatron_generation-<node>-<gpu>`); `/api/v0/nodes` maps node_id → node_ip; `getent hosts IP` → hostname.
3. `curl ".../api/v0/logs/stream?node_id=…&actor_id=…&lines=60&suffix=out"` (and `suffix=err`) streams that rank's live stdout/stderr without node access.
4. Baseline for "too slow": chain P⁗2 seg 1 went engine-ready → controller in ~1 min; chain P⁗ seg 7 (MINF resume) had its first step dump 33 min after job start; vLLM resumes bring the controller up ~7 min after start.

**Why:** the launcher has no startup watchdog, so one wedged rank holds the whole allocation for the 8 h wall; and the nvl72d074-T14 history made a wrong attribution tempting — the state API settles it.

**How to apply:** when a segment shows no controller 25 min after start, run steps 1-3 before speculating about nodes; report the single RUNNING task and its host; cancelling is rkirby's call (the manager now has standing authorisation for suspected hangs); propose `ExcNodeList` for the queued segments only with the host in hand. Related: [[minf-hang-rule-false-positive]], [[swe-p4-q4-run-to-70]], [[swe-prefix-keep-arm]].

---
name: minf-startup-hang-prepare-for-generation
description: "2026-09-27 - a MINF segment hung at startup with one rank stuck in prepare_for_generation; transient, not a node fault; diagnose with the Ray state API"
metadata:
  type: project
---

chain P⁗2 (MINF from scratch, prefix cache kept across refits, seed 1234) segment 2, job 4033270, hung during startup on 2026-09-27 and burned 61 min of an 8 h wall on 64 idle nodes before being cancelled at 09:33.

**Signature** (differs from a mid-run stall — see [[minf-hang-rule-false-positive]]): the SingleController log is never created at all, the driver log freezes mid-init, and **zero files are modified anywhere under the run directory**. No traceback, no `NVSHMEMX_ERROR`, no NCCL watchdog. The driver's last line was a `MegatronPolicyWorker` bringing up its text-generation server, i.e. the final init stage before the first weight sync.

**Diagnosis recipe (Cross Train Experiment's, and the thing that made it fast):** query the Ray state API for tasks still RUNNING cluster-wide. Here exactly one was left — `MegatronPolicyWorker[rank=23].prepare_for_generation`, running 44 min on nvl72d180-T11 while the other 255 ranks finished theirs within ~2 min. That rank's own log ended at "Initialized persistent inference engine" and never reached "Coordinator started". All 256 actors and 64/64 Ray nodes showed ALIVE, so actor liveness proves nothing.

**It was transient, not a node fault.** The replacement segment 4033271 was allocated the *same* nvl72d180-T11 and came up clean in 8 min (controller up, "Skipping replay buffer restore", restored from step_16, 32 pending prompts regenerated, sync done). So do not blacklist the node on one occurrence; a coordinator/ZMQ start race or a transient GPU wedge fits the evidence better. I had initially mis-attributed it to nvl72d074 purely because that node had a prior nvshmem failure — wrong, and worth not repeating.

**How to apply:** a MINF segment that has produced no SC log and no writes anywhere ~30 min after start is safe to cancel; the singleton chain rolls to the next segment and resumes from the rolling rung, costing only startup. Related: [[feedback-cancel-relaunch-on-hang]], [[swe-p4-q4-run-to-70]].

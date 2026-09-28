---
name: splice-foreign-index-startup-cost
description: "Splice/cross-train runs stall SingleControllerActor.__init__ for ~11 min because ForeignRolloutSource._build_index() scans the foreign rollout JSONLs synchronously; cost scales with the injected step range"
metadata:
  type: project
---

A cross-train/splice run that injects foreign rollouts emits a ladder of `SingleController ping failed after Ns unresponsive: GetTimeoutError` warnings during startup — 60s, 120s, ... up to ~660s — then proceeds normally.

Root cause (found by the Cross Train Experiment session, 2026-09-24, after I flagged the pattern): `ForeignRolloutSource._build_index()` runs SYNCHRONOUSLY inside `SingleControllerActor.__init__`, scanning the source run's rollout JSONLs to build the (target_step, prompt_idx) -> group_id index. For `foreign_rollout.steps=1:20` against chain M (vLLM from scratch, no overlap, replica 1) that is 20 files totalling ~234 GB, hence ~11 minutes of unresponsive actor construction. The scan is currently unparallelised and not async, so actor-construction time scales LINEARLY with the injected step range, on top of normal Ray and model-load cost. Revisit before using a much larger range.

The watchdog itself is NOT splice-specific: it lives at `nemo_rl/examples/run_grpo_single_controller.py:~198` in the shared entrypoint, and that file is byte-identical (md5 1d39bbe008fa887099d5cc3f2c2c857a) across the swe_dump and swe_replay_splice workspaces. It only fires when `__init__` is slow, which is why the ordinary arms never print it.

DIAGNOSTIC VALUE: `SingleController ping failed` is ZERO in every normal arm — verified across chain J (MINF from scratch, prefix cache on, replica 2, ran to step_50), chain I (replica 1) and chain L (MINF from chain K step_10). So on a NON-splice run this warning is anomalous and worth investigating; on a splice run it is expected and benign provided the ladder stops rather than escalating and real progress follows. Configured `stall_timeout_s=7500`, so the ~660 s ladder is far inside the limit.

**Why:** the peer initially reported this as "normal MINF startup"; it is not — no ordinary arm produces it. Recording so nobody re-derives it, and so a future large-range splice is not launched blind to the scaling.

**How to apply:** for a splice run expect roughly 11 min of silent actor construction per ~234 GB of injected rollouts and do not apply the hang rule during it; judge health by whether the ping ladder terminates and `_sync_weights` / `rollout_pump: starting` follow. Related: [[swe-dump-runs-nopg]].

---
name: feedback-never-prune-live-arms
description: "User rule (2026-09-26) - never delete checkpoints, replay buffers or dumps from a chain that is still running; cleanup passes apply only to finished/stopped arms"
metadata:
  type: feedback
---

Do not prune anything belonging to an actively running arm. The earlier cleanup directive ("remove anything that isn't the latest checkpoint and has a HF conversion") applies **only to finished or stopped chains**. On a live chain, leave every rung, `replay_buffer.pt` and dump in place no matter how many newer rungs exist and no matter whether the HF export has verified.

Concretely: chain P⁗ (MINF from scratch, prefix cache kept across refits) step_20 was fully HF-exported and two rungs behind the head, and the user still rejected the deletion. A peer session reporting "step_N is now eligible under your gate" is not authorisation.

**Why:** a running chain can still need earlier state, and a deletion during a run cannot be undone from a job that is mid-flight; the user wants the disk-pressure tradeoff decided by them, per arm, not by a rule applied continuously.

**How to apply:** when quota gets tight, report the candidates and their sizes and wait — do not delete. Only after an arm is stopped and will not be restarted does the HF-gated prune apply. Related: [[swe-splice-experiments]], [[feedback-monitor-ops-only]], [[feedback-arm-labels]].

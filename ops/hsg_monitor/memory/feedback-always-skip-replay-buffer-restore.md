---
name: feedback-always-skip-replay-buffer-restore
description: "rkirby standing rule 2026-09-28 — every resume of every arm always uses +checkpointing.load_replay_buffer=false, no exceptions"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a7f966fb-b99e-4db0-bb54-a178ecde5ae0
  modified: 2026-09-28T21:32:33.780Z
---

rkirby, 2026-09-28, verbatim: "FROM NOW ON EVERYTHING IS ALWAYS RUN WITH
LOAD_REPLAY_BUFFER=FALSE ALWAYS!"

This supersedes my earlier per-arm judgment call (leave V2/V3 on the
default, reserve the flag for chain V's specific incident) — that call was
wrong in a way that had a real, silent cost.

**Why:** chain V3 (seed 4321) hit a Slurm requeue mid-rollout-collection.
The forced driver restart went through NeMo RL's checkpoint-resume path,
which restores the replay buffer by default. That pulled pre-crash in-flight
rollouts (started under the old weight version) into the first post-restart
step's commit instead of discarding them — `target_step_00002.jsonl` had 768
rows instead of 512, mixing weight_version 1 and 2. No traceback, no error;
only visible by reading the rollout dump row counts. The trainer's batch
size held (512/step every time, per token_level dumps), but the prompt
curriculum permanently shifted ~2 blocks ahead of V/V2 from step 4 on —
found by [[swe-datadiff-investigation]]'s successor session, not by any
monitoring I was doing.

**How to apply:** `+checkpointing.load_replay_buffer=false` on every launch
and every resume, unconditionally — not just after a crash, not just for
one arm. On `swe_vllm_parity` this is now baked into
`launch_swe_vllm_parity.sh` as `RESUME_OVERRIDES`, always appended, so it
never needs to be typed per-call (commit `4bc747be`, pushed to
`rkirby/swe-v2-vllm-parity`). It's a no-op on a fresh start (nothing to
resume). If porting the launcher pattern to another workspace, bake it in
the same way rather than relying on remembering to pass it.

**Precondition to know about, not to gate on:** the flag requires the
checkpoint's `pending_rollouts.pt` to exist and be `complete=True`
(`single_controller.py:479/507`); it raises cleanly with a clear message if
not, rather than silently doing the wrong thing — that's the failure mode
you want, not the default's silent wrong-behavior. Every checkpoint on these
arms so far has satisfied it.

See [[swe-vllm-parity-arm-v]] for the arm state this decision affects.

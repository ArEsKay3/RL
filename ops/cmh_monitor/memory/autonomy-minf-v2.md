---
name: autonomy-minf-v2
description: User granted standing authorization (2026-09-11 ~22:35) to launch minf-v2 without waiting and to keep relaunching/debugging autonomously until a full 4-hour 86-node run completes
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7aa0a354-aa1b-4011-8140-8d3122eb7495
  modified: 2026-09-12T22:01:53.678Z
---

User said: "Don't wait for me. Once you have nccl or nvshmem launch and keep debugging until you get a full 4H 86
node." Context: nvshmem smoke 3683477 passed; nccl smoke 3683907 was mid-run.

**Why:** the user is done with per-step approvals for this campaign; each round trip costs queue time.

**Update 2026-09-12 15:05:** user said "Keep this 4H run to completion, but then don't submit any followups for
now." So: monitor job 3688117 to its 15:47 walltime, report the final step count/checkpoints and the optimizer
bit-compare result, and do NOT relaunch or submit anything afterwards until told.

**How to apply (original):** pick the refit backend from the nccl smoke result (nccl if it passes, else nvshmem), launch
`EXP_NAME=minf-v2 MINF=1 ...` from pipeline_A, monitor, and on any failure diagnose, fix (committing real fixes
to the branch, no rationale comments in code), and relaunch under the same EXP_NAME (it resumes from its
checkpoint dir) without asking. Still respect: no code-comment notes ([[feedback-terse-code-comments]]), no Lustre
run-time Gym venv builds, and don't push to origin unless asked. Report each launch/failure/fix as it happens.
Success = a minf-v2 job that runs to its 4h walltime (or 100 steps) with checkpoints landing every 10 steps.

---
name: minf-hang-rule-false-positive
description: 2026-09-26 - a MINF arm satisfied both halves of the 20-min hang rule and still recovered by itself after ~50 min; raise the threshold before cancelling
metadata: 
  node_type: memory
  type: project
  originSessionId: 824af663-9c1e-4cb4-9c9e-fd1feb14c5fa
  modified: 2026-09-27T05:42:24.831Z
---

chain P⁗ (MINF from scratch, prefix cache kept across refits), job 4022657, went fully silent from 11:39:42 to ~12:30 on 2026-09-26: SingleController log unwritten for 50 min, no `| step N | HH:MM:SS` engine line for 20 min, zero files written under `dumps/` or `checkpoints/` for 15+ min, step-27 token dump frozen at `chunk_003` of 4 (chunks 1-3 had arrived 1-3 min apart), step-28 rollout bar already at 16/16. Both halves of the MINF hang rule were met and the correct action under the rule was `scancel 4022657`.

The cancel was refused by the permission classifier, and the job recovered on its own — it went on to complete steps 27, 28, 29, 30 and 31 at normal pace, with step_30 landing complete at 13:27. Cancelling would have thrown away five steps for nothing.

**Why:** the 20-minute thresholds are calibrated for generation stalls; a MINF arm in the training/refit phase can legitimately produce no engine lines, no dumps and no SC output for the better part of an hour, and no available signal distinguished that from a dead run.

Second instance 2026-09-26 22:13-22:41 (job 4022658, step 55): SC stdout silent 27 min after the 3rd of 4 chunks, no rung for 55 min; cause was one Gym group whose `Collecting rollouts` bar in ray-driver.log sat at 15/16 for 55 min (58 min total); its release dispatched the last chunk (dumps/token_level/step_00055_chunk_004.pt at 22:40:56) and MegatronPolicyWorker ranks logged `Executing pipeline` seconds later. Real-time signals that separate head-of-line waiting from a dead run: the `Collecting rollouts N/16 [elapsed<eta]` bars in ray-driver.log (live), the SC's `.err` file next to its worker `.out` (INFO `train_pump: step S chunk K: 8 group(s), 8K/32 dispatched` lines; a step is 32 groups of 16 completions, closing on 32/32), and MegatronPolicyWorker `[PE n] ... Executing pipeline` lines in ray-driver.log during training.

**How to apply:** treat the 20-min rule as "start investigating," not "cancel." Before cancelling a MINF arm, require a much longer silence (60+ min with zero writes anywhere) or a real traceback, and say explicitly in the report that the arm may simply be slow. Related: [[swe-splice-experiments]], [[feedback-monitor-ops-only]], [[swe-prefix-keep-arm]].

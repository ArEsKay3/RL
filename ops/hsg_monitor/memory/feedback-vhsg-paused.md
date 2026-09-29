---
name: feedback-vhsg-paused
description: "rkirby 2026-09-29 ~19:00 CDT: all V-HSG arms (vLLM-parity MINF on HSG, seeds 42/1234/4321) are PAUSED; running segments finish their wall, nothing new is submitted or resumed until he says"
metadata:
  type: feedback
---

rkirby, 2026-09-29 (HSG, typed directly): "Let existing runs finish but otherwise I'm pausing all V-HSG."

State at the order: V-HSG (seed 42, 7546054) TIMEOUT 18:21 CDT at step_8 rung + rolling; V-HSG2 (seed 1234, 7548578) FAILED 16:56 on a node death at rolling step_2; V-HSG3 (seed 4321, 7548591) running to its 19:39 wall, rungs step_5/step_8. No followers exist.

**Why:** the Log Analyzer session reported 20-70 % of rollouts per step lost to a harness exit-1 failure since ~15:40 CDT (Gym cache on NFS home shared by three 64-node jobs, LOG_LEVEL=CRITICAL hiding the traceback); the parity session was told to stand down at the same time.

**How to apply:** no submit, no resume, no cancel on any V-HSG run; let V-HSG3 reach its wall; report the final state. Resuming later means fresh submissions from the rungs with +checkpointing.load_replay_buffer=false, after the Gym cache is moved off home. Related: [[hsg-cluster-facts]], [[feedback-hsg-batch-queue]].

---
name: feedback-vhsg-stopped-20261001
description: "rkirby 2026-10-01 19:49 CDT: kill the V-HSG arms and tell the owner (VLLM Parity session) to stand down for now; run manager cancelled all five jobs at 19:50"
metadata:
  type: feedback
---

rkirby, 2026-10-01 19:49 CDT (typed directly): "you can kill the V-HSG, and inform the owner we are stopping and to stand down for now."

Executed 19:50 CDT by the run manager (scancel, followers first so no afterany fired): 7603673 and 7601140 (followers), 7598853 (queued seed-1234 segment), then running 7598839 (V-HSG-np-r2 seed 42, 0:26 in) and 7592517 (V-HSG3-np-r2 seed 4321, 2:31 in). All CANCELLED per sacct. The VLLM Parity session was told to stop its top-up and stand down; the Eval Runner was told no more r2 rungs will close.

Final state of the three no-prefix parity arms (nano35-swe-v2-from0-parity-minf-noprefix-hsg-r2-*-20260930): seed 42 rungs 5-30 + rolling step_34; seed 1234 rungs 5-45 + rolling step_49; seed 4321 rungs 5-35 + rolling step_39. Checkpoints, dumps and HF exports untouched. Earlier today's history: OpenHands cache wipe incident ([[hsg-parity-tree-deletion-20261001]]), two NODE_FAIL losses (seed 42 17:54, seed 1234 19:15).

**Why:** rkirby's call; no reason given to the run manager.

**How to apply:** nothing on the V-HSG arms is submitted, resumed or chained until rkirby says so; resuming later is a fresh submission from the rolling checkpoints with +checkpointing.load_replay_buffer=false. This supersedes [[feedback-vhsg-paused]] (09-29 pause) as the current order. Related: [[swe-verified-eval-vhsg-np-r2]].

---
name: feedback-rkirby-timeslices-the-block
description: "rkirby 2026-09-27: he allocates the shared 64-node reservation block between chain U and chain V manually, by telling the monitor; stop surfacing the allocation as a decision"
metadata:
  type: feedback
---

rkirby, 2026-09-27 13:55 PDT: "I will time slice the U and V work manually by talking to you. Don't worry."

This closes the "first validated smoke takes the block" question. Do not re-raise who should get the free 64 reservation nodes, do not relay either team's case for the slot, and do not run a submit command on behalf of a session whose own permission classifier blocked it.

**Why:** two sessions ("Get Latest Main?" for chain U, "VLLM parity" for chain V) were both validating 16-node smokes for the same block, and I had escalated the allocation to him twice in twenty minutes. He wants the scheduling decision, not the analysis of it.

**How to apply:** keep reporting both arms' state in the tick — job, elapsed, gate cleared, errors — because that is the evidence he time-slices on. Tell each team that rkirby directs the block and that neither should submit into it without his word. Answer a direct question about capacity if asked; do not volunteer a recommendation. Related: [[feedback-monitor-ops-only]], [[feedback-cron-table-reports]], [[main915-prefix-stitch-storm]], [[swe-vllm-parity-arm-v]].

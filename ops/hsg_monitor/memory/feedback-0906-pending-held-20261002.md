---
name: feedback-0906-pending-held-20261002
description: "rkirby 2026-10-02 12:35 CDT \"pause all pending runs\" - all four pending 09-06 upstream 64n vLLM jobs user-held by the Run Manager; release is his call"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7b1a546f-f8f3-4d0d-8430-81566ea147fd
  modified: 2026-10-02T17:36:01.898Z
---

rkirby ordered "pause all pending runs" at 12:35 CDT 2026-10-02. The Run Manager placed user holds on every pending 09-06 upstream 64-node vLLM job: 7615130 (seed 42 follower, resumes from step_9), 7615131 (seed 1234 follower), 7615132 (seed 4321 follower) and 7615113 (seed 4321 lead, already owner-held). The running seed 1234 lead 7615111 was left alone (wall 12:46).

**Why:** his explicit word; pending jobs stay out of the queue until he says otherwise, and the afterany followers must not fire into an unattended queue.

**How to apply:** do not release any of these without rkirby's explicit instruction. When he does, release the follower only after confirming the lead's rolling checkpoint is complete, and keep +checkpointing.load_replay_buffer=false on every segment. See [[swe-nrl-0906-and-0921-workspaces]] and [[feedback-hsg-batch-queue]].

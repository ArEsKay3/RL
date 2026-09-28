---
name: feedback-eval-requeue-until-done
description: rkirby 2026-09-27 - no one-resume limit on SWE-Bench eval shards; re-queue aborted/retry-limit shards at every tick until every eval completes
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bf3c842c-2229-4c76-9865-a934d05e45ad
  modified: 2026-09-27T17:17:44.803Z
---

rkirby, verbatim via the run manager 10:20 PDT 2026-09-27: "I have not such one resume rule submit until you get it done." There is no one-resume limit for SWE-Bench Verified eval shards (chains Q⁗, P⁗2, Q⁗2 and any later arm): a shard whose continuation chain hits "Infra retry limit (3)" is re-queued (sbatch of its own `nel_eval.sbatch`, which resumes from cached attempts) at every monitor tick until it writes `.shard_done`. He did not ask for an eval-concurrency cut under the AWS ECS DescribeTasks throttling; completion beats API politeness.

**Why:** the throttling has cost only wall-clock: every abort preserved its verified attempts through the cache (79 abort events across two arms, 0 attempts lost by 10:14 PDT 2026-09-27), so more resumes are cheap and the alternative (a 9-shard partial rung) is a real data loss.

**How to apply:** at each tick re-queue every dead shard; record every resume in runs.json `resume_jobs[shard]` with the earlier resumes kept under `previous` so the record stays auditable; never reset `.nel_infra_retries` (a resumed job gets no continuations, so each resume is one attempt window and the next tick re-queues again). If re-queueing genuinely cannot keep up, say so with numbers instead of slowing down pre-emptively. Supersedes the "one resume then ask rkirby" rule in [[swe-verified-eval-chainQ4]], [[swe-verified-eval-chainP42]] and [[swe-verified-eval-chainQ42]].

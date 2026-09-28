---
name: feedback-cancel-relaunch-on-hang
description: "User grant (2026-09-27) - standing authorisation to cancel and relaunch a segment when I suspect it is hung, without asking first"
metadata:
  type: feedback
---

rkirby: "You are approved to cancel and relaunch the job if you suspect it's hung."

This is standing authorisation to `scancel` a RUNNING segment on a suspected hang and let the singleton chain roll to its follower — no per-incident approval needed. It does **not** extend to cancelling queued segments (that still requires a repeatable code failure or an explicit stop rule), to jobs the user holds, or to modifying queued jobs (e.g. `scontrol update ExcNodeList`), which remains his call or the owning session's.

**Why:** the first hang cost 61 minutes of an 8 h wall on 64 idle nodes while I escalated and waited, because `mcp__slurm-broker__slurm_cancel_job` was refused by the permission classifier on that occasion even though the identical call had succeeded twice earlier the same night.

**How to apply:** evidence first, then act — state the specific signals (driver/SC mtime, zero writes anywhere under the run dir, absent controller actor, engine step lines) in the report. Remember the false positive in [[minf-hang-rule-false-positive]]: a MINF arm mid-step can go quiet for ~50 min and recover, so a *running* arm needs a long silence, whereas a *startup* that never produces an SC log and writes nothing anywhere is a much safer call. If the broker refuses, surface the exact command to the user rather than retrying repeatedly. Related: [[feedback-never-prune-live-arms]], [[swe-p4-q4-run-to-70]].

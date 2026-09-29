---
name: swe-strip-policy-keep-hf
description: "Reclaim ~368 GB/rung by deleting checkpoints/step_N/policy where step_N/hf exists, keeping each arm's last checkpoint whole; script and the 2026-09-27 run that freed 7.73 TiB"
metadata:
  type: project
---

A rung is `checkpoints/step_N/` = **430 GB**: `policy/` 368 GB (Megatron training state) + `hf/` 62 GB (the evaluable export) + a few small `.pt`/json files. **`hf/` is what evals need; `policy/` is only needed to resume training.** So any rung that has been exported and is not a resume point can drop `policy/` and keep everything that matters.

Script: `/home/rkirby/.claude/jobs/e247f0cd/tmp/strip_policy.sh` (`dry` | `go`). Guards: an explicit arm allow-list, requires `step_N/hf` to exist, never touches the **highest-numbered checkpoint** of an arm, and path-pattern-refuses anything outside `$R/*/checkpoints/step_*`. Uses `find -type f -delete` before `rm -rf` so it streams instead of buffering (the login node OOM-kills buffering deletes).

**Run 2026-09-27 20:45 on rkirby's word** ("remove all but the last checkpoint where there is a HF conversion") across chain P⁗, chain P⁗2, chain Q⁗, chain Q⁗2: 22 rungs stripped.

| arm | stripped | kept whole |
|---|---|---|
| chain P⁗ | 20,25,30,35,40,45,50,55 (5/10/15 already stripped earlier) | step_60 |
| chain P⁗2 | 5,10,15,20 | step_22 (held resume point, has no hf) |
| chain Q⁗ | 5,10,15,20,25,30,35 | step_40 |
| chain Q⁗2 | 5,10,15 | step_20 (held resume point) |

**Result: 92.77 -> 85.04 TiB used, free 7.23 -> 14.96 TiB (7.73 TiB reclaimed).** Verified after: every rung still has its 25-file `hf/`, and only each arm's last checkpoint retains `policy/`. Both held arms (P⁗2 4033272, Q⁗2 4045357) remain resumable.

**Why:** checkpoints were ~42 TiB of a 100 TiB quota with only ~1.8 TiB on running arms; see [[swe-gym-results-is-the-disk]] for the other half of the accounting.

**How to apply:** safe to repeat on any finished arm once its rungs are HF-exported and merged. Evidence it is safe: chain P⁗'s step_5/10/15 had already been policy-stripped before this run and the arm trained to step_60 without issue. Do NOT strip an arm's latest checkpoint - that is the resume point, and on a held arm it may have no `hf/` at all.

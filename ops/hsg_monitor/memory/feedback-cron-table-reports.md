---
name: feedback-cron-table-reports
description: User rule (2026-09-23) - every cron/monitor status update is delivered as a table (one row per step/arm), including no-change ticks; no one-line prose updates
metadata:
  type: feedback
---

Every scheduled monitor report (eval campaigns, training chains) must be a markdown table: one row per step or arm, columns for status, shards done/running/queued/dead, verified attempts, graded-0.0 failures by class, and notes (new jobs, incidents, merges, results). Even "nothing changed" ticks use the table; at most two sentences of prose below it (training progress, anything needing the user).

**Why:** the user asked for it explicitly ("update so all your cron updates are done in table form", 19:10 PDT 2026-09-23) after several one-line no-change ticks; tables make step-by-step status scannable and comparable tick to tick.

**How to apply:** encode the table format in every CronCreate prompt (see the chain J monitor c93256bc); when reporting results, add a second small table for that step's metrics. Related: [[feedback-arm-labels]], [[swe-verified-eval-chainJ]].

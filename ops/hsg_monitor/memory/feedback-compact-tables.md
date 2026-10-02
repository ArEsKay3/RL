---
name: feedback-compact-tables
description: "rkirby 2026-10-01: the wide tick tables were not rendering as tables; every future update uses compact tables (few columns, short cells, '-' for empty, rungs in a separate table)"
metadata:
  type: feedback
---

rkirby, 2026-10-01 20:4x CDT: "The way you are showing the runs is not a table. Can you correct that for all future updates and show me the table?"

**Why:** the 11-column tick rows with long free-text cells (gym progress strings, rung lists, multi-clause failure notes) and empty trailing cells did not render as a table in his terminal.

**How to apply:** every monitor report uses up to three compact tables: (1) segments: arm | job | state | elapsed/left | ckpt | guards | step | failures, one short phrase per cell, "-" for empty; (2) rungs: arm | rungs | rolling; (3) aux: kind | running | pending. Arm labels stay letter+tag ([[feedback-arm-labels]]) but short; put the long tag once in prose if needed. No cell longer than a few words; no nested lists; no empty cells. Related: [[feedback-cron-table-reports]], [[feedback-monitor-ops-only]].

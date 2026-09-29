---
name: feedback-monitor-ops-only
description: "User rule (2026-09-23 11:1x) - the combined SWE chain monitor is operations only; stop running the close_vs_length audit and stop reporting or interpreting its metrics"
metadata:
  type: feedback
---

On the combined SWE-E2E v2 chain monitor, do operations only. Do NOT submit `analysis/logs/refresh_all.sbatch`, do NOT regenerate `tools/close_vs_length.py`, and do NOT report or interpret geo-mean P, 10th-pct P, low share, think p95, mean length, or matched-step comparisons between arms. Drop the per-step reward / tokens / truncation / gen_kl_error numbers from tick reports too.

Report only: which segments are running and queued, elapsed and wall-clock end, current step and latest checkpoint, dump/trainer failure counts, the MINF engine guard counters (`Coordinator: removed engine` and `post_process_requests` must stay 0), hang-rule evidence, segment handovers, and stop-rule completion.

**Why:** the user said "Your metrics are outdated stop looking at them only manage the running of the experiments" on 2026-09-23 at 11:1x. A separate session owns the loop-token-share analysis, which supersedes the close-probability statistic this monitor had been producing.

**How to apply:** this OVERRIDES the AUDIT and per-chain metric lines in the recurring monitor prompt, which still asks for them. If the user later asks for analysis again, resume it. Related: [[swe-dump-runs-nopg]], [[feedback-arm-labels]] (the letter-plus-tag naming rule still applies to every arm mention).

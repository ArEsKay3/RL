# SWE-Bench Verified: chain F (MINF from vLLM step 10) HF exports

Recipe: akamehra's certified `swe-bench-verified-nano-3.5` config 20260907_223032_df6becfd (NEL toolchain 569ff7e8, Harbor + OpenHands SDK 1.17.0, 500 problems x 5 repeats, temperature 1.0 / top-p 0.95, vLLM TP2 DP2), replayed with only model path, served name, output dir, cache mounts, cluster hostname (local sbatch) and provenance tags changed. pass@1 and pass@5 with CIs are the evaluator's own numbers (report.md / eval-*.json); pass@3 is computed from the merged results.jsonl with the unbiased estimator (n=5). Attempts graded 0.0 because the solver failed (agent crash: context-window overflow; turn-budget termination at the 200-turn cap, which the adapter proxy returns as an error that litellm surfaces as RateLimitError (no HTTP 429 from vLLM); agent SIGTERM) are included as zeros and listed separately.

| step | run id | pass@1 [95% CI] | pass@3 | pass@5 [bootstrap CI] | attempts | solver failures graded 0 |
|---|---|---|---|---|---|---|
| 10 | 20260921_164150_45f30c03 | 0.4944 [0.4832, 0.5056] | 0.6174 | 0.662 [0.620, 0.704] | 2500 | 10 |
| 15 | 20260921_165541_ebfcd098 | 0.4952 [0.4838, 0.5066] | 0.6238 | 0.678 [0.636, 0.718] | 2500 | 1 |
| 20 | 20260921_172445_d6e41fc8 | 0.5152 [0.5040, 0.5264] | 0.6384 | 0.682 [0.642, 0.722] | 2500 | 2 |
| 25 | 20260921_173957_591f0a06 | 0.5036 [0.4927, 0.5145] | 0.6190 | 0.662 [0.620, 0.704] | 2500 | 1 |
| 26 | 20260921_175548_50a903f1 | 0.5068 [0.4958, 0.5178] | 0.6234 | 0.666 [0.624, 0.708] | 2500 | 2 |
| 30 | 20260921_162059_2c4debae | 0.5096 [0.4984, 0.5208] | 0.6274 | 0.666 [0.624, 0.706] | 2500 | 0 |

Reference: akamehra main chain (MINF from scratch, original) step_10, same recipe: pass@1 0.4976 [0.4860, 0.5092].

Infrastructure incidents (no attempts lost; every step has 2,500 distinct problem/repeat results): admin cancellation sweeps at 12:34 (user vedmonds, 46 shards) and 14:15 (root, 26 jobs) PDT 2026-09-21, resumed from cached results via re-queued `nel eval run --resume` jobs; Slurm preemptions of 8 shards (resumed by the recipe's own continuations); 3 transient vLLM startup failures (retried); one eval-runner crash (aiohttp 'Session is closed', step_15 shard 5, resumed). Known harness bug: preempted shards write a false .shard_done; all final merges were verified at n_results = 2500.

- step_10: shard 4 primary failed at vLLM init (port collision) and was retried; 7 shards cancelled by the 12:34 sweep and resumed from cache; shards 2 and 7 preempted at 13:48/14:xx and resumed; shard 8 cancelled again by the 14:15 root sweep and resumed a second time; all 2,500 attempts accounted
- step_15: shard 5 primary crashed mid-run (aiohttp 'Session is closed' in the Fargate client) and was resumed by its continuation; all 10 shards cancelled by the 12:34 sweep and re-queued; shards 3 and 9 cancelled again by the 14:15 root sweep and resumed a second time; shard 3's first resume died at vLLM init (TCPStore ping) and was retried; all 2,500 attempts accounted
- step_20: 5 shards cancelled by the 12:34 sweep (resumed), then cancelled again by the 14:15 root sweep (resumed a second time); the 5 original shards were preempted at 14:31 and resumed by their continuations (4 of them left transient false completion markers); the final attempt of shard 0 (p397 r2) sat 50 min without model calls before finishing at 18:55; all 2,500 attempts accounted
- step_25: shard 2 primary failed at vLLM init (TCPStore ping) and was retried; 9 shards cancelled by the 12:34 sweep, re-queued, cancelled again by the 14:15 root sweep and resumed a second time from cached results; all 2,500 attempts accounted
- step_26: 9 shards cancelled by the 12:34 sweep and again by the 14:15 root sweep, resumed twice from cached results; all 2,500 attempts accounted
- step_30: 6 of 10 shards cancelled by the 12:34 admin sweep and resumed from cached results (13:00-14:20); no startup failures; benign single-request ServerDisconnected errors only


## Addendum 2026-09-22 16:20 PDT: full failure-class disclosure from the merged records

The "solver failures graded 0.0" column above counts attempts the harness graded 0.0 **without verification** (scoring_details.method = solve_failed); those counts are confirmed by the merged results.jsonl. The records also annotate attempts whose agent session ended abnormally but whose repository state **was still verified** and scored on its real test outcome (method = harbor with an error annotation). Those attempts were not called out before; they are included in every pass@k above with their actual reward. Classes: context_window_exceeded = the agent crashed on a 262,144-token context overflow after producing a patch; turn_budget_exhausted = the adapter proxy terminated the session at the 200-turn cap (surfaced by litellm as RateLimitError); solve_timeout = the agent hit the 10,800 s solve cap.

| step | graded 0.0 unverified | context overflow, verified (n / passed) | turn budget, verified (n / passed) | solve timeout, verified (n / passed) |
|---|---|---|---|---|
| 10 | 10 | 14 / 4 | 3 / 2 | 1 / 0 |
| 15 | 1 | 10 / 1 | 1 / 1 | 1 / 0 |
| 20 | 2 | 3 / 1 | 0 / 0 | 0 / 0 |
| 25 | 1 | 1 / 1 | 0 / 0 | 0 / 0 |
| 26 | 2 | 1 / 0 | 0 / 0 | 0 / 0 |
| 30 | 0 | 0 / 0 | 0 / 0 | 0 / 0 |

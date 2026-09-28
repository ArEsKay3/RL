# SWE-Bench Verified: run A (vLLM from scratch) HF exports

Run: `runs/nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918` (vLLM rollout engine from step 0, the shared vLLM lineage; training segments were still queued on a singleton dependency when these evaluations ran, so rungs beyond step_35 may appear later). Checkpoints evaluated: rungs step_5..step_35 (in-place HF exports under checkpoints/step_N/hf, each verified: 14 shards, 65,827,374,264 bytes, 6,513 tensors, world-readable). step_10 was not re-run: its export is byte-identical (all 14 shards, weight map and config) to chain F's checkpoints/step_10/hf, which chain F (MINF from vLLM step 10) already evaluated as its step_10, so that result is reused below. Evaluation order 20, 35, 30, 25, 15, 5 (one step per 15 min while the previous was healthy), submitted 18:38-19:49 PDT 2026-09-22.

Recipe: akamehra's certified `swe-bench-verified-nano-3.5` config 20260907_223032_df6becfd (NEL toolchain 569ff7e8, Harbor + OpenHands SDK 1.17.0, 500 problems x 5 repeats, temperature 1.0 / top-p 0.95, vLLM TP2 DP2, 10 shards x 1 node x 4 GPUs on batch_long), replayed with only model path, served name, output dir, cache mounts, cluster hostname (local sbatch) and provenance tags changed. pass@1 and pass@5 with CIs are the evaluator's own numbers from the merged eval json (pass@1 = mean reward over all 2,500 attempts; 95% CI = evaluator's normal-approximation interval; bootstrap CIs also reported); pass@3 is computed from the merged results.jsonl with the unbiased estimator (per problem 1 - C(5-c,3)/C(5,3), averaged over 500 problems). Attempts graded 0.0 because the solver failed before producing a patch are included as zeros and listed separately; attempts whose agent session ended abnormally but whose patch was still verified are listed in the disclosure table and carry their real test outcome.

| step | run id | pass@1 [95% CI] | pass@3 | pass@5 [bootstrap CI] | attempts | solver failures graded 0 |
|---|---|---|---|---|---|---|
| 35 | 20260923_014924_403c0408 | 0.5112 [0.5001, 0.5223] | 0.6284 | 0.668 [0.628, 0.710] | 2500 | 0 |
| 30 | 20260923_020415_4e422a1d | 0.5176 [0.5064, 0.5288] | 0.6372 | 0.676 [0.634, 0.716] | 2500 | 1 |
| 25 | 20260923_021911_7aa1a925 | 0.5112 [0.5001, 0.5223] | 0.6296 | 0.670 [0.628, 0.710] | 2500 | 1 |
| 20 | 20260923_013841_c00ced3e | 0.5080 [0.4965, 0.5195] | 0.6328 | 0.674 [0.634, 0.716] | 2500 | 1 |
| 15 | 20260923_023411_62bafb4e | 0.5116 [0.5000, 0.5232] | 0.6370 | 0.676 [0.636, 0.716] | 2500 | 1 |
| 10 (reused from chain F step_10: identical export) | 20260921_164150_45f30c03 | 0.4944 [0.4832, 0.5056] | 0.6174 | 0.662 [0.620, 0.704] | 2500 | 10 |
| 5 | 20260923_024912_44327956 | 0.4964 [0.4845, 0.5083] | 0.6306 | 0.678 [0.638, 0.718] | 2500 | 10 |

References (same recipe, same users/rkirby/evaluation/jobs tooling): chain G (MINF from scratch, no prefix cache) = ../chainG-minf-from0-nopg-noprefix-swe/results.md (pass@1 step_5 0.4920, 10 0.4908, 15 0.4900, 20 0.4788, 25 0.4800, 30 0.4816, 32 0.4772, 35 0.4752, 36 0.4640); chain F (MINF from vLLM step 10) = ../chainF-minf-fromvllm10-swe/results.md (step_10 0.4944, 15 0.4952, 20 0.5152, 25 0.5036, 26 0.5068, 30 0.5096); akamehra main chain (MINF from scratch, original) step_10 0.4976 [0.4860, 0.5092].

## Full failure-class disclosure (from the merged results.jsonl)

`graded 0.0 unverified` = scoring_details.method = solve_failed (no patch to test; counted as zero). The other columns are attempts whose agent session ended abnormally but whose repository state was still verified (method = harbor with an error annotation): context overflow = 262,144-token context exceeded after a patch existed; turn budget = adapter proxy terminated the session at 200 turns (litellm surfaces it as RateLimitError); solve timeout = 10,800 s solve cap. All are included in pass@k with their real reward.

| step | graded 0.0 unverified (classes) | context overflow, verified (n / passed) | turn budget, verified (n / passed) | solve timeout, verified (n / passed) |
|---|---|---|---|---|
| 35 | 0 () | 3 / 0 | 0 / 0 | 0 / 0 |
| 30 | 1 (context_window_exceeded 1) | 1 / 0 | 0 / 0 | 0 / 0 |
| 25 | 1 (context_window_exceeded 1) | 2 / 0 | 0 / 0 | 0 / 0 |
| 20 | 1 (context_window_exceeded 1) | 5 / 0 | 0 / 0 | 0 / 0 |
| 15 | 1 (context_window_exceeded 1) | 8 / 3 | 1 / 0 | 0 / 0 |
| 10 | 10 (agent_nonzero_exit_-15 1, context_window_exceeded 8, turn_budget_exhausted 1) | 14 / 4 | 3 / 2 | 1 / 0 |
| 5 | 10 (agent_nonzero_exit_-15 1, context_window_exceeded 7, turn_budget_exhausted 2) | 16 / 4 | 1 / 0 | 2 / 1 |

## Infrastructure incidents (no attempts lost; every merged step has 2,500 distinct problem/repeat results)

- step_35: none: all 10 primaries (3938912-3938921) completed on their first job (3h57m-5h47m); merged by the last shard's auto-merge; no preemptions, startup failures or Fargate aborts
- step_30: none: all 10 primaries (3939331-3939340) completed on their first job (4h07m-5h29m); merged by the last shard's auto-merge; no preemptions, startup failures or Fargate aborts
- step_25: none: all 10 primaries (3939654-3939663) completed on their first job (4h13m-5h19m); merged by the last shard's auto-merge; no preemptions, startup failures or Fargate aborts
- step_20: none: all 10 primaries (3938631-3938640) completed on their first job (4h23m-5h20m); merged by the last shard's auto-merge; no preemptions, startup failures or Fargate aborts
- step_15: none: all 10 primaries (3939915-3939924) completed on their first job (4h31m-6h24m); merged by the last shard's auto-merge; no preemptions, startup failures or Fargate aborts
- step_10: shard 4 primary failed at vLLM init (port collision) and was retried; 7 shards cancelled by the 12:34 sweep and resumed from cache; shards 2 and 7 preempted at 13:48/14:xx and resumed; shard 8 cancelled again by the 14:15 root sweep and resumed a second time; all 2,500 attempts accounted [chain F run of the identical step_10 export]
- step_5: none: all 10 primaries (3940227-3940236) completed on their first job (5h09m-6h47m); merged by the last shard's auto-merge; no preemptions, startup failures or Fargate aborts

Generated 2026-09-23 02:49 PDT from runs.json by write_results_a.py; per-attempt ids of every failure class are in runs.json (result.solver_failure_ids, result.terminated_but_verified_ids).

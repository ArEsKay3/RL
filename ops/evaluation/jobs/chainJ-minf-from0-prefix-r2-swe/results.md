# SWE-Bench Verified: chain J (MINF from scratch, prefix cache on, replica 2) HF exports

Run: `runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921` (MINF rollout engine from step 0, prefix caching ON, overlap_param_gather off, seed 42; independent replica 2 of the prefix-cache-on MINF-from-scratch arm; training ran on the patched dynamic_engine guard for its whole life, save_period 5, step_50 stop rule). Checkpoints evaluated: every rung step_5..step_50 (in-place HF exports under checkpoints/step_N/hf, each verified: 14 shards, 65,827,374,264 bytes, 6,513 tensors, world-readable); rolling checkpoints (non-multiples of 5) were never evaluated. Provenance: the step_5 and step_10 exports were produced 12:41 PDT 2026-09-23 by another session (hf-export array 3961162, swe_minf tree) and were byte-compared against my own re-exports (all 14 shards + weight_map identical) before use; all other exports are mine. Rungs 5-30 were submitted in parallel 14:40-18:38 PDT 2026-09-23 (user: fast turnaround, no gating); rungs 35-50 were exported and submitted as each landed.

Recipe: akamehra's certified `swe-bench-verified-nano-3.5` config 20260907_223032_df6becfd (NEL toolchain 569ff7e8, Harbor + OpenHands SDK 1.17.0, 500 problems x 5 repeats, temperature 1.0 / top-p 0.95, vLLM TP2 DP2, 10 shards x 1 node x 4 GPUs on batch_long), replayed with only model path, served name, output dir, cache mounts, cluster hostname (local sbatch) and provenance tags changed. pass@1 and pass@5 with CIs are the evaluator's own numbers from the merged eval json (pass@1 = mean reward over all 2,500 attempts; 95% CI = evaluator's normal-approximation interval; bootstrap CIs also reported); pass@3 is computed from the merged results.jsonl with the unbiased estimator (per problem 1 - C(5-c,3)/C(5,3), averaged over 500 problems). Attempts graded 0.0 because the solver failed before producing a patch are included as zeros and listed separately; attempts whose agent session ended abnormally but whose patch was still verified are listed in the disclosure table and carry their real test outcome.

| step | run id | pass@1 [95% CI] | pass@3 | pass@5 [bootstrap CI] | attempts | solver failures graded 0 |
|---|---|---|---|---|---|---|
| 50 (final) | 20260924_094153_4803e2c2 | 0.5012 [0.4900, 0.5124] | 0.6260 | 0.674 [0.634, 0.716] | 2500 | 9 |
| 45 | 20260924_074154_02b0894e | 0.4924 [0.4814, 0.5034] | 0.6094 | 0.648 [0.606, 0.690] | 2500 | 13 |
| 40 | 20260924_051151_06bc4448 | 0.5072 [0.4964, 0.5180] | 0.6206 | 0.660 [0.618, 0.700] | 2500 | 7 |
| 35 | 20260924_024138_c796f7fe | 0.4968 [0.4853, 0.5083] | 0.6240 | 0.666 [0.626, 0.708] | 2500 | 8 |
| 30 | 20260924_013847_0b1fda5b | 0.4912 [0.4798, 0.5026] | 0.6170 | 0.664 [0.622, 0.704] | 2500 | 11 |
| 25 | 20260923_232200_3bf2cf95 | 0.4920 [0.4810, 0.5030] | 0.6078 | 0.644 [0.602, 0.686] | 2500 | 17 |
| 20 | 20260924_012624_01426eb6 | 0.4840 [0.4725, 0.4955] | 0.6102 | 0.648 [0.606, 0.690] | 2500 | 16 |
| 15 | 20260924_013824_c4df7847 | 0.4856 [0.4748, 0.4964] | 0.6002 | 0.646 [0.604, 0.688] | 2500 | 8 |
| 10 | 20260923_213955_da32a898 | 0.4928 [0.4811, 0.5045] | 0.6274 | 0.674 [0.634, 0.716] | 2500 | 12 |
| 5 | 20260924_013836_d151e923 | 0.4964 [0.4849, 0.5079] | 0.6234 | 0.670 [0.628, 0.712] | 2500 | 9 |

References (same recipe, same users/rkirby/evaluation/jobs tooling): chain G (MINF from scratch, no prefix cache) = ../chainG-minf-from0-nopg-noprefix-swe/results.md (pass@1 step_5 0.4920, 10 0.4908, 15 0.4900, 20 0.4788, 25 0.4800, 30 0.4816, 32 0.4772, 35 0.4752, 36 0.4640); run A (vLLM from scratch) = ../runA-vllm-from0-swe/results.md (5 0.4964, 10 0.4944, 15 0.5116, 20 0.5080, 25 0.5112, 30 0.5176, 35 0.5112); chain F (MINF from vLLM step 10) = ../chainF-minf-fromvllm10-swe/results.md (10 0.4944, 15 0.4952, 20 0.5152, 25 0.5036, 26 0.5068, 30 0.5096); akamehra main chain (MINF from scratch, original) step_10 0.4976 [0.4860, 0.5092].

## Full failure-class disclosure (from the merged results.jsonl)

`graded 0.0 unverified` = scoring_details.method = solve_failed (no patch to test; counted as zero). The other columns are attempts whose agent session ended abnormally but whose repository state was still verified (method = harbor with an error annotation): context overflow = 262,144-token context exceeded after a patch existed; turn budget = adapter proxy terminated the session at 200 turns (litellm surfaces it as RateLimitError); solve timeout = 10,800 s solve cap. All are included in pass@k with their real reward.

| step | graded 0.0 unverified (classes) | context overflow, verified (n / passed) | turn budget, verified (n / passed) | solve timeout, verified (n / passed) |
|---|---|---|---|---|
| 50 | 9 (context_window_exceeded 9) | 24 / 2 | 3 / 1 | 3 / 0 |
| 45 | 13 (agent_nonzero_exit_137 1, context_window_exceeded 11, turn_budget_exhausted 1) | 26 / 5 | 1 / 1 | 6 / 0 |
| 40 | 7 (context_window_exceeded 6, turn_budget_exhausted 1) | 27 / 2 | 0 / 0 | 1 / 0 |
| 35 | 8 (context_window_exceeded 7, turn_budget_exhausted 1) | 17 / 0 | 2 / 0 | 0 / 0 |
| 30 | 11 (context_window_exceeded 10, turn_budget_exhausted 1) | 32 / 1 | 4 / 1 | 1 / 0 |
| 25 | 17 (context_window_exceeded 15, turn_budget_exhausted 2) | 29 / 5 | 1 / 0 | 0 / 0 |
| 20 | 16 (agent_nonzero_exit_-15 1, context_window_exceeded 14, turn_budget_exhausted 1) | 18 / 7 | 2 / 1 | 1 / 0 |
| 15 | 8 (context_window_exceeded 8) | 19 / 2 | 2 / 0 | 1 / 0 |
| 10 | 12 (agent_nonzero_exit_-15 1, context_window_exceeded 10, turn_budget_exhausted 1) | 16 / 6 | 0 / 0 | 0 / 0 |
| 5 | 9 (context_window_exceeded 9) | 13 / 3 | 1 / 0 | 0 / 0 |

## Infrastructure incidents (no attempts lost; every merged step has 2,500 distinct problem/repeat results)

- step_50: none: all 10 primaries (3978053-3978062) completed on their first job in 4 h 31 min to 7 h 56 min; no infra retries or continuations ran
- step_45: none at the job level: all 10 primaries (3976772-3976781) completed on their first job in 5 h 21 min to 8 h 18 min, no infra retries or continuations ran; one record-level agent crash (shard 5 p159 r3, agent SIGKILLed in the sandbox, exit 137, 33 min after solver entry) graded 0.0 unverified
- step_40: none: all 10 primaries (3974212-3974221) completed on their first job in 5 h 32 min to 7 h 06 min; no infra retries or continuations ran
- step_35: none: all 10 primaries (3970691-3970700) completed on their first job (4h26m-6h08m); merged by the last shard's auto-merge at ~01:50; no preemptions, startup failures or Fargate aborts
- step_30: none: all 10 primaries (3969364-3969373) completed on their first job (4h40m-6h10m); merged by the last shard's auto-merge at ~00:50; no preemptions, startup failures or Fargate aborts
- step_25: shard 1 primary 3966401 died at vLLM startup (DistNetworkError EADDRINUSE port 19827, transient) and its recipe retry 3969065 re-ran the shard from scratch (5h42m); the other 9 primaries completed on their first job (4h34m-6h11m) after a 2-hour queue wait on Priority; merged by the last shard's auto-merge at ~00:35; no preemptions or Fargate aborts
- step_20: none: all 10 primaries (3969159-3969168) completed on their first job (4h49m-6h35m); merged by the last shard's auto-merge at ~01:05; no preemptions, startup failures or Fargate aborts
- step_15: none: all 10 primaries (3969338-3969347) completed on their first job (5h17m-7h00m); merged by the last shard's auto-merge at ~01:38; no preemptions, startup failures or Fargate aborts
- step_10: shard 2 primary 3964030 died at vLLM startup (torch.distributed TCPStore rendezvous timeout, transient) and its recipe retry 3964525 re-ran the shard from scratch (started 18:19 after 2h55m queued, 5h22m run); shard 3 primary 3964031 aborted at 17:10 on a Fargate provisioning timeout (p272-class 'ECS task not RUNNING' x3, 11 secondary 'Session is closed' tracebacks) and its continuation 3964519 resumed from 90 cached attempts (started 18:19, 4h30m run); the other 8 primaries completed on their first job (5h28m-7h23m); merged by the last shard's auto-merge at ~23:35
- step_5: none: all 10 primaries (3969354-3969363) completed on their first job (4h53m-7h16m); merged by the last shard's auto-merge at ~01:55; no preemptions, startup failures or Fargate aborts

Generated 2026-09-24 10:42 PDT from runs.json by write_results_j.py; per-attempt ids of every failure class are in runs.json (result.solver_failure_ids, result.terminated_but_verified_ids).

# SWE-Bench Verified: chain G (MINF from scratch, no prefix cache) HF exports

Run: `runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920` (MINF rollout engine from step 0, overlap_param_gather=false, enable_prefix_caching=false; training stopped 08:24 PDT 2026-09-22 after step_36). Checkpoints evaluated: the rungs step_5..step_35 (in-place HF exports under checkpoints/step_N/hf) plus step_32 and the final step_36 (hf_exports/step_N/hf); every export verified (14 shards, 65,827,374,264 bytes, 6,513 tensors). The user asked for the final checkpoint to be prioritised; evaluation order was 32, 30, 25, 20, 15, 36, 35, 10, 5 (36 and 35 as soon as their exports landed).

Recipe: akamehra's certified `swe-bench-verified-nano-3.5` config 20260907_223032_df6becfd (NEL toolchain 569ff7e8, Harbor + OpenHands SDK 1.17.0, 500 problems x 5 repeats, temperature 1.0 / top-p 0.95, vLLM TP2 DP2, 10 shards x 1 node x 4 GPUs on batch_long), replayed with only model path, served name, output dir, cache mounts, cluster hostname (local sbatch) and provenance tags changed. pass@1 and pass@5 with CIs are the evaluator's own numbers from the merged eval json (pass@1 = mean reward over all 2,500 attempts = mean per-problem fraction of 5 attempts that pass; 95% CI = evaluator's normal-approximation interval; bootstrap CIs also reported); pass@3 is computed from the merged results.jsonl with the unbiased estimator (per problem 1 - C(5-c,3)/C(5,3), averaged over 500 problems). Attempts graded 0.0 because the solver failed (agent crash before any patch: context-window overflow, turn-budget termination, agent SIGTERM, empty-output sandbox fault) are included as zeros and listed separately; attempts whose agent session ended abnormally but whose patch was still verified are listed in the disclosure table below and carry their real test outcome.

| step | run id | pass@1 [95% CI] | pass@3 | pass@5 [bootstrap CI] | attempts | solver failures graded 0 |
|---|---|---|---|---|---|---|
| 36 (final) | 20260922_153736_d2c24cd2 | 0.4640 [0.4523, 0.4757] | 0.5984 | 0.642 [0.600, 0.684] | 2500 | 9 |
| 35 | 20260922_154659_68896c34 | 0.4752 [0.4637, 0.4867] | 0.6034 | 0.650 [0.608, 0.692] | 2500 | 5 |
| 32 | 20260922_141829_76b9d21c | 0.4772 [0.4656, 0.4888] | 0.6062 | 0.652 [0.610, 0.694] | 2500 | 8 |
| 30 | 20260922_142849_2e6c545a | 0.4816 [0.4698, 0.4934] | 0.6166 | 0.660 [0.618, 0.702] | 2500 | 9 |
| 25 | 20260922_144412_07485ed7 | 0.4800 [0.4685, 0.4915] | 0.6060 | 0.644 [0.602, 0.686] | 2500 | 4 |
| 20 | 20260922_145907_f03e11ee | 0.4788 [0.4675, 0.4901] | 0.6018 | 0.648 [0.606, 0.690] | 2500 | 9 |
| 15 | 20260922_151421_eca1931a | 0.4900 [0.4787, 0.5013] | 0.6116 | 0.650 [0.608, 0.692] | 2500 | 8 |
| 10 | 20260922_160256_a4f9c534 | 0.4908 [0.4794, 0.5022] | 0.6158 | 0.660 [0.618, 0.702] | 2500 | 11 |
| 5 | 20260922_161812_8244ecad | 0.4920 [0.4805, 0.5035] | 0.6208 | 0.666 [0.624, 0.708] | 2500 | 8 |

References (same recipe): chain F (MINF from vLLM step 10) step_10 (= run A's vLLM step_10) pass@1 0.4944 [0.4832, 0.5056], step_15 0.4952, step_20 0.5152, step_25 0.5036, step_26 0.5068, step_30 0.5096 (see ../chainF-minf-fromvllm10-swe/results.md); akamehra main chain (MINF from scratch, original) step_10 pass@1 0.4976 [0.4860, 0.5092].

## Full failure-class disclosure (from the merged results.jsonl)

`graded 0.0 unverified` = scoring_details.method = solve_failed (no patch to test; counted as zero). The other columns are attempts whose agent session ended abnormally but whose repository state was still verified (method = harbor with an error annotation): context overflow = 262,144-token context exceeded after a patch existed; turn budget = adapter proxy terminated the session at 200 turns (litellm surfaces it as RateLimitError); solve timeout = 10,800 s solve cap. All are included in pass@k with their real reward.

| step | graded 0.0 unverified (classes) | context overflow, verified (n / passed) | turn budget, verified (n / passed) | solve timeout, verified (n / passed) |
|---|---|---|---|---|
| 36 | 9 (agent_no_output 1, context_window_exceeded 6, turn_budget_exhausted 2) | 16 / 3 | 0 / 0 | 13 / 0 |
| 35 | 5 (context_window_exceeded 4, turn_budget_exhausted 1) | 19 / 2 | 3 / 0 | 12 / 3 |
| 32 | 8 (context_window_exceeded 7, max_tokens_zero 1) | 8 / 2 | 1 / 0 | 8 / 1 |
| 30 | 9 (context_window_exceeded 9) | 9 / 1 | 3 / 0 | 2 / 0 |
| 25 | 4 (context_window_exceeded 4) | 3 / 2 | 0 / 0 | 5 / 0 |
| 20 | 9 (context_window_exceeded 9) | 9 / 1 | 2 / 0 | 5 / 0 |
| 15 | 8 (context_window_exceeded 8) | 9 / 2 | 0 / 0 | 1 / 0 |
| 10 | 11 (agent_nonzero_exit_-15 1, context_window_exceeded 9, turn_budget_exhausted 1) | 8 / 1 | 4 / 1 | 3 / 0 |
| 5 | 8 (context_window_exceeded 8) | 14 / 1 | 0 / 0 | 0 / 0 |

## Infrastructure incidents (no attempts lost; every merged step has 2,500 distinct problem/repeat results)

Three AWS Fargate sandbox-provisioning aborts ('ECS task not RUNNING after ~300 s' three times for one attempt -> eval loop retries exhausted -> shard run aborted with skip_failed=False; the recipe's afternotok continuation resumed each from its cached attempts as infra retry 1/3), one Slurm preemption (resumed by the recipe's continuation after it removed the known false completion marker), three transient vLLM startup failures (torch.distributed TCPStore/EADDRINUSE class, retried by the recipe) and one sandbox fault that zeroed a single attempt (step_36 p141 r4: the Fargate container's working directory vanished mid-run, so the harness saw 'Agent produced no output' and graded 0.0 without verification; disclosed as class agent_no_output, not a model failure). No admin cancellations. Eval-json `model_errors` undercounts solver failures for steps whose shard was resumed from cache (it only counts the final job's own errors); the record-level counts above are authoritative.

- step_36: shards 1 and 4 primaries (3925302, 3925305) failed at vLLM startup (torch.distributed TCPStore class, transient) and were retried by the recipe (3925324, 3925317); the shard 1 retry 3925324 was PREEMPTED at 15:14 with 229/250 verified (false completion marker removed by the PREEMPTED continuation 3925598, which completed at 17:07); shard 0 p141 r4 was graded 0.0 unverified after the Fargate container's working directory vanished mid-run (class agent_no_output: a sandbox fault, not a model failure; 1 of 2,500 attempts); the other primaries completed on their first job (8h21m-9h39m); merged by shard 0's auto-merge at 18:16
- step_35: none: all 10 primaries (3925532-3925541) completed on their first job (7h06m-10h11m); merged by shard 8's auto-merge at ~18:58; no preemptions, startup failures or Fargate aborts
- step_32: none: all 10 primaries (3924210-3924219) completed on their first job (6h36m-9h24m; shard 7's last attempt p138 r1 ran to the 3 h solve cap, hit a Fargate provisioning timeout at verify time and was retried in full by the eval loop); no preemptions, startup failures or aborts
- step_30: 1 incident(s), see incidents[]
- step_25: none: all 10 primaries completed on their first job
- step_20: 1 incident(s), see incidents[]
- step_15: 1 incident(s), see incidents[]
- step_10: none: all 10 primaries (3925942-3925951) completed on their first job; no preemptions, startup failures or Fargate aborts
- step_5: shard 5 primary 3926191 aborted at 12:19 on a Fargate provisioning timeout (p264 r4, 3 retries exhausted; 9 secondary 'Session is closed' tracebacks); continuation 3926233 resumed from 153 cached attempts and completed at 14:11; the other 9 primaries completed on their first job (5h37m-7h28m); no preemptions

Generated 2026-09-22 19:01 PDT from runs.json by write_results_g.py; per-attempt ids of every failure class are in runs.json (result.solver_failure_ids, result.terminated_but_verified_ids).

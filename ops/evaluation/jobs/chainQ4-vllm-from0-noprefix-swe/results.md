# SWE-Bench Verified: chain Q⁗ (vLLM from scratch, vLLM prefix caching OFF) HF exports

Run: `runs/nano35-swe-v2-from0-noprefix-vllm-20260926` (chain Q⁗: vLLM rollout engine from the base model akamehra/swe_e2e_corrected/base_model/step_18/hf, seed 42, with vLLM prefix caching explicitly DISABLED - policy.generation.vllm_cfg.enable_prefix_caching: false; every earlier vLLM arm (run A, chain K, chains M/Q/R/S) left that key absent, which NeMo RL resolves to ON for compute capability >= 8, so this is the first vLLM arm with prefix caching off - overlap_param_gather false (run A and chain K: true), refit_backend nccl, kv_cache_management_mode persist, save_period 5, load_replay_buffer false, 32 prompts x 16 generations (GBS 512), in_order lag 1; the arm generates all of its own rollouts from step 1, so its rungs are directly comparable to run A / chain K / chain P⁗ / chain J / chain G at matched steps; endpoint step_60 (rkirby moved all arms from step_70 to step_60 at 21:10 PDT 2026-09-26) with the stop rule enforced by the run manager; training in 8 h segments 4022856 -> 4022857 -> 4022858 -> 4022859 -> 4033276 -> 4033277, launched 08:43 PDT 2026-09-26). Checkpoints evaluated: every rung step_5..step_60 as it landed (in-place HF exports under checkpoints/step_N/hf, each verified: 14 shards, 65,827,374,264 bytes, 6,513 tensors, world-readable); rolling checkpoints (non-multiples of 5) were never evaluated. Exports: steps 5/10/15 by array 4033665 (2026-09-26); later rungs by single-task arrays recorded in runs.json export_job.
Recipe: akamehra's certified `swe-bench-verified-nano-3.5` config 20260907_223032_df6becfd (NEL toolchain 569ff7e8, Harbor + OpenHands SDK 1.17.0, 500 problems x 5 repeats, temperature 1.0 / top-p 0.95, vLLM TP2 DP2, 10 shards x 1 node x 4 GPUs on batch_long), replayed with only model path, served name, output dir, cache mounts, cluster hostname (local sbatch) and provenance tags changed. pass@1 and pass@5 with CIs are the evaluator's own numbers from the merged eval json (pass@1 = mean reward over all 2,500 attempts; 95% CI = evaluator's normal-approximation interval; bootstrap CIs also reported); pass@3 is computed from the merged results.jsonl with the unbiased estimator (per problem 1 - C(5-c,3)/C(5,3), averaged over 500 problems). Attempts graded 0.0 because the solver failed before producing a patch are included as zeros and listed separately; attempts whose agent session ended abnormally but whose patch was still verified are listed in the disclosure table and carry their real test outcome.

| step | run id | pass@1 [95% CI] | pass@3 | pass@5 [bootstrap CI] | attempts | solver failures graded 0 |
|---|---|---|---|---|---|---|
| 40 (final) | 20260927_144125_d778efca | 0.4956 [0.4842, 0.5070] | 0.6170 | 0.656 [0.614, 0.698] | 2500 | 2 |
| 35 | 20260927_122236_c379d1fa | 0.5016 [0.4906, 0.5126] | 0.6174 | 0.656 [0.614, 0.698] | 2500 | 4 |
| 30 | 20260927_092444_44c1d409 | 0.5032 [0.4918, 0.5146] | 0.6320 | 0.682 [0.640, 0.722] | 2500 | 8 |
| 25 | 20260927_063115_3cbbcac6 | 0.4956 [0.4838, 0.5074] | 0.6296 | 0.676 [0.634, 0.716] | 2500 | 10 |
| 20 | 20260927_033205_32582d4b | 0.4924 [0.4808, 0.5040] | 0.6232 | 0.672 [0.632, 0.712] | 2500 | 17 |
| 15 | 20260927_023145_ca7a5f2c | 0.4920 [0.4807, 0.5033] | 0.6172 | 0.664 [0.622, 0.704] | 2500 | 18 |
| 10 | 20260927_023132_47a7c837 | 0.4940 [0.4825, 0.5055] | 0.6220 | 0.668 [0.628, 0.708] | 2500 | 12 |
| 5 | 20260927_023118_551c174e | 0.4904 [0.4786, 0.5022] | 0.6204 | 0.662 [0.620, 0.704] | 2500 | 20 |

References (same recipe, same users/rkirby/evaluation/jobs tooling): chain P⁗ (MINF from scratch, prefix cache kept across refits) = ../chainP4-minf-from0-keepprefix-swe/results.md (pass@1 step_5 0.4932, 10 0.4840, 15 0.5020, 20 0.5080, 25 0.5112, 30 0.5072, later rungs there); chain G (MINF from scratch, no prefix cache) = ../chainG-minf-from0-nopg-noprefix-swe/results.md (pass@1 step_5 0.4920, 10 0.4908, 15 0.4900, 20 0.4788, 25 0.4800, 30 0.4816, 32 0.4772, 35 0.4752, 36 0.4640); run A (vLLM from scratch) = ../runA-vllm-from0-swe/results.md (5 0.4964, 10 0.4944, 15 0.5116, 20 0.5080, 25 0.5112, 30 0.5176, 35 0.5112); chain F (MINF from vLLM step 10) = ../chainF-minf-fromvllm10-swe/results.md (10 0.4944, 15 0.4952, 20 0.5152, 25 0.5036, 26 0.5068, 30 0.5096); chain J (MINF from scratch, prefix cache on, replica 2) = ../chainJ-minf-from0-prefix-r2-swe/results.md (5 0.4964, 10 0.4928, 15 0.4856, 20 0.4840, 25 0.4920, 30 0.4912, 35 0.4968, 40 0.5072, 45 0.4924, 50 0.5012); chain K (vLLM from scratch, seed 1234) = ../chainK-vllm-from0-seed1234-swe/results.md (5 0.4908, 10 0.5020, 15 0.4984, 20 0.5012, 25 0.5124, 28 0.5020); akamehra main chain (MINF from scratch, original) step_10 0.4976 [0.4860, 0.5092].

## Full failure-class disclosure (from the merged results.jsonl)

`graded 0.0 unverified` = scoring_details.method = solve_failed (no patch to test; counted as zero). The other columns are attempts whose agent session ended abnormally but whose repository state was still verified (method = harbor with an error annotation): context overflow = 262,144-token context exceeded after a patch existed; turn budget = adapter proxy terminated the session at 200 turns (litellm surfaces it as RateLimitError); solve timeout = 10,800 s solve cap. All are included in pass@k with their real reward.

| step | graded 0.0 unverified (classes) | context overflow, verified (n / passed) | turn budget, verified (n / passed) | solve timeout, verified (n / passed) |
|---|---|---|---|---|
| 40 | 2 (context_window_exceeded 2) | 3 / 0 | 1 / 0 | 0 / 0 |
| 35 | 4 (agent_nonzero_exit_-15 1, agent_nonzero_exit_137 1, context_window_exceeded 2) | 3 / 1 | 2 / 0 | 0 / 0 |
| 30 | 8 (context_window_exceeded 7, turn_budget_exhausted 1) | 6 / 2 | 1 / 1 | 0 / 0 |
| 25 | 10 (context_window_exceeded 10) | 12 / 3 | 4 / 1 | 2 / 0 |
| 20 | 17 (context_window_exceeded 16, turn_budget_exhausted 1) | 12 / 3 | 5 / 0 | 0 / 0 |
| 15 | 18 (context_window_exceeded 15, turn_budget_exhausted 3) | 20 / 4 | 3 / 1 | 0 / 0 |
| 10 | 12 (agent_nonzero_exit_137 2, context_window_exceeded 7, turn_budget_exhausted 3) | 22 / 2 | 3 / 0 | 0 / 0 |
| 5 | 20 (context_window_exceeded 20) | 16 / 5 | 1 / 0 | 0 / 0 |

## Infrastructure incidents (no attempts lost; every merged step has 2,500 distinct problem/repeat results)

- step_40: 24 infrastructure abort events on shards 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (24 AWS ECS DescribeTasks-throttling Fargate aborts, 0 vLLM startup deaths), each recovered from cached attempts by the afternotok continuation; 6 retry-limit stops (shards 0 twice, 5, 7 twice, 4) re-queued under rkirby's re-queue-until-done rule, resumes per shard {"0": ["4047843", "4048653"], "5": ["4048518"], "7": ["4048519", "4049129"], "4": ["4049717"]} all completed their shards; 0 attempts lost; run manager confirmed the step_40 stop 07:27
- step_35: 20 infrastructure abort events on shards 0, 1, 2, 3, 4, 5, 7, 8, 9 (18 AWS ECS DescribeTasks-throttling Fargate aborts, 2 vLLM startup deaths), each recovered from cached attempts by the afternotok continuation; 5 retry-limit stops (shards 0, 9 twice, 2, 4) re-queued under rkirby's re-queue-until-done rule, resumes per shard {"0": ["4048517"], "9": ["4048791", "4049325"], "2": ["4049439"], "4": ["4049440"]} all completed their shards; 0 attempts lost
- step_30: 6 Fargate/ECS DescribeTasks-throttling aborts on 4 shards (shard 0 primary 4041455; shard 2 primary 4041457; shard 7 primary 4041462 + continuation 4041478; shard 9 primary 4041464 + continuation 4041479), each recovered by the afternotok continuation from cached attempts (shards 7 and 9 finished on their retry-2/3 continuations 4045562 and 4045986); 0 attempts lost; no re-queues; run manager confirmed the step_40 stop 07:27
- step_25: 1 incident: shard 5 primary 4037877 (nvl72d092-T09) FAILED after 3 h 56 min with a Fargate abort ('ECS task not RUNNING' x18, aborting attempt p310 r2); 179 verified attempts intact; continuation 4037890 resumed from cache as infra retry 1/3 and finished the shard at 05:07; all other 9 primaries completed on their first job; 0 attempts lost
- step_20: none: all 10 primaries completed on their first job
- step_15: none: all 10 primaries completed on their first job
- step_10: none: all 10 primaries completed on their first job
- step_5: 1 incident: shard 3 primary 4033970 PREEMPTED by Slurm at 00:42 PDT 2026-09-27 after 5 h 10 min with 203 verified attempts; continuation 4034001 resumed from cache and finished the shard; all other 9 primaries completed on their first job; 0 attempts lost

Generated 2026-09-27 15:26 PDT from runs.json by write_results_q4.py; per-attempt ids of every failure class are in runs.json (result.solver_failure_ids, result.terminated_but_verified_ids).

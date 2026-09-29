---
name: swe-dump-runs
description: "SWE-E2E v2 dump runs (2026-09-18): clone users/rkirby/workspaces/swe_dump (branch rkirby/swe-v2-dump 33a9bf6a4, async_rl.dump instrumentation; launch_swe_dump.sh ENGINE=vllm|minf); run A vLLM from scratch ...-vllm_dump-20260918 (seg 1 done: steps 1-8, rung step_5, resume step_8), run B MINF from step_10 ...-minf_dump-from10-20260918 (seg 1 done: steps 11-17, rung step_15, resume step_17); segments 2 pending (3847610 est 22:08, 3847611 est 22:18); dump layout, sizes (~10 GB/step rollouts), monitor cron 3f7b7c84"
metadata:
  type: project
---

User approval 2026-09-18 ~12:20 PDT (after the review in [[swe-dump-inventory-and-vllm-rerun-plan]]): full decoded text
in the dumps, vLLM `processed_logprobs`, the two new runs independent of each other and of the running MINF chain
([[swe-minf-64n-run]]), names carry the date and `vllm_dump` / `minf_dump`; vLLM run starts from scratch, MINF run
continues from the chain's step_10; "if needed create a new clone" -> the live swe_minf checkout was left untouched.

**Workspace:** `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/`
- `nemo_rl`: local clone of swe_minf/nemo_rl, branch `rkirby/swe-v2-dump`, commit **f37173595** (pushed to
  git@github.com:ArEsKay3/RL.git) = 952eaf85b + "feat(sc): offline rollout and token-level dumps". Gym submodule
  354babf7e checked out (needed: launcher mounts it; Gym results now go to the run dir via results_dir).
- `Megatron-LM`: detached worktree at 880de0fce (same fork tree the MINF chain mounts); mounted only for ENGINE=minf.
  ENGINE=vllm uses the container's Megatron-LM, like the reference run 3541201.
- `launch_swe_dump.sh`: copy of launch_swe_minf.sh with ENGINE=vllm|minf, RUN_DATE, dump overrides
  (`async_rl.dump.dir=$RESULTS_DIR/dumps`, `+env.nemo_gym.results_dir=$RESULTS_DIR/gym_results`), optional
  SEED_CHECKPOINT precheck, SMOKE=1 (8+8 nodes, short QOS, 2 steps, checkpointing off).
- `tools/`: copies of minf_engine_load.py and export_checkpoint_hf_v2.sbatch.

**Code (commit f37173595):** `async_rl.dump {enabled, dir, rollout_text, token_level}` (pydantic, defaults off);
`nemo_rl/experience/rollout_dump.py`: `RolloutDumpWriter.write_group` called in RolloutManager.generate_and_push
after a successful commit (rollout_manager.py), `write_token_level_chunk` called in
SingleControllerActor._advantage_stage after advantages (single_controller.py; the stage now takes step/chunk_index
from the pump: step = version_during_step + 1 = the `train step N/` number, chunk 1-based); setup.py builds the
writer. Also forwards `step=` into compute_and_apply_seq_logprob_error_masking (fixes the NRL_LOGPROB_DUMP_DIR
filename on SC). Overlays: `swe_sc_cmh_dump_vllm.yaml` (defaults stream4 + vllm_cfg.logprobs_mode processed_logprobs
+ dump block), `swe_sc_cmh_dump_minf.yaml` (defaults swe_sc_cmh_minf.yaml + dump block).

**Dump layout** under `<run>/dumps/`:
- `rollouts/target_step_NNNNN.jsonl` (NNNNN = sampler target_step, 0-based; trains in `train step NNNNN+1`), one line
  per completion: sample_id (`<group_id>_g<i>`, joins to token_level), group_id, journal_id, target_step,
  start/end_weight_version, prompt_idx, task_name, environment, prompt_metadata (instance_id, dataset...), Gym task/
  rollout index, reward, truncated, num_assistant_turns, generation_length, total_tokens, messages (role, n_tokens,
  is_invalid_tool_call, has_malformed_thinking), rollout_metrics (numeric), env_extras_numeric, and with rollout_text:
  prompt_messages (content) + full_result (whole Gym result incl. per-turn prompt_str/generation_str, tool calls).
- `token_level/step_NNNNN_chunk_CCC.pt`: sample_ids, tags (weight_version, rollout_* tags), input_lengths (int32),
  input_ids (int32, flat trimmed), token_mask (uint8, flat), logprob_offset=1, generation_logprobs / prev_logprobs /
  reference_policy_logprobs (fp32, flat, positions 1..L-1), advantages (fp32 flat), rewards, sample_mask_before/after,
  seq_mult_prob_error, step, chunk_index, trainer_version. ~0.8 GB per full step.
- Gym per-instance SWE artifacts: `<run>/gym_results/swebench_results_<session>/`.

**Run B seed:** `runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918/checkpoints/step_10` = cp -a of
the MINF chain's step_10 (165 files, 460,825,839,205 bytes, byte-count verified 12:44 PDT; the duplicated hf/ export was
removed afterwards); latest_checkpoint_status.json written with step 10. Resume needs
`+checkpointing.load_replay_buffer=false` (journal regenerates the same in_order prompts; MINF keeps raw_logprobs + torch
sampler like the chain).

**Smokes submitted 2026-09-18 12:48 PDT (QOS short, 16 nodes, 2 steps):** vLLM **3844754**
(runs/nano35-swe-v2-vllm_dump-smoke-16n/runs/20260918-1248), MINF **3844757**
(runs/nano35-swe-v2-minf_dump-smoke-16n/runs/20260918-1248). They must show "Rollout dump enabled" in the driver log,
`[token-dump]` lines per chunk, `target_step_*.jsonl` rows with full_result text, and no `[rollout-dump] FAILED` /
`[token-dump] FAILED`. Then submit the chains (independent singletons): run A `ENGINE=vllm bash ./launch_swe_dump.sh`
x ~8 segments (0 -> 60), run B `SEED_CHECKPOINT=<minf chain>/checkpoints/step_10 ENGINE=minf bash ./launch_swe_dump.sh
+checkpointing.load_replay_buffer=false` x ~7 segments (10 -> 60), WITHOUT EXCLUDE_NODES (user dropped the exclusion 2026-09-18 14:31); the MINF chain's two pending segments are held (JobHeldUser) so the dump chains get nodes first.

2026-09-18 12:30-13:03: `chmod -R o+rX` on swe_minf/nemo_rl/3rdparty/Gym-workspace/Gym/results DONE (user asked for world
read on the existing Gym dumps): exit 0, 0 errors, full-depth check of one session and depth-4 check of all sessions found
0 entries lacking o+r / dirs lacking o+x; the path down to results/ was already world-traversable.

Smokes started 14:28 PDT (3844754 on nvl72d013, 3844757 on nvl72d184); both configs loaded (dump overlays + the
`+env.nemo_gym.results_dir` and `async_rl.dump.dir` overrides accepted), workers initializing at 14:32.
14:51 PDT: both smokes committing rollouts; `dumps/rollouts/target_step_00001.jsonl` verified (vLLM 32 rows / 204 MB,
MINF 16 rows / 160 MB): all row keys present, full_result has response.output with per-turn generation_str, instance ids,
rewards; Gym results land in `<run>/gym_results/`. prompt_messages content was empty (Gym prompt log has token ids only)
-> commit **33a9bf6a4** adds `prompt_input` (responses_create_params.input) to each row; the chains launch from it.
Token-level dumps pending the first training chunk.

**Chains submitted 2026-09-18 15:01-15:10 PDT (no EXCLUDE_NODES, code 33a9bf6a4, smokes still running at submit time):**
- Run A vLLM from scratch, EXP `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918`, 8 segments: 3847571, 3847610,
  3847633, 3847664, 3847696, 3847720, 3847768, 3847792 (run dirs runs/20260918-1502,1503,1504,1505,1506,1507,1509,1510).
- Run B MINF from step_10, EXP `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918`, 7 segments: 3847572,
  3847611, 3847639, 3847668, 3847697, 3847721, 3847793 (run dirs 1502,1503,1504,1505,1506,1507,1510).
- 3847669/3847670 were cancelled and resubmitted as 3847792/3847793 because they shared the 1505 minute stamp with the
  round-4 pair (a run-dir collision only affects Gym server text logs + provenance.txt).
- Both chains are singletons on their own names; the MINF chain's pending segments 3823398/3823419 are held (JobHeldUser)
  so these get nodes first; its segment 10 (3811946) is running until ~18:31.
Monitor both like the MINF chain (setup-hang 30-min rule, OOM / IB / hang = resubmit, code traceback = stop) plus dump
health: `dumps/rollouts/target_step_*.jsonl` growing, `dumps/token_level/step_*_chunk_*.pt` per chunk, no
`[rollout-dump] FAILED` / `[token-dump] FAILED` in the SC actor log. Stop each at its step_60 rung (A: 60 steps from 0,
B: from 10). Expected: ~8 steps per 4h segment.
Monitor cron for both dump chains: **3f7b7c84** (19,39,59 past the hour, session-only, expires 2026-09-25); the MINF chain's
cron is e839236d. Smoke status 15:11: MINF smoke step 1 done (reward 0.453, gen_kl 0.0012, 128 valid) with token dumps
step 1 chunks 1-4 + step 2 chunk 1; vLLM smoke at step 1 chunk 3; 0 dump failures, 0 tracebacks in either.

**Smokes PASSED (2026-09-18 ~15:13-15:18 PDT):** vLLM 3844754 COMPLETED 49:40, MINF 3844757 COMPLETED 44:46, 2 train steps
each (vLLM rewards 0.391/0.367, gen_kl 0.0013/0.0014; MINF 0.453 + step 2), 0 dump failures, dumps for both steps
(vLLM 5 token files 383 MB, MINF 6 files 375 MB; 256 rollout rows each). vLLM driver tracebacks = `RuntimeError: cannot
schedule new futures after shutdown` from VllmAsyncGenerationWorker at teardown, benign. The SC worker .out can be cut
at exit (MINF smoke lost its `train step 2/2` line there; the driver log has it) -> use the driver log for the last step.
**Size finding:** rollout text dumps are ~21.6 MB per rollout (full_result with tool outputs) = 5.5 GB per 256 rows ->
~11 GB per GBS-512 step, ~660 GB per 60-step run uncompressed; token-level ~1.5 GB/step (~90 GB/run). Fits the quota
(12.5 of 100 TB used) but consider offline gzip of finished target_step_*.jsonl (5-10x).
**Both chains started 2026-09-18:** run A seg 1 (3847571) at 15:23:19 on nvl72d013-T[01-13,15-17], nvl72d106-T[01-06,08-17],
nvl72d184-T[01-07,09,11-18], nvl72d186-T[01-04,06-17]; run B seg 1 (3847572) at 15:23:50 on nvl72d032-T[01-11,13-17],
nvl72d073-T[01-03,05-06,08-18], nvl72d090-T[01-10,12-15,17-18], nvl72d142-T[01-16]. Walltimes end ~19:23. Together with the
MINF chain's segment 10 that is 192 nodes running for this user at once. Run dirs runs/20260918-1502 for both first segments.
15:41 PDT setup check, both first segments healthy: run A 3847571 SC actor up (session 15:26:16), dump enabled, Gym init
167 s (cold cache for the new run), first vLLM refit 5.9 s, 16 rollout rows already; run B 3847572 SC up (session 15:26:19),
dump enabled, Gym init 18 s, 128 engine ranks ready, resumed from the copied step_10 (target_step=10, 32 spares, shortfall 0),
first nvshmem refit 38.6 s, 16 rollout rows. 0 tracebacks, 0 OOM/IB, 0 dump failures in either. Expect first train steps
~16:10-16:25 (A: steps 1-2, B: steps 11-12).
16:02 PDT (both ~39 min): no train step yet in either (first pair expected ~16:10-16:25); run A 48 rollout rows / 232 MB,
run B 48 rows / 171 MB; 0 dump failures; run B 38 MaxSequenceLengthOverflowError (HTTP 400), 37 in flight on 32 engines.
Run A's driver "Traceback" lines are vLLM's context overflow: `ValueError: Prompt length (196796) fills or exceeds
max_model_len (196608)` -> "Exception in ASGI application" (HTTP 500 instead of MINF's 400); benign, same phenomenon as
the MINF overflow. Count run A overflows with `grep -c "fills or exceeds max_model_len"`, not Traceback.
17:50 PDT (both ~2h27m): run A steps 1-4 done (rewards 0.344, 0.541, 0.256, 0.402; gen_kl 0.0013-0.0015; 511-512 valid),
step_4 resume point 17:19:02, 10 token files (3.0 GB), 2160 rollout rows in 6 files (41.9 GB, ~19 MB/row), 12 vLLM
context overflows, 17 warn-only health timeouts, 0 dump failures. Run B steps 11-14 done (rewards 0.277, 0.336, 0.270,
0.271; gen_kl 0.0015-0.0017; 510-512 valid), step_14 resume point 17:31:23, 9 token files (3.1 GB), 2112 rows (44.6 GB),
116 overflows, 23 health timeouts, 0 dump failures. Rollout dumps run ~10-11 GB per step as estimated.
18:04 PDT (both ~2h39m): run A still at 4 steps (step 5 wave: 8/32 groups dispatched), 11 token files (3.1 GB), 2208 rows
(42.5 GB), 12 overflows, 0 dump failures; run B still at 14 steps (step 15 wave generating, 40 in flight), 9 token files,
2112 rows (44.6 GB), 120 overflows, 0 dump failures. First segments end ~19:23; their successors then queue on Priority.
18:25 PDT (both ~3h00m): run A steps 5 (reward 0.254, gen_kl 0.0015, 511) and 6 (0.371, 0.0014, 512) done; step_5 = first
permanent rung, step_6 resume point saved 18:19:44; 15 token files (4.6 GB), 3088 rows (65 GB), 12 overflows, 30 warn-only
health timeouts, 0 dump failures. Run B still at step 14 (step 15 wave 8/32 dispatched), 10 token files (3.2 GB), 2464 rows
(49 GB), 120 overflows, 32 health timeouts, 0 dump failures. First segments end ~19:23.
18:45 PDT (both ~3h20m): run A unchanged at 6 steps (step 7 wave generating), 15 vLLM overflows, 3120 rows (65 GB), 15 token
files. Run B steps 15 (reward 0.229, gen_kl 0.0016, 512) and 16 (0.316, 0.0017, 512) done; step_15 = first new permanent rung
of run B, step_16 resume point saved 18:34:59; 15 token files (4.9 GB), 3216 rows (69 GB), 123 overflows, 32 health timeouts,
0 dump failures; step 17 wave 8/32 dispatched with 141 in flight. First segments end ~19:23.
19:05 PDT (both ~3h40m, walltime ~19:23): run A still 6 steps (step 7 chunk 1 dispatched), 17 token files (4.9 GB), 3600 rows
(72 GB), 15 vLLM overflows, 34 health timeouts, 0 dump failures; run B still 16 steps (step 17 chunk 1 dispatched, 33 in
flight), 15 token files (4.9 GB), 3248 rows (70 GB), 155 overflows, 0 dump failures. Next: confirm TIMEOUT for 3847571 /
3847572 after 19:23, verify step_6 / step_16 (or later) resume points, then report start estimates for 3847610 / 3847611.

**Segment 1 of both dump chains DONE 2026-09-18 ~19:23 PDT, both TIMEOUT at 4h, 0 incidents, 0 dump failures.**
- Run A (3847571, TIMEOUT 04:00:16): 8 steps (1-8), rewards 0.344, 0.541, 0.256, 0.402, 0.254, 0.371, 0.432, 0.289;
  gen_kl 0.0013-0.0015; 511-512 valid. step_5 rung + step_8 resume point (saved 19:10:56, complete). Dumps: 10 rollout
  jsonl files (86 GB), 20 token files (6.1 GB). 15 vLLM context overflows (= the 15 driver "Traceback" lines), 35 health.
- Run B (3847572, TIMEOUT 04:00:15): 7 steps (11-17), rewards 0.277, 0.336, 0.270, 0.271, 0.229, 0.316, 0.234;
  gen_kl 0.0015-0.0017; 510-512 valid. step_15 rung + step_17 resume point (saved 19:22:21, complete). Dumps: 8 rollout
  files (88 GB), 19 token files (6.3 GB). 155+ MINF overflows, 39 health, 0 tracebacks.
- Successors pending on Priority: 3847610 (A seg 2) est. 22:08, 3847611 (B seg 2) est. 22:18 (as of 19:27).
Monitoring note: never `cat dumps/rollouts/*.jsonl | wc -l` (70+ GB, times out); use `ls | wc -l` + `du -sm`; rows ~= 512
per completed target step.


**STOP TARGET CHANGED (user, 2026-09-20):** stop each dump chain at checkpoints/step_35 (not 60); capacity rule threshold 27. Cron 3f7b7c84 was recreated with the new rule. Segment 2 (3847610/3847611) ran 2026-09-19 from ~13:10; segment 3 (3847633/3847639) running 2026-09-20; run A at step_20, run B at step_26 as of 2026-09-20 midday.


**2026-09-19 evening:** segment 3 of both chains (3847633 steps 20-23, 3847639 steps 25-28) FAILED at 20:59/21:08 because the live checkout was deleted at 20:37 (see [[swe-dump-checkout-deletion-incident]]); tree restored; 3847664/3847668 pending on Priority.


**ALL RUNS HELD (user, 2026-09-19 ~21:30 PDT):** every queued SWE segment is on JobHeldUser (main chain 3823398/3823419; dump run A 3847664 3847696 3847720 3847768 3847792; dump run B 3847668 3847697 3847721 3847793). No segment is running. Do not release, resubmit or cancel anything until the user says so; crons only report state.

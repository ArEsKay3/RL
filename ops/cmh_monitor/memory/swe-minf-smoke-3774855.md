---
name: swe-minf-smoke-3774855
description: "First SWE-E2E v2 + MINF smoke on the akamehra/swe-v2 base: job 3774855 (nano35-swe-v2-minf-smoke-16n, 8 train + 8 gen nodes, QOS short 2h, 2 steps), submitted 2026-09-15 20:20 from users/rkirby/workspaces/swe_minf/launch_swe_minf.sh SMOKE=1; where its logs are and what it must prove"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7aa0a354-aa1b-4011-8140-8d3122eb7495
  modified: 2026-09-16T04:47:50.780Z
---

Submitted 2026-09-15 20:20 local (CMH) as Slurm job **3774855**, EXP_NAME `nano35-swe-v2-minf-smoke-16n`, from
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_minf/launch_swe_minf.sh` with
`SMOKE=1` (nemo_rl `rkirby/swe-v2-minf` 2bc89d09b, fork Megatron-LM 880de0fce mounted, container rl-gym.63635108-zstd).
Shape: 8 train + 8 gen nodes (TP4 CP4 EP8 training; TP4 EP4 MINF engines x8), 8 prompts x 16, GBS 128,
min_groups_for_streaming_train 2, max_buffered 24, max_inflight 16, checkpointing off, grpo.max_num_steps=2,
QOS short, walltime 2h, W&B ultra-v3-swe-e2e-convergence / nano35-swe-v2-minf-smoke-16n.

Paths (all under `$U/rkirby/runs/nano35-swe-v2-minf-smoke-16n/`, U=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users):
- run dir `runs/20260915-2020/` (provenance.txt, `slurm/3774855.{out,err}`), `logs/` (driver + nemo_gym/)
- Ray logs `ray_logs/3774855-logs/` (ray-driver.log is the main one)

What it must prove, in order: (1) fingerprint mismatch bypassed (NRL_IGNORE_VERSION_MISMATCH=1) and megatron-core
imports from the MOUNTED tree (fork symbols: `_run_post_refit_hooks`, prefix-cache coordinator); (2) MegatronGeneration
builds on the 8 gen nodes at max_model_len 196608 with buffer_size_gb 70 (no OOM at KV/cuda-graph allocation);
(3) Gym policy_model + swe_agents_train/val come up against the 8 MINF URLs (setup.py `_spinup_gym_behind_megatron_generation`);
(4) SingleController.run() first `_sync_weights` = nvshmem refit (NVSHMEM_MAX_CTAS=2) trainer->engine succeeds;
(5) OpenHands rollouts flow (tool parser qwen3-coder-tool, reasoning parser nemotron-v3-reasoning, token ids) and a
train pump starts after 2 groups; (6) step 1 completes (step 2 unlikely within 2h given SWE rollout lengths).

Env knobs that reach the workers via sbatch env inheritance (not the driver prefix): NVSHMEM_MAX_CTAS=2,
NRL_MINF_SAMPLING_BACKEND=torch, NRL_MINF_LOGPROBS_MODE=raw_logprobs, NRL_IGNORE_VERSION_MISMATCH=1.

**Attempt 1 (3774855) FAILED at 5 min (20:26), before any refit/rollout.** Startup itself was fine: fingerprint
bypass worked, megatron-core imported from the MOUNTED tree, config loaded, 32+32 workers up, both groups built
their models. Then `lm_policy-0-0` and `megatron_generation-0-0` (rank 0 of the two parallel worker groups) BOTH
converted the HF checkpoint and saved into the same cache dir `$HF_HOME/nemo_rl/model__scratch_..._step_18_hf/`
at once -> mixed-layout torch_dist checkpoint, no common.pt -> all 64 ranks died in `load_common_state_dict` with
`_pickle.UnpicklingError: invalid load key, '\x02'`. The v1 driver never hits this (policy before engine). Fix =
port of upstream #3864's conversion-cache half (staged save + atomic rename + `megatron_conversion_is_complete`)
committed as 952eaf85b on rkirby/swe-v2-minf (pushed); the corrupt cache dir was deleted. **Attempt 2 = job 3775390**,
submitted 2026-09-15 20:48, same EXP_NAME, run dir `runs/20260915-2048/`, Ray logs `ray_logs/3775390-logs/`.

**Attempt 2 (3775390) PASSED: COMPLETED exit 0 in 54:18 (20:48-21:42), both train steps done.** Evidence in
ray-driver.log: both groups saved into `.model__*.staging-*` dirs, one publish + "Completed conversion already
published ... discarding the staged copy" (line 549); 32 gen ranks "Initialized persistent inference engine" +
hypercorn servers; Gym policy_model/swe_agents_train/swe_agents_val up; setup 1428 s total (generation init 237 s,
policy init 122 s, NeMo-Gym init 1187 s = the long pole, SWE agent venv/r2egym install on a cold cache); first
nvshmem refit 4.2 s, second 2.9 s ("[adam-fix] finalize_async_save before refit optimizer offload" printed);
step 1: 1396 s total (exposed_generation 1078 s, training 196 s, logprobs 107 s), reward 0.4375, loss 0.0098,
gen_kl_error 0.0012, 1 sequence masked for mult_prob_error>2; step 2: 129 s (overlapped rollouts), loss 0.0032,
gen_kl_error 0.0015. 16 prompt groups completed at 20-22 min each; 4 requests hit MaxSequenceLengthOverflowError
(engine returns HTTP 400 "reduce the length of the messages", same shape as vLLM); ~3000 benign
"Clamping num_tokens_to_generate" warnings because Gym requests the full 196608 budget. W&B
nvidia/ultra-v3-swe-e2e-convergence run v01kzs9j. The "Stack (most recent call first)" dumps at the end are Ray
worker teardown, not crashes.

Cluster at submit time: 0 idle nodes in batch, short QOS had 27 pending / 7 running. Full-scale relaunch
(`bash launch_swe_minf.sh` = runbook shape, 64 nodes, 4h normal QOS) needs the user's explicit go-ahead.
Related: [[swe-e2e-base-vs-minf-fork]], [[cmh-cpus-per-worker-140]], [[gym-venvs-not-prebaked-in-nightly]].

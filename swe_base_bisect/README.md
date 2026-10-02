# NeMo RL base bisection for the SWE length-growth question (HSG, 2026-10)

## Question

Some MINF and vLLM seeds on the newer NeMo RL bases show growing sequence length;
the 08-20 base (b59165262) with vLLM did not. Three seeds per base are needed to
see it. This directory documents how the bases were built so the runs can be
reproduced and compared.

## Bases

| base  | NeMo RL commit | Megatron-Bridge / Megatron-LM pin | Gym pin | vLLM | container (users/rkirby/containers/) |
|-------|----------------|-----------------------------------|---------|------|--------------------------------------|
| 08-20 | b59165262 (reference only; the v2 arms ran the private fork 880de0fce = upstream MLM 14346b65 + 29 patches) | 8c46dc42 / 14346b65 | c3bac963 | 0.25.1 | CMH image nemo-rl:fd45cb8-67127697-gym |
| 09-06 | c0165c8c upstream, "ci: Bump Megatron-Bridge to 5ed9799" | 5ed97996c / 1e7598cbf | fd5e84d6b | 0.25.1 | nemo-rl_fd45cb8-67127697-gym.sqsh |
| 09-21 | de764eae saumishr/RL (= upstream 880a37a39 + 23 fork commits) | 5ed97996c / 1e7598cbf | 0f4710ecf | 0.26.0 in pyproject, 0.25.1 in the image | rl-gym.b7a4d95-68736277-gym.sqsh (what the 09-21 team used) |
| 09-24 | 4aaa48fab upstream ("main915" workspace; the name predates the 09-24 re-pin) | 1f8873bb0 / 6a3660905 | 267305e2a | 0.26.0 | nemo-rl_4aaa48f-69764523-gym.sqsh (default); fd45cb8 for chain U/W continuity |

Pin bumps between 08-20 and 09-21, all in NeMo RL upstream: Bridge a43d71b3 (08-26,
MLM 731b7914), a0edb276 (09-01, MLM 7c9c3a02, first MLM with the M2N refit backend
dc931ad2f), c0165c8c (09-06, MLM 1e7598cb). NeMo RL side of the MINF M2N refit:
5d49fbf4 (09-11). vLLM 0.25.1 -> 0.26.0: 3491eed5 (09-20).

## Branch scheme (all on the ArEsKay3 forks)

`rkirby/swe-stack/<date>-<tag>` (RL), `mlm-swe-stack/<date>-<tag>`,
`bridge-swe-stack/<date>-<tag>`, `gym-swe-stack/<date>-<tag>`; tags
`20260906-upstream`, `20260921-saumishr`. The 09-24 base keeps its original names
(`rkirby/swe-main915-latest`, `mlm-main915-latest`, `mlm-bridge-main915-latest`,
`rkirby/gym-main915`).

Each RL branch = base commit + the same series of recipe/dump/MINF-fix commits
(ported from the 09-24 branch; see `swe_main915_workspace/HANDOFF.md` for what the
series contains) + the base-specific adaptations listed below.

Each MLM branch = the base's own pin + the 5 patches in
`swe_main915_workspace/patches/megatron-lm-main915-latest/` (multi-EOS, zero-token
prefill guard, prompt_token_ids fallback, untrack artifact, hybrid prefix-skip fix).
Each Gym branch = the base's own pin + the 3 `openai_utils` fixes
(Bedrock auth kwargs, tool-message `name`, `compact_prompt_token_ids`).

Workspaces on Lustre: `users/rkirby/workspaces/swe_nrl_20260906_upstream`,
`swe_nrl_20260921_saumishr`, `swe_main915` (09-24). Each holds `nemo_rl/` (with
`3rdparty/Gym-workspace/Gym/` cloned explicitly), `Megatron-LM/`,
`Megatron-Bridge/`, and a launcher.

## Base-specific adaptations

09-21: straight rebase; one manual merge in `single_controller_utils/config.py`;
the nemo.lens shim commit was dropped (every image used has nemo.lens).

09-06 (70 upstream commits older than 09-21):
- Gym actor API is `spinup_nemo_gym_actor`; the data-plane restore block and the
  inference_optimized ETP hunk do not exist yet and were dropped.
- `megatron_worker.py` keeps the 09-06 `InferenceConfig` structure; only the
  env-selectable sampling backend (`NRL_MINF_SAMPLING_BACKEND`) was carried.
- Gym pin needs upstream 7bfccecc1 (normalize `defer_loading` None -> False);
  without it every swe_agents request fails pydantic validation with the openai
  client in the fd45cb8 image.
- NeMo RL needs upstream 9f962dce ("Handle MTP position_ids correctly", 09-13);
  without it the first training forward asserts in `hybrid_model.py:594`.
- MINF refit: the base predates the M2N `nccl_reshard` Megatron refit, so the
  recipe sets `refit_transport: null` and `refit_backend`. The `nccl` copy
  service hangs on its first coalesced P2P batch on 64 ranks even with MLM
  79be654e3 + 325c9936b cherry-picked (jobs 7606633, 7607484); `nvshmem`
  completes the sync and is what the recipe uses. 08-20 also ran nvshmem.
  `mlm-swe-stack/20260906-upstream` therefore carries those two extra MLM picks.

All bases: `ray.sub` writes the node IPs it already resolved into a job-local
hosts file and binds it over `/etc/hosts` in the sandbox container. The head-node
sandbox generates one nginx upstream per worker port (2304 at 16 nodes) and nginx
resolves them all at startup; one DNS miss aborted nginx and ray.sub tore the job
down about 110 s in (6 of the first 9 HSG smokes). The pyxis "failed to create
container filesystem" message seen in those jobs is the teardown symptom.

## Masking and penalties

Same posture on every base, matching the 08-20 behaviour:
`should_mask_flagged_samples: false`, `penalize_invalid_tool_call` /
`penalize_malformed_thinking` false, `*_advantage` null, `reward_penalties` flags
false. The sequence-level logprob-error mask (mult_prob_error > 2.0) stays on, as it
was at 08-20. Every launch carries `checkpointing.load_replay_buffer=false`.

## Launching

```
cd /lustre/fsw/portfolios/llmservice/users/rkirby/workspaces/<workspace>
SMOKE=1 ENGINE=vllm bash launch_swe_nrl0906.sh        # 8+8 nodes, 2 steps, QOS short
SEED=1234 ENGINE=vllm bash launch_swe_nrl0921.sh      # 32+32 nodes, 4 h, QOS normal
ENGINE=minf bash launch_swe_main915_hsg.sh             # 09-24 base
```

Run names: `nano35-swe-nrl0906upstream-64n-<engine>-hsg-seed<seed>-<date>`,
`nano35-swe-nrl0921saumishr-...`, `nano35-swe-main915-64n-<engine>-hsg-<date>`.
W&B: entity `nvidia`, project `ultra-v3-swe-e2e-convergence`, run name = job name.
Run dirs: `/lustre/fsw/portfolios/llmservice/users/rkirby/runs/<run name>/`
(`checkpoints/`, `dumps/rollouts/target_step_*.jsonl`, `gym_results/`,
`ray_logs/<jobid>-logs/`, `runs/<timestamp>/slurm/<jobid>.out`).
`CONTAINER=...` and `EXP_NAME=...` override the launcher defaults; `EXCLUDE_NODES`
is honoured. Containers were imported from nvcr.io/nvidian/nemo-rl with
`users/rkirby/containers/import_rl_gym_*.sbatch`.

## Smoke results (16 nodes, 2 steps)

| base | engine | job | reward step 1 / 2 | valid | notes |
|------|--------|-----|-------------------|-------|-------|
| 09-06 | vLLM | 7609888 | 0.51 / 0.40 | 128/128, 0 masked | after Gym + MTP fixes |
| 09-06 | MINF nvshmem | 7609890 | 0.49 / 0.42 | 128/128, 0 masked | nccl attempts 7606633, 7607484 hung |
| 09-21 | vLLM | 7606202 | 0.49 / 0.38 | 128/128, 0 masked | |
| 09-21 | MINF | 7606203 | 0.44 / 0.38 | 127/128 | 1 logprob-error mask |
| 09-24 | vLLM | 7606205 (fd45cb8), 7607461 (4aaa48f) | 0.43 / 0.40, 0.41 / 0.39 | 128/128 | |
| 09-24 | MINF | 7600779 (rl-gym.69725534), 7607464 (4aaa48f) | 0.45 / 0.41, 0.47 / 0.38 | 128/128 | |

Retired image: the CI sqsh `rl-gym.69725534.sqsh`; its vLLM `Worker` already
defines `synchronize_device`, which collides with NeMo RL's worker extension
(job 7605212), and the 09-21 code fails at import on it (`fsdp` extra undeclared).

## Caveats for reading the seeds

- The 09-21 image carries vLLM 0.25.1 although the 09-21 pyproject pins 0.26.0;
  the 09-21 team ran that way, so this mirrors them. The 09-24 image has 0.26.0.
- 08-20 (nvshmem), 09-06 (nvshmem copy service) and 09-21/09-24 (M2N nccl_reshard)
  are three different MINF refit paths. vLLM arms share the reshard refit across
  all bases.
- Smoke run directories accumulate rollout dumps across attempts; per-step counts
  above 128 are from earlier failed attempts of the same smoke.

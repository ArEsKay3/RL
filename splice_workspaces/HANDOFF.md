# HANDOFF — SWE-E2E v2 splice / replay family (chains P … AC)

Written 2026-09-28 12:3x PDT by the "Cross Train Experiment" session for rkirby's possible move to HSG.
Everything below is code + docs; checkpoints, dumps and gym_results stay on CMH under
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/` (RUNS) and the workspaces under
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/` (W).

## 1. Repositories and branches (all on rkirby's forks)

| tree on CMH | branch @ commit | GitHub | pushed |
|---|---|---|---|
| `W/swe_replay_splice/nemo_rl` (LIVE, bind-mounted by pending jobs) | local `rkirby/swe-v2-dump` @ `4f779abc` = GitHub `rkirby/swe-v2-dump` 7f8a2b9d + the splice commit | `git@github.com:ArEsKay3/RL.git` **`rkirby/swe-v2-splice`** (this branch; also carries `splice_workspaces/`) | yes (2026-09-28) |
| `W/swe_replay_splice/Megatron-LM` (fork MLM for MINF arms) | `880de0fce` + ONE uncommitted working-tree change in `megatron/core/inference/engines/dynamic_engine.py` that is byte-identical to GitHub branch `rkirby/mlm-880de0fce-dynengine-logprob-guard` @ `91cb08ea` ("Guard prefill logprob append when a request has no generated tokens") | `git@github.com:ArEsKay3/Megatron-LM.git` `rkirby/mlm-880de0fce-dynengine-logprob-guard` | yes (already) |
| `W/swe_replay_splice/nemo_rl/3rdparty/Gym-workspace/Gym` (submodule) | `354babf7` = upstream `https://github.com/NVIDIA-NeMo/Gym.git` ("feat(token-id-capture) #2124") | upstream | n/a |
| `W/swe_915/nemo_rl` (not this session's; pushed for completeness) | `rkirby/swe-915-nomask` @ `3a7fde92c` | `ArEsKay3/RL` `rkirby/swe-915-nomask` | yes (2026-09-28) |
| `W/swe_915/Megatron-LM` (not this session's) | `fork-915` @ `a012970be` (880de0fce + #7256 guard + EOS fix) | `ArEsKay3/Megatron-LM` `rkirby/fork-915` | yes (2026-09-28) |
| overlays / launchers / masks / renders / tests / READMEs (were outside git) | — | this branch, `splice_workspaces/<workspace>/` | yes |

Never force-push. Never checkout/stash/reset inside `W/swe_replay_splice`, `W/swe_mask_replay`, `W/swe_range_replay`,
`W/swe_prefix_keep`, `W/swe_lr0` while any job of this family is pending or running: they are bind-mounted read-write
(the trees were wiped twice by an unknown actor, 2026-09-19 and 2026-09-24 22:24; `git checkout --` restored them).

## 2. The stack every arm ran on

* NeMo RL: `rkirby/swe-v2-dump` lineage (7f8a2b9d = old SWE-E2E v2 base + offline rollout/token-level dumps) + splice commit
  4f779abc (`nemo_rl/experience/foreign_rollout_source.py`, wiring in `nemo_rl/algorithms/single_controller.py`).
* Recipes: `examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_dump_vllm.yaml` (vLLM) / `swe_sc_cmh_dump_minf.yaml` (MINF),
  defaults chain `… -> swe_sc_cmh_{vllm,minf} -> swe_sc_cmh_stream4 -> swe_sc_cmh_common -> ../../configs/grpo_math_1B.yaml`.
  seed 42, 32 prompts x 16 completions = 512 rows/step, in_order sampler, max_num_steps 1156 (stops only by scancel).
* Container `…/users/akamehra/containers/rl-gym.63635108-zstd.sqsh`; sandbox `nemo-skills-sandbox-no-sync.sqsh`;
  base model `…/users/akamehra/swe_e2e_corrected/base_model/step_18/hf`; data
  `…/swe_e2e_corrected/data/large_root_cause_curriculum_with_mercor_ots_plus_singlefile_swerebench_overlap_fix.jsonl`,
  val `swe_public_datasets_val_swebench.jsonl`; SIF dir `…/swe_e2e_corrected/sif`.
* 64 nodes (GB200 NVL72, 4 GPUs each): 32 train (TP4 CP4 EP32) + 32 generation; CPUS_PER_WORKER=140; 8 h segments;
  hero-res QOS, batch_long partition, reservation sla_res_nemotron_sw_post; account nemotron_sw_post.
* Launcher chain: `W/swe_replay_splice/launch_swe_splice.sh` (copy in `splice_workspaces/swe_replay_splice/`) ->
  `examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh swe …` -> `ray.sub`. Env knobs: ENGINE=vllm|minf, EXP_NAME,
  RUN_DATE, WALLTIME, SLURM_QOS, SLURM_PARTITION, SLURM_RESERVATION, SLURM_DEPENDENCY (merged with singleton), DRY_RUN=1
  (renders TRAIN_CMD, never runs Hydra), SEED_CHECKPOINT (launcher refuses unless the copy is in the run's checkpoints/),
  EXCLUDE_NODES, WANDB_API_KEY (required), W&B project ultra-v3-swe-e2e-convergence. Trailing args = Hydra overrides.
* Overlays are extra bind mounts over `/opt/nemo-rl/...` (and over the MLM mount target for Megatron files), never edits
  of the live tree: see each `splice_workspaces/<ws>/launch_*.sh` EXTRA_MOUNTS line.

## 3. Workspaces (now in `splice_workspaces/`)

| dir | what | validated by |
|---|---|---|
| `swe_replay_splice/launch_swe_splice.sh` | plain splice launcher (chains P, Q, R, R′, Q⁗, Q⁗2, AA, AB, AC) | production |
| `swe_lr0/` | `patches/setup.py` = splice setup.py + `load_optim` forwarded into Bridge CheckpointConfig; `launch_swe_lr0.sh` mounts it (chains P′/Q′) | jobs 3994875/3995012 |
| `swe_prefix_keep/` | 4-file overlay adding `mcore_generation_config.invalidate_prefix_cache_on_weight_update` (Megatron `inference/config.py`, `engines/dynamic_engine.py`; NeMo RL `megatron_worker.py`, `megatron/config.py`), `patches.diff`, `launch_swe_prefix_keep.sh`, tests, renders/launch logs (chains P‴, P⁗, P⁗2) | cpu job 4009386 PASS |
| `swe_mask_replay/` | 5-file overlay: `+foreign_rollout.mask_file=<csv>` (sample_mask 0 for listed replayed sample_ids, rows kept) + the manager's `async_rl.dump.token_ids=off|digest|full` patch; `masks/*.csv`; `launch_swe_mask_replay.sh`; tests; renders; launch logs (chains X, Y, Z) | cpu job 4053246 PASS |
| `swe_range_replay/` | no code: renders, TRAIN_CMD token files, `test_overrides_range.py` + `test_range_replay.py` + sbatch, README (chains AA, AB, AC) | cpu jobs 4057445/4057715/4058117/4058559 |
| `token_ids_dump/` | the manager's patch + README that the mask overlay embeds | — |
| `run_readmes/` | per-arm provenance files copied from the run dirs | — |

## 4. Arms

Source runs for replay: chain G = `RUNS/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920`
(MINF from scratch, the "bad" lineage), chain M = `RUNS/nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923`
(vLLM from scratch, clean). A replay arm needs the source run's `dumps/rollouts/target_step_*.jsonl` (index) AND
`dumps/token_level/step_*_chunk_*.pt` (tensors) for the replayed steps.

| arm | purpose | run dir (under RUNS) | jobs | final state | key result |
|---|---|---|---|---|---|
| chain P | MINF replaying chain M steps 1-20, live MINF after | `nano35-swe-v2-splice-minf-runMdata-to20-20260924` | 3985905 | TIMEOUT 8 h, 31 closed, rungs 5..30,31 | replayed rows bit-identical to M; step_5/20 within 1 bf16 ULP of M |
| chain Q | vLLM replaying M 1-20, live vLLM after (control) | `…-splice-vllm-runMdata-to20-20260924` | 3987245 | 33 closed, rungs 5..30,33 | P=Q=M replay identity; only prev_logprobs differ |
| chain R | vLLM replaying G 1-10, live after | `…-splice-vllm-runGdata-to10-20260924` | 3989910 (died, r2egym cache), **4002932** | cancelled 09-25 14:19 at 26 closed, rungs 5/10/15/20/25 + 26 | LOOPS: loop share 4.18 % (11-15), 6.48 % (16-20) vs ≤1 % on clean arms |
| chain R′ | vLLM replaying G 1-16 | `…-splice-vllm-runGdata-to16-20260924` | 3990303 (died), 4002933 | cancelled 09-25 10:14 at 21 closed, rungs 5/10/15/20 + 21 | as R |
| chain P′ | MINF from P step_20, lr 0 (frozen weights) | `nano35-swe-v2-fromP20-lr0-minf-20260924` | 3994875 (earlier 3993703/3993952/3994235/3994408 failed on the traps in §6) | 15 live steps 21-35, rungs 20/25/30/35 | weights bit-identical to P20 except router expert_bias (+0.001/step) |
| chain Q′ | vLLM twin of P′ | `nano35-swe-v2-fromP20-lr0-vllm-20260924` | 3995012 | rungs 20/25/30/35 | engine-neutral at fixed weights |
| chain P″ | MINF from P20, normal lr, refit_backend=nccl | `nano35-swe-v2-fromP20-nccl-minf-20260925` | 4004676 | FAILED at first refit (NCCL watchdog on the refit group) | nccl refit backend unusable on this lineage |
| chain P‴ | MINF from P20, nvshmem, prefix cache KEPT across refits | `nano35-swe-v2-fromP20-nvshmem-keepprefix-minf-20260925` | 4009432, 4016063 (nvshmem init fail), 4016064, 4016982 (quota) | 30 closed (21-50), rungs 20/25/30/35/40/45/50 | knob has no measurable effect on engine-vs-trainer logprobs |
| chain P⁗ | MINF from scratch, prefix cache kept | `nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925` | 4013305, 4013423, 4022656, 4022657, 4022658 (+4022659-61 cancelled) | FINAL step_60 (00:28 09-27), rungs 5..60 | pass@1 0.49-0.53, above G/J from step 15 |
| chain Q⁗ | vLLM from scratch, vLLM prefix caching OFF | `nano35-swe-v2-from0-noprefix-vllm-20260926` | 4022856, 4022857, 4022858 (+4022859/4033276/4033277 cancelled) | FINAL step_40 (07:27 09-27), rungs 5..40 | pass@1 flat 0.49-0.50 |
| chain P⁗2 | P⁗ replica, grpo.seed=1234 | `nano35-swe-v2-from0-nvshmem-keepprefix-minf-seed1234-20260926` | 4033269, 4033270 (startup hang), 4033271; **4033272/4033273 HELD** | ON HOLD at 22 closed, rungs 5/10/15/20 + 22 | pass@1 5-20 within seed noise of P⁗ |
| chain Q⁗2 | Q⁗ replica, seed 1234 | `nano35-swe-v2-from0-noprefix-vllm-seed1234-20260927` | 4045355, 4045356; **4045357/4045358/4045363 HELD** | ON HOLD at 20 closed, rungs 5/10/15/20 | pass@1 level with run A at 5-20 |
| chain X | masked replay TEST: G 1-10 with 127 rewarded-deep rows sample_mask 0, live to 30 | `nano35-swe-v2-splice-vllm-runGdata-to10-maskX-rewarded-deep-20260927` | 4055486, 4055489 | FINAL step_30 (04:38 09-28), rungs 5..30 | does NOT loop: 2.86 % (11-15), 2.53 % (16-20) vs R 4.18/6.48 % |
| chain Y | masked replay CONTROL: 127 random rewarded rows | `…-maskY-random-rewarded-20260927` | **4055490/4055491 HELD** (never ran) | target step_30 | — |
| chain Z | masked replay MIRROR on chain M: 213 punished-deep rows | `nano35-swe-v2-splice-vllm-runMdata-to10-maskZ-punished-deep-20260927` | **4055492/4055493 HELD** (never ran) | target step_30 | — |
| chain AA | step-range ablation: G steps 8-10 only replayed, live 1-7 and 11-25 | `nano35-swe-v2-splice-vllm-runGdata-8to10-20260928` | 4061467 (cancelled 08:46 09-28 on rkirby's "put AA on hold"), **4061468 HELD** | ON HOLD at 10 closed, rungs 5/10; live 11+ not run | replayed 8-10 identical to G |
| chain AB | G steps 1-7 only replayed, live 8-25 | `nano35-swe-v2-splice-vllm-runGdata-1to7-20260928` | 4061469 (afterany 4061467+4061468), 4061470 | gated, never ran | — |
| chain AC | control: M steps 8-10 replayed | `nano35-swe-v2-splice-vllm-runMdata-8to10-20260928` (empty scaffold) | not submitted | rendered only | — |

Order rkirby last set (2026-09-28 ~00:55): X → AA → AB → Y → Z; AA/AB target step_25 (never confirmed), X/Y/Z step_30.
Stop = the run manager's trigger: when `checkpoints/step_N` is complete (140 files, `policy/weights/iter_0000000`,
`config.yaml`, `training_info.json`, status json; no `tmp_*`), scancel BOTH job ids of the arm by name.
Evals (SWE-Bench Verified pass@1) are the "SWE Verified Eval Runner" session's: job dirs under
`…/users/rkirby/evaluation/jobs/chain*-swe/results.md`.

## 5. Exact launch / resume commands (CMH form)

Two identical submissions per arm = two 8 h segments; the second has `SLURM_DEPENDENCY=afterany:<seg1>` and shares the
name (singleton). A segment that starts with rungs present resumes from the latest rung automatically.

Plain splice (P/Q/R/Q⁗/Q⁗2/AA/AB/AC):
```
cd W/swe_replay_splice && ENGINE=vllm RUN_DATE=YYYYMMDD WALLTIME=8:00:00 SLURM_QOS=hero-res SLURM_PARTITION=batch_long \
  SLURM_RESERVATION=sla_res_nemotron_sw_post EXP_NAME=<run name> [SLURM_DEPENDENCY=afterany:<id>[:<id>]] \
  bash launch_swe_splice.sh +foreign_rollout.source_run=<RUNS/source> +foreign_rollout.steps=<a:b> \
  +checkpointing.load_replay_buffer=false policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false \
  [grpo.seed=1234] [+policy.generation.vllm_cfg.enable_prefix_caching=false]      # Q⁗/Q⁗2 only; no foreign_* on them
```
Masked replay (X/Y/Z): same env, `bash W/swe_mask_replay/launch_swe_mask_replay.sh` plus
`+foreign_rollout.mask_file=W/swe_mask_replay/masks/<csv> +async_rl.dump.token_ids=digest` (exact lines:
`splice_workspaces/swe_mask_replay/README.md`, `train_cmd_chainX.txt`, `logs/launch_chain*_seg*_*.out`).
Step-range ablation (AA/AB/AC): plain launcher with `+foreign_rollout.steps=8:10` or `1:7`
(`splice_workspaces/swe_range_replay/README.md`, `train_cmd_chain{AA,AB,AC}.txt`).
Prefix-keep MINF (P‴/P⁗/P⁗2): `ENGINE=minf … bash W/swe_prefix_keep/launch_swe_prefix_keep.sh
+policy.generation.mcore_generation_config.invalidate_prefix_cache_on_weight_update=false
policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false +checkpointing.load_replay_buffer=false
[SEED_CHECKPOINT=<run>/checkpoints/step_20 for the P20-seeded P‴] [grpo.seed=1234]` (`train_cmd_pk.txt`, `logs/`).
lr-0 frozen-weight resumes (P′/Q′): `SEED_CHECKPOINT=<own copy of P step_20> ENGINE=minf|vllm …
bash W/swe_lr0/launch_swe_lr0.sh policy.megatron_cfg.optimizer.lr=0.0 policy.megatron_cfg.optimizer.min_lr=0.0
policy.megatron_cfg.scheduler.lr_warmup_init=0.0 +policy.megatron_cfg.checkpoint.load_optim=false
+checkpointing.load_replay_buffer=false policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false`;
to freeze fully also set `policy.megatron_cfg.moe_router_bias_update_rate=0.0`.
Resume a held arm: `scontrol release <seg id>` (AA: 4061468; Y: 4055490 then 4055491 follows; Z: 4055492/93;
P⁗2: 4033272; Q⁗2: 4045357). Validate any new override set BEFORE submitting with the workspace's `test_*.sbatch`
(cpu partition, driver venv, replays the entrypoint's Hydra path); `DRY_RUN=1` alone never runs Hydra.
Hold trick when a launcher job must not start: pre-submit `sbatch --hold -A nemotron_sw_post -p cpu -N1 -t 1 -J <EXP_NAME> --wrap true`,
launch (blocks on singleton), `scontrol hold <job>`, then scancel the placeholder.

## 6. Traps (each one cost a segment or a day)

1. **afterany fires on cancellation.** Gates are `afterany:<seg1>:<seg2>` of the previous arm; cancelling a pending
   gate job releases whatever depends on it. To insert or reorder: `scontrol hold` every downstream id FIRST, cancel
   nothing. Stopping an arm = cancel both of its ids (by name); never cancel a pending seg-1 alone (its seg 2 starts).
   A hang-cancel of a running seg 1 makes its seg 2 the automatic relaunch.
2. **lr 0 cannot load Adam state**: DistributedOptimizer matches param groups by (max_lr, min_lr, …) -> KeyError;
   `checkpointing.save_optimizer=false` does NOT gate the load. Fix = `+policy.megatron_cfg.checkpoint.load_optim=false`
   via the `swe_lr0` setup.py overlay (weights-only rungs; compare weights, not optimizer).
3. **Hydra `+` rule**: `+key=` is required when the key is absent from the whole yaml defaults chain and crashes
   ("An item is already at …") when it exists. Walk the chain with an exact-key parse: `checkpointing.load_replay_buffer`
   is ABSENT (needs `+`, default True in single_controller.py:416 — every resume segment must carry
   `+checkpointing.load_replay_buffer=false`); `override_opt_param_scheduler` EXISTS (no `+`);
   `policy.generation.vllm_cfg.enable_prefix_caching` absent (needs `+`; absent = vLLM default = ON on Hopper/Blackwell;
   the `false` you see in saved configs is the MINF knob `mcore_generation_config.enable_prefix_caching`).
4. **Resume artefacts**: a resumed step regenerates its pending prompts and APPENDS to the same
   `dumps/rollouts/target_step_N.jsonl` (>512 rows) — dedupe by `sample_id`; the token_level `.pt` holds the 512 trained.
5. **Replayed groups write token_level chunks but NO rollout jsonl** (`generate_and_push_foreign` skips the writer),
   and their sample/group ids are re-minted, so post-hoc identity checks join to the source by `input_ids` content
   (Data Difference's `W/swe_dump/analysis/datadiff/verify_replay.py <arm_run> <src_run> <mask_csv>`).
6. **Splice startup**: a "SingleController ping failed" ladder (up to ~40 min for 20 steps) while
   `ForeignRolloutSource._build_index` scans the source jsonl; not a hang. The first live step after a fresh start,
   after replay and after a resume takes ~60 min; head-of-line groups of 60-70 min are normal (in_order sampler).
   Before calling a hang, check Gym `worker-*.out` mtimes and jsonl writes in the last 15 min. The real-time SC log is
   `ray_logs/<job>-logs/ray/session_*/logs/worker-*.out` (grep `rollout_pump: starting`); the driver log lags.
7. **MINF single-rank startup hang** (P⁗2 seg 2): no SC log, zero writes; diagnose via the head node's Ray state API
   (`/api/v0/tasks?limit=2000&detail=1`: one old RUNNING `prepare_for_generation`), cancel, let seg 2 relaunch.
8. **Gym cache is in-tree and unversioned**: `Gym/cache/swe_agents/*_setup` (r2egym, swebench, multilingual, rebench,
   OpenHands) must exist before a fresh Gym server starts (no egress to rebuild); restore by targeted rsync from a
   working Gym cache, never the 5 TB `swe_openhands_setup`.
9. **Quota**: 100 TiB uid quota; a rung is 369 GB, dumps ~10 GB/step, gym_results dominate (~0.14 TiB/step all-in).
   A full disk kills every queued segment at container start.
10. **MINF nccl refit backend** (P″) dies at the first refit; nvshmem works (advisory "currently broken" is noise).

## 7. What must change on HSG

* Every launcher pins CMH absolute paths (W, RUNS, akamehra's `swe_e2e_corrected` tree, HF_HOME under
  `nemotron_sw_pre`, the two containers, SIF dir) and CMH Slurm names (account, hero-res QOS, batch_long partition,
  reservation, `cpu` partition for tests) — edit the header block of each `launch_*.sh` and the test sbatch files.
* Node shape: recipes are `swe_sc_cmh_*` for GB200 4-GPU nodes (32+32 nodes, TP4/CP4/EP32, CPUS_PER_WORKER=140).
  Different GPUs per node change node counts and the parallelism block; port the recipe rather than the launcher only.
* Copy to HSG: the container images, base model `step_18/hf`, the two data jsonl, the SIF dir, and for any replay arm
  the source run's `dumps/rollouts/target_step_00000..00019.jsonl` + `dumps/token_level/step_00001..00020_chunk_*.pt`
  (chain G: ~10 GB/step); for seeded arms the seed rung (369 GB) copied into the new run's `checkpoints/`.
* Rebuild the Gym in-tree setup cache (trap 8) from a working CMH Gym cache or with egress.
* Re-render with `DRY_RUN=1`, then run the workspace `test_*.sbatch` in the HSG container before the first submit.
* W&B project/API key and the eval runner's NEL recipe are separate systems; results so far are in the
  `results.md` files named in §4.

## 8. Where things are on CMH (not portable)

RUNS/<run>/{checkpoints,dumps/rollouts,dumps/token_level,gym_results,ray_logs/<job>-logs}; per-arm READMEs in the run
dirs (copies in `splice_workspaces/run_readmes/`); manager's run-dir map `W/swe_dump/analysis/run_dir_map.md`;
Data Difference's tooling `W/swe_dump/analysis/datadiff/` (spec EXPERIMENTS.md, verify_replay.py, loop-share scoring);
my validators `W/engine_loop_test/analysis/optim/{compare_replay_dumps*.py,weights_identity_check.*,turn_mismatch_by_refit.py}`
and `W/engine_loop_test/ops/test_overrides.sbatch`; monitor script used for the ticks:
`~/.claude/jobs/824af663/tmp/tick_xyz.sh` (session-local). Memory notes: `~/.claude/projects/-home-rkirby-workspaces-nemo-rl-workspace/memory/swe-*.md`.

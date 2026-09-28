---
name: swe-e2e-base-vs-minf-fork
description: "SWE-E2E v2 + MINF on the akamehra/swe-v2 base (Option A, user's parity requirement): analysis of the runbook base vs our fork, the curated cherry-pick list, and the IMPLEMENTED workstream at users/rkirby/workspaces/swe_minf (branch rkirby/swe-v2-minf 2bc89d09b pushed, fork MLM 880de0fce mount, launch_swe_minf.sh helper with SMOKE=1)"
metadata:
  type: project
---

User pivot 2026-09-15: the MR52 RLVR route is parked; they were handed the "CMH-1 SWE-E2E v2 64-node runbook"
(reference job 3541201, EXP nano35-swe-v2-stream128-inorder1-cmh-64n, W&B ultra-v3-swe-e2e-convergence) and want
MINF on top of it. Worktree: $U/akamehra/code/post-training-pipeline-lightning-sc/nemo_rl_swe_e2e_e919 (git ops need
`-c safe.directory=*`; commits are fetchable by SHA from NVIDIA-NeMo/RL).

Findings (all verified with git on 2026-09-15):
- nemo_rl base 7cfa1bed (Sep 1) = 5057d1a1 (Aug 29) + 3 SWE yamls in examples/nemo_gym/nemotron-3.5-nano/. The branch
  forked from main at b59165262 (Aug 21) and carries 14 commits, none on main by patch-id (MR52 README: 3 of them
  superseded differently upstream). Our fork base 5d49fbf4e is 107 main commits later.
- MINF under SingleController does NOT exist at the base: single_controller_utils/setup.py asserts
  "The Megatron generation backend does not support non-colocated inference in SingleController." Upstream MInf/SC
  work missing there = ~4.8k lines (setup.py +1307, config.py +790, rollout_checkpoint.py +632, megatron_worker +1349,
  megatron_generation +416, weight sync +141; PRs #3727 #3730 #4009 #3912 #3762 #3739 #3300). Base config lacks
  sampling_backend/logprobs_mode/recompute.
- All 5 of our nemo_rl fork commits conflict at the base (adam drain, MTP #4098, dumps, genrm identity, port fix).
- Bridge 8c46dc42 (Aug 14) is 200 commits behind our 5ed97996c; MLM 14346b65a (Aug 12) is 273 behind 1e7598cbf;
  all 5 MLM fork commits conflict on the old MLM (multi-EOS, sampler, debug print, stitching fallback, strict parser).
- Gym 354babf7e (Aug 20) is an ancestor of our pins (163 commits to fd5e84d6b); SWE harness configs
  (swebench_openhands_training.yaml, vllm_model_for_training.yaml) and their keys are unchanged at fd5e84d6b/3615823e.
  run_with_mixed_prompts is not consumed by Gym or nemo_rl code at all (yaml-only).
- Old container rl-gym.63635108-zstd: Bridge/MLM editable installs at the same /opt/nemo-rl/3rdparty paths;
  apptainer present in both it and our NGC image.
- Forward-port (SWE onto our fork 7961ce437): recipe commits 304990066, a23f72a4d, 7cfa1bed (+3cbbe3e09 pyxis
  isolation) apply cleanly; the SC-branch fixes (ec9848f6d, 684c0f9d9, e919a5190, d1bf7ab3f, e0655221e, fd184d606,
  611da00ee replay-free resume, 5057d1a1 metrics) all conflict. `load_replay_buffer` on main exists only in the async
  v1 path; the SC prompt-journal resume is branch-only.
- SWE recipe specifics that matter for a MINF overlay: max_total_sequence_length 196608 (RLVR is 73728),
  vllm max_num_batched_tokens 32768 (validated ref used 8480), backend simple, in_order lookahead 1, MTP 5, GBS 512,
  32 prompts x 16, checkpoint save_period 5, 0 Gym GPU nodes, SIF_DIR apptainer images via EXTRA_MOUNTS,
  NRL_TQ_SKIP_RUNTIME_ENV_INSTALL=1 (branch-only env), TRAIN_ENTRYPOINT='--no-sync ./examples/run_grpo_single_controller.py'.

**How to apply:** recommend building SWE+MINF on our fork (main@Sep 11 + MINF fixes) with the NGC container, porting
the 3 SWE yamls + a swe_sc_minf overlay (copy the rlvr_sc_minf.yaml policy.generation block) and akamehra's
nano35_launch.sh swe case/SIF handling; treat the SC-branch fixes as optional re-implementations. Do not try to rebase
MINF onto the Aug-29 base. Related: [[pipeline-b-mr52-branch]], [[minf-v2-9-15-run]].

**Update 2026-09-15 (user insists on the old base for parity; unknown regression between Aug 21 lineage and main):**
The earlier HSG MINF campaign stack already IS "MINF on the old lineage" and is the parity-compatible starting point:
- nemo_rl fork branch `rkirby/rlvr-nolap-repro` (29e352165) = ee12fc57d lineage (merge-base with akamehra/swe-v2 =
  98b2829c3) + MINF commits: 2dcbd24b4 serve inference from every DP group, ee7093955 load inference weights instead
  of refit, a32895771 raw_logprobs default + logprob dumps, 0c148c8b7 tolerate megatron-core w/o fork prefix-cache
  symbols, 20a694d56 sampling backend selectable, 0998b013e Adam-save drain (origin of our cherry-pick), plus debug
  dumps (d63cb42c3, 41eae1a03) and upstream #3599 (bca48bc44). Sequentially cherry-picking the six MINF commits onto
  7cfa1bed is CLEAN (trial 2026-09-15).
- Bridge fork branch `rkirby/rlvr-nolap-repro` (618a5da36) = the container's Bridge 8c46dc42 + pointer-only commits
  -> no Bridge mount needed; only the nested Megatron-LM path
  /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM needs mounting.
- MLM fork branch `rkirby/rlvr-nolap-repro` (880de0fce) = the container's MLM 14346b65a + 30 fork commits (Siddharth's
  DP frontend stack, prefix-cache salt, Mamba prefix-skip fix, multi-EOS 8b880867e, Gumbel fp32 sampler 2c43cacff,
  qwen3-coder parser parity 367b7b53f/fcc4d283c, adistomar fused-MoE buffers). The old MLM has no compact-id
  stitching gate and no implicit_reasoning_end_markers, so the stitching fallback and strict-parser commits are moot.
- The SC gap remains: non-colocated MInf in SingleController needs upstream #3727 (dd4b98835) which conflicts heavily
  on the base (23 hunks incl. grpo.py, single_controller.py, setup.py, generation/interfaces.py + docs/tests); its
  helper #3569 (89cbf16d9) conflicts in 3 hunks once the fork MINF commits are in; #3718 applies. The base asserts
  "Megatron generation backend does not support non-colocated inference in SingleController".
- Old container rl-gym.63635108-zstd has nvidia_nvshmem_cu13 3.7.2 + nvshmem4py (nvshmem refit OK) and apptainer.
- The old-schema MINF overlay = pipeline_A commit 8f2364c:RLVR/nemotron-3.5-nano/configs/rlvr_minf.yaml
  (mcore_generation_config keys valid for that nemo_rl age; add sampling_backend only with 20a694d56).

**Decision 2026-09-15: Option A (build MINF on akamehra/swe-v2 7cfa1bed).** Curated list (verified):
- nemo_rl, from fork branch rkirby/rlvr-nolap-repro, all apply clean in this order: 2dcbd24b4 (frontend placement on every
  MP coordinator + coordinator routing policy + cg sizing config; developed on a 16-engine SWE-RL run; the base already
  has report_dp_openai_server_base_url but not this), ee7093955 (skip_weight_load=False for the dedicated inference
  policy; the same must be applied to the SC path #3727 adds in setup.py:429), a32895771 (logprobs_mode raw_logprobs +
  dumps), 0c148c8b7 (tolerate mcore without fork prefix-cache symbols; safety), 20a694d56 (sampling_backend config),
  0998b013e (Adam drain: base's _requires_nvrx_cuda_cache_release exists and the bug was measured on this lineage ->
  needed). Skip: dumps 41eae1a03/d63cb42c3 (conflict, debug), bca48bc44 (#3599 v1-only), sauramishra trio (already in
  base), pointer commits.
- upstream: #3727 dd4b98835 only (9 code hunks in grpo.py, single_controller.py, setup.py, generation/interfaces.py,
  megatron_generation.py + 6 docs/tests files, measured after the six MINF commits). #3569/#3718 are NOT prerequisites
  (skip). Later MInf PRs (#3864 #3730 #4009 #3912 #3762 #3739 #3300) skipped.
- Megatron-LM: mount fork tree 880de0fce (rkirby/rlvr-nolap-repro) over the nested Bridge/3rdparty/Megatron-LM path;
  all 30 commits are fork-only vs NVIDIA main; 8 touch shared files but only inference-gated code plus one debug print,
  EXCEPT 5f0d342d8 (adistomar) which removes the Mamba SSM `intermediate_ssm_states = x  # Dummy pointer` kernel bug
  (NVIDIA #6598) present in the container's MLM 14346b65a and fixed upstream by 1e7598cbf -> the MINF run trains with
  the fixed kernel while the vLLM reference trained with the buggy one (parity caveat). Parsers available:
  nemotron-v3-reasoning, qwen3-coder-tool.
- Bridge: container's 8c46dc42 as-is (fork Bridge = pointer-only). Gym 354babf7e unchanged. Container 63635108-zstd.
- Config: old-schema overlay from pipeline_A 8f2364c rlvr_minf.yaml + sampling_backend torch; check 192K context vs
  buffer_size_gb/max_tokens; NVSHMEM_MAX_CTAS=2 via exported env (launcher lacks EXTRA_TRAIN_ENV).


**Update 2026-09-15 evening: Option A IMPLEMENTED (not yet run).** Workstream dir
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_minf/`:
- `nemo_rl/` = clone of git@github.com:ArEsKay3/RL.git, branch `rkirby/swe-v2-minf` (pushed) = 7cfa1bed + six MINF
  cherry-picks (a2bd45f88 d6e88f241 8f881aac3 e5fd44259 e9e980652 9ef882715) + 31fe8640a "feat(sc): serve
  non-colocated Megatron in-engine inference in SingleController" (hand port of #3727's SC half: no assertion in
  _build_clusters, megatron branch in _build_generation constructing MegatronGeneration(cluster=inference_cluster,
  skip_weight_load=False) which starts engine+HTTP server in __init__, Gym spun up behind a Future once the engine is
  live, trainer in parallel; _validate_megatron_generation rejects colocated/fleet-health/kv-recompute mismatch;
  MegatronGeneration.worker_group property; Generation union) + 2bc89d09b swe_sc_cmh_minf.yaml overlay
  (defaults: swe_sc_cmh_stream4.yaml; old-schema MINF block, prefix caching ON with longest_prefix, buffer 70GB,
  max_tokens 16384, max_model_len = 196608 via interpolation, 32 gen nodes). Gym submodule 354babf7e; Bridge/Automodel
  not checked out (container's). The wholesale `git cherry-pick dd4b98835` was ABORTED (needed RemoteHeldPortReservation
  etc. absent at base); the port is the minimal replacement.
- `Megatron-LM/` = detached git worktree (from pipeline_B's nested MLM repo) at fork 880de0fce, mounted via EXTRA_MOUNTS
  onto /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM (container venv python3.13,
  megatron_core editable finder -> that path; verified 2026-09-15).
- `launch_swe_minf.sh` = runbook env (akamehra assets read-only, container rl-gym.63635108-zstd, SIF_DIR, sandbox) +
  ours: HF_HOME nemotron_sw_pre/users/rkirby/hf_home, RESULTS_DIR/PERSISTENT_CACHE under users/rkirby/{runs,persistent_cache}/$EXP_NAME,
  NRL_IGNORE_VERSION_MISMATCH=1 (fingerprint check is a hard RuntimeError otherwise), NVSHMEM_MAX_CTAS=2,
  NRL_MINF_SAMPLING_BACKEND=torch, NRL_MINF_LOGPROBS_MODE=raw_logprobs (on this lineage sampling backend/logprobs mode
  are ENV knobs read in megatron_worker.py, not config keys), WANDB_PROJ ultra-v3-swe-e2e-convergence (runbook's).
  SMOKE=1 -> EXP nano35-swe-v2-minf-smoke-16n, 8 train + 8 gen nodes, QOS short 2h, TP4 CP4 EP8, 8 prompts x16,
  GBS 128, min_groups 2, max_buffered 24, max_inflight 16, checkpointing off, NRL_MAX_STEPS=2. Full = runbook shape
  (32+32, TP4 CP4 EP32, 32 prompts, GBS 512, min_groups 8, max_buffered 96, normal QOS 4h), EXP
  nano35-swe-v2-stream128-inorder1-cmh-64n-minf. DRY_RUN=1 renders clean (2026-09-15 20:15).
- MLM #6672 (fbbc142b8 "Fix Refit Cache for mamba", user asked for it): NOT applied as a commit, but the fork tree
  880de0fce already carries the same fix in its original form: refit.py reshard_model_weights() -> execute_reshard_plan()
  then _run_post_refit_hooks(tgt_core) -> MambaMixer.post_refit() -> _refresh_A_neg_exp_cache(); nemo_rl drives refit
  only through swap_model_weights (megatron_worker.py:1033/1092) -> covered. Upstream renamed it refresh_cache() and
  moved the trigger into execution.py; porting would conflict with the fork version. Told the user; no action.
- Parity caveats to state in reports: fork MLM fixes the Mamba SSM dummy-pointer kernel bug (#6598) the reference's
  container MLM has; prefix caching is on for MINF (vLLM reference had it on too); Gym spinup is serialized behind the
  engine load (no port reservation) so setup is a few minutes longer than upstream's #3727 flow.

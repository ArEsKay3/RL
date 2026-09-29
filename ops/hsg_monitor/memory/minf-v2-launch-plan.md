---
name: minf-v2-launch-plan
description: "User's plan for the first full 86-node V2 RLVR run on CMH — EXP_NAME=minf-v2, 4h walltime, W&B nano35-rlvr-convergence, refit backend chosen by the nccl smoke (nccl if it passes, else nvshmem)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7aa0a354-aa1b-4011-8140-8d3122eb7495
  modified: 2026-09-12T22:53:47.948Z
---

Decided 2026-09-11 evening. Once a smoke passes on CMH, launch one 4-hour full run:

```
EXP_NAME=minf-v2 MINF=1 [EXTRA_HYDRA_OVERRIDES="policy.generation.mcore_generation_config.refit_backend=nccl"] \
  bash RLVR/nemotron-3.5-nano/scripts/launch_dolphin_convergence_v1_86n.sh
```

(run from `nemo-rl-workspace/pipeline_A/`). Defaults give 86 nodes (32 train + 32 gen + 2 gym + 20 judges),
100 steps, checkpoint every 10, walltime 4:00:00, W&B project `nano35-rlvr-convergence` run `minf-v2`,
checkpoints under `users/rkirby/runs/minf-v2`. Dry run verified clean on 2026-09-11.

Refit backend rule (user's call): `rlvr_minf.yaml` sets `refit_backend: nvshmem`. Smoke 3683477 tests nvshmem;
smoke 3683907 (`rlvr-smoke-minf-cmh-nccl`) tests nccl via the override above. If the nccl smoke completes its two
steps with a working refit, minf-v2 uses nccl (pass the same override); otherwise leave nvshmem.

**Result so far:** smoke 3683477 (nvshmem) PASSED on 2026-09-11 21:54 — 2 steps, nvshmem copy-service refit
between them, `ray.sub exiting (exit_code=0)`, 32 min wall (~9 min judges, ~2 min Ray, ~5 min Gym spinup, 2 rollout
batches of ~6 min, step-2 time 12 s). Gumbel-max PR#16 PRESENT on every rank. Generation KL error 0.0008 / 0.0020.
First CMH job ever to complete. nccl smoke 3683907 also PASSED (2 steps, nccl refit, exit 0, ~50 min wall incl. a
20-min apt-get hang on one node rescued live). Decision: minf-v2 uses refit_backend=nccl.

**Launched:** minf-v2 attempt 1 = job 3684379, submitted 2026-09-11 ~22:47 with the nccl override, 86 nodes
(64+2 het-group 0, 20 het-group 1), 4h walltime. Logs `users/rkirby/runs/minf-v2/3684379-logs/`, sbatch logs under
`users/rkirby/runs/minf-v2/runs/*/slurm/`. Relaunches must reuse EXP_NAME=minf-v2 to resume from checkpoints.

**Attempt 1 (3684379, nccl) FAILED** at 34 min in `setup()` during the initial refit: all 264 actors up, 71/71
Gym servers, then `MegatronPolicyWorker.swap_weights_via_reshard` -> `nccl_copy_service._ensure_nccl_connected`
-> `ncclRemoteError: connect to 10.67.10.124<36085> returned Connection refused, exceeded error retry count after
35 attempts` (peer = nvl72d230-T15). nccl worked on the 8-node smoke but not on 66 nodes. Per the user's rule,
attempt 2 uses nvshmem (config default, passed smoke 3683477).

**Attempt 2 = job 3685551** (nvshmem, no override), submitted 2026-09-12 ~00:33. Logs
`users/rkirby/runs/minf-v2/3685551-logs/`. The peer node in attempt 1 (nvl72d230-T15) had all its workers at
init_complete, so the refusal was a NCCL bootstrap/listener problem, not a dead node. **Attempt 2 (3685551, nvshmem) FAILED at 31 min**, but PAST the refit: 264/264 actors, 71/71 servers, nvshmem
refit OK, first rollout batch 74% done, then `AsyncTrajectoryCollector aborting: 1 batch-worker failure(s)
exceeded max_generation_failures=0`. Root cause: Megatron-LM chat endpoint raised `Prefix stitching requires
compact_prompt_token_ids` (9 requests, genrm_simple_agent) because Gym's TokenIDLogProbMixin whitelists only
prompt_token_ids/generation_token_ids/generation_log_probs and drops compact_prompt_token_ids. Fix committed on the
fork chain: Megatron-LM 949a7bf8e (fallback to prompt_token_ids when no media slots) -> Megatron-Bridge 38efd6a9d
-> nemo_rl 2f38c4b7d -> pipeline 4280a78 (EXPECTED_* guards updated). Pushed 2026-09-12: fork branches rkirby/nemorl-with-minf on ArEsKay3/{Megatron-LM,Megatron-Bridge,RL} and pipeline rlvr-convergence-minf on GitLab at 4e1f508 (docs commit on top). Branches
`rkirby/nemorl-with-minf` are now checked out (not detached) in all three submodules.

**Mechanism, verified in code (2026-09-12):** the failing MINF request is the SimpleAgent's second call after a
policy TOOL CALL in a GenRM task that defines tools (~3% of genrm rows; genrm_compare server log shows a
`POST /browse_url` tool execution). The rebuilt assistant tool-call message carries the Mixin token ids (no
compact) and MINF's stitching gate rejects it. The "reasoning-only re-call" path does NOT reach the engine:
policy_model has `sequential_reasoning_allowed: false`, and Gym short-circuits a trailing assistant message with
empty content and no tool_calls before calling the engine. Dataset-provided assistant turns (34% of genrm rows)
carry no token ids and never trigger the gate. Both engines return a message-level token bundle (vLLM via
NeMo RL's `attach_token_information_to_chat_response_choices` patch; MINF natively, plus compact_prompt_token_ids);
Gym normalizes both to the same ForTraining message; only MINF *consumes* token ids on input.

**Reasoning-parser comparison (verified 2026-09-12):** vLLM 0.25.1 `BaseThinkingReasoningParser.extract_reasoning`
(nano_v3 plugin inherits via DeepSeekR1) and MINF `NemotronV3ReasoningParser` (inherits MINF DeepSeekR1) agree on
every reasoning-only case with enable_thinking=true: unclosed `<think>` -> everything is reasoning, content empty;
closed with nothing after -> reasoning + empty content; enable_thinking=false -> both surface reasoning as content.
Gym then treats both identically (re-wrap in think tags, reasoning item only, agent loops once, Gym's
sequential_reasoning_allowed=false short-circuit returns an empty completion without calling the engine). Only
asymmetry found: MINF's parser honours `implicit_reasoning_end_markers` (e.g. `<tool_call>`) when tools are
requested, so a tool call emitted inside an unclosed think block is parsed and executed on MINF but swallowed as
reasoning on vLLM. Chat template (nano v3.5) is the same HF jinja on both engines.

**Attempt 3 = job 3688117** (nvshmem + stitching fallback), submitted 2026-09-12 06:25 local (run dir
`runs/20260912-062500`). Logs `users/rkirby/runs/minf-v2/3688117-logs/`. Sat PENDING (Resources) behind a
~450-job normal-QOS backlog; at 11:10 SLURM began a start (prolog on a node subset, transient COMPLETING) that
did not hold and the job went back to pending. Run-dir names are local submit time; dry runs also create run dirs. **Attempt 3 STARTED 2026-09-12 11:47:49**
(all 86 nodes); walltime ends ~15:47. Timeline: ray sruns ~+9 min, 264/264 actors 12:00, 71/71 servers +
first rollouts 12:10, step 1 done ~12:35 (719 s step time, KL err 0.0015), `[adam-fix] nvrx_cuda_cache_release=True
storage_can_move=True offload_for_refit=True` printed, first checkpoint (`step_1`) saved via NVRx async save.
Step 2 rollouts overlap training (async GRPO). Steps 1-10 done by 14:50 (times 720/559/1096/564/1286/584/1183/
585/999/593 s; slow steps = trainer waiting on 1-2 straggler rollout groups, `exposed_generation`). Checkpoints:
ft_save_period=1/ft_keep_latest_k=1 -> per-step latest only; save_period=10/keep_top_k=1e6 -> `step_10` is the
first PERMANENT rung (14:50). **COMPLETED to walltime:** TIMEOUT at 15:47:59 (4:00:10), 14 optimizer steps, zero errors/stitching failures/
server deaths. Checkpoints left: `step_10` (permanent rung) and `step_14` (latest, resume point;
latest_checkpoint_status.json -> 14). W&B nvidia/nano35-rlvr-convergence run minf-v2. Rewards 0.75-0.88, KL err
~0.0016. Per user instruction (15:05) NO follow-up jobs were submitted. optim_bitcmp step_10 vs step_14 run as a
read-only CPU check afterwards (result: every optimizer tensor DIFFERS, see [[adam-fix-bitcmp-result]]).

If nvshmem also fails at the
initial refit, next candidates: `refit_backend=gloo` (config comment names gloo and nccl as fallbacks), or check
NCCL_SOCKET_IFNAME / interface selection on CMH.

**Why:** upstream's configs annotate nvshmem as currently broken (NVIDIA-NeMo/RL#3646) while the validator still
accepts it; the user wants the safer backend if it demonstrably works here.

**How to apply:** after both smokes report, launch minf-v2 with the chosen backend and set up a monitor. Note
`checkpoint_must_save_by` in rlvr.yaml is 23h35m, so on a 4h job only the every-10-step checkpoints land.
Related: [[v2-stack-needs-newer-container]], [[cmh-cpus-per-worker-140]].

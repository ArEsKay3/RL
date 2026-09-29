#!/usr/bin/env bash
# Launch the CMH SWE-E2E recipe of launch_swe_main915.sh (nemo_rl main,
# rkirby/swe-main915-latest) with Megatron in-engine inference running the
# vLLM numerical-parity adapter: the Megatron-LM-parity tree (mlm-main915-latest
# + santhnm2 vllm-numerical-parity-main + the two eval-mode gate fixes) is
# mounted instead of Megatron-LM, the recipe is swe_sc_cmh_parity_minf.yaml
# (chain U's MINF recipe + inference_vllm_parity, TP4/EP1/ETP4, max_tokens 8480),
# and the staged CUTLASS-DSL/quack packages reach the Megatron policy-worker
# venv through a bind-mounted .pth (tools/stage_parity_site.sbatch stages them
# from this stack's own container).
#
#   DRY_RUN=1 SMOKE=1 bash launch_swe_main915_parity.sh   # render only
#   SMOKE=1 RESERVATION=1 bash launch_swe_main915_parity.sh   # 16-node wiring test
#   RESERVATION=1 bash launch_swe_main915_parity.sh           # 32 + 32 nodes, 8 h segment
#
# Trailing arguments are forwarded as Hydra overrides. Same knobs as the base
# launcher (RESERVATION, SMOKE, EXP_NAME, RUN_DATE, ENABLE_PREFIX_CACHING).
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${WS}/nemo_rl"
MLM_TREE="${WS}/Megatron-LM-parity"
PARITY_SITE="${WS}/parity_site"
PARITY_PTH="${WS}/parity_paths.pth"
TORCH_EXT_DIR="${TORCH_EXT_DIR:-${WS}/torch_extensions}"
PTXAS_PATH="${PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
MW_SITE_PACKAGES=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages
BRIDGE_TREE="${WS}/Megatron-Bridge"
BRIDGE_MOUNT_TARGET=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MLM_MOUNT_TARGET="${BRIDGE_MOUNT_TARGET}/3rdparty/Megatron-LM"

SWE_BASE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra
MINE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby
MY_HF_HOME=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/rkirby/hf_home

: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
ENGINE="${ENGINE:-minf}"
[[ "${ENGINE}" == "minf" ]] || { echo "ERROR: the parity launcher is MINF-only (got ENGINE=${ENGINE})" >&2; exit 1; }
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

if [[ "${RESERVATION:-0}" == "1" ]]; then
  export SLURM_PARTITION="${SLURM_PARTITION:-batch_long}"
  export SLURM_QOS="${SLURM_QOS:-hero-res}"
  export SLURM_RESERVATION="${SLURM_RESERVATION:-sla_res_nemotron_sw_post}"
  export WALLTIME="${WALLTIME:-8:00:00}"
fi

export MODEL_PATH="${SWE_BASE}/swe_e2e_corrected/base_model/step_18/hf"
export TRAIN_PATH="${SWE_BASE}/swe_e2e_corrected/data/large_root_cause_curriculum_with_mercor_ots_plus_singlefile_swerebench_overlap_fix.jsonl"
export VAL_PATH="${SWE_BASE}/swe_e2e_corrected/data/swe_public_datasets_val_swebench.jsonl"
# rl-gym.63635108 lacks nemo.lens, which nemo_rl main's telemetry module
# hard-imports (see the fix commits on the nemo_rl branch). fd45cb8 is the
# V2-line container verified to carry nemo_lens + venvs + libXrender.
export CONTAINER="${MINE}/containers/nemo-rl_fd45cb8-67127697-gym.sqsh"
export SANDBOX_CONTAINER="${SWE_BASE}/containers/nemo-skills-sandbox-no-sync.sqsh"
export SIF_DIR="${SWE_BASE}/swe_e2e_corrected/sif"
export HF_HOME="${MY_HF_HOME}"

export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_ACCOUNT=nemotron_sw_post
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=140
export NUM_GYM_NODES=0

export USE_SNAPSHOT=0
export NRL_TQ_SKIP_RUNTIME_ENV_INSTALL=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_IGNORE_VERSION_MISMATCH=1
export TRAIN_ENTRYPOINT='--no-sync ./examples/run_grpo_single_controller.py'

export WANDB_PROJ="${WANDB_PROJ:-ultra-v3-swe-e2e-convergence}"
export SLURM_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"data_loading","description":"Async GRPO SWE: train GPUs idle during long agentic rollout collection (can exceed 3h) and per-step validation"}}'

if [[ "${ENGINE}" == "minf" ]]; then
  export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_parity_minf.yaml
  export NVSHMEM_MAX_CTAS="${NVSHMEM_MAX_CTAS:-2}"
  export NRL_MINF_SAMPLING_BACKEND="${NRL_MINF_SAMPLING_BACKEND:-torch}"
  export NRL_MINF_LOGPROBS_MODE="${NRL_MINF_LOGPROBS_MODE:-raw_logprobs}"
  ENGINE_OVERRIDES=(
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
    "policy.generation.mcore_generation_config.enable_prefix_caching=${ENABLE_PREFIX_CACHING:-true}"
  )
  FULL_EXP_NAME="nano35-swe-main915-64n-parity-minf-${RUN_DATE}"
  SMOKE_EXP_NAME="nano35-swe-main915-parity-minf-smoke-16n"
else
  export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_dump_vllm.yaml
  ENGINE_OVERRIDES=(
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
  )
  FULL_EXP_NAME="nano35-swe-main915-64n-vllm-${RUN_DATE}"
  SMOKE_EXP_NAME="nano35-swe-main915-vllm-smoke-16n"
fi

SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  export EXP_NAME="${EXP_NAME:-${SMOKE_EXP_NAME}}"
  export SLURM_QOS="${SLURM_QOS:-short}"
  export WALLTIME="${WALLTIME:-2:00:00}"
  export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-8}"
  export NUM_GEN_NODES="${NUM_GEN_NODES:-8}"
  export NRL_MAX_STEPS="${NRL_MAX_STEPS:-2}"
  # select_segment_nodes requires num_nodes divisible by segment_size; the
  # full run's 16 divides 32 but not the smoke's 8 (job 4013945: ValueError).
  export SEGMENT_SIZE="${SEGMENT_SIZE:-8}"
  SHAPE_OVERRIDES=(
    policy.megatron_cfg.tensor_model_parallel_size=4
    policy.megatron_cfg.context_parallel_size=4
    policy.megatron_cfg.expert_model_parallel_size=8
    grpo.num_prompts_per_step=8
    policy.train_global_batch_size=128
    async_rl.min_groups_for_streaming_train=2
    async_rl.max_buffered_rollouts=24
    async_rl.max_inflight_prompts=16
    checkpointing.enabled=false
    # floor = ceil(num_prompts_per_step * min_step_batch_fraction). Two
    # attempts (4015949 lost 1/8, 4016238 lost 3/8 even after replacement)
    # show a real per-rollout drop rate on this SWE dataset/container combo
    # (Gym: "extra_forbidden" on aws_region_name, 672 occurrences in one
    # run -- agent-harness-side, unrelated to anything ported this session).
    # The smoke exists to prove the MINF/nemo_rl wiring, not batch quality
    # at n=8; 0.4 (floor=4) gives real headroom against that drop rate
    # rather than re-guessing the exact fraction after each attempt.
    async_rl.rollout_failure.min_step_batch_fraction=0.4
  )
else
  export EXP_NAME="${EXP_NAME:-${FULL_EXP_NAME}}"
  export SLURM_QOS="${SLURM_QOS:-normal}"
  export WALLTIME="${WALLTIME:-4:00:00}"
  export SEGMENT_SIZE="${SEGMENT_SIZE:-16}"
  export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
  export NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
  export NRL_MAX_STEPS="${NRL_MAX_STEPS:-1000000}"
  SHAPE_OVERRIDES=(
    policy.megatron_cfg.tensor_model_parallel_size=4
    policy.megatron_cfg.context_parallel_size=4
    policy.megatron_cfg.expert_model_parallel_size=32
    grpo.num_prompts_per_step=32
    policy.train_global_batch_size=512
    async_rl.min_groups_for_streaming_train=8
    async_rl.max_buffered_rollouts=96
  )
fi

export PERSISTENT_CACHE="${PERSISTENT_CACHE:-${MINE}/persistent_cache/${EXP_NAME}}"
export RESULTS_DIR="${RESULTS_DIR:-${MINE}/runs/${EXP_NAME}}"
mkdir -p "${PERSISTENT_CACHE}" "${RESULTS_DIR}"

if [[ -n "${SEED_CHECKPOINT:-}" ]]; then
  SEED_TARGET="${RESULTS_DIR}/checkpoints/$(basename "${SEED_CHECKPOINT}")"
  [[ -d "${SEED_TARGET}/policy/weights/iter_0000000" && -f "${SEED_TARGET}/config.yaml" && -f "${SEED_TARGET}/pending_rollouts.pt" ]] || {
    echo "ERROR: ${SEED_TARGET} is missing or incomplete; copy ${SEED_CHECKPOINT} there first" >&2; exit 1; }
  echo "Seed checkpoint present: ${SEED_TARGET}"
fi

DUMP_OVERRIDES=(
  "async_rl.dump.dir=${RESULTS_DIR}/dumps"
  "+env.nemo_gym.results_dir=${RESULTS_DIR}/gym_results"
)
mkdir -p "${RESULTS_DIR}/dumps" "${RESULTS_DIR}/gym_results"

# nano35_launch.sh's standard mounts cover nemo_rl/, examples/configs/ and
# examples/nemo_gym/nemotron-3.5-nano/, but not loose files directly under
# examples/ -- our TRAIN_ENTRYPOINT script lives there and isn't covered, so
# it silently runs from whatever fd45cb8 baked (a stale driver script that
# predates the draft_refit_enabled() helper: smoke 4013863,
# TypeError: 'Eagle3DraftConfig' object is not subscriptable).
[[ -f "${MLM_TREE}/megatron/core/inference/vllm_parity.py" ]] || {
  echo "ERROR: ${MLM_TREE} is not on the parity branch (no megatron/core/inference/vllm_parity.py)" >&2; exit 1; }
[[ -d "${PARITY_SITE}/quack" && -d "${PARITY_SITE}/nvidia_cutlass_dsl/python_packages/cutlass" ]] || {
  echo "ERROR: ${PARITY_SITE} is missing quack / cutlass; run: sbatch --partition=batch_long --qos=hero-res --reservation=sla_res_nemotron_sw_post tools/stage_parity_site.sbatch" >&2; exit 1; }
mkdir -p "${TORCH_EXT_DIR}"
cat > "${PARITY_PTH}" <<PTH
${PARITY_SITE}
${PARITY_SITE}/nvidia_cutlass_dsl/python_packages
import os; os.environ.setdefault('TRITON_PTXAS_BLACKWELL_PATH', '${PTXAS_PATH}'); os.environ.setdefault('TORCH_EXTENSIONS_DIR', '${TORCH_EXT_DIR}')
PTH

EXTRA_MOUNTS="${WS}:${WS},${SWE_BASE}/swe_e2e_corrected:${SWE_BASE}/swe_e2e_corrected,${MINE}/runs:${MINE}/runs,${MINE}/persistent_cache:${MINE}/persistent_cache,${HF_HOME}:${HF_HOME},${WORKTREE}/examples/run_grpo_single_controller.py:/opt/nemo-rl/examples/run_grpo_single_controller.py"
if [[ "${ENGINE}" == "minf" ]]; then
  [[ -f "${MLM_TREE}/megatron/core/resharding/refit.py" ]] || { echo "ERROR: Megatron-LM tree missing at ${MLM_TREE}" >&2; exit 1; }
  # This tree is main-tracked (6a366090 + our patches), not the old fork, so
  # the Mamba post-refit fix lives upstream under its own name rather than
  # _run_post_refit_hooks. Sanity-check on our own patch commits being
  # present instead -- confirms the mount is this branch, not something stale.
  # 9900d8be5 is our text-only compact_prompt_token_ids robustness fix on the
  # mlm-main915-latest branch (base 6a366090, the MLM commit nemo_rl main's
  # own Megatron-Bridge pin resolves to as of 2026-09-27).
  timeout 10 git -C "${MLM_TREE}" merge-base --is-ancestor 9900d8be5 HEAD 2>/dev/null || {
    echo "ERROR: ${MLM_TREE} is not on the expected mlm-main915-latest branch (missing the multi-EOS patch)" >&2; exit 1; }
  # The container's own baked-in megatron.bridge predates nemo_rl main's pin
  # (1f8873bb) and 422s importing megatron.core APIs 6a366090 added (job
  # 4057075: ModuleNotFoundError on dist_checkpointing.strategies.async_utils).
  # Mount a real Bridge checkout at that exact commit, not just Megatron-LM.
  [[ -d "${BRIDGE_TREE}/src/megatron/bridge" ]] || { echo "ERROR: Megatron-Bridge tree missing at ${BRIDGE_TREE}" >&2; exit 1; }
  timeout 10 git -C "${BRIDGE_TREE}" merge-base --is-ancestor 1f8873bb00a8ddf3af811649f0a7efdb2363570c HEAD 2>/dev/null || {
    echo "ERROR: ${BRIDGE_TREE} is not on the expected commit (1f8873bb, nemo_rl main's pin)" >&2; exit 1; }
  EXTRA_MOUNTS="${EXTRA_MOUNTS},${BRIDGE_TREE}:${BRIDGE_MOUNT_TARGET},${MLM_TREE}:${MLM_MOUNT_TARGET},${PARITY_PTH}:${MW_SITE_PACKAGES}/zz_parity_paths.pth"
fi
export EXTRA_MOUNTS

cd "${WORKTREE}"
echo "engine: ${ENGINE} (vLLM numerical parity)  config: ${CONFIG_PATH}"
echo "nemo_rl: $(git rev-parse --short HEAD) ($(git branch --show-current)) dirty=$(git status --porcelain | wc -l)"
if [[ "${ENGINE}" == "minf" ]]; then
  echo "Megatron-LM mount: $(git -C "${MLM_TREE}" rev-parse --short HEAD 2>/dev/null || echo unknown) -> ${MLM_MOUNT_TARGET}"
  echo "prefix_caching: ${ENABLE_PREFIX_CACHING:-true}"
  echo "parity site: ${PARITY_SITE} via ${MW_SITE_PACKAGES}/zz_parity_paths.pth  ptxas: ${PTXAS_PATH}  torch extensions: ${TORCH_EXT_DIR}"
else
  echo "Megatron-LM: container tree (no mount)"
fi
echo "container: ${CONTAINER}"
echo "wandb: ${WANDB_PROJ} / ${EXP_NAME}"
echo "env.should_mask_flagged_samples: $(awk '/^  should_mask_flagged_samples:/ {print $2; f=1} END {if (!f) print "unset(defaults true)"}' examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_common.yaml)"
echo "dumps: ${RESULTS_DIR}/dumps  gym results: ${RESULTS_DIR}/gym_results"
exec bash examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh swe "${SHAPE_OVERRIDES[@]}" "${DUMP_OVERRIDES[@]}" "${ENGINE_OVERRIDES[@]}" "$@"

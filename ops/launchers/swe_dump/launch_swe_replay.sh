#!/usr/bin/env bash
# Sibling of launch_swe_dump.sh (copied, not edited in place, to avoid touching
# the shared launcher other live chains depend on) for driving
# examples/nemo_gym/nemotron-3.5-nano/replay_train.py through the identical
# CMH SWE-E2E v2 multi-node bootstrap (same container, mounts, shape logic,
# nano35_launch.sh call) instead of run_grpo_single_controller.py. No
# generation engine is ever constructed by replay_train.py itself; ENGINE
# still selects which yaml config chain (vllm/minf) to load and therefore the
# node shape, but the generation-node pool it allocates sits idle (see
# replay_train.py's own docstring -- nano35_launch.sh hard-requires
# NUM_GEN_NODES > 0 with no bypass).
#
# Replay-specific inputs are Hydra overrides forwarded as trailing arguments,
# same mechanism as +checkpointing.load_replay_buffer=false:
#   +replay.source_run=/path/to/source/run/dir
#   +replay.steps=1:10
#
#   DRY_RUN=1 SMOKE=1 ENGINE=vllm bash launch_swe_replay.sh +replay.source_run=... +replay.steps=1:2   # render only
#   SMOKE=1 ENGINE=vllm bash launch_swe_replay.sh +replay.source_run=... +replay.steps=1:2              # 16-node smoke
#
# SEED_CHECKPOINT=/path/to/step_N makes the launcher refuse to submit unless a
# copy of that checkpoint already sits in the run's checkpoints/ directory
# (relevant for a future +replay.weights_path= run seeded from a real
# checkpoint rather than a fresh HF import).
# Trailing arguments are forwarded as Hydra overrides.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${WS}/nemo_rl"
MLM_TREE="${WS}/Megatron-LM"
MLM_MOUNT_TARGET=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM

SWE_BASE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra
MINE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby
MY_HF_HOME=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/rkirby/hf_home

: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
ENGINE="${ENGINE:?ENGINE=vllm or ENGINE=minf is required}"
case "${ENGINE}" in
  vllm|minf) ;;
  *) echo "ERROR: ENGINE must be vllm or minf (got ${ENGINE})" >&2; exit 1 ;;
esac
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

export MODEL_PATH="${SWE_BASE}/swe_e2e_corrected/base_model/step_18/hf"
export TRAIN_PATH="${SWE_BASE}/swe_e2e_corrected/data/large_root_cause_curriculum_with_mercor_ots_plus_singlefile_swerebench_overlap_fix.jsonl"
export VAL_PATH="${SWE_BASE}/swe_e2e_corrected/data/swe_public_datasets_val_swebench.jsonl"
export CONTAINER="${SWE_BASE}/containers/rl-gym.63635108-zstd.sqsh"
export SANDBOX_CONTAINER="${SWE_BASE}/containers/nemo-skills-sandbox-no-sync.sqsh"
export SIF_DIR="${SWE_BASE}/swe_e2e_corrected/sif"
export HF_HOME="${MY_HF_HOME}"

export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_ACCOUNT=nemotron_sw_post
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=140
export NUM_GYM_NODES=0
export SEGMENT_SIZE=16

export USE_SNAPSHOT=0
export NRL_TQ_SKIP_RUNTIME_ENV_INSTALL=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_IGNORE_VERSION_MISMATCH=1
export TRAIN_ENTRYPOINT='--no-sync ./examples/nemo_gym/nemotron-3.5-nano/replay_train.py'

export WANDB_PROJ="${WANDB_PROJ:-ultra-v3-swe-e2e-convergence}"
export SLURM_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"data_loading","description":"Async GRPO SWE: train GPUs idle during long agentic rollout collection (can exceed 3h) and per-step validation"}}'

if [[ "${ENGINE}" == "minf" ]]; then
  export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_dump_minf.yaml
  export NVSHMEM_MAX_CTAS="${NVSHMEM_MAX_CTAS:-2}"
  export NRL_MINF_SAMPLING_BACKEND="${NRL_MINF_SAMPLING_BACKEND:-torch}"
  export NRL_MINF_LOGPROBS_MODE="${NRL_MINF_LOGPROBS_MODE:-raw_logprobs}"
  MINF_OVERRIDES=(
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
    policy.generation.mcore_generation_config.enable_prefix_caching=false
  )
  FULL_EXP_NAME="nano35-swe-v2-replay-minf_dump-${RUN_DATE}"
  SMOKE_EXP_NAME="nano35-swe-v2-replay-minf_dump-smoke-16n"
else
  MINF_OVERRIDES=()
  export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_dump_vllm.yaml
  FULL_EXP_NAME="nano35-swe-v2-replay-vllm_dump-${RUN_DATE}"
  SMOKE_EXP_NAME="nano35-swe-v2-replay-vllm_dump-smoke-16n"
fi

SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  export EXP_NAME="${EXP_NAME:-${SMOKE_EXP_NAME}}"
  export SLURM_QOS="${SLURM_QOS:-short}"
  export WALLTIME="${WALLTIME:-2:00:00}"
  export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-8}"
  export NUM_GEN_NODES="${NUM_GEN_NODES:-8}"
  export NRL_MAX_STEPS="${NRL_MAX_STEPS:-2}"
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
  )
else
  export EXP_NAME="${EXP_NAME:-${FULL_EXP_NAME}}"
  export SLURM_QOS="${SLURM_QOS:-normal}"
  export WALLTIME="${WALLTIME:-4:00:00}"
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

EXTRA_MOUNTS="${WS}:${WS},${SWE_BASE}/swe_e2e_corrected:${SWE_BASE}/swe_e2e_corrected,${MINE}/runs:${MINE}/runs,${MINE}/persistent_cache:${MINE}/persistent_cache,${HF_HOME}:${HF_HOME}"
if [[ "${ENGINE}" == "minf" ]]; then
  [[ -f "${MLM_TREE}/megatron/core/resharding/refit.py" ]] || { echo "ERROR: Megatron-LM tree missing at ${MLM_TREE}" >&2; exit 1; }
  grep -q "_run_post_refit_hooks" "${MLM_TREE}/megatron/core/resharding/refit.py" || { echo "ERROR: ${MLM_TREE} lacks the Mamba post-refit cache refresh" >&2; exit 1; }
  EXTRA_MOUNTS="${EXTRA_MOUNTS},${MLM_TREE}:${MLM_MOUNT_TARGET}"
fi
export EXTRA_MOUNTS

cd "${WORKTREE}"
echo "engine: ${ENGINE}  config: ${CONFIG_PATH}"
echo "nemo_rl: $(git rev-parse --short HEAD) ($(git branch --show-current)) dirty=$(git status --porcelain | wc -l)"
if [[ "${ENGINE}" == "minf" ]]; then
  echo "Megatron-LM mount: $(git -C "${MLM_TREE}" rev-parse --short HEAD 2>/dev/null || echo unknown) -> ${MLM_MOUNT_TARGET}"
else
  echo "Megatron-LM: container tree (no mount)"
fi
echo "dumps: ${RESULTS_DIR}/dumps  gym results: ${RESULTS_DIR}/gym_results"
exec bash examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh swe "${SHAPE_OVERRIDES[@]}" "${DUMP_OVERRIDES[@]}" "${MINF_OVERRIDES[@]}" "$@"

#!/usr/bin/env bash
# Launch the CMH SWE-E2E v2 recipe with Megatron in-engine inference (MINF)
# from the rkirby/swe-v2-minf checkout, following the CMH-1 SWE-E2E v2 64-node
# runbook. SMOKE=1 shrinks it to 8 train + 8 generation nodes, 2 steps, short QOS.
#
#   DRY_RUN=1 SMOKE=1 bash launch_swe_minf.sh    # render only
#   SMOKE=1 bash launch_swe_minf.sh              # 16-node wiring test
#   bash launch_swe_minf.sh                      # 32 + 32 nodes, runbook shape
#
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

export MODEL_PATH="${SWE_BASE}/swe_e2e_corrected/base_model/step_18/hf"
export TRAIN_PATH="${SWE_BASE}/swe_e2e_corrected/data/large_root_cause_curriculum_with_mercor_ots_plus_singlefile_swerebench_overlap_fix.jsonl"
export VAL_PATH="${SWE_BASE}/swe_e2e_corrected/data/swe_public_datasets_val_swebench.jsonl"
export CONTAINER="${SWE_BASE}/containers/rl-gym.63635108-zstd.sqsh"
export SANDBOX_CONTAINER="${SWE_BASE}/containers/nemo-skills-sandbox-no-sync.sqsh"
export SIF_DIR="${SWE_BASE}/swe_e2e_corrected/sif"
export HF_HOME="${MY_HF_HOME}"

export SLURM_PARTITION=batch
export SLURM_ACCOUNT=nemotron_sw_post
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=140
export NUM_GYM_NODES=0
export SEGMENT_SIZE=16

export USE_SNAPSHOT=0
export NRL_TQ_SKIP_RUNTIME_ENV_INSTALL=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_IGNORE_VERSION_MISMATCH=1
export TRAIN_ENTRYPOINT='--no-sync ./examples/run_grpo_single_controller.py'
export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_minf.yaml

export NVSHMEM_MAX_CTAS="${NVSHMEM_MAX_CTAS:-2}"
export NRL_MINF_SAMPLING_BACKEND="${NRL_MINF_SAMPLING_BACKEND:-torch}"
export NRL_MINF_LOGPROBS_MODE="${NRL_MINF_LOGPROBS_MODE:-raw_logprobs}"

export WANDB_PROJ="${WANDB_PROJ:-ultra-v3-swe-e2e-convergence}"
export SLURM_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"data_loading","description":"Async GRPO SWE: train GPUs idle during long agentic rollout collection (can exceed 3h) and per-step validation"}}'

SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  export EXP_NAME="${EXP_NAME:-nano35-swe-v2-minf-smoke-16n}"
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
  export EXP_NAME="${EXP_NAME:-nano35-swe-v2-stream128-inorder1-cmh-64n-minf}"
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

export EXTRA_MOUNTS="${WS}:${WS},${MLM_TREE}:${MLM_MOUNT_TARGET},${SWE_BASE}/swe_e2e_corrected:${SWE_BASE}/swe_e2e_corrected,${MINE}/runs:${MINE}/runs,${MINE}/persistent_cache:${MINE}/persistent_cache,${HF_HOME}:${HF_HOME}"

[[ -f "${MLM_TREE}/megatron/core/resharding/refit.py" ]] || { echo "ERROR: Megatron-LM tree missing at ${MLM_TREE}" >&2; exit 1; }
grep -q "_run_post_refit_hooks" "${MLM_TREE}/megatron/core/resharding/refit.py" || { echo "ERROR: ${MLM_TREE} lacks the Mamba post-refit cache refresh" >&2; exit 1; }

cd "${WORKTREE}"
echo "nemo_rl: $(git rev-parse --short HEAD) ($(git branch --show-current)) dirty=$(git status --porcelain | wc -l)"
echo "Megatron-LM mount: $(git -C "${MLM_TREE}" rev-parse --short HEAD 2>/dev/null || echo unknown) -> ${MLM_MOUNT_TARGET}"
exec bash examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh swe "${SHAPE_OVERRIDES[@]}" policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false policy.generation.mcore_generation_config.enable_prefix_caching=false "$@"

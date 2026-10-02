#!/usr/bin/env bash
# Launch the CMH SWE-E2E v2 recipe on Megatron in-engine inference running the
# vLLM numerical-parity adapter (arm V). Same shape, data, model and dumps as
# chain P'''' (MINF from scratch, prefix cache kept across refits); the only
# change is the audited parity profile from
# Megatron-LM/docs/inference/vllm_numerical_parity.md.
#
#   DRY_RUN=1 SMOKE=1 bash launch_swe_vllm_parity.sh   # render only
#   SMOKE=1 bash launch_swe_vllm_parity.sh             # 16-node wiring test
#   bash launch_swe_vllm_parity.sh                     # 32 + 32 nodes
#
# Trailing arguments are forwarded as Hydra overrides.
set -euo pipefail

WS=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_vllm_parity
WORKTREE="${WS}/nemo_rl"
MLM_TREE="${WS}/Megatron-LM"
MLM_MOUNT_TARGET=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
PARITY_SITE="${WS}/parity_site"
PARITY_PTH="${WS}/parity_paths.pth"
MW_SITE_PACKAGES=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages

SWE_BASE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra
MINE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby
MY_HF_HOME=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/rkirby/hf_home

: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
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
export TRAIN_ENTRYPOINT='--no-sync ./examples/run_grpo_single_controller.py'

export WANDB_PROJ="${WANDB_PROJ:-ultra-v3-swe-e2e-convergence}"
export SLURM_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"data_loading","description":"Async GRPO SWE: train GPUs idle during long agentic rollout collection (can exceed 3h) and per-step validation"}}'

NCCL_RESHARD="${NCCL_RESHARD:-0}"
if [[ "${NCCL_RESHARD}" == "1" ]]; then
  # The container's Bridge predates the local-view / streaming-import APIs.
  # Mount patched Python sources only, preserving the separate parity MCore mount.
  BRIDGE_TREE="${BRIDGE_TREE:-${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}"
  [[ -f "${BRIDGE_TREE}/src/megatron/bridge/models/conversion/param_mapping.py" ]] || {
    echo "ERROR: BRIDGE_TREE must point to the patched Megatron-Bridge checkout" >&2; exit 1; }
  grep -q 'def local_hf_param_specs' "${BRIDGE_TREE}/src/megatron/bridge/models/conversion/param_mapping.py" || {
    echo "ERROR: apply the Bridge refit backport to BRIDGE_TREE first" >&2; exit 1; }
  export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_parity_minf_nccl_reshard.yaml
else
  export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_parity_minf.yaml
fi
export NVSHMEM_MAX_CTAS="${NVSHMEM_MAX_CTAS:-2}"
export NRL_MINF_SAMPLING_BACKEND="${NRL_MINF_SAMPLING_BACKEND:-torch}"
export NRL_MINF_LOGPROBS_MODE="${NRL_MINF_LOGPROBS_MODE:-raw_logprobs}"

PTXAS_PATH="${PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
TORCH_EXT_DIR="${TORCH_EXT_DIR:-${WS}/torch_extensions}"
mkdir -p "${TORCH_EXT_DIR}"

MINF_OVERRIDES=(
  policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
  policy.generation.mcore_generation_config.enable_prefix_caching=true
)
FULL_EXP_NAME="nano35-swe-v2-from0-parity-minf-${RUN_DATE}"
SMOKE_EXP_NAME="nano35-swe-v2-parity-minf-smoke-16n"

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

DUMP_OVERRIDES=(
  "async_rl.dump.dir=${RESULTS_DIR}/dumps"
  "+env.nemo_gym.results_dir=${RESULTS_DIR}/gym_results"
)
mkdir -p "${RESULTS_DIR}/dumps" "${RESULTS_DIR}/gym_results"

# Standing rule (rkirby, 2026-09-28): every resume skips the replay-buffer
# restore, regardless of trigger (wall timeout, Slurm requeue, manual). A
# mid-collection requeue that restored the default buffer mixed pre-crash
# in-flight rollouts into the wrong step and shifted chain V3's prompt
# curriculum ~2 blocks ahead of V/V2 (2026-09-28). No-op on a fresh start.
RESUME_OVERRIDES=(
  "+checkpointing.load_replay_buffer=false"
)

[[ -f "${MLM_TREE}/megatron/core/inference/vllm_parity.py" ]] || {
  echo "ERROR: ${MLM_TREE} is not on the parity branch" >&2; exit 1; }
grep -q "_run_post_refit_hooks" "${MLM_TREE}/megatron/core/resharding/refit.py" || {
  echo "ERROR: ${MLM_TREE} lacks the Mamba post-refit cache refresh" >&2; exit 1; }
[[ -d "${PARITY_SITE}/quack" && -d "${PARITY_SITE}/nvidia_cutlass_dsl/python_packages/cutlass" ]] || {
  echo "ERROR: ${PARITY_SITE} is missing quack / cutlass; run tools/stage_parity_site.sbatch" >&2; exit 1; }

# A .pth in the Megatron policy-worker venv reaches every worker interpreter
# whatever Ray does with runtime_env, and touches no other venv in the image.
cat > "${PARITY_PTH}" <<PTH
${PARITY_SITE}
${PARITY_SITE}/nvidia_cutlass_dsl/python_packages
import os; os.environ.setdefault('TRITON_PTXAS_BLACKWELL_PATH', '${PTXAS_PATH}'); os.environ.setdefault('TORCH_EXTENSIONS_DIR', '${TORCH_EXT_DIR}')
PTH

EXTRA_MOUNTS="${WS}:${WS}"
EXTRA_MOUNTS="${EXTRA_MOUNTS},${SWE_BASE}/swe_e2e_corrected:${SWE_BASE}/swe_e2e_corrected"
EXTRA_MOUNTS="${EXTRA_MOUNTS},${MINE}/runs:${MINE}/runs"
EXTRA_MOUNTS="${EXTRA_MOUNTS},${MINE}/persistent_cache:${MINE}/persistent_cache"
EXTRA_MOUNTS="${EXTRA_MOUNTS},${HF_HOME}:${HF_HOME}"
EXTRA_MOUNTS="${EXTRA_MOUNTS},${MLM_TREE}:${MLM_MOUNT_TARGET}"
if [[ "${NCCL_RESHARD}" == "1" ]]; then
  EXTRA_MOUNTS="${EXTRA_MOUNTS},${BRIDGE_TREE}/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src"
fi
EXTRA_MOUNTS="${EXTRA_MOUNTS},${WORKTREE}/nemo_rl/models/generation/megatron/megatron_worker.py:/opt/nemo-rl/nemo_rl/models/generation/megatron/megatron_worker.py"
EXTRA_MOUNTS="${EXTRA_MOUNTS},${WORKTREE}/nemo_rl/models/generation/megatron/config.py:/opt/nemo-rl/nemo_rl/models/generation/megatron/config.py"
EXTRA_MOUNTS="${EXTRA_MOUNTS},${PARITY_PTH}:${MW_SITE_PACKAGES}/zz_parity_paths.pth"
export EXTRA_MOUNTS

cd "${WORKTREE}"
echo "engine: minf (vLLM parity)  config: ${CONFIG_PATH}"
echo "nemo_rl: $(git rev-parse --short HEAD) ($(git branch --show-current)) dirty=$(git status --porcelain | wc -l)"
echo "Megatron-LM mount: $(git -C "${MLM_TREE}" rev-parse --short HEAD) ($(git -C "${MLM_TREE}" branch --show-current)) -> ${MLM_MOUNT_TARGET}"
echo "parity site: ${PARITY_SITE} via ${MW_SITE_PACKAGES}/zz_parity_paths.pth"
echo "ptxas: ${PTXAS_PATH}  torch extensions: ${TORCH_EXT_DIR}"
echo "dumps: ${RESULTS_DIR}/dumps  gym results: ${RESULTS_DIR}/gym_results"
exec bash examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh swe "${SHAPE_OVERRIDES[@]}" "${DUMP_OVERRIDES[@]}" "${MINF_OVERRIDES[@]}" "${RESUME_OVERRIDES[@]}" "$@"

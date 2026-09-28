#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_minf/tools/logs/%x_%A_%a.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_minf/tools/logs/%x_%A_%a.err

# Submit with the job name "hf-export-<EXP_NAME>" and an array over the steps:
#   env -u TMPDIR sbatch -J hf-export-nano35-swe-v2-stream128-inorder1-cmh-64n-minf --array=5,10,15,20,25,30,35 export_checkpoint_hf_v2.sbatch
# The prefix keeps the export out of the training chain's singleton dependency,
# which keys on the bare EXP_NAME. Logs land in logs/<jobname>_<A>_<a>.

set -euo pipefail

readonly STEP="${SLURM_ARRAY_TASK_ID:?submit this file as a Slurm array}"
readonly EXP_NAME="${SLURM_JOB_NAME#hf-export-}"
readonly MINE="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby"
readonly SWE_BASE="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra"
readonly WS="${MINE}/workspaces/swe_minf"
readonly RUN_ROOT="${MINE}/runs/${EXP_NAME}/checkpoints"
readonly CONTAINER="${EXPORT_CONTAINER:-${SWE_BASE}/containers/rl-gym.63635108-zstd.sqsh}"
readonly CHECKPOINT="${RUN_ROOT}/step_${STEP}"
readonly BASE_HF="${EXPORT_BASE_HF:-${SWE_BASE}/swe_e2e_corrected/base_model/step_18/hf}"
readonly OUTPUT_HF="${CHECKPOINT}/hf"
readonly OUTPUT_TMP="${CHECKPOINT}/hf.exporting.${SLURM_ARRAY_JOB_ID}_${STEP}"
readonly HF_HOME_DIR="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/rkirby/hf_home"
readonly MLM_MOUNT_TARGET="/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

if [[ -f "${OUTPUT_HF}/model.safetensors.index.json" ]]; then
  echo "HF export already complete: ${OUTPUT_HF}"
  exit 0
fi
if [[ -e "${OUTPUT_HF}" || -e "${OUTPUT_TMP}" ]]; then
  echo "ERROR: refusing to overwrite incomplete output: ${OUTPUT_HF} or ${OUTPUT_TMP}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}/config.yaml" || ! -d "${CHECKPOINT}/policy/weights/iter_0000000" ]]; then
  echo "ERROR: incomplete Megatron checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${WS}/Megatron-LM/megatron/core/resharding/refit.py" ]]; then
  echo "ERROR: Megatron-LM tree missing at ${WS}/Megatron-LM" >&2
  exit 1
fi

MOUNTS="${WS}/nemo_rl/nemo_rl:/opt/nemo-rl/nemo_rl"
MOUNTS+=",${WS}/nemo_rl/examples/converters:/opt/nemo-rl/examples/converters"
MOUNTS+=",${WS}/Megatron-LM:${MLM_MOUNT_TARGET}"
MOUNTS+=",${MINE}/runs:${MINE}/runs"
MOUNTS+=",${SWE_BASE}/swe_e2e_corrected:${SWE_BASE}/swe_e2e_corrected"
MOUNTS+=",${HF_HOME_DIR}:${HF_HOME_DIR}"

echo "Exporting ${CHECKPOINT} to ${OUTPUT_HF} (container ${CONTAINER})"
echo "nemo_rl: $(git -C "${WS}/nemo_rl" rev-parse --short HEAD)  Megatron-LM: $(git -C "${WS}/Megatron-LM" rev-parse --short HEAD)"

srun --nodes=1 --ntasks=1 \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" \
  --export=ALL,HF_HOME="${HF_HOME_DIR}",NRL_IGNORE_VERSION_MISMATCH=1,PYTHONPATH=/opt/nemo-rl,CHECKPOINT="${CHECKPOINT}",OUTPUT_TMP="${OUTPUT_TMP}" \
  bash -c '
    set -euo pipefail
    cd /opt/nemo-rl
    /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
      examples/converters/convert_megatron_to_hf.py \
      --config="${CHECKPOINT}/config.yaml" \
      --megatron-ckpt-path="${CHECKPOINT}/policy/weights/iter_0000000" \
      --hf-ckpt-path="${OUTPUT_TMP}"
  '

for file in \
  __init__.py \
  chat_template.jinja \
  configuration_nemotron_h.py \
  generation_config.json \
  modeling_nemotron_h.py \
  special_tokens_map.json \
  tokenizer.json \
  tokenizer_config.json \
  ultra_v3_reasoning_parser.py; do
  if [[ -f "${BASE_HF}/${file}" && ! -e "${OUTPUT_TMP}/${file}" ]]; then
    cp "${BASE_HF}/${file}" "${OUTPUT_TMP}/${file}"
  fi
done

test -f "${OUTPUT_TMP}/config.json"
test -f "${OUTPUT_TMP}/model.safetensors.index.json"
test -f "${OUTPUT_TMP}/tokenizer.json"

# The converter writes shards owner-only; open them to the group before publishing.
chmod -R g+rX "${OUTPUT_TMP}"
mv "${OUTPUT_TMP}" "${OUTPUT_HF}"
echo "HF export complete: ${OUTPUT_HF}"

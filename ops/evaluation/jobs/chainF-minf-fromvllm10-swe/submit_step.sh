#!/usr/bin/env bash
# Usage: ENV_FILE=/path/to/oci-eval.env ./submit_step.sh STEP [HF_DIR]
# Submits one chain F (MINF from vLLM step 10) HF export to the pinned SWE-Bench Verified recipe, local sbatch mode.
set -euo pipefail
STEP="${1:?step}"; [[ "$STEP" =~ ^[0-9]+$ ]]
MINE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby
A=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra
JOB_DIR=$MINE/evaluation/jobs/chainF-minf-fromvllm10-swe
RUN=$MINE/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920
CHECKPOINT="${2:-$RUN/checkpoints/step_${STEP}/hf}"
MODEL_NAME="rkirby-nano35-swe-v2-stream128-inorder1-minf-fromvllm10-step-${STEP}"
OUT_ROOT=$MINE/nemo-evaluator-rundirs/nano_v35/swebench-verified-fixed
SOURCE_CONFIG=$A/nemo-evaluator-rundirs/nano_v35/swebench-verified-fixed/20260907_223032_df6becfd/full_config.yaml
VENV=$A/.efb-runner-cache/pinned-venvs/nel-next/569ff7e80669ff9e200b15666702c11994723403
UV=$HOME/.local/bin/uv
ENV_FILE="${ENV_FILE:-/lustre/fsw/portfolios/nemotron/projects/nemotron_n3_post/eval/.frontier_eval/.env}"
HOSTNAME_OVERRIDE="${NEL_CLUSTER_HOSTNAME:-}"
test -x "$UV"; test -x "$VENV/bin/nel"; test -s "$ENV_FILE"; test -s "$SOURCE_CONFIG"; test -d "$CHECKPOINT"
exec 9>"$JOB_DIR/eval-submit.lock"
flock -n 9 || { echo 'Another submission is active'; exit 1; }
test ! -e "$JOB_DIR/step_${STEP}-eval-attempted"
if grep -l -F "$CHECKPOINT" "$JOB_DIR"/step_*-eval-submission.log "$OUT_ROOT"/*/full_config.yaml 2>/dev/null; then
  echo "An existing submission references $CHECKPOINT; refusing duplicate" >&2; exit 1
fi
python3 -I "$JOB_DIR/verify_export.py" "$CHECKPOINT"
"$UV" run --quiet --no-progress --no-project --env-file "$ENV_FILE" -- \
  "$VENV/bin/python" -c 'import boto3; boto3.client("sts", region_name="us-east-2").get_caller_identity(); print("Exact launcher env AWS STS: OK")'
CFG="$JOB_DIR/chainF_step${STEP}_config.yaml"
python3 "$JOB_DIR/derive_config.py" "$SOURCE_CONFIG" "$CFG" "$CHECKPOINT" "$MODEL_NAME" "$OUT_ROOT" "$MINE/cache" "$HOSTNAME_OVERRIDE"
diff "$SOURCE_CONFIG" "$CFG" > "$JOB_DIR/step_${STEP}-config.diff" || true
export NEMO_EVALUATOR_TRUST_PRE_CMD=1 NEMO_EVALUATOR_TRUST_UNLISTED_TASKS=1
touch "$JOB_DIR/step_${STEP}-eval-attempted"
"$UV" run --quiet --no-progress --no-project --env-file "$ENV_FILE" -- \
  "$VENV/bin/nel" eval run "$CFG" 2>&1 | tee "$JOB_DIR/step_${STEP}-eval-submission.log"

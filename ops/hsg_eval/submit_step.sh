#!/usr/bin/env bash
# Usage: ARM=<jobdir name> RUN=<run dir name under runs/> ./submit_step.sh STEP [HF_DIR]
# Replays the certified swe-bench-verified-nano-3.5 config (oci-hsg, commit 81337a5b, toolchain 569ff7e8) for one HF export, local sbatch mode.
set -euo pipefail
STEP="${1:?step}"; [[ "$STEP" =~ ^[0-9]+$ ]]
ARM="${ARM:?ARM=<jobdir name, e.g. chainV-minf-from0-parity-swe>}"
RUN="${RUN:?RUN=<run dir name under runs/>}"
MINE=/lustre/fsw/portfolios/llmservice/users/rkirby
TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_DIR=$MINE/evaluation/jobs/$ARM
CHECKPOINT="${2:-$MINE/runs/$RUN/checkpoints/step_${STEP}/hf}"
MODEL_NAME="rkirby-${ARM}-step-${STEP}"
OUT_ROOT=$MINE/nemo-evaluator-rundirs/nano_v35/swebench-verified-fixed
SOURCE_CONFIG=$TEMPLATE/source_full_config_20260918_115143_3acfec8b.yaml
VENV=$MINE/.efb-runner-cache/pinned-venvs/nel-next/569ff7e80669ff9e200b15666702c11994723403
UV=$HOME/.local/bin/uv
ENV_FILE="${ENV_FILE:-/lustre/fsw/portfolios/nemotron/projects/nemotron_n3_post/eval/.frontier_eval/.env}"
HOSTNAME_OVERRIDE="${NEL_CLUSTER_HOSTNAME:-}"
test -x "$UV"; test -x "$VENV/bin/nel"; test -s "$ENV_FILE"; test -s "$SOURCE_CONFIG"; test -d "$CHECKPOINT"
mkdir -p "$JOB_DIR" "$MINE/cache/huggingface" "$MINE/cache/vllm" "$OUT_ROOT"
exec 9>"$JOB_DIR/eval-submit.lock"
flock -n 9 || { echo 'Another submission is active'; exit 1; }
test ! -e "$JOB_DIR/step_${STEP}-eval-attempted"
if grep -l -F "$CHECKPOINT" "$JOB_DIR"/step_*-eval-submission.log "$OUT_ROOT"/*/full_config.yaml 2>/dev/null; then
  echo "An existing submission references $CHECKPOINT; refusing duplicate" >&2; exit 1
fi
python3 -I "$TEMPLATE/verify_export.py" "$CHECKPOINT"
"$UV" run --quiet --no-progress --no-project --env-file "$ENV_FILE" -- \
  "$VENV/bin/python" -c 'import boto3; boto3.client("sts", region_name="us-east-2").get_caller_identity(); print("Exact launcher env AWS STS: OK")'
CFG="$JOB_DIR/${ARM}_step${STEP}_config.yaml"
python3 "$TEMPLATE/derive_config.py" "$SOURCE_CONFIG" "$CFG" "$CHECKPOINT" "$MODEL_NAME" "$OUT_ROOT" "$MINE/cache" "$ARM" "$HOSTNAME_OVERRIDE"
diff "$SOURCE_CONFIG" "$CFG" > "$JOB_DIR/step_${STEP}-config.diff" || true
export NEMO_EVALUATOR_TRUST_PRE_CMD=1 NEMO_EVALUATOR_TRUST_UNLISTED_TASKS=1
touch "$JOB_DIR/step_${STEP}-eval-attempted"
"$UV" run --quiet --no-progress --no-project --env-file "$ENV_FILE" -- \
  "$VENV/bin/nel" eval run "$CFG" 2>&1 | tee "$JOB_DIR/step_${STEP}-eval-submission.log"

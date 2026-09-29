#!/usr/bin/env bash
# Usage: ./merge_step.sh RUN_DIR   -- merges a finished sharded run into RUN_DIR/report.md (idempotent: skips if report.md exists)
set -euo pipefail
RUN_DIR="${1:?run dir}"
MINE=/lustre/fsw/portfolios/llmservice/users/rkirby
VENV=$MINE/.efb-runner-cache/pinned-venvs/nel-next/569ff7e80669ff9e200b15666702c11994723403
ENV_FILE="${ENV_FILE:-/lustre/fsw/portfolios/nemotron/projects/nemotron_n3_post/eval/.frontier_eval/.env}"
test -d "$RUN_DIR"; test -s "$ENV_FILE"
[ "$(find "$RUN_DIR" -maxdepth 2 -name .shard_done | wc -l)" -eq 10 ] || { echo "not all 10 shards done"; exit 1; }
if [ -s "$RUN_DIR/report.md" ]; then echo "report already present: $RUN_DIR/report.md"; cat "$RUN_DIR/report.md"; exit 0; fi
cd /tmp && "$HOME/.local/bin/uv" run --quiet --no-progress --no-project --env-file "$ENV_FILE" -- "$VENV/bin/nel" eval merge "$RUN_DIR"
test -s "$RUN_DIR/report.md" && cat "$RUN_DIR/report.md"

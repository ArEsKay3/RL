#!/bin/bash
# Resolve a job's live SingleController log from its job id (HSG run root). Never uses runs/latest,
# which is re-pointed at submit time and goes stale the moment a continuation is queued.
R=/lustre/fsw/portfolios/llmservice/users/rkirby/runs
jid=$1
exp=$(timeout 30 squeue -j "$jid" -h -o "%j" 2>/dev/null)
[ -z "$exp" ] && exp=$(timeout 30 sacct -j "$jid" -n -X -o JobName%80 2>/dev/null | head -1 | tr -d ' ')
ld=$(ls -dt "$R/$exp/ray_logs/${jid}-logs" "$R/$exp/ray_logs/${jid}-"[0-9]*"-logs" 2>/dev/null | head -1)
[ -z "$ld" ] && { echo "NO_LOGDIR for $jid ($exp)" >&2; exit 1; }
c="$(dirname "$(readlink -f "$0")")/sc_$(basename "$ld").path"
if [ -s "$c" ] && [ -f "$(cat "$c")" ]; then cat "$c"; exit 0; fi
sess=$(ls -d "$ld"/ray/session_* 2>/dev/null | sort | tail -1)
sc=$(grep -lm1 "SingleControllerActor" "$sess"/logs/worker-*.out 2>/dev/null | head -1)
[ -z "$sc" ] && { echo "NO_SC in $ld" >&2; exit 1; }
echo "$sc" > "$c"; echo "$sc"

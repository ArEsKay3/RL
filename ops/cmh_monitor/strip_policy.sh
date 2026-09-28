#!/bin/bash
# Delete checkpoints/step_N/policy only where step_N/hf exists and step_N is NOT the
# highest-numbered checkpoint of its arm. Restricted to four explicitly named run dirs.
R=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs
ARMS="nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925
nano35-swe-v2-from0-nvshmem-keepprefix-minf-seed1234-20260926
nano35-swe-v2-from0-noprefix-vllm-20260926
nano35-swe-v2-from0-noprefix-vllm-seed1234-20260927"
MODE=${1:-dry}
for a in $ARMS; do
  [ -d "$R/$a/checkpoints" ] || { echo "SKIP missing $a"; continue; }
  # highest-numbered checkpoint on this arm - always preserved whole
  last=$(ls -d $R/$a/checkpoints/step_* 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1)
  echo "=== $a   (keeping step_$last whole)"
  for d in $(ls -d $R/$a/checkpoints/step_* 2>/dev/null | sort -t_ -k2 -n); do
    n=$(basename "$d" | sed 's/step_//')
    [ "$n" = "$last" ] && { echo "   step_$n  KEEP (last checkpoint)"; continue; }
    [ -d "$d/hf" ]     || { echo "   step_$n  KEEP (no hf export)"; continue; }
    [ -d "$d/policy" ] || { echo "   step_$n  already stripped"; continue; }
    case "$d" in $R/*/checkpoints/step_*) ;; *) echo "   REFUSE $d"; continue;; esac
    if [ "$MODE" = "go" ]; then
      find "$d/policy" -type f -delete 2>/dev/null; rm -rf "$d/policy" 2>/dev/null
      echo "   step_$n  STRIPPED policy (hf kept)"
    else
      echo "   step_$n  would strip policy (368G), hf kept"
    fi
  done
done

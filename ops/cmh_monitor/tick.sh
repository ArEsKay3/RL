#!/bin/bash
# Self-discovering monitor tick: no hardcoded job ids, no manual edit on segment rollover.
R=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs
CACHE=$(dirname "$(readlink -f "$0")")

arm_letter() {
  case "$1" in
    *-vllm_dump-20260918)                    echo "A (vLLM from scratch)";;
    *-minf_dump-from10-20260918)             echo "B (MINF from MINF step 10)";;
    *-minf_dump-nopg-noprefix-20260919)      echo "C (MINF from MINF step 10, no prefix cache)";;
    *-vllm_dump-from10-nopg-20260919)        echo "D (vLLM from MINF step 10)";;
    *-minf_dump-from0-20260920)              echo "E (MINF from scratch, prefix cache on)";;
    *-minf_dump-fromvllm10-20260920)         echo "F (MINF from vLLM step 10)";;
    *-minf_dump-from0-nopg-noprefix-20260920) echo "G (MINF from scratch, no prefix cache)";;
    *-minf_dump-from0-prefix-nopg-r1-20260921) echo "I (MINF from scratch, prefix cache on, replica 1)";;
    *-minf_dump-from0-prefix-nopg-r2-20260921) echo "J (MINF from scratch, prefix cache on, replica 2)";;
    *-vllm_dump-from0-seed1234-20260922)     echo "K (vLLM from scratch, seed 1234)";;
    *-minf_dump-fromk10-prefix-nopg-20260923) echo "L (MINF from chain K step_10)";;
    *-vllm_dump-from0-nopg-r1-20260923)      echo "M (vLLM from scratch, no overlap, replica 1)";;
    *-vllm_dump-from0-nopg-r2-20260923)      echo "N (vLLM from scratch, no overlap, replica 2)";;
    *-minf_dump-from0-prefix-nopg-r3-20260923) echo "O (MINF from scratch, prefix cache on, replica 3)";;
    *runGdata-8to10*)                        echo "AA (step-range replay: chain G steps 8-10 only, no mask, live elsewhere)";;
    *runGdata-1to7*)                         echo "AB (step-range replay: chain G steps 1-7 only, no mask, live elsewhere)";;
    *runMdata-8to10*)                        echo "AC (step-range replay control: chain M steps 8-10 only, no mask)";;
    *maskX*)                                 echo "X (masked replay, test: chain G data 1-10, deep-think rewarded rollouts masked)";;
    *maskY*)                                 echo "Y (masked replay, control: chain G data 1-10, random rewarded rollouts masked)";;
    *maskZ*)                                 echo "Z (masked replay, mirror: chain M data 1-10, punished deep-think rollouts masked)";;
    *-splice-minf-runMdata-to20-20260924)     echo "P (MINF on chain M data to step 20, then live)";;
    *-splice-vllm-runMdata-to20-20260924)     echo "Q (vLLM on chain M data to step 20, then live)";;
    *-splice-vllm-runGdata-to10-20260924)     echo "R (vLLM on chain G data to step 10, then live)";;
    *-splice-vllm-runGdata-to16-20260924)     echo "R-prime (vLLM on chain G data to step 16, then live)";;
    *-fromP20-lr0-minf-20260924)             echo "P-prime (MINF from chain P step_20, lr=0, frozen weights)";;
    *-fromP20-lr0-vllm-20260924)             echo "Q-prime (vLLM from chain P step_20, lr=0, frozen weights)";;
    *-fromP20-nccl-minf-20260925)            echo "P-double-prime (MINF from chain P step_20, normal lr, refit nccl)";;
    *-fromP20-nvshmem-keepprefix-minf-*)     echo "P-triple-prime (MINF from chain P step_20, nvshmem, prefix cache kept across refits)";;
    *keepprefix*minf-seed1234*)              echo "P-quadruple-prime-2 (MINF from scratch, prefix cache kept across refits, seed 1234)";;
    *from0*keepprefix*minf*|*keepprefix*from0*minf*)  echo "P-quadruple-prime (MINF from scratch, prefix cache kept across refits)";;
    *noprefix*vllm-seed1234*)                echo "Q-quadruple-prime-2 (vLLM from scratch, vLLM prefix caching OFF, seed 1234)";;
    *from0*noprefix*vllm*)                   echo "Q-quadruple-prime (vLLM from scratch, vLLM prefix caching OFF)";;
    nano35-swe-915-64n-vllm-*)               echo "S (swe_915 vLLM from scratch, masking off)";;
    nano35-swe-915-64n-minf-*)               echo "T (swe_915 MINF from scratch, masking off)";;
    nano35-swe-main915*smoke*)               echo "? (main@9-15 smoke, not a lettered arm)";;
    nano35-swe-main915*minf*)                echo "U (MINF from scratch, NeMo RL main 09-24 stack, masking off)";;
    nano35-swe-main915*vllm*)                echo "W (main@9-15 vLLM from scratch, masking off)";;
    *-parity-minf-smoke-*)                   echo "? (vLLM-parity smoke, not a lettered arm)";;
    *-from0-parity-minf-seed1234-*)          echo "V2 (MINF from scratch, vLLM-parity build, seed 1234)";;
    *-from0-parity-minf-seed4321-*)          echo "V3 (MINF from scratch, vLLM-parity build, seed 4321)";;
    *-from0-parity-minf-*)                   echo "V (MINF from scratch, vLLM-parity build)";;
    *-cmh-64n-minf)                          echo "main chain (MINF from scratch, original)";;
    *)                                       echo "? ($1)";;
  esac
}

timeout 120 squeue -u rkirby -h -o "%i|%j|%T|%M|%q|%r" 2>/dev/null > "$CACHE/sq_raw.txt"
echo "harbor=$(grep -c nel-eval-harbor "$CACHE/sq_raw.txt") held=$(grep -v nel-eval-harbor "$CACHE/sq_raw.txt" | grep -c JobHeldUser)"

grep -v nel-eval-harbor "$CACHE/sq_raw.txt" | awk -F'|' '$3=="RUNNING" && $2 !~ /^ *hf-export/' | while IFS='|' read -r jid exp st el qos rsn; do
  jid=$(echo "$jid" | tr -d ' '); el=$(echo "$el" | tr -d ' ')
  echo "=== chain $(arm_letter "$exp")  job $jid  elapsed $el"
  ld=$(ls -dt "$R/$exp/ray_logs/${jid}-logs" "$R/$exp/ray_logs/${jid}-"[0-9]*"-logs" 2>/dev/null | head -1)
  [ -z "$ld" ] && { echo "  logdir: NONE YET"; continue; }
  drv="$ld/ray-driver.log"
  # cache keyed by logdir basename: self-invalidates on rollover, never needs manual removal
  c="$CACHE/sc_$(basename "$ld").path"
  if [ -s "$c" ] && [ -f "$(cat "$c")" ]; then sc=$(cat "$c"); else
    sess=$(ls -d "$ld"/ray/session_* 2>/dev/null | sort | tail -1)
    sc=$(grep -lm1 "SingleControllerActor" "$sess"/logs/worker-*.out 2>/dev/null | head -1)
    [ -n "$sc" ] && echo "$sc" > "$c"
  fi
  st_json="$R/$exp/checkpoints/latest_checkpoint_status.json"
  ck=$(python3 -c "import json,sys;print(json.load(open('$st_json'))['last_checkpoint_step'])" 2>/dev/null || echo "?")
  echo "  checkpoint=$ck  driver=$( [ -f "$drv" ] && date -r "$drv" +%H:%M:%S || echo none )  SC=$( [ -n "$sc" ] && date -r "$sc" +%H:%M:%S || echo none )"
  if [ -f "$drv" ]; then
    clean=$(sed 's/\x1b\[[0-9;]*m//g' "$drv")
    echo "  guard: removed_engine=$(echo "$clean" | grep -c 'Coordinator: removed engine') post_process_requests=$(echo "$clean" | grep -c post_process_requests)"
  case "$exp" in *main915-64n*)
    echo "  U-health: noncontig=$(grep -c 'Non-contiguous messages found' "$drv" 2>/dev/null) compact_prompt_err=$(grep -c 'compact_prompt_token_ids' "$drv" 2>/dev/null) malformed_think=$(grep -c 'malformed_think_tag_rate' "$drv" 2>/dev/null) empty_final=$(grep -c 'empty_final_answer_rate' "$drv" 2>/dev/null) penalty_metrics=$(grep -o "'[a-z_/]*penalt[a-z_/]*': [0-9.e-]*" "$sc" 2>/dev/null | grep -v "'kl_penalty': 0.0" | grep -vc "_penalt[a-z_/]*': 0.0$") memory_only_mode=$(grep -c 'Running in memory-only mode' "$drv" 2>/dev/null)";;
  esac
    eng=$(grep -ao '| step [0-9]* | [0-9:]*' "$drv" 2>/dev/null | tail -1)
    [ -n "$eng" ] && echo "  engine: newest${eng#*|} (count $(grep -ac '| step [0-9]* | [0-9]' "$drv" 2>/dev/null))"
    echo "  exceptions: $(echo "$clean" | grep -E '^\(.*(Error|Exception):' | sed -E 's/.*\) //; s/\(([0-9]+)\)/(N)/g; s/\[repeated.*//' | sort | uniq -c | tr '\n' ';' | cut -c1-160)"
    echo "  gym: $(echo "$clean" | grep -oE 'Collecting rollouts: +[0-9]+%\|[^|]*\| *[0-9]+/16 \[[0-9:]+' | tail -3 | tr '\n' ' ')"
  fi
  if [ -n "$sc" ]; then
    echo "  resume(SC): restoring=$(grep -c 'Restoring dataloader state' "$sc") skip_replay=$(grep -c 'Skipping replay buffer restore' "$sc") regenerated=$(grep -c 'regenerated .* pending prompt' "$sc")"
  fi
  [ -n "$sc" ] && echo "  last train step: $(grep -oE 'train step [0-9]+/' "$sc" | tail -1)  failures: rollout-dump=$(grep -c '\[rollout-dump\] FAILED' "$sc") token-dump=$(grep -c '\[token-dump\] FAILED' "$sc") died=$(grep -ciE 'actor.*(died|dead)|RayActorError' "$sc")"
  echo "  newest dumps: $(ls -t --time-style=+%H:%M "$R/$exp/dumps/rollouts" -l 2>/dev/null | awk 'NR>1&&NR<4{printf "%s@%s(%s) ", $7,$6,$5}')"
  # cross-tick delta: automatic liveness verdict, no manual resampling needed
  now=$(date +%s)
  # grep -c exits 1 on a zero count, so capture plainly and normalise rather than using || echo
  dbytes=$(du -sb "$R/$exp/dumps/rollouts" 2>/dev/null | cut -f1); dbytes=${dbytes:-0}
  bars=0; [ -f "$drv" ] && bars=$(grep -c 'Collecting rollouts:' "$drv" 2>/dev/null); bars=${bars:-0}
  # replay arms emit no Gym bars, so track driver size and token-dump count as engine-agnostic signals
  dsize=$( [ -f "$drv" ] && stat -c %s "$drv" 2>/dev/null ); dsize=${dsize:-0}
  toks=$(ls "$R/$exp/dumps/token_level" 2>/dev/null | wc -l); toks=${toks:-0}
  stf="$CACHE/tickstate_$exp.txt"
  if [ -s "$stf" ]; then
    read -r p_now p_bytes p_bars p_ck p_dsize p_toks < "$stf"
    p_dsize=${p_dsize:-0}; p_toks=${p_toks:-0}
    dt=$(( now - p_now ))
    echo "  delta over ${dt}s: dump_bytes=+$(( dbytes - p_bytes )) gym_bars=+$(( bars - p_bars )) driver_bytes=+$(( dsize - p_dsize )) token_dumps=+$(( toks - p_toks )) checkpoint=${p_ck}->${ck}"
    if [ "$(( dbytes - p_bytes ))" -eq 0 ] && [ "$(( bars - p_bars ))" -eq 0 ] && [ "$(( dsize - p_dsize ))" -eq 0 ] && [ "$(( toks - p_toks ))" -eq 0 ] && [ "$ck" = "$p_ck" ] && [ "$dt" -ge 1200 ]; then
      echo "  *** NO PROGRESS on any signal for ${dt}s - investigate before any cancel ***"
    fi
  else
    echo "  delta: first observation, no baseline yet"
  fi
  echo "$now $dbytes $bars $ck $dsize $toks" > "$stf"
  echo "  rungs: $(ls -1 "$R/$exp/checkpoints" 2>/dev/null | grep '^step_' | sort -t_ -k2 -n | tr '\n' ' ')"
  case "$exp" in
    *-from0-nvshmem-keepprefix-minf-*)
      if [ -d "$R/$exp/checkpoints/step_60" ]; then
        ids=$(timeout 30 squeue -u rkirby -h -n "$exp" -o "%i" 2>/dev/null | tr "\n" " ")
        echo "  *** TRIGGER: chain P-quadruple-prime reached step_60 - STOP RULE: verify files then: scancel $ids ***"
      fi
      ;;
    *noprefix*vllm-seed1234*)
      if [ -d "$R/$exp/checkpoints/step_60" ]; then
        ids=$(timeout 30 squeue -u rkirby -h -n "$exp" -o "%i" 2>/dev/null | tr "\n" " ")
        echo "  *** TRIGGER: chain Q-quadruple-prime-2 reached step_60 - STOP RULE: verify files then: scancel $ids ***"
      fi
      ;;
    *from0*noprefix*vllm*)
      if [ -d "$R/$exp/checkpoints/step_40" ]; then
        ids=$(timeout 30 squeue -u rkirby -h -n "$exp" -o "%i" 2>/dev/null | tr "\n" " ")
        echo "  *** TRIGGER: chain Q-quadruple-prime reached step_40 - STOP RULE: verify files then: scancel $ids ***"
      fi
      ;;
    *keepprefix*minf-seed1234*)
      if [ -d "$R/$exp/checkpoints/step_60" ]; then
        ids=$(timeout 30 squeue -u rkirby -h -n "$exp" -o "%i" 2>/dev/null | tr "\n" " ")
        echo "  *** TRIGGER: chain P-quadruple-prime-2 reached step_60 - STOP RULE: verify files then: scancel $ids ***"
      fi
      ;;
    *mask[XYZ]*)
      if [ -d "$R/$exp/checkpoints/step_30" ]; then
        ids=$(timeout 30 squeue -u rkirby -h -n "$exp" -o "%i" 2>/dev/null | tr "\n" " ")
        echo "  *** TRIGGER: chain $(arm_letter "$exp") reached step_30 - STOP RULE: verify files then: scancel $ids ***"
      fi
      ;;
    *-8to10-*|*-1to7-*)
      if [ -d "$R/$exp/checkpoints/step_25" ]; then
        ids=$(timeout 30 squeue -u rkirby -h -n "$exp" -o "%i" 2>/dev/null | tr "\n" " ")
        echo "  *** TRIGGER: chain $(arm_letter "$exp") reached step_25 - STOP RULE: verify files then: scancel $ids ***"
      fi
      ;;
  esac
done

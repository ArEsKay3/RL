#!/usr/bin/env bash
# One eval tick: submit evals for verified HF exports not yet attempted, re-queue dead shards, merge finished runs, print a per-step table.
set -uo pipefail
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINE=/lustre/fsw/portfolios/llmservice/users/rkirby
STEPS="${EVAL_STEPS:-5 10 15 20 25 30 35 40 45 50 55 60}"
squeue -u rkirby -h -o "%i %T" > "$T/.sq.txt" 2>/dev/null
NSUB=0; MAXSUB="${EVAL_MAX_SUBMIT:-1}"
[ -e "$T/PAUSE_SUBMIT" ] && { MAXSUB=0; echo "submissions PAUSED ($T/PAUSE_SUBMIT)"; }
[ -e "$T/PAUSE_RESUME" ] && echo "resumes PAUSED ($T/PAUSE_RESUME)"
while IFS=$'\t' read -r ARM RUN LABEL; do
  J=$MINE/evaluation/jobs/$ARM; mkdir -p "$J"; [ -s "$J/runs.json" ] || echo '{}' > "$J/runs.json"
  echo "## chain $LABEL  [$ARM]"
  for S in $STEPS; do
    HF=$MINE/runs/$RUN/checkpoints/step_${S}/hf
    if [ -f "$HF/model.safetensors.index.json" ] && [ ! -e "$J/step_${S}-eval-attempted" ] && [ "${EVAL_SUBMIT:-1}" = 1 ] && [ "$NSUB" -lt "$MAXSUB" ]; then
      if python3 -I "$T/verify_export.py" "$HF" >/dev/null 2>&1; then
        echo "  submitting step_$S"
        ARM=$ARM RUN=$RUN "$T/submit_step.sh" "$S" > "$J/.submit_${S}.out" 2>&1 && \
          python3 "$T/record_run.py" "$J" "$S" "$HF" "rkirby-${ARM}-step-${S}" >/dev/null && { echo "  submitted step_$S"; NSUB=$((NSUB+1)); } || echo "  SUBMIT FAILED step_$S (see $J/.submit_${S}.out)"
      else
        echo "  step_$S hf present but failed verify_export"
      fi
    fi
  done
  [ -e "$T/PAUSE_RESUME" ] || python3 "$T/resume_dead_shards.py" "$J" 2>&1 | grep -E "resumed|FAILED|SUBMIT" | grep -v "resumed: 0" | sed 's/^/  resume: /'
  python3 - "$J" "$T" <<'PY'
import json, os, subprocess, sys, glob
J, T = sys.argv[1:3]
d = json.load(open(os.path.join(J, "runs.json")))
sq = dict(l.split() for l in open(os.path.join(T, ".sq.txt")) if l.strip())
print("  | step | run_id | shards done | live jobs | verified rows | report |")
print("  |---|---|---|---|---|---|")
for step in sorted(d, key=lambda s: int(s[5:])):
    r = d[step]; rd = r["run_dir"]
    done = sum(os.path.exists(f"{rd}/shard_{s}/.shard_done") for s in range(10))
    ids = set(map(str, r["job_ids"])) | {v["job_id"] for v in r.get("resume_jobs", {}).values() if v.get("job_id")}
    for s in range(10):
        p = f"{rd}/shard_{s}/.nel_job_chain"
        if os.path.exists(p): ids |= set(open(p).read().split())
    live = sum(1 for i in ids if i in sq and sq[i] in ("RUNNING", "PENDING", "COMPLETING"))
    rows = 0
    for f in glob.glob(f"{rd}/shard_*/harbor___swebench_verified_1_0/swebench-verified_1.0/verified_log.jsonl"):
        rows += sum(1 for _ in open(f, errors="ignore"))
    rep = "merged" if os.path.exists(f"{rd}/report.md") else ("MERGE NEEDED" if done == 10 else "-")
    if done == 10 and not os.path.exists(f"{rd}/report.md") and rows >= 2500:
        subprocess.run([f"{T}/merge_step.sh", rd], capture_output=True)
        rep = "merged" if os.path.exists(f"{rd}/report.md") else "merge failed"
    if os.path.exists(f"{rd}/report.md") and "result" not in r:
        out = subprocess.run([sys.executable, f"{T}/pooled_check.py", rd], capture_output=True, text=True).stdout.strip()
        try:
            pc = json.loads(out)
            r["result"] = dict(pooled=pc, report_md=open(f"{rd}/report.md").read())
            json.dump(d, open(os.path.join(J, "runs.json"), "w"), indent=1)
            rep = f"merged pass@1={pc['pass1']} rows={pc['rows']} complete={pc['complete']}"
        except Exception as e:
            rep = f"merged (pooled check error {e})"
    elif "result" in r:
        pc = r["result"]["pooled"]; rep = f"pass@1={pc['pass1']} rows={pc['rows']} complete={pc['complete']}"
    print(f"  | {step} | {r['run_id']} | {done}/10 | {live} | {rows} | {rep} |")
PY
done < "$T/arms.tsv"

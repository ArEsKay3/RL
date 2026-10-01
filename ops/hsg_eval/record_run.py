"""Record a submission in JOB_DIR/runs.json from its step_N-eval-submission.log.

Usage: record_run.py JOB_DIR STEP CHECKPOINT MODEL_NAME
"""
import datetime, json, os, re, sys

jd, step, ck, model = sys.argv[1:5]
log = open(os.path.join(jd, f"step_{step}-eval-submission.log"), errors="replace").read()
run_dirs = sorted(set(re.findall(r"(/lustre\S+/swebench-verified-fixed/(\d{8}_\d{6}_[0-9a-f]{8}))/", log)))
assert len(run_dirs) == 1, run_dirs
run_dir, run_id = run_dirs[0]
jobs = [int(j) for j in re.findall(r"Submitted batch job (\d+)", log)]
assert len(jobs) == 10, jobs
p = os.path.join(jd, "runs.json")
d = json.load(open(p)) if os.path.exists(p) else {}
d[f"step_{step}"] = dict(checkpoint=ck, model_name=model, run_id=run_id, run_dir=run_dir, job_ids=jobs,
                         submitted_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), resume_jobs={})
tmp = p + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, p)
print(json.dumps(d[f"step_{step}"]))

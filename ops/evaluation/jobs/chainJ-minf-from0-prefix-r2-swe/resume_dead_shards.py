"""Queue a resume job for every shard that has no live job and no .shard_done (re-runs the shard's own nel_eval.sbatch, which calls `nel eval run --resume`).

Usage: resume_dead_shards.py [STEP ...]   (default: all steps in runs.json, in the order 26 25 20 15 10 5)
Idempotent: shards with a resume job recorded in runs.json that is still queued/running are skipped.
"""
import glob, json, os, subprocess, sys, datetime

J = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(J, "runs.json")
d = json.load(open(P))
order = [f"step_{s}" for s in (26, 25, 20, 15, 10, 5)]
steps = [f"step_{s}" for s in sys.argv[1:]] or [s for s in order if s in d]
sq = subprocess.run(["squeue", "-u", "rkirby", "-h", "-o", "%i %t"], capture_output=True, text=True).stdout.split()
live = {sq[i] for i in range(0, len(sq), 2)}
rows = []
for step in steps:
    rd = d[step]["run_dir"]; d[step].setdefault("resume_jobs", {})
    for s in range(10):
        sd = f"{rd}/shard_{s}"
        if os.path.exists(f"{sd}/.shard_done"):
            rows.append((step, s, "done", "")); continue
        log_ids = [os.path.basename(l)[6:-4] for l in glob.glob(f"{sd}/logs/slurm-*.log")]
        chain = open(f"{sd}/.nel_job_chain").read().split() if os.path.exists(f"{sd}/.nel_job_chain") else []
        queued = []
        for l in glob.glob(f"{sd}/logs/slurm-*.log"):
            for line in open(l, errors="replace"):
                if "Auto-resume follow-up queued:" in line:
                    queued.append(line.split("queued:")[1].split()[0])
        prev_resume = d[step]["resume_jobs"].get(str(s))
        if any(j in live for j in log_ids + chain + queued) or (prev_resume and prev_resume["job_id"] in live):
            rows.append((step, s, "live", "")); continue
        r = subprocess.run(["sbatch", f"{sd}/nel_eval.sbatch"], capture_output=True, text=True)
        jid = r.stdout.strip().split()[-1] if r.returncode == 0 else ""
        d[step]["resume_jobs"][str(s)] = dict(job_id=jid, submitted_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                              error=None if r.returncode == 0 else r.stderr.strip()[:200])
        rows.append((step, s, "resumed" if jid else "SUBMIT FAILED", jid or r.stderr.strip()[:80]))
json.dump(d, open(P, "w"), indent=1)
for step in steps:
    print(f"{step}: " + ", ".join(f"s{s}:{st}{(' ' + info) if info else ''}" for stp, s, st, info in rows if stp == step))
print("resumed:", sum(1 for r in rows if r[2] == "resumed"), "live:", sum(1 for r in rows if r[2] == "live"), "done:", sum(1 for r in rows if r[2] == "done"), "failed:", sum(1 for r in rows if r[2] == "SUBMIT FAILED"))

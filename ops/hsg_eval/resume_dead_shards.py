"""Queue a resume job for every shard that has no live job and no complete .shard_done (re-runs the shard's own nel_eval.sbatch, which calls `nel eval run --resume`).

Usage: resume_dead_shards.py JOB_DIR [STEP ...]   (default: all steps in JOB_DIR/runs.json, highest first)
Idempotent: shards with a resume job recorded in runs.json that is still queued/running are skipped.
A .shard_done with fewer than 250 verified results (preempted shard) counts as incomplete.
"""
import datetime, glob, json, os, subprocess, sys

J = os.path.abspath(sys.argv[1])
P = os.path.join(J, "runs.json")
d = json.load(open(P))
steps = [f"step_{s}" for s in sys.argv[2:]] or sorted((s for s in d if s.startswith("step_")), key=lambda s: -int(s[5:]))
sq = subprocess.run(["squeue", "-u", "rkirby", "-h", "-o", "%i %t"], capture_output=True, text=True).stdout.split()
live = {sq[i] for i in range(0, len(sq), 2)}

def verified(sd):
    n = 0
    for f in glob.glob(f"{sd}/harbor___swebench_verified_1_0/*/results.jsonl"):
        n += sum(1 for _ in open(f, errors="ignore"))
    return n

rows = []
for step in steps:
    if "result" in d[step]:
        continue
    rd = d[step]["run_dir"]; d[step].setdefault("resume_jobs", {})
    for s in range(10):
        sd = f"{rd}/shard_{s}"
        if os.path.exists(f"{sd}/.shard_done") and verified(sd) >= 250:
            rows.append((step, s, "done", "")); continue
        log_ids = [os.path.basename(l)[6:-4] for l in glob.glob(f"{sd}/logs/slurm-*.log")]
        chain = open(f"{sd}/.nel_job_chain").read().split() if os.path.exists(f"{sd}/.nel_job_chain") else []
        queued = []
        for l in glob.glob(f"{sd}/logs/slurm-*.log"):
            for line in open(l, errors="replace"):
                if "Auto-resume follow-up queued:" in line:
                    queued.append(line.split("queued:")[1].split()[0])
        prev = d[step]["resume_jobs"].get(str(s))
        primary = [str(d[step]["job_ids"][s])] if len(d[step].get("job_ids", [])) == 10 else []
        known = [j for j in dict.fromkeys(primary + log_ids + chain + queued + ([prev["job_id"]] if prev and prev.get("job_id") else [])) if j]
        if any(j in live for j in known):
            rows.append((step, s, "live", "")); continue
        fresh = subprocess.run(["squeue", "-h", "-j", ",".join(known), "-o", "%i"], capture_output=True, text=True).stdout.split() if known else []
        if fresh:
            rows.append((step, s, "live", "")); continue
        r = subprocess.run(["sbatch", f"{sd}/nel_eval.sbatch"], capture_output=True, text=True)
        jid = r.stdout.strip().split()[-1] if r.returncode == 0 else ""
        rec = dict(job_id=jid, submitted_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   error=None if r.returncode == 0 else r.stderr.strip()[:200])
        if prev:
            rec["previous"] = [prev] + prev.pop("previous", [])
        d[step]["resume_jobs"][str(s)] = rec
        rows.append((step, s, "resumed" if jid else "SUBMIT FAILED", jid or r.stderr.strip()[:80]))
tmp = P + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, P)
for step in steps:
    print(f"{step}: " + ", ".join(f"s{s}:{st}{(' ' + info) if info else ''}" for stp, s, st, info in rows if stp == step))
print("resumed:", sum(1 for r in rows if r[2] == "resumed"), "live:", sum(1 for r in rows if r[2] == "live"), "done:", sum(1 for r in rows if r[2] == "done"), "failed:", sum(1 for r in rows if r[2] == "SUBMIT FAILED"))

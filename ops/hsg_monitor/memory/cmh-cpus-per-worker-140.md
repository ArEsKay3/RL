---
name: cmh-cpus-per-worker-140
description: "Root cause of every \"ray-head never starts / step creation disabled / Requested nodes are busy\" stall on CMH — tools/launch.sh hardcodes CPUS_PER_WORKER=144 (HSG node size) but CMH GPU nodes have CPUTot=140; fixed via the CLUSTER case in dolphin_convergence_common.sh"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7aa0a354-aa1b-4011-8140-8d3122eb7495
  modified: 2026-09-11T22:33:59.095Z
---

CMH (`aws-cmh-slurm-1`) GPU nodes report `CPUTot=140`. `tools/launch.sh` line ~404 sets
`CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"` (the HSG node size), which bypasses ray.sub's scontrol
auto-detection. The ray-head and ray-worker sruns then request `--cpus-per-task=144` on 140-CPU nodes, and
SLURM retries forever printing `step creation temporarily/still disabled, retrying (Requested nodes are busy)`.
Symptoms: sandbox (`--overlap`) and judge services come up fine, but no `ray-head.log`, no ray-worker step in
`sacct`, and the job idles until the GPU reaper or walltime kills it (jobs 3628529, 3632991, 3634104, 3680166).

Diagnosed 2026-09-11 on smoke job 3680166. Fix: `_default_CPUS_PER_WORKER=140` in the CMH arm of the
`case "$CLUSTER"` block in `dolphin_convergence_common.sh`, exported before launch.sh runs (HSG arm = 144).

**Why:** three days and four multi-hour allocations were spent chasing srun ordering and `--overlap`; none of
that was the cause. The reorder of sandbox after ray-head in ray.sub ([[ray-sub-sandbox-reorder-fix]]) is
harmless but was NOT the fix, and the "slow vLLM patching" theory in [[rlvr-minf-gpu-idle-reaper]] was wrong:
setup never ran because the step never launched.

**How to apply:** if a job shows "Requested nodes are busy" for ray-head/ray-worker, first compare
`scontrol show node <n> | grep CPUTot` with the `--cpus-per-task` in the sbatch .err trace. On any new cluster,
add its CPUTot to the CLUSTER case. Verify a fix landed by grepping the job's .out for
`Using user-provided CPUS_PER_WORKER=140` and `.err` for zero "step creation" lines.

---
name: ray-sub-sandbox-reorder-fix
description: "The ray.sub reorder (sandbox srun after ray-head/ray-workers) was NOT a fix for the ray-head stall on CMH; discarded 2026-09-12, backup patch at ~/workspaces/ray-sub-sandbox-reorder.patch. Root cause was CPUS_PER_WORKER=144 on 140-CPU nodes."
metadata:
  type: project
---

On 2026-09-08 jobs 3628529 and 3632991 sat in "Requested nodes are busy / step creation still disabled" and I
blamed upstream `nemo_rl/ray.sub` launching the sandbox srun (all nodes, `--overlap`) before ray-head and
ray-workers. Reordering the sruns changed nothing, and neither did adding `--overlap` to ray-head. The real
blocker was `--cpus-per-task=144` on nodes with CPUTot=140 ([[cmh-cpus-per-worker-140]]). With that fixed, the
unmodified upstream `ray.sub` (nemo_rl 2f38c4b7d) ran smokes 3683477 and 3683907 and the full minf-v2 job 3688117.

The user declined committing the reorder (2026-09-12, "Don't want those tools"); the working-tree change was
reverted with `git checkout -- ray.sub` and a copy kept at `/home/rkirby/workspaces/ray-sub-sandbox-reorder.patch`.

**Why:** stops me re-deriving the wrong theory the next time ray-head fails to start.

**How to apply:** do not reapply the patch. If ray-head never starts, first compare `CPUS_PER_WORKER` with
`scontrol show node <gpu-node> | grep CPUTot`, then check the sbatch log for the srun retry message.

---
name: workspace-layout-restructure
description: "/home/rkirby/workspaces/nemo-rl-workspace was meant to be a parent dir for multiple workstreams/clones, but the post-training pipeline repo was cloned directly into it; user wants it moved into a subdirectory at a quiet point between debug steps"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7aa0a354-aa1b-4011-8140-8d3122eb7495
  modified: 2026-09-12T02:41:31.266Z
---

**Done 2026-09-11:** the checkout now lives at `/home/rkirby/workspaces/nemo-rl-workspace/pipeline_A/`
(branch `rlvr-convergence-minf`, rebased onto origin, cluster-support commit on top). `cluster-viz/` stays at the
workspace root. All paths in other memory notes that say `nemo-rl-workspace/nemo_rl/...` or
`nemo-rl-workspace/RLVR/...` now mean `nemo-rl-workspace/pipeline_A/nemo_rl/...` etc.

The user pointed out (2026-09-11) that `/home/rkirby/workspaces/nemo-rl-workspace/` is supposed to hold several
workstreams and clones side by side, but the `post-training/pipeline` GitLab repo (with its `nemo_rl` submodule
tree) was cloned straight into that directory, so its `.git` sits at the workspace root next to unrelated things
like `cluster-viz/`.

**Why:** the directory name reflects the intended layout; a repo at the top level blocks adding other clones
cleanly and makes `git status` at the workspace root show unrelated artifacts as untracked.

**How to apply:** at a quiet moment (no running job that bind-mounts this checkout), move the pipeline checkout
into `nemo-rl-workspace/pipeline_A/` (name chosen by the user on 2026-09-11; the `_A` suffix leaves room for
`pipeline_B`, ... as parallel pipeline workstreams), and keep `cluster-viz/` and future clones as siblings.
Do NOT do this while a job launched with `USE_SNAPSHOT=0` is running: ray.sub bind-mounts `nemo_rl/` and the
Megatron-Bridge/Megatron-LM submodules from this path live, so moving it would break the run. After moving:
`git submodule status --recursive` must still be clean, the `ray-sub-sandbox-reorder-fix` patch path in memory
and the `/home/rkirby/workspaces/ray-sub-sandbox-reorder.patch` backup still apply, the Claude memory dir
(`-home-rkirby-workspaces-nemo-rl-workspace`) is keyed to the workspace root so it stays valid if sessions still
start from `nemo-rl-workspace/`. Confirm the new path with the user before moving.

---
name: pipeline-b-mr52-branch
description: "pipeline_B = the rlvr-convergence-minf-mr52 pipeline branch (fc708e1) checked out on Lustre for CMH launches; nemo_rl fork 6e20063f2 inside the submodule; needs CONTAINER=, PERSISTENT_CACHE=, HF_HOME= overrides on CMH; the branch's default image has no /opt/gym_venvs although the launcher thinks it does"
metadata:
  type: project
---

Set up 2026-09-15 at the user's request ("try out this new branch for launching experiments").

Location: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/pipeline_B`
(symlink `~/workspaces/nemo-rl-workspace/pipeline_B` -> there). Launch with cwd on the /scratch path, not via
the symlink: `nano35_rlvr_common.sh` uses `pwd -L` and refuses any PIPELINE_ROOT outside
EXTERNAL_VLLM_SHARED_ROOT=/scratch/fsw (only that root is bind-mounted; /lustre is a symlink to /scratch here).

Stack: pipeline `rlvr-convergence-minf-mr52` @ fc708e1 (upstream main + 2 recipe commits by sauramishra/rkirby);
nemo_rl submodule pointer stays on upstream 8cd44c4b7 by design; fork `rkirby/nemorl-with-minf` checked out inside it,
now at 7961ce437 = 6e20063f2 + 3 cherry-picks done 2026-09-15 at the user's request (aabb324b1 Gym pointer -> 3615823e
"isolate prompt cohort retries", 363dc6950 per-prompt identity in sync rollouts, 7961ce437 data-plane port fix #4103);
NOT PUSHED yet. Megatron-Bridge fork @ 6b43a2c9e; Megatron-LM fork @ 38bb3fd9f; Gym @ 3615823e (shallow clones fail
with "transport 'file' not allowed"; fix = `git fetch origin <sha>` then checkout).
The recipe was validated on CMH by sauramishra (jobs 3668422, 3670374, 3671057 smokes/86n; 3698210 = 86-node SC run,
~28 steps in 4 h). Built-in defaults are CMH paths (kajalj policy copy, sauramishra akamehra-mirror blend,
akamehra judges, GPUS_PER_NODE=4, aarch64 X11 mounts); `SITE_OVERLAY=` is for other sites only. ray.sub now
auto-detects CPUS_PER_WORKER, so the 140-CPU issue is moot on this branch.

**Overrides needed on CMH (verified by DRY_RUN=1 on 2026-09-15):**
- `CONTAINER=$U/rkirby/containers/nemo-rl_fd45cb8-67127697-gym.sqsh` -- the branch default
  (`sauramishra/containers/rl-gym.nightly-20260910.sqsh`) has NO /opt/gym_venvs (unsquashfs: 0 venv interpreters
  vs 39 in the NGC image), yet the launcher's USE_IMAGE_GYM proof only compares Gym commits (both fd5e84d6b) and
  picks "image /opt/gym_venvs (prebuilt)". With the default image Gym would rebuild venvs at spinup.
- `PERSISTENT_CACHE=$U/rkirby/persistent_cache` (warm uv/inductor/triton caches; default `$U/rkirby/cache` is cold).
- `HF_HOME=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/rkirby/hf_home` -- bashrc exports the
  /lustre spelling, which does not exist inside the container.
- Optional `SLURM_QOS=short` (priority 200 vs 100, MaxWall 2 h, <=64 nodes) for the 1:30 smoke.

Smoke = `launch_nano35_rlvr_v2_smoke_small.sh` (SC driver, 10 nodes: 4 train + 1 gen + 1 gym + 2 GenRM + 2 nl2bash,
2 steps, W&B project nano35-rlvr-main-tot-smoke); `MINF=1` selects rlvr_sc_minf_smoke_small.yaml.

Fork parser change (2026-09-14, Megatron-LM 38bb3fd9f): `qwen3-coder-tool` is now strict (no implicit
`<tool_call>` reasoning end, matching vLLM's separate-parser pairing); `qwen3-coder-tool-combined` opts in.
rlvr_minf.yaml still lists `qwen3-coder-tool`, so MINF now matches vLLM on the unclosed-think tool-call case.

**How to apply:** launch from the /scratch checkout with the four overrides above; keep pipeline_A for the
original `rlvr-convergence-minf` line. Related: [[workspace-layout-restructure]], [[v2-stack-needs-newer-container]],
[[gym-venvs-not-prebaked-in-nightly]].

**Smoke 2026-09-15:** job 3764514 (USE_IMAGE_GYM=1, image Gym code) was cancelled while pending after the user clarified
they want the container venvs but the checkout Gym code. Resubmitted as job 3764675 (`nano35-rlvr-v2-minf-smoke-small`,
SC + MINF, 10 nodes, QOS short) with `USE_IMAGE_GYM=0 GYM_VENV_DIR=/opt/gym_venvs`: Gym 3615823e mounted over
/opt/nemo-rl/3rdparty/Gym-workspace/Gym, NEMO_GYM_VENV_DIR=/opt/gym_venvs (editable installs resolve to the mount;
skip_venv_if_present keeps uv from re-syncing). The launcher prints a misleading "[INFO] ... building venvs into
/opt/gym_venvs. Prefetch them first." in this mode; harmless. Logs: users/rkirby/runs/nano35-rlvr-v2-minf-smoke-small/3764675-logs/.

**Result:** job 3764675 PASSED on 2026-09-15: 2/2 steps, driver `ray.sub exiting (exit_code=0)` at 10:31 (34 min wall:
~35 min queue on QOS short, 9.5 min judges, 7.4 min setup incl. 439 s Gym spinup, step 0 281 s, step 1 83 s), nvshmem
refit ran between the steps (Pipeline complete, ~0.7 s weight_sync), 0 dropped/replaced prompt groups, 0 venv builds
(image venvs + mounted Gym 3615823e worked), gen/policy KL error ~0.0015/0.0016, W&B
nvidia/nano35-rlvr-main-tot-smoke/runs/n2x7bnai. Teardown noise ("Env 'nemo_gym' shutdown failed: Get timed out",
raylet retry messages, CANCELLED by 6428) is self-cleanup, not failure. Fork branch still unpushed (3 cherry-picks).

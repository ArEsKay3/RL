---
name: v2-stack-needs-newer-container
description: "The V2 nemo_rl pin imports nemo.lens (added 2026-08-28), so rl-gym.63635108 fails the vLLM patch step; the CMH default container is now the imported NGC image nvcr.io/nvidian/nemo-rl:fd45cb8-67127697-gym at users/rkirby/containers/nemo-rl_fd45cb8-67127697-gym.sqsh"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7aa0a354-aa1b-4011-8140-8d3122eb7495
  modified: 2026-09-12T03:29:43.437Z
---

Smoke job 3680442 (2026-09-11) was the first CMH job where Ray actually started (after the 140-CPU fix,
[[cmh-cpus-per-worker-140]]). It then failed setup on 7/8 nodes: `setup_command.sh` runs `_apply_vllm_patches`
with the container's prebuilt vllm worker venv, which imports `nemo_rl.telemetry.instrumentation` ->
`from nemo.lens import ...` -> `ModuleNotFoundError: No module named 'nemo'`. `nemo-lens` became a dependency in
NeMo RL commit 2dfbf776d (2026-08-28, PR #3655); the default container `rl-gym.63635108` predates it. The
fingerprint check also shows the image was built for older Automodel/Gym/Megatron-Bridge pins.

`RLVR/nemotron-3.5-nano/README.md` (branch `rlvr-convergence-minf`) states the container must be CI image
`rl-gym.66843268`, "passed as CONTAINER=; the script default is still the older rl-gym.63635108.sqsh". On HSG the
CI artifact dir is `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/nemo_rl_ci/sqsh_files/` (rotates ~6 weeks);
that path exists on CMH but has no 66843268. Candidate newer images on CMH being fingerprint-probed on 2026-09-11:
sauramishra `rl-gym.nightly-20260910.sqsh` (73 GB), joyang `nemo-rl-nightly-gym-20260905-152240.sqsh`,
sauramishra `rl-gym.allfeatures-v2.sqsh` (2026-08-29). `rl-gym.nightly-66319088-arm64.sqsh` is a broken image
(pyxis "Invalid image format").

**Final resolution (2026-09-11 evening):** imported NGC `nvcr.io/nvidian/nemo-rl:fd45cb8-67127697-gym` via
`nemo-rl-workspace/import_ngc_sqsh.sh` (enroot on a CPU node, ~25 min, 90 GB) to
`/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/containers/nemo-rl_fd45cb8-67127697-gym.sqsh`.
Probe: fingerprint matches README pins, nemo/lens present, libXrender present, and `/opt/gym_venvs` is prebaked
(32 resources servers incl. ether0, 3 model servers incl. local_vllm_model, 4 agents). This is the CMH default.
NGC tag naming is `<nemo_rl sha7>-<CI job>-gym`; `nightly-gym` is a moving alias (= 18cce9a-67327114-gym on
2026-09-11, same submodule pins). There is no 66843268 tag on NGC. The earlier local candidate
(sauramishra's rl-gym.nightly-20260910) had NO Gym venvs, which is why it failed; ether0 was restored in
rlvr.yaml once this image proved to carry libXrender.

**Earlier resolution attempt (superseded):** probes showed sauramishra's
`/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra/containers/rl-gym.nightly-20260910.sqsh`
(built from NeMo RL fd45cb8e4, 2026-09-09) matches all three README pins and its pyproject/uv.lock hashes equal
the branch's; it has nemo/lens everywhere. Set as the CMH `_default_CONTAINER`. joyang's 20260905 image and
sauramishra's allfeatures-v2 have nemo-lens but wrong pins. It lives in another user's dir; copy to
`users/rkirby/containers/` (73 GB) if it needs to be durable.

**Why:** a container/pin mismatch costs a full allocation and only surfaces after Ray is up, ~10 min in.

**How to apply:** before launching the V2 stack on CMH, set `CONTAINER=` (and `_default_CONTAINER` in the CMH arm)
to an image whose `/opt/nemo_rl_container_fingerprint` reports Gym `fd5e84d6b`, Automodel `1814c6c93`,
Megatron-Bridge `5ed97996c`, and whose ray venvs contain `nemo/lens`. If none exists on CMH, copy 66843268 from HSG
or import a fresh CI image. Alternatives: `NRL_FORCE_REBUILD_VENVS=true` (slow, and the setup_command patch step
still uses the stale venv) — not a real fix.

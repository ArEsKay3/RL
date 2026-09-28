---
name: gym-venvs-not-prebaked-in-nightly
description: "rl-gym nightly images ship no prebaked NeMo Gym venvs, so multi-node Gym breaks and a run-time venv build (node overlay or Lustre) is never acceptable; the fix is a container with /opt/gym_venvs prebaked (NGC <sha>-<job>-gym tags). Probe /opt/gym_venvs and libXrender when picking a container."
metadata: 
  node_type: memory
  type: project
  originSessionId: 7aa0a354-aa1b-4011-8140-8d3122eb7495
  modified: 2026-09-12T04:17:46.934Z
---

**Superseded 2026-09-11 evening:** both problems below are moot with the NGC `-gym` image now used as the CMH
default (see [[v2-stack-needs-newer-container]]): it has all Gym venvs prebaked and libXrender, so the Lustre
`NEMO_GYM_VENV_DIR` override was removed from the code and ether0 was restored in rlvr.yaml. Lesson kept: when
picking a container, probe `/opt/gym_venvs` too, not just the fingerprint. The user confirmed a run-time Gym venv
build on Lustre can take >4 h and is never acceptable. Leftover partial venvs sit in
`users/rkirby/persistent_cache/gym_venvs/` and can be deleted.

Smoke job 3681287 (2026-09-11, container `rl-gym.nightly-20260910`) got the driver running, then died in Gym
spinup with `RuntimeError: Process 'ether0' finished unexpectedly!`. Two independent causes:

1. **No prebaked Gym venvs.** The Dockerfile only prefetches Gym venvs when `NEMO_GYM_PREFETCH_CONFIGS` is set;
   the nightly was built without it, so every one of the ~71 servers logged "Creating virtual environment" at
   run time. The image sets `NEMO_GYM_VENV_DIR=/opt/gym_venvs`, which is inside each node's container overlay,
   so venvs built by the NemoGym actor's node do not exist on the other Gym node:
   `/opt/gym_venvs/responses_api_models/local_vllm_model/.venv/bin/python: No such file` (141 hits).
   Fix: CMH arm of the CLUSTER case sets `_default_NEMO_GYM_VENV_DIR=.../users/rkirby/persistent_cache/gym_venvs`
   and common.sh appends `NEMO_GYM_VENV_DIR=...` to `EXTRA_TRAIN_ENV` (must be a driver-command prefix; the
   container ENV would otherwise win). `skip_venv_if_present: true` in rlvr.yaml means later runs reuse them.
2. **ether0 needs libXrender.** Its resources server imports rdkit.Chem.Draw -> `libXrender.so.1` missing from
   the image. `grep -c ether0` over the full 22 GB dolphin v41 blend = 0, so the server is dead weight; its
   config line is commented out in `RLVR/nemotron-3.5-nano/configs/rlvr.yaml` with the reason.

Also learned: the sacct `Comment` on these jobs is the submit-time reaper-exemption request written by
tools/launch.sh, NOT a kill reason. `CANCELLED by 6428` (rkirby's uid) = ray.sub's own cleanup after the driver
exited; `CANCELLED by 146504` = an external actor (the reaper), seen on 3628529 and 3634104 at ~1h31.

**Why:** each of these costs a ~20-minute 11-node allocation to discover; the venv problem only shows up on
multi-node Gym layouts, which a single-node test would never catch.

**How to apply:** if a driver dies in Gym spinup, grep the driver log for `No such file` under /opt/gym_venvs and
for `finished unexpectedly`, then read `logs/nemo_gym/<server>.log` under the run dir for the server's own
traceback. First run with a fresh Lustre venv root is slow (uv builds ~70 venvs, minutes each); subsequent runs
should skip. Related: [[v2-stack-needs-newer-container]], [[cmh-cpus-per-worker-140]].

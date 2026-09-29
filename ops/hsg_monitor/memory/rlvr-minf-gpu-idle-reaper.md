---
name: rlvr-minf-gpu-idle-reaper
description: "CMH's OccupiedIdleGPUsJobReaper (90-min GPU-idle kill, acts as uid 146504) reaped RLVR jobs whose Ray steps never launched; with the CPU fix a normal MINF startup reaches 264/264 actors in ~12 min, so it is not a risk for ordinary launches"
metadata:
  type: project
---

Jobs 3628529 and 3634104 (2026-09-08) were CANCELLED by the reaper after ~90 min of idle GPUs. In every reaped job
the ray-head/ray-worker steps never launched because of the 144-vs-140 CPU request ([[cmh-cpus-per-worker-140]]),
so the GPUs were idle by construction. My earlier "slow CPU-only vLLM patching" explanation was wrong.

Measured on minf-v2 job 3688117 (2026-09-12): job start 11:47:49, Ray sruns up by ~+9 min, 264/264 actors at
12:00, first rollouts 12:10. Smokes 3683477 and 3683907 behaved the same. Startup never came near 90 min.

Telling a reaper kill from a self-cancel: the cancelling uid in the slurm log is 146504 for the reaper and 6428
(rkirby) for our own cleanup. The sacct Comment field is the submit-time exemption tag, not a kill reason.

**Why:** an 86-node allocation lost to the reaper costs hours of queue time; knowing it is not a startup risk
avoids adding keep-alive hacks to setup_command.sh.

**How to apply:** nothing extra for normal launches. Remaining exposures: a container without prebaked Gym venvs
(Gym then builds ~70 venvs at run time, see [[gym-venvs-not-prebaked-in-nightly]]), a wrong CPUS_PER_WORKER, or
any other reason the Ray steps never launch. Related: [[rlvr-minf-path-overrides]].

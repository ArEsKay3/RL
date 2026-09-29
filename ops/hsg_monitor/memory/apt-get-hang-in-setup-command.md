---
name: apt-get-hang-in-setup-command
description: "A ray-worker stuck at \"+ bash setup_command.sh\" with no VLLM PATCH output and actors stuck at 28/32 means apt-get update is hanging on that node (no route to apt mirrors); fixed with timeout 60 in tools/launch.sh; live workaround is pkill apt-get inside the worker container"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7aa0a354-aa1b-4011-8140-8d3122eb7495
  modified: 2026-09-12T05:31:11.860Z
---

Symptom (CMH, 2026-09-11, jobs 3681287 and 3683907): one ray-worker log stops at
`+ bash .../setup_command.sh`, never prints `[VLLM PATCH]`, `actors online` sits at 28/32, and the job dies at
the 30-min WORKER_DEADLINE. Process tree inside the container showed `apt-get update -qq` with its http/https
methods in poll for 18+ min. Cause: `tools/launch.sh` SETUP_COMMAND began with
`command -v zstd || { apt-get update && apt-get install zstd; }`; the NGC gym image lacks zstd. CMH compute nodes
DO normally reach an apt mirror — 7/8 nodes in job 3683907 installed zstd successfully in under 3 min (logs show
"Setting up zstd") — but the occasional node cannot, and there apt-get hangs indefinitely rather than failing. So
this is a per-node network fault, not a cluster-wide egress block.

Fix committed on `rlvr-convergence-minf` (pipeline_A): the attempt is wrapped in `timeout 60`.

**Why:** cost a smoke each time and looked like a node problem; it is a network-egress + container-content
interaction that only shows on some nodes.

**How to apply:** live rescue that worked:
`srun --jobid=<id> --het-group=0 --overlap -w <node> -N1 -n1 --container-name=ray-worker bash -c 'pkill -TERM -f "^apt-get update"'`
(the `|| true` lets setup continue and the worker joins within ~30 s). Use `--het-group=0`; the slurm-broker
exec tool cannot target hetjob components. Don't `pkill -f apt-get` from a shell whose own command line contains
"apt-get" — it kills itself.

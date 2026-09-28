---
name: adam-fix-bitcmp-result
description: "Optimizer-state bit compare of minf-v2 step_10 vs step_14 (job 3693058, 2026-09-12): all six dp_group_idx_0 tensors DIFFER in >99.5% of bits, so the saved Adam state advances between rungs and the Adam-staleness fix is doing its job"
metadata:
  type: project
---

Ran `/home/rkirby/workspaces/nemo-rl-workspace/optim_bitcmp.py` as a CPU-only sbatch (job 3693058, 16 min,
nemo_rl container) on `.../users/rkirby/runs/minf-v2/step_{10,14}/policy/weights/iter_0000000`, group
`dp_group_idx_0`. Six tensors: exp_avg, exp_avg_sq, param for two shards (563.7M and 1.916B elements).

Result: identical-bit fraction 0.35-0.45% on every tensor -> DIFFERS. Shard 2 deltas are Adam-sized (exp_avg
max 2.1e-4, exp_avg_sq max 9.6e-10). Shard 1 exp_avg/exp_avg_sq print max|delta| 9.47e29 and rel=nan: that is
~110 padding/allocator-sentinel elements that slip under the script's 1e30 "real value" filter, not model state.
Raw output was at /tmp/rk_optim_bitcmp_10_14.txt (tmp, may be gone).

**Why:** the NeMo RL patch drains the NVRx async save before optimizer offload; without it a rung's optimizer
state could be a stale copy of an earlier step, which would show up here as identical tensors.

**How to apply:** to check future rungs, sbatch a CPU job running
`python optim_bitcmp.py A_DIR B_DIR --label-a step_N --label-b step_M [--group dp_group_idx_0]` on two surviving
checkpoints. Only permanent rungs (every 10 steps) and the single latest step survive; an attempt against step_11
failed with "metadata is None" because ft_keep_latest_k=1 had rotated it out. Related: [[minf-v2-launch-plan]].

---
name: swe-optimizer-state-check
description: 2026-09-22 optimizer-state check on the SWE dump chains (chain G, run A, main chain, chain I): saved Adam state advances step_5 -> step_10 in all four; second-moment landscape agrees across MINF and vLLM runs (corr 0.79-0.81, step-size proxy within 3 %); how to rerun
metadata:
  type: project
---

Tool: users/rkirby/workspaces/engine_loop_test/analysis/optim/optim_stats.py (+ optim_stats.sbatch: interactive-QOS GPU node,
rl-gym container, `uv run --no-sync python`, mounts runs + workspace; 170 s for 8 checkpoints - the dp_group_idx_0 optimizer
shards load in ~15 s each). Output logs/optim_stats_3939996.out. Based on ~/workspaces/nemo-rl-workspace/optim_bitcmp.py.

Facts: the Adam-drain fix (swe_dump commit 9ef88271, "drain async save before offloading the optimizer") is on the branch all
chains ran; the offload path (`offload_before_refit` -> finalize_async_save -> move_optimizer("cpu"), rebinding every state
tensor with .to()) ran 67x on chain G and the main chain, 0x on run A (NcclReshardWeightSynchronizer never offloads).
Bit-compare step_5 vs step_10, group dp_group_idx_0 (chained_0 shard 563.7M elems, chained_1 shard 957.9M elems; exp_avg /
exp_avg_sq fp32, param stored as int16): every tensor DIFFERS in all four runs (90-95 % of real values changed) -> no
stale-save bug on the SWE chains. Cross-run at step_10, chained_1 shard (clean Adam magnitudes: rms m 9e-9, rms sqrt v
4.3e-9, mean |m|/sqrt v 0.20, 11 % zero v in all runs): corr(log exp_avg_sq) G-A 0.81, G-main 0.80, G-I 0.80, A-main 0.79,
A-I 0.81, main-I 0.79; rms sqrt v ratios 0.97-1.03; Adam-step proxy ratios 0.97-1.03 -> MINF optimizers are not reset,
stale or drifted relative to vLLM's. Caveat: chained_0 shard statistics are contaminated by non-Adam values that pass a
|x|<1e6 mask (rms m ~10-75, param RMS ~2.5e4 - the int16 "param" is probably not a bf16 view; the shard may hold fp32
router/bias params); its cross-run corr(log v) is still 0.67-0.71 for every pair and MINF runs differ from each other as
much as from run A. Combined with the weight-space result (identical update magnitude/sparsity across runs), the
optimizer offload shows no training-side fingerprint; `policy.offload_optimizer_for_refit=false` remains a clean ablation.

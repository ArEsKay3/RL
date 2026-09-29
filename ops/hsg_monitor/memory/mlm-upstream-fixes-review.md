---
name: mlm-upstream-fixes-review
description: 2026-09-20 agent review of upstream Megatron-LM fixes since the MINF fork point (fork 880de0fce branched from upstream main 14346b65a, 2026-08-12); ranked list of fixes absent from the fork and the report path
metadata:
  type: project
---

Report: `swe_dump/analysis/mlm_upstream_fixes_review.md` (clone used for history: /home/rkirby/.claude/jobs/e247f0cd/tmp/mlm_review, temporary).
Fork 880de0fce (2026-09-10) = upstream main 14346b65a (2026-08-12) + 29 patches; 12 of those have upstream equivalents (gumbel-max fp32 sampler, Mamba dummy-pointer #6598, Mamba refit-cache refresh #6672, prefix-clamp walk-back #6930, persistent vLLM MoE buffers #6618, prefix-cache dummy-block redirect #6895, HYBRID CG sizing, coordinator scaling); fork-only: multi-EOS termination, Qwen3-Coder parser parity, debug dumps. Upstream main since then: 456 commits, 118 fix-like. `dev` has nothing relevant to the Mamba2/MINF path.

Absent from the fork, ranked by the agent:
1. `487bb3948` (#6481) -1 sentinels in real rows of the per-step KV block table reach the FA paged-attention kernels; the only absent fix with a direct path to different sampled tokens / logprobs (9 lines).
2. `79be654e3` (#6823) refit local same-rank copies not stream-ordered after current-stream producers (nvshmem/nccl/gloo/nixl services); stale or partially updated weights possible, mainly colocated.
3. `94939b85d` (#6442) assertion crash when a chunked-prefill request is hidden under KV pressure with async overlap.
4. `8dc7ed381` (#7081) MoE expert-bias token counts cast to bf16 by Float16Module (bias_update_rate 1e-3 is on); training-side, common to both engines.
5. Lifecycle/async fixes 3972b96bb 8da558d5f 6ffe9f732 6807fb8e4 (neutral under kv_cache_management_mode=persist), 19abd1fc4 ZMQ init deadlock, a145d2a19 abort on client disconnect, f7b220977 tool-call-inside-think parsing.
Dismissed as N/A: batch-invariant/NVLS/FlashInfer/MXFP8/GTP/MFSDP/Muon, tanh clamp (model is relu2), tied-embedding replica id, TE is_first_microbatch, megatron/training checkpoint/RNG fixes; note e2c2e55bf removes the mcore async-save modules NeMo RL imports (rebase break).

**How to apply:** cherry-pick 1-4 onto a new fork worktree before the next MINF run; verify 1 on the dumps (large |gen-prev| tail) and 4 by printing the expert-bias count dtype at runtime.

2026-09-25 nvshmem baseline (measured, not inferred): every MINF arm on this lineage emits TWO distinct nvshmem messages, both benign so far.
1. `refit_backend="nvshmem" is currently broken; prefer "nccl" or "gloo"` (NVIDIA-NeMo/RL#3646) from megatron_worker.py:904-909 - warns and proceeds, there is NO fallback.
2. `WARNING:megatron.core.resharding.nvshmem_copy_service.core.gpu_resource_manager:gpu_resource_manager.py:108: Could not determine nvidia-nvshmem-cu12 package version for NVSHMEM safety check` (repeated once per rank) - the copy service cannot version-check the package so it SKIPS its safety check rather than failing.
Counts are 2 lines each on chain T (swe_915 MINF from scratch), chain P (MINF splice, 41 refit cycles) and chain G (MINF from scratch, no prefix cache, 30 refit cycles); zero failures on all three. So the version-guard skip is lineage-wide and asymptomatic. **If post-refit corruption ever appears, look there first** - it sits right next to the absent #6823 nvshmem copy-service stream-ordering fix. Cross-checked with the Cross Train Experiment session 2026-09-25.

nvshmem refit COST baseline (2026-09-25, measured across four arms). The refit duration is the `weight_sync` line in the SingleController timing block - there is NO separate "refit took Ns" line in the driver log on this stack, so do not go looking for one. Values: chain P (MINF splice, 28 refits) min 4.65 s / median 4.93 s / max 7.42 s / mean 5.52 s; chain P-prime (lr0, 9 refits) median 5.60 s; chain T (swe_915 MINF from scratch, masking off) first two refits 6.90 s and 6.92 s. So **nvshmem costs 5-7 s per refit regardless of prefix caching or seed**. `exposed_generation: 0.00s (0.0%)` on chain T confirms the engine is fully paused during refit, consistent with the job 3688117 verification. chain P-double-prime (4004676, refit nccl) will be the FIRST nccl datapoint on this lineage - weight_sync is the line to diff. chain P-triple-prime (4009432, nvshmem + prefix cache kept across refits) should stay inside the nvshmem band; its interesting measurement is instead the per-turn engine-vs-trainer logprob mismatch on post-refit turns.

2026-09-25 16:25 **chain P-double-prime (MINF from chain P step_20, normal lr, refit NCCL, job 4004676) FAILED at its FIRST refit** - exit 1:0, FAILED after 21:26, zero closed steps, zero weight_sync lines. Six NCCL watchdog timeouts, all `WorkNCCL(SeqNum=1, OpType=COALESCED, NumelIn=0, NumelOut=0, Timeout(ms)=600000) ran for 600005 ms`, PG GUID `refit`, all 256 ranks. Stack: swap_weights_via_reshard (nemo_rl/models/generation/megatron/megatron_worker.py:1033) -> swap_model_weights -> reshard_model_weights -> execute_reshard_plan -> batch_isend_irecv -> _coalescing_manager. SeqNum=1 with zero numel = the first collective on the refit PG never matched across ranks; not bandwidth, not size, nothing partial.

**Scope the claim correctly** (correction from the Cross Train Experiment session, verified by me): do NOT say "nccl refit deadlocks" generally. Our fork HEAD is 880de0fce (2026-09-10), branched from upstream 2026-08-21, and I confirmed `git cat-file -e` finds NEITHER c69a6427 (2026-09-15, "synchronize parameters before inference refit") NOR 5d49fbf4 (2026-09-11, nccl_m2n / refit_transport for Megatron generation) in the tree. Those are exactly the fixes for a first-op batch_isend_irecv mismatch. Honest statement: **on THIS fork, refit_backend=nccl is unusable and nvshmem is the only working MINF refit path - the opposite of the #3646 advisory text.** A fair nccl-vs-nvshmem test needs a tree carrying c69a6427 and 5d49fbf4.

Control quality: chain P-triple-prime (4009432, nvshmem, same chain P step_20 seed, same 64 nodes, same workload, only the refit backend differs) was healthy at the same moment with rollouts being written and zero exceptions. NOT resubmitted - user's standing rule is never resubmit after a code failure.

2026-09-25 CORRECTION to the basis of the nccl-refit claim above. I repeated SHAs c69a6427 / 5d49fbf4 secondhand from the Cross Train Experiment session without checking they resolve. They do NOT resolve in either clone - `git cat-file -e` fails on both in swe_replay_splice/Megatron-LM (880de0fce) AND in swe_main915/Megatron-LM (c04ebc896, NOT shallow, 9748 commits), and `git log --all --grep` for their subjects and for "nccl_m2n"/"refit_transport" returns nothing. **Treat those SHAs as unconfirmed.**
The ROBUST evidence for the same conclusion is code content, which I did verify:
- swe_main915/Megatron-LM HAS `nccl_m2n` in megatron/core/resharding/refit.py, resharding/README.md and resharding/copy_services/__init__.py
- swe_replay_splice/Megatron-LM has ZERO occurrences of `nccl_m2n` anywhere under megatron/core/resharding
So the old fork's resharding module has no m2n path at all, which is a sufficient and checkable explanation for chain P-double-prime's first-collective deadlock, and it means **refit_backend=nccl is plausibly viable on the new main-based pin though it was not on the fork**. Check code content, not quoted SHAs, when scoping this.

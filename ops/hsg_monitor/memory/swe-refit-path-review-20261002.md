---
name: swe-refit-path-review-20261002
description: 2026-10-02 deep review of the vLLM refit path across NeMo RL (b5916526..4aaa48fa), Megatron-LM (14346b65a..6a3660905) and Megatron Bridge (8c46dc425..1f8873bb0); the two fixes that matter and which stacks carry them
metadata:
  type: project
---

Review done 2026-10-02 for rkirby (question: overlap of refit with decode, missing params, precision, "layernorm syncing to vLLM wrong").

- Layernorm fix = NeMo RL f8ed2952 (#3660, Saurabh Mishra, 2026-08-18): the SC split-API train step ended with a bare finish_grad_sync and skipped finalize_model_grads, so sequence-parallel layernorm grads were never all-reduced across TP; norm weights drifted per TP rank and Bridge's replicated export shipped one rank's copy to vLLM. Bug lived 2026-07-07..08-18. Present in every stack we run (v2 upstream base 08-21, parity fork, 09-06, 09-21, 09-24).
- Stale-refit fix = NeMo RL c69a6427 (#4086, wdykas, 2026-09-15): with use_distributed_optimizer + overlap_param_gather=true the optimizer step defers the param all-gather to the next forward pre-hook, so a refit between step and forward exports a stitched buffer (own DP shard fresh, other shards one step stale). Fix adds sync_params_before_refit (start_param_sync(force_sync=True) + cuda synchronize) called by SC and GRPO before the weight synchronizer. Present in 09-21 and 09-24 only; parity (v2) fork and 09-06 lack it. On the old pin the MINF generation worker did force the gather, vLLM did not. overlap_param_gather=false (rkirby's rule, enforced by the audit) makes it moot on every stack.
- In-flight generation is never paused on the SingleController path at either pin: _sync_weights only gates new dispatch, aborts stale groups only on the non-Gym path, and vLLM requests continue through the swap on their old KV cache (recompute_kv_cache_after_weight_updates=false in our configs). #3839 pause-for-refit only touches the legacy async collector.
- No parameter class is renamed, dropped or recast on the BF16 vLLM export at either pin; NeMo RL passes no weight_dtype, tensors leave Megatron in their native dtype. Bridge mapping validation became fatal (08-24) and our logs show no unmapped params. Our vLLM engines realize the TRITON unquantized MoE backend, so the FlashInfer-TRTLLM layerwise refit path (#3545/#3659) is inactive.
- MLM: resharding gained batched NCCL rounds (#7189, 32 params/batch) and a same-rank copy stream-ordering fix (#6823, moot with vLLM in separate processes); nothing changes which Mamba/norm/router tensors are params or how replicated params are synchronized. v2 fork carries a pre-merge Mamba decode-cache refresh that runs after the final synchronize (MINF-only CUDA-graph race).

**Why:** rkirby asked for a deep dive across all three repos; these are the verified conclusions so future sessions do not redo the archaeology.

**How to apply:** when comparing arms, check each arm's stack and overlap_param_gather setting against the two fixes above before attributing differences to the engine. See [[feedback-minf-defaults-noprefix-nopg]], [[swe-nrl-0906-and-0921-workspaces]], [[swe-main915-workspace]].

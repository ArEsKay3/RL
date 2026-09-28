---
name: swe-vllm-parity-branch-caveats
description: "What the Megatron vLLM-parity branch changes outside the inference_vllm_parity gate, and what the audit does not cover"
metadata: 
  node_type: memory
  type: project
  originSessionId: a7f966fb-b99e-4db0-bb54-a178ecde5ae0
  modified: 2026-09-27T19:54:20.617Z
---

Reviewing `santhnm2/Megatron-LM:vllm-numerical-parity` against rkirby's base
`880de0fce8` on 2026-09-27, three things are not gated by
`inference_vllm_parity` despite the flag's "training continues to use the
original forward path" docstring:

1. `tl.exp` → `fast_exp` (`exp2(log2e·x)`) in `ssd_chunk_scan.py`,
   `ssd_chunk_state.py`, `ssd_state_passing.py` — the *shared* Mamba SSD forward
   kernels. The backward kernels keep `tl.exp`.
2. `autotune_configs()` (the fork's determinism wrapper) stripped from 4 of 5
   call sites. Harmless in practice: the wrapper is a no-op unless
   `MAMBA_DETERMINISTIC` or torch deterministic mode is on, and the SWE runs set
   neither.
3. Two parity paths gated only on `not self.training`, which is the wrong gate:
   NeMo RL runs eval-mode forwards on the *training* model, because
   `get_logprobs_presharded` calls `model.eval()`.
   - `ExtendedRMSNorm.forward` — put the compiled vLLM-order grouped gated norm
     into every trainer logprob pass and into plain non-parity MINF.
   - `SelfAttention.get_query_key_value_tensors` — routed the trainer's logprob
     pass through the vLLM parity QKV, which handed vLLM-layout tensors to TE's
     context-parallel flash attention at CP=4 and killed smoke 4050516 with
     `SystemError: <class 'UserWarning'> returned a result with an exception set`.
   **Both fixed on `rkirby/vllm-parity-armV`** (`517934242`, `d37db1077`); the
   second uses `InferenceMode.is_active()`, which `hybrid_block.py:311` already
   uses. Every other parity site is inference-only (`ssm_prefill`/`ssm_decode`)
   or attribute-gated by the adapter. Report both upstream.

`docs/inference/vllm_numerical_parity.md` states its own limits: "full byte
parity is pending"; sampling/RNG lifecycle and cached recurrent/KV state are
"separate unfinished acceptance gates"; the audited profile is TP4/EP1/ETP4
with **prefix caching disabled** and does not establish equivalence with the
EP4 layout; and repeated training-to-generation refits — what an RL arm does
~60 times — were never validated. It also records that two native vLLM runs
matched byte-for-byte while a third differed by 145,934 bytes on the same
checkpoint and prefix, so the reference itself is not bit-reproducible.

Replica size is unchanged by the EP1/ETP4 switch: `EP×ETP = 4 = TP`, so a
generation replica stays 4 GPUs either way and per-GPU expert memory is the
same; only the sharding axis changes.

Context for arm V: [[swe-vllm-parity-arm-v]].

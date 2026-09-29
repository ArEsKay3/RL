---
name: feedback-parity-is-engine-vs-engine
description: rkirby — vLLM parity is MINF-vs-vLLM; train-vs-inference metrics cannot measure it
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a7f966fb-b99e-4db0-bb54-a178ecde5ae0
  modified: 2026-09-27T23:28:49.312Z
---

rkirby, 2026-09-27, on chain V reporting: "This is about difference between
training and inference and you are meant to have parity with a different
inference engine (vllm) not training."

`num_masked_seqs_by_logprob_error`, `token_mult_prob_error`, `gen_kl_error` and
the `generation_token_ids == token_ids` check all compare the **trainer's**
recomputed logprobs or ids against the **engine's**, inside one arm. They are
train-vs-inference consistency checks. An arm can be perfectly self-consistent
and still diverge completely from vLLM.

**Why:** chain V exists to show Megatron in-engine inference matches vLLM. That
is an engine-vs-engine question. Self-consistency metrics are structurally
incapable of answering it, so reporting them as parity evidence is worse than
reporting nothing — it looks like a result.

**How to apply:** report those metrics on the guard/health line, never the
parity line. Parity evidence for an RL arm is the trajectory against the vLLM
arms at matched steps: `rollout_length` mean/p50 (the statistic
[[swe-length-growth-investigation]] turned on, which separates the engines by
step 15-25), SWE-Bench Verified pass@1 at the rungs, and loop share at steps
22-30. Step 1 discriminates nothing — [[swe-splice-experiments]]'s P′/Q′ result
already showed the engines are neutral at fixed weights, so divergence only
accumulates over training. Until ~step 15 the honest status is "running,
healthy, nothing drifting."

Matched comparators for chain V (same base, same data order, same seed):
chain Q⁗ and run A on the vLLM side, chain P⁗/J/G on the MINF side.
See [[swe-vllm-parity-arm-v]].

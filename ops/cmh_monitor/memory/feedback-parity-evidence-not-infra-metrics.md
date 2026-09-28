---
name: feedback-parity-evidence-not-infra-metrics
description: "rkirby 2026-09-27: num_masked_seqs_by_logprob_error and generation_token_ids==token_ids are train-vs-inference checks, NOT vLLM-vs-MINF parity evidence; report them as engine health"
metadata:
  type: feedback
---

rkirby, 2026-09-27 (relayed via the VLLM parity session, which had set the opposite standing instruction that morning and retracted it):

**Stop reporting `num_masked_seqs_by_logprob_error` as a parity signal for chain V.** It compares the TRAINER's recomputed logprobs against the ENGINE's logprobs *inside a single arm* — a train-vs-inference consistency check. chain V's question is parity between two INFERENCE engines, MINF and vLLM. A MINF arm can sit at 0 masked every step and still diverge completely from vLLM; the metric structurally cannot tell you. Report it on the guard-counter / engine-health line instead.

Same correction applies to the token-ids patch pitch: `generation_token_ids == token_ids` is also engine-vs-trainer. The ids remain worth dumping for prefix integrity and per-turn boundary reconstruction (see [[swe-token-ids-dump-patch]]), just not as parity evidence.

**What chain V is actually judged on:**
1. `rollout_length/swe_agents_train` mean and p50 against the vLLM arms at matched steps — the statistic the whole length-growth investigation turned on; it separates MINF from vLLM by step 15-25. Matched comparator is chain Q⁗ (vLLM from scratch, prefix caching off, same base), then run A. If parity works, chain V's length curve sits with the vLLM arms, not with chain P⁗/J/G.
2. SWE-Bench Verified pass@1 at the rungs — run A / chain K / chain Q⁗ ~0.512-0.518 from step 15, chain G sagged to 0.464. Decisive, and days out.
3. Loop share at steps 22-30, the splice criterion.

Step 1 discriminates nothing early, as expected from the P′/Q′ frozen-weight null: at step 1, chain Q⁗ 27,618 mean / reward 0.3789, chain V 26,055 / 0.3750, chain P⁗ 25,195 / 0.3848. Divergence accumulates over training, which is why this has to be an arm and not a bench test.

**Why:** infra metrics were reading as parity results in the monitor ticks. Until ~step 15 the honest chain V status is "running, healthy, nothing drifting" — refits, guard counters, exception strings, rungs.

**How to apply:** in every tick, keep masked counts on the health line with guard counters. Do not present any single-arm train-vs-inference statistic as evidence about MINF-vs-vLLM. Related: [[feedback-monitor-ops-only]], [[swe-length-growth-investigation]], [[swe-vllm-parity-arm-v]].

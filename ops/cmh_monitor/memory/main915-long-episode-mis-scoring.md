---
name: main915-long-episode-mis-scoring
description: "main@9-15 mis-scores long multi-turn episodes: seq_mult_prob_error>2.0 on 58% of 150-200-turn rollouts vs 0.3% on the v2 stack; median flips 1.018 -> 105 past ~150 turns"
metadata:
  type: project
---

2026-09-27, found from smoke 4053262 (main@9-15, after `ee863535e` + `35552a16c`). The stitch storm ([[main915-prefix-stitch-storm]]) is fixed and episodes are alive — but the arm mis-scores long episodes at a rate that makes it a different experiment from the v2 arms.

`seq_mult_prob_error` by assistant-turn bucket, joined token_level row -> rollout jsonl on `sample_id`:

| turns | main@9-15 n / err>2.0 | median err | chain V (v2 control) n / err>2.0 | median err |
|---|---|---|---|---|
| 0-24 | 1 / 0 (0%) | 1.0167 | 47 / 0 (0%) | 1.0164 |
| 25-49 | 37 / 1 (2.7%) | 1.0139 | 374 / 0 (0%) | 1.0151 |
| 50-74 | 68 / 5 (7.4%) | 1.0163 | 469 / 0 (0%) | 1.0148 |
| 75-99 | 39 / 2 (5.1%) | 1.0161 | 342 / 0 (0%) | 1.0159 |
| 100-149 | 56 / 10 (17.9%) | 1.0180 | 488 / 1 (0.2%) | 1.0160 |
| **150-200** | 55 / **32 (58.2%)** | **105.24** | 328 / 1 (0.3%) | 1.0159 |

**The median flips, not just the tail** — normal (1.018) through 100-149, then 105.24 at 150-200. Past ~150 turns, catastrophic mis-scoring is the default outcome. Whole-distribution view: body identical to v2 (p25/p50 1.0135/1.0166 vs 1.0141/1.0167) but p90 900, p99 794,811, max 1.9e8.

chain V is a clean control: 2,048 rows over the same turn distribution, 328 of them at 150-200 turns, median 1.0148-1.0164 in **every** bucket. Long episodes are not inherently hard to score.

Per-batch cost: `num_masked_seqs_by_logprob_error` 30/128 and 20/128 (vs 0-2 of 512 on v2 arms). High-error rows have 188 median turns vs 76.5, and mean reward 0.16 vs 0.42.

**Ruled out:** dump-wiring from `35552a16c` (4051868 showed 24/17 before it existed; 4022822 showed 0/0 when episodes were dead stubs). Retokenization in `ee863535e` (the substitution is numerically identity — `compact_prompt_tokens` is None for non-VLM). The exception-based splice fallback (`grep -rc "Skipping prefix replacement"` = 0 everywhere in 4053262).

**Also ruled out:** PP-stage disagreement on `ssm_chunk_alignment` (the recipe runs `pipeline_model_parallel_size=1`); context overflow (`max_total_sequence_length` 196,608 vs a 144,974-token high-error median); `mamba_cache_mode: align` (under `vllm_kwargs`, never read in the MINF path); the exception-based splice fallback; within-call chunking misalignment (`(capacity // alignment) * alignment` rounds down robustly); and the unaligned splice boundary itself — **v2 uses the identical last-EOS `_replace_prefix_tokens` logic and neither tree checks the boundary against SSM alignment**, so the boundary is not the variable, the consumer is.

**Best open lead (2026-09-27, unconfirmed):** `contexts/mamba_slot_allocator.py`, `compute_and_store_offsets` endpoint cache. Two deltas vs v2:
1. Snapshot filter quantum coarsened: `offset % mamba_chunk_size == 0` -> `offset % ssm_chunk_alignment == 0`, where `__init__` asserts `ssm_chunk_alignment % mamba_chunk_size == 0`, so it is a multiple — strictly **fewer** snapshots recorded.
2. **The `is_last_chunk` guard was removed:** v2 has `if is_last_chunk and last_aligned_abs == prompt_len and prompt_len > 0:` with the comment *"Only valid on the final chunk (otherwise the live state is mid-prompt)"*; main@9-15 has `if chunk_end > 0 and chunk_end % bs == 0:`, extending it to non-final boundaries. The two trees make **opposite claims about correctness**. If v2's is right, main@9-15 writes a mid-prompt live state into the cache labelled as a block-boundary state — occasional (only when a chunk end lands block-aligned), length-dependent, and persistent once written.

`dynamic_engine.py:2690-2745`'s chunk-snapping guard is **fail-safe** and not the culprit: a dropped snapshot costs a prefix-cache hit, not correctness.

Traced one level further (Get Latest Main?, confirmed): the endpoint cache writes `eos_bids_cpu`, which feeds a **hash-indexed commit** — docstring says *"all_hashes covers intermediate_bids + eos_bids"* — used for **future prefix-cache matching**. So a possibly-wrong snapshot is committed to the index later turns match against, which is the persistence mechanism.

**The precise open question for whoever picks this up:** does the live recurrent state at a non-final block-aligned `chunk_end` actually equal the true end-of-sequence state, or does something downstream distinguish "mid-prompt" from "terminal" when matching against this hash? That is kernel / hash-matching semantics, reachable by static reading.

Reproduce the diff: `diff -u <v2>/Megatron-LM/megatron/core/inference/contexts/mamba_slot_allocator.py <main915>/...` (716 vs 735 lines).

**Why:** chain U cannot be compared to the v2 arms while it discards 16-23% of every batch, wrongly, on exactly the longest episodes.

**How to apply:** reproduce on a **16-node smoke** — 4053262 produced 55 episodes past 150 turns and failed 32. No 64-node arm needed. Analysis is torch-free via `swe_dump/tools/pt_numpy.py`: join `token_level` `sample_ids` to the rollout jsonl, bucket on `num_assistant_turns`. Related: [[swe-main915-workspace]], [[feedback-parity-evidence-not-infra-metrics]].

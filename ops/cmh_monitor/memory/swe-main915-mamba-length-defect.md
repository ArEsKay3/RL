---
name: swe-main915-mamba-length-defect
description: "ROOT-CAUSED 2026-09-28: the chain U long-episode logprob corruption is Megatron's _compute_prefix_match applying the KV-only prefix skip on block-aligned continuation chunks of a hybrid model (memory-only prefix caching, no Mamba cache); NOT the mamba_slot_allocator EOS-cache change"
metadata:
  node_type: memory
  type: project
  originSessionId: 51d77432-fbb8-4536-b679-a1714b74af54
  modified: 2026-09-28T14:40:27.716Z
---

Found 2026-09-27 while validating the [[swe-main915-workspace]] stitch fix; root-caused
2026-09-28. Symptom: on every main-based MINF smoke, `num_masked_seqs_by_logprob_error`
elevated and `seq_mult_prob_error` catastrophic (max 6e7) on 48-58% of 150-200-turn
episodes, while chain V (v2 stack, same recipe, same 196608 max length) is clean.

**Root cause (Megatron-LM, `megatron/core/inference/contexts/dynamic_context.py`,
`_compute_prefix_match`):** the generic KV-cache prefix skip
`prefix_skip_tokens = min(num_matched*block, chunk_len-1)` fires whenever
`finished_chunk_token_count % block_size == 0` and the chunk's blocks hash-match. The
hybrid-model guard that zeroes it (`elif self.is_hybrid_model and finished == 0`) only
covers the FIRST chunk. A hybrid model's Mamba state is restored only on the first chunk
(and only when a Mamba cache exists), so a block-aligned CONTINUATION chunk that matches
skips up to a whole chunk of tokens the recurrent layers never see: the ">= 2 computed
tokens" clamp leaves exactly 256 computed tokens per 16384-token chunk. Attention still
reads the matched KV, so output stays mostly coherent -- errors are SPARSE (only the
>1-nat tail of |engine-trainer| jumps 10-50x; 85% of tokens still agree to bf16), engine
is the confident side, tokens are copy-from-context ('PY', 'EOF', identifiers).

**Why only the main stack:** the v2 NeMo RL worker passed
`prefix_caching_mamba_gb=mcore_generation_config.get("prefix_caching_mamba_gb", 50)`
(a 50 GB Mamba state cache by default). NeMo RL main only forwards the knob when the
recipe sets it; ours does not, so the engine logs "Running in memory-only mode"
(`mamba_slot_allocator is None`). In that mode every turn re-prefills its whole prompt in
`max_tokens - active_decode_count` chunks with no Mamba-alignment snapping (the snapping
guard requires the allocator). A request prefilling ALONE gets 16384-token chunks, every
boundary block-aligned, so every continuation chunk skips. Lone prefills happen late in a
rollout step when only the longest conversations survive -> the per-position step at
~131072 tokens (2^17 = 8 x 16384) and the 150-200-turn concentration. With the Mamba
cache active the same bug can still fire (continuation chunk matching KV past the Mamba
frontier) but rarely -- chain V shows 0.01 spikes/1e4 tokens at every position.

**Evidence (smoke 4057393 dumps, preserved at
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-main915-minf-smoke-16n/dumps.4057393-prefix-skip-bug`):**
- Engine-reported cached tokens per turn (`full_result.per_turn_metrics.token_usages[i].cache_read_tokens`,
  must be 0 in memory-only mode): 487/25557 turns > 0 (min 256, median 16128 =
  `(16382//256)*256`, max 175872); 279 of the 334 turns containing a >3-nat spike have
  cached > 0 vs 208 of 25223 spike-free turns. Cached>0 fraction by prompt length
  0.3/1.0/1.4/2.8/15.3/26.8% tracks spiked fraction 0.2/0.4/0.7/1.5/13.5/25.3%.
- Engine step logs (ray-driver.log `real config [N]: p P + d D`): single-prefill chunk
  sizes are 16,3xx or exactly 256; all 27 256-token chunks are solo prefills (0 D);
  their share rises 0% -> 30% across the step.
- Spike rate by absolute position: 1-9/1e4 below 131072, 25-80 above; chain V 0.01-0.02
  everywhere (nano35-swe-v2-from0-parity-minf-20260927 dumps).
- Model is NoPE (`position_embedding_type="none"` via nemotron_h_bridge.py), so RoPE is
  irrelevant; attention kernels/kv append kernel identical between stacks; async
  scheduler in both stacks; EOS-cache code (`compute_and_store_offsets`) never runs in
  memory-only mode -- the earlier `is_last_chunk` lead was a red herring.

Analysis tooling (torch-free, `swe_dump/tools/pt_numpy.py`) lives in
`/home/rkirby/.claude/jobs/51d77432/tmp/`: `spike_tokens.py`, `spike_position.py`,
`spike_pos_ptonly.py`, `diff_density.py`, `behavior_vs_position.py`,
`cached_tokens_per_turn.py` (job tmp dir; copy out if needed long-term).

**Fix:** Megatron-LM `mlm-main915-latest`: in `_compute_prefix_match`, hybrid models
zero `prefix_skip_tokens` on every chunk except the first-chunk Mamba-backed restore
(`elif self.is_hybrid_model:` instead of `... and finished == 0`). Upstreamable as-is
(present on Megatron-LM main). Separately, the recipe should set
`policy.generation.mcore_generation_config.prefix_caching_mamba_gb` (v2 used 50) so
chain U runs the same Mamba-cache regime as chain V; memory-only mode also re-prefills
whole 145K prompts every turn.

**Status: FIX VALIDATED 2026-09-28** by smoke 4065835 (fix mounted, memory-only regime
unchanged, 2 steps, dumps at `.../runs/nano35-swe-main915-minf-smoke-16n/dumps`): 255/256
sequences unmasked (was ~215), `seq_mult_prob_error` max 3.46 / err>2 on 1 sequence (was
6.3e7 / 39), 150-200-turn bucket 0.0% err>2 with median 1.016 (was 48.1% / 1.405), spike
rate 0.00-0.05/1e4 at every position incl. >=131072 (was 25-80), 0/25382 turns with engine
cached tokens (was 487), solo prefill chunks 16384 tokens (no 256-token skip chunks), 0
stitch / non-contiguous errors. Side effect: without the (wrong) skip, memory-only mode
recomputes whole prompts every turn, so long-conversation steps are slower -- another reason
to set `prefix_caching_mamba_gb`. Second smoke 4065840 (fix + `++policy.generation.
mcore_generation_config.prefix_caching_mamba_gb=50`, chain V's regime, run dir
`.../runs/nano35-swe-main915-minf-smoke-16n-mambacache50`) also clean: exit 0 in 37:40
(faster -- turns reuse the Mamba cache), 256/256 unmasked, max err 1.052, err>2 = 0 in every
turn bucket (150-200: n=62), spikes 0.00-0.12/1e4, 27251/27274 turns with legitimate cached
tokens, no memory-only warning, 0 errors. Recipe now declares `prefix_caching_mamba_gb: 50`
in `swe_sc_cmh_minf.yaml` (nemo_rl `rkirby/swe-main915-latest`). Write-up + scripts:
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_main915/analysis/prefix_skip_bug/REPORT.md`.
Nothing pushed; report/PR to Megatron-LM pending rkirby.

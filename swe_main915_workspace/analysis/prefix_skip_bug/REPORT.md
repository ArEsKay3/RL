# Megatron-LM dynamic inference: KV-only prefix skip on continuation chunks desyncs hybrid (Mamba) models

Date: 2026-09-28. Stack: NeMo RL `rkirby/swe-main915-latest` (base `4aaa48fab`, NeMo RL `main`
2026-09-27), Megatron-LM `mlm-main915-latest` (base `6a3660905a2736b5670baed1ca5954372937918b`, the
commit Megatron-Bridge `1f8873bb` pins), Nemotron-3.5-nano (NemotronH hybrid MoE, 6 attention layers,
`position_embedding_type="none"`). Recipe: SWE-agent RL, `max_total_sequence_length=196608`,
`block_size_tokens=256`, `max_tokens=16384`, `enable_chunked_prefill=true`, `enable_prefix_caching=true`.

## Symptom

On every main-based MINF run, the trainer-vs-engine logprob check (`seq_mult_prob_error`,
`num_masked_seqs_by_logprob_error`) flagged 15% of all sequences and **48-58% of 150-200-turn
episodes** (max error 6e7). The v2 stack running the same recipe (chain V) is clean at every length.

The disagreement is sparse, not a corrupted state: the fraction of generated tokens with
|engine - trainer| > 0.01 nats is ~15% at every position (bf16 noise) while the > 1 nat tail jumps
from 0.02% below position 131072 to 1.4% above it. In 4096 of 4175 spikes (> 3 nats) the engine is
the confident side (logprob -0.01..-1 vs trainer -17..-30) and the token is copied from elsewhere
in the context (`PY`, `EOF`, identifiers). Spiked assistant turns show a 13x higher invalid-tool-call
rate than spike-free turns at the same positions.

## Root cause

`megatron/core/inference/contexts/dynamic_context.py`, `DynamicInferenceContext._compute_prefix_match`:

```python
block_aligned = finished % self.block_size_tokens == 0
if num_matched > 0 and block_aligned:
    prefix_skip_tokens = min(num_matched * self.block_size_tokens, prefill_chunk_length - 1)
else:
    prefix_skip_tokens = 0

if self.is_hybrid_model and self.mamba_slot_allocator is not None and finished == 0:
    ...  # Mamba-backed skip, restored in add_request
elif self.is_hybrid_model and finished == 0:      # <-- only the FIRST chunk was guarded
    prefix_skip_tokens = 0
```

A hybrid model's recurrent state is restored only on the first chunk (`add_request` queues
`_pending_mamba_restores` only when `finished_chunk_token_count == 0`), and only when a Mamba prefix
cache exists. On a **continuation chunk whose start is block-aligned and whose blocks hash-match**,
the generic KV skip is applied anyway: the chunk skips `num_matched * 256` tokens, the ">= 2 computed
tokens" clamp leaves exactly `256` computed tokens per 16384-token chunk, and the Mamba layers never
see the skipped span. Attention still reads the matched KV blocks, so generation stays mostly coherent
and the corruption shows up as sparse, confident mispredictions.

Why it is routine here: NeMo RL `main` no longer defaults `prefix_caching_mamba_gb` (the v2 worker
passed `mcore_generation_config.get("prefix_caching_mamba_gb", 50)`), so the engine runs prefix
caching in **memory-only mode** (`prefix_caching_mamba_gb` unset -> `mamba_slot_allocator is None`,
engine logs "Running in memory-only mode"). In that mode every turn re-prefills its whole prompt in
`max_tokens - active_decode_tokens` chunks, the Mamba-alignment chunk snapping in
`schedule_chunked_prefill` is skipped (it requires the allocator), and a request prefilling **alone**
gets exact 16384-token chunks so every boundary is block-aligned and every continuation chunk skips.
Lone prefills happen late in a rollout step when only the longest conversations remain, hence the
per-position step at ~131072 tokens (8 x 16384) and the 150-200-turn concentration. With a Mamba cache
the same code path can still fire whenever a continuation chunk matches KV past the Mamba frontier,
just rarely.

## Evidence (smoke 4057393, dumps preserved at
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-main915-minf-smoke-16n/dumps.4057393-prefix-skip-bug`)

1. Engine-reported cached tokens per turn (`num_cached_tokens`, surfaced by OpenHands as
   `token_usages[i].cache_read_tokens`; must be 0 in memory-only mode): 487 / 25557 turns > 0,
   min 256, **median 16128 = (16382 // 256) * 256**, max 175872 (90% of a 195K prompt).
   279 of the 334 turns containing a > 3-nat spike have cached > 0; 208 of 25223 spike-free turns do.
   Cached>0 fraction by prompt length 0.3 / 1.0 / 1.4 / 2.8 / 15.3 / 26.8% tracks the spiked
   fraction 0.2 / 0.4 / 0.7 / 1.5 / 13.5 / 25.3%.
2. Engine step log (`real config [N]: p P + d D`): single-request prefill chunks are ~16,3xx tokens
   or **exactly 256**; all 27 256-token chunks are solo prefills (0 decode requests); their share
   rises from 0% to 30% of prefills across the step.
3. Spike rate by absolute position: 1-9 per 1e4 tokens below 131072, 25-80 above. Chain V (v2 stack,
   same recipe, 50 GB Mamba cache): 0.01-0.02 everywhere.
4. Ruled out: the `mamba_slot_allocator.py` EOS-cache change (never executes in memory-only mode;
   ordering strictly sequential anyway), RoPE (NoPE model), attention / KV-append kernels
   (byte-identical between stacks), the async scheduler (enabled in both stacks), KV allocator
   eviction (pool 86015 blocks, ~16K in use).

## Fix

Megatron-LM commit `475167fa4` on `mlm-main915-latest`
(`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_main915/Megatron-LM`):
hybrid models zero `prefix_skip_tokens` on every chunk except the first-chunk Mamba-backed restore
(`elif self.is_hybrid_model:` instead of `elif self.is_hybrid_model and finished == 0:`). Matched
blocks remain deduplicated through the existing dummy-block write redirect; non-hybrid models are
unchanged. Applies to Megatron-LM `main` as-is.

## Validation

Smoke 4065835 (fix mounted, memory-only regime unchanged, 16 nodes, 2 steps, exit 0, 57:56):

| metric | before (4057393) | after (4065835) |
|---|---|---|
| sequences masked by logprob error | 24 + 17 of 256 | 1 + 0 of 256 |
| `seq_mult_prob_error` max / err>2 | 6.3e7 / 39 | 3.46 / 1 |
| 150-200-turn bucket err>2 / median | 48.1% / 1.405 | 0.0% / 1.016 |
| spikes per 1e4 tokens at >=131072 | 25-80 | 0.00-0.05 |
| turns with engine cached tokens > 0 | 487 | 0 / 25382 |
| solo prefill chunk sizes | 256 x27, 16384 x9 | 16384 x47, tails |

Second smoke 4065840 adds `++policy.generation.mcore_generation_config.prefix_caching_mamba_gb=50`
(chain V's regime) on top of the fix; run dir
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-main915-minf-smoke-16n-mambacache50`.

## Reproduction / tooling

Torch-free analysis scripts (need `swe_dump/tools/pt_numpy.py`): `spike_tokens.py`,
`spike_position.py`, `spike_pos_ptonly.py`, `diff_density.py`, `behavior_vs_position.py`,
`cached_tokens_per_turn.py`, `final_validation.py` in this directory; each takes a run dir with
`dumps/token_level/*.pt` and `dumps/rollouts/*.jsonl` (NeMo RL `async_rl.dump`).

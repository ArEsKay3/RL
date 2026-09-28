# Megatron-LM upstream fixes since the MINF fork point — review (2026-09-20)

Fork under review: `880de0fce` (ArEsKay3/Megatron-LM, branch `rkirby/nemorl-with-minf`), mounted read-only from
`.../workspaces/swe_dump/Megatron-LM` and `.../workspaces/swe_minf/Megatron-LM` (both at `880de0fce`; the
`nemo_rl/3rdparty/Megatron-LM-workspace/Megatron-LM` submodule dir is absent in swe_dump).
Upstream: `https://github.com/NVIDIA/Megatron-LM.git`, `main` at `fb6a123a0` (2026-09-20 09:04 UTC), `dev` at `d909db3a6`.
No live tree was modified; all history work was done in a fresh blobless clone at
`/home/rkirby/.claude/jobs/e247f0cd/tmp/mlm_review` (upstream + fork remotes). No `git fetch` was run in the live trees.

Workload assumed (from `examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_{minf,common}.yaml` in swe_dump/nemo_rl):
Nemotron 3.5 nano hybrid (NemotronH, `mlp_hidden_act=relu2`, 128 routed experts top-6, untied embeddings),
MINF with `transformer_impl=inference_optimized`, TP4/EP4/ETP1/PP1/CP1, `inference_grouped_gemm_backend=vllm`,
`cuda_graph_impl=local` + `use_cuda_graphs_for_non_decode_steps=true`, chunked prefill, prefix caching
(longest_prefix routing), `mamba_inference_ssm_states_dtype=float32`, `kv_cache_management_mode=persist`
(no recompute after weight updates), `async_sched_mode=ASYNC`, TorchSampling, `logprobs_mode=raw_logprobs`,
temperature 1.0 / top_p 1.0 / top_k none, `max_new_tokens=196608`, `refit_backend=nvshmem`; training with
DistributedOptimizer, `overlap_param_gather=true`, `overlap_grad_reduce=false`, precision-aware optimizer,
`moe_router_enable_expert_bias=true` with `moe_router_bias_update_rate=1e-3`, `freeze_moe_router=true`,
aux loss coeff 0, async dist-ckpt save.

## Top 5

1. **`487bb3948` Avoid passing -1 sentinels into inference kernels (#6481, 2026-08-13) — ABSENT.** Real rows of the
   per-step MHA block table can carry `-1` in trailing columns; the paged-attention kernels (FA2/FA3/FA4 `block_table`/
   `page_table`) treat `-1` as a real block index. Upstream masks them to the dummy block. Out-of-range KV reads can
   perturb attention output (worst case NaN/garbage for some rows) and therefore sampled tokens. 9-line, isolated
   change to `dynamic_context.py` → **cherry-pick first**; it is the one absent fix that can directly change tokens.
2. **`79be654e3` Fix order local refit copies after source updates (#6823, 2026-09-10) — ABSENT in all four copy
   services (nvshmem, nccl, gloo, nixl).** The same-rank ("local") copies run on a side stream that is never made to
   wait for the current stream, so they can read source tensors before the producers on the current stream (param
   all-gather, dtype harmonisation, dequant/`_ensure_sendable`) have finished. In the colocated setup most pairs are
   local. This is exactly a "stale/partially-updated weights after refit" bug. 4 one-line `wait_stream` additions →
   **cherry-pick**.
3. **`94939b85d` Handle hidden chunked prefill during async overlap (#6442, 2026-08-13) — ABSENT.** Under KV pressure
   the chunked-prefill request can be absent from the active batch for a step; the fork still asserts exactly one row
   → hard crash (`The active chunked-prefill request must have exactly one row`). Our config runs chunked prefill +
   async overlap + 196k sequences → **cherry-pick** (5 lines).
4. **`8dc7ed381` Preserve exact expert token counts for expert-bias updates (#7081, 2026-09-09) — ABSENT and live
   here.** `router.local_tokens_per_expert` is a float32 buffer in the fork; `Float16Module`
   (`convert_module_to_dtype_except_fp32_marked`) casts unmarked float buffers to bf16 and NeMo RL's
   `MoEFloat16Module._maintain_float32_expert_bias` restores only `expert_bias`/`qb_*`, not the count buffer. With
   `moe_router_enable_expert_bias=true`, `bias_update_rate=1e-3` the bias update direction is computed from bf16 token
   counts. Upstream uses an int64 buffer and an exact integer comparison. Training-side (affects routing drift over the
   run, identical for vLLM- and MINF-served chains) → **cherry-pick + verify at runtime** (`router.local_tokens_per_expert.dtype`).
5. **Refit/inference lifecycle hardening that is NOT in the fork but is mostly neutral for this config:** `3972b96bb`
   (#7256), `8da558d5f` (#7053), `6ffe9f732` (#6851), `6807fb8e4` (#7031), `d0612089d` (#6895 refinement). The token-
   affecting parts target RECOMPUTE suspend/resume (stale pending async logits) or prompt logprobs of partial prefill
   chunks (we skip prompt logprobs, and we run `persist`). They matter if `recompute_kv_cache_after_weight_updates`
   is ever turned on, and `6807fb8e4` changes which Mamba boundaries get cached (perf/determinism). **Verify / port
   on the next rebase; not a cherry-pick emergency.**

Also present-and-verified (no action): Gumbel-max fp32 sampler (`268be1327` = fork `2c43cacff`), Mamba
`selective_state_update` dummy-pointer fix (`6e5a8c11f` = fork `5f0d342d8`), Mamba decode-cache refresh after refit
(`fbbc142b8` ≙ fork `f7f625318` post_refit hook, reached via `swap_model_weights → reshard_model_weights`),
Mamba prefix-skip clamp walk-back (`47f079ae2` ≙ fork `43004649e`), vLLM fused-MoE persistent buffers (`55b2e6ceb`
≙ fork `d276ccdfa`), prefix-cache write redirect to the dummy block (`d0612089d` ≙ fork `b7638dbbf`+`e6ba7b82c`,
upstream version slightly more complete), prefix cache re-salted per weight generation (fork `ce0e6bc43`; upstream
has `_weight_epoch`/`_weight_scoped_salt`).

## 1. Fork point and ranges

| item | value |
|---|---|
| fork HEAD | `880de0fce84321c04a81a052722f1b909014bac6` 2026-09-10 15:34 -0500 "debug: announce PR#6598 and PR#15 once per process" |
| merge-base with upstream `main` | `14346b65a2d0790e451919858f7771078105c5f0` 2026-08-12 05:06 UTC "Eliminate the ShardedObject all_gather_object on the FullyParallel load path (#5551)" |
| upstream `main` head | `fb6a123a09f524a2abfca81a78b6478358ca77e0` 2026-09-20 09:04 UTC |
| upstream `main` commits in `14346b65a..fb6a123a0` | **456** (all first-parent squash merges; 2026-08-12 → 2026-09-20) |
| fix-like titles among them (fix/bug/race/deadlock/hang/leak/prevent/avoid/harden/correct/restore/guard/revert) | 118 |
| fork-only commits `14346b65a..880de0fce` | 31 (29 patches + 2 merges) |
| upstream `dev` head | `d909db3a6` 2026-09-20; merge-base(dev, main) = `f41ec5495` 2026-07-20 |
| `dev` commits not on `main` | 701 (`git rev-list`), 653 non-merge per `git cherry`, 646 with no patch-equivalent on main, 331 touch priority paths, 271 of those have no same-title commit anywhere on main |

Commits by touched area on `main` since the merge-base (a commit can count in several; fix-like count in parentheses):

| area | commits |
|---|---|
| megatron/core/inference | 50 (18) |
| megatron/core/ssm | 28 (7) |
| megatron/core/resharding | 13 (6) |
| megatron/core/transformer/moe | 28 (10) |
| megatron/core/distributed (non-FSDP) | 11 (4) |
| megatron/core/distributed/fsdp | 45 (9) |
| megatron/core/optimizer | 29 (14) |
| megatron/core/dist_checkpointing | 4 (0) |
| other core model/transformer/TE/fusions/tensor_parallel/models | 128 (37) |
| megatron/training + legacy + rl | 102 (22) |
| only CI/docs/tests/examples | 117 |

Full lists: `/home/rkirby/.claude/jobs/e247f0cd/tmp/main_commits.txt` (hash|date|author|subject),
`main_commits_areas.txt` (hash|date|subject|areas), `main_commits_files.txt`, `dev_cherry.txt`, `dev_priority_unique.txt`.

## 2. Fork-only patches (`14346b65a..880de0fce`) and their upstream status

| fork commit | subject | upstream equivalent |
|---|---|---|
| `b7638dbbf`, `e6ba7b82c` | prefix-cache: don't rewrite hash-matched blocks; kernel skips dummy block | `d0612089d` (#6895) — upstream also redirects a chunk that resumes inside an earlier-matched partial block (`req.num_matched_prefix_blocks`, absent in fork) |
| `7747d352c`, `f9b0ca9db` | named module logger in dynamic engine | `aaa4dfd9c` (#6001) |
| `09ceec82e`…`c694bd05d`, `245761fc0`, `2661c803d` | tokenizer off event loop, detok in HTTP server, coordinator wire format, SO_REUSEPORT frontends, bcast dp addr | `114888079` (#6223), `ed6ac270b` (#6497) — upstream versions differ substantially (coordinator.py 425 / handlers.py 291 / inference_client.py 349 changed lines vs fork) |
| `1ae2bb73d`, `19ea4ab52` | prefix-affinity routing, kv-routing cost fn, HYBRID CG sizing | `ed6ac270b` (#6497), `634e8b5ca` (#6667) |
| `f7f625318` | refresh Mamba decode cache after refit (`post_refit` hook) | `fbbc142b8` (#6672) (`refresh_cache` inside `execute_reshard_plan`) |
| `d276ccdfa` | persistent vLLM fused-MoE buffers | `55b2e6ceb` (#6618) |
| `111b44de8` | skip renorm for unfiltered rows in processed log-probs | folded into `f76d34522` (#6993) |
| `43004649e` | keep Mamba state available when clamping prefix skip | `47f079ae2` (#6930) (same logic, `torch.isin` refactor) |
| `ce0e6bc43` | scope prefix cache to weight generation | upstream has `_weight_epoch` + `_weight_scoped_salt` |
| `a7bba0d9d`, `8253359a2`, `880de0fce` | debug: logit dumps for improbable tokens, sampler/PR announcements | fork-only (debug) |
| `8b880867e` | honor all model-declared EOS tokens | fork-only (upstream still terminates on `termination_id` only) |
| `2c43cacff` | Gumbel-max exponential-race sampling in fp32 | `268be1327` (#7182) (upstream adds `MIN_SAMPLING_TEMPERATURE`, `is_no_op_top_{k,p}`, fp32 processed-logprob buffer) |
| `5f0d342d8` (+ `9d47c94d4`, `f8d49fce8`, `79a94d30d`, `3e7addc97`) | Mamba `selective_state_update` no-intermediate-state dummy pointer | `6e5a8c11f` (#6598) identical |
| `fcc4d283c`, `367b7b53f` | Qwen3-Coder tool parser: vLLM argument coercion + grammar (fork PR#15) | fork-only; upstream has a different parser fix `f7b220977` (#6411, absent in fork) |

`git cherry upstream/main 880de0fce 14346b65a` marks all 29 as `+` (no byte-identical upstream patch); the mapping above is by content.

## 3. Ranked table of fixes that matter for this workload

Present = whether `880de0fce` already contains the fix (by code comparison/grep in the live tree). Action: CP = cherry-pick.

| # | SHA | date | title (PR) | area | why it matters here | present | action |
|---|---|---|---|---|---|---|---|
| 1 | `487bb3948` | 2026-08-13 | Avoid passing -1 sentinels into inference kernels (#6481) | inference/KV block table | `_cpu_mha_block_table[:real_bs]` copies rows that still hold `-1` for unallocated blocks; attention kernels read them as real block ids. Can corrupt attention output → sampled tokens/logprobs. Same code lines exist unchanged in fork (`dynamic_context.py:2506-2508`). | no | **CP** |
| 2 | `79be654e3` | 2026-09-10 | Fix order local refit copies after source updates (#6823) | resharding | adds `copy_stream.wait_stream(current_stream)` before local copies in nvshmem/nccl/gloo/nixl services; fork only has the trailing `current_stream.wait_stream(copy_stream)`. Without it local copies can read pre-update source data → silently stale slices after refit. | no | **CP** |
| 3 | `94939b85d` | 2026-08-13 | Handle hidden chunked prefill during async overlap (#6442) | inference/chunked prefill | assertion crash when the chunked-prefill request is not admitted for a step under memory pressure (`text_generation_controller.py:2332` in fork still asserts `== 1`). | no | **CP** |
| 4 | `8dc7ed381` | 2026-09-09 | Preserve exact expert token counts for expert-bias updates (#7081) | MoE router (training) | count buffer int64 + exact comparison; fork's float32 buffer is cast to bf16 by `Float16Module`/`MoEFloat16Module` (only `expert_bias` is restored to fp32). Expert-bias updates are enabled in this run. | no | **CP**, verify dtype |
| 5 | `19abd1fc4` | 2026-08-25 | Fix deadlock when initializing ZMQ communication between inference engine ranks (#6779) | inference/coordinator | PUB/SUB slow-joiner: MP ranks can miss the first broadcast → hang at engine start (`RankedPubSub.wait_for_subscribers`). Startup robustness for 64-86 node jobs. | no | CP when convenient |
| 6 | `6807fb8e4` | 2026-09-02 | Add current prefix-cache pairwise coverage and fixes (#7031) | inference/prefix cache + Mamba slots | Mamba state is cached at every block-aligned chunk end (fork: only on the final chunk when prompt is block aligned) → fewer Mamba back-offs on hits, more deterministic reuse; correct `num_cached_tokens`; `merge_lists` tolerates a None first segment; routing_indices reconstruction. | no | port on rebase / verify |
| 7 | `3972b96bb` | 2026-09-14 | Harden dynamic inference request lifecycle handling (#7256) | inference/engine+request | (a) partial prefill chunk's selected logprob now uses the known next prompt token (this is what moved the chunked-prefill *prompt-logprob* goldens `7f7c424fc`, `ad29cd0d3`); (b) stop-word trimming across eviction checkpoints keeps tokens/logprobs aligned; (c) RECOMPUTE resume preserves row order; (d) `is_prefill` logprob append guarded; (e) API: `finished_requests` flattened, `RequestRecord.serialize` removed, `add_request` rejects duplicate ids. We skip prompt logprobs, use no stop words, run `persist` → no token effect here; API break for NeMo RL on rebase. | no | verify on rebase |
| 8 | `8da558d5f` | 2026-09-04 | Add current async scheduling pairwise coverage and fixes (#7053) | inference/async sched | clears pending async logits on `engine.reset()` and on RECOMPUTE suspend; compacts every `active_request_metadata` field (fork compacts the three it has); asserts logprobs present when requested; RoPE `inv_freq` moved to GPU lazily. Stale-logit fix only bites in RECOMPUTE mode. | no | verify (needed if recompute-after-refit is enabled) |
| 9 | `6ffe9f732` | 2026-08-28 | Enable async scheduling by default (#6851) | inference/async sched | besides the default, clears stale pending logits when a lifecycle reset removed all requests (again RECOMPUTE). We already set `AsyncScheduleMode.ASYNC`. | no | ignore (verify with recompute) |
| 10 | `d0612089d` | 2026-08-26 | Prevent overwriting matched prefix cache blocks with recomputed KV (#6895) | inference/prefix cache | fork has its own variant; upstream additionally protects the partial block carried from the previous chunk when it was itself hash-matched. Fork can rewrite such a shared block with recomputed (same-token, low-bit different) KV → run-to-run nondeterminism, not stale content. | partial | port refinement on rebase |
| 11 | `f7b220977` | 2026-08-13 | Fix tool call reasoning boundary (#6411) | server/parsers | `<tool_call>` inside an unterminated `<think>` now ends reasoning and is parsed; fork returns it as reasoning with no tool call → the agent loop sees no tool call. Changes trajectory shape for agentic rollouts if the model ever does this. (Prior dump analysis found parser behaviour identical across engines.) | no | verify on dumps (count `<tool_call>` before `</think>`) |
| 12 | `a145d2a19` | 2026-09-02 | Abort inference requests when the HTTP client disconnects (#6916) | server | fork endpoints have no disconnect detection; requests orphaned by Gym-side timeouts keep decoding up to 196k tokens, consuming KV/compute. Capacity/latency, not token values. | no | CP medium |
| 13 | `af6d4a985` | 2026-08-28 | initialize NCCL on idle ranks (#6955) | resharding/NCCL | hang when a valid plan leaves a rank with no ops (first NCCL P2P lazily creates the communicator collectively). Only for `refit_backend=nccl`. | no | CP if nccl backend is used |
| 14 | `325c9936b` | 2026-09-14 | bound NCCL refit P2P groups to avoid kernel-plan split (#7189) | resharding/NCCL | NCCL deadlock when one ncclGroup holds a whole model of P2P ops (pytorch#174288); planner caps params per batch (32) and windows task ids (256). Only `nccl` backend; fork planner has no execution batching. | no | CP if nccl backend is used |
| 15 | `f76d34522` | 2026-09-03 | Avoid emitting NaN/Inf logprobs (#6993) | inference/sampling | tempered fallback for -inf *processed* logprobs, `top_p>=1`/`top_k<=0` normalised as no-op, `MIN_SAMPLING_TEMPERATURE`. With `raw_logprobs`, temperature 1.0, top_p 1.0 the fork already yields the same distribution (top_p=1.0 is a no-op filter in `filter_logits`; only the dispatch flag treats it as active). | no | ignore unless switching to processed logprobs / FlashInfer sampler |
| 16 | `4449eec63` | 2026-09-17 | Fix slow request serialization for raw Tensors (#7352) | inference/coordinator | msgpack of raw tensors on the engine→coordinator→API path; with 100k+ token prompts serialization cost is real. Perf only. | no | consider on rebase |
| 17 | `9b9dbb7f0` | 2026-09-17 | EP=1 inference dispatchers get EP>1 metadata (#7426) | MoE inference | EP=1 only; inference runs EP4. | no | ignore |
| 18 | `ad740d823` | 2026-08-26 | preserve batch invariance for Nemotron Nano (#6893) | MoE/SSM/attention inference | only under `batch_invariant_mode` (not enabled). | no | ignore |
| 19 | `4e45f55f5`, `cc99d8e89` | 2026-09-14 / 08-25 | FlashInfer NVLS routing buffer race; batch-invariant NVLS | MoE inference dispatcher | NVLS dispatcher + FlashInfer backend only; default dispatcher is `nccl` all-gather and backend is `vllm`. | no | ignore |
| 20 | `dd468df08` | 2026-09-11 | Do not pass is_first_microbatch to TE when it is not maintained (#6965) | TE linear (training) | flag only tracked for fp8/fp4/kitchen; bf16 run passes a constant after the first forward → no effect. | no | ignore |
| 21 | `723db5a72` | 2026-08-20 | broadcast padding mask for expert bias counts (#6114) | MoE router | fork's `routing_map & (~padding_mask)` would raise a shape error if a padding mask were passed; training runs, so no mask is passed here. | no | ignore (verify if packed-seq padding masks are enabled) |
| 22 | `2e12e2cf7`, `e81281377`, `0e48cdf3e`, `89624531a`, `bb251bc0a` | Aug–Sep | MoE aux-loss counting/dtype/fusion flag, op-fuser precision override, GTP grouped-GEMM path | MoE training | aux loss coeff 0, no TE precision config, no GTP → no effect. | no | ignore |
| 23 | `c230f5959`, `3a179c123`/`26c7dc869`, `d5ff7ea72`, `32a03b77c`, `976bff9ce` | Aug–Sep | ChainedOptimizer nested step sync; FusedAdam empty-shard workaround (MFSDP v2 only); detach fp32 model params; DistOpt torch_dist resharding; Muon padding | optimizer | nested chained optimizers/Muon/MFSDP v2/fp32-param models are not used; `32a03b77c` only matters when resuming with a different DP size. | no | ignore / verify on DP-rescaled resume |
| 24 | `154f66547` | 2026-09-08 | dp_cp_group for tied-embeddings replica_id (#6888) | dist ckpt (model sharded_state_dict) | `tie_word_embeddings=False` for this model. | no | ignore |
| 25 | `def07afe7`, `23cc208ac`, `e2c2e55bf`, `b212e6d14` | Aug–Sep | RNG state per rank on load; persistent ckpt worker termination; "Remove mcore async"; NVRx version guard | megatron/training + dist_checkpointing | NeMo RL uses its own checkpoint path (imports only `dist_checkpointing.strategies.{torch,async_utils}`); `e2c2e55bf` deletes `filesystem_async.py`/`state_dict_saver.py` and reshapes `torch.py`/`async_utils.py` → API break for NeMo RL's async save on the next rebase; `b212e6d14` turns a missing nvrx version into "no async support" instead of an assert. | no | plan for rebase |
| 26 | `f7b5409a3`, `a6e191fda` | 2026-09-19 / 09-15 | FP32-output-logit fixes (TE GEMM output ownership; fused CE grad dtype; fused clamped sq-relu) | tensor_parallel/fusions | NeMo RL computes GRPO logprobs itself (no MLM fused CE); `_linear_forward` view-vs-owner change has no numeric effect. | no | ignore |
| 27 | `9d8183bb9` (#7003) + follow-ups in `vllm_fused_moe.py` | 2026-09-03 | TanH soft-clamped squared ReLU / SiTU-GLU incl. inference kernels | activations | feature, not a fix; this checkpoint has `mlp_hidden_act=relu2` and no clamp (HF config of exported step_5). | no | ignore |
| 28 | `6e5a8c11f` | 2026-08-27 | Mamba no-intermediate-state dummy pointer (#6598) | ssm | identical one-line fix in fork (`5f0d342d8`). | **yes** | none |
| 29 | `fbbc142b8` | 2026-08-20 | Fix Refit Cache for mamba (#6672) | ssm/resharding | fork refreshes `_A_neg_exp_cache` via `MambaMixer.post_refit()` from `reshard_model_weights`; NeMo RL reaches it through `swap_model_weights`. Upstream refreshes inside `execute_reshard_plan` before the final `synchronize()`; both order the refresh before graph replay. | **yes** | none |
| 30 | `47f079ae2` | 2026-08-28 | Mamba prefix cache clamp for <2 token chunk edge case (#6930) | inference/Mamba prefix | fork `43004649e` has the same walk-back. | **yes** | none |
| 31 | `268be1327` | 2026-09-18 | Gumbel-max exponential race in fp32 (#7182) | sampling | fork `2c43cacff`; only cosmetic/normalisation deltas remain (see #15). | **yes** | none |
| 32 | `55b2e6ceb`, `634e8b5ca`, `aaa4dfd9c` | Aug | persistent vLLM MoE buffers; HYBRID CG sizing default; named logger | inference | fork carries pre-merge versions. | **yes** | none |

Refactors that are not fixes but will make the next rebase non-trivial: `1386505fe` (hybrid per-layer configs),
`c2382665e` (backend selection via `BackendSpecProvider`), `42b8a6911`/`a1a09b084` (hybrid specs immutable at runtime),
`de4a82e5c` (SSM ops dir moves; `mamba_ssm.py` → `ops/mamba2/mamba_ssm.py`), `731b79146` (SSM state handoff),
`3972b96bb` (engine API), `e2c2e55bf` (mcore async removal).

## 4. Fixes that can change sampled tokens or returned logprobs

- `487bb3948` (absent): wrong block ids reach the attention kernel for real rows → attention values for those rows can
  be wrong; this is the only absent fix with a direct, per-step path to different tokens and different raw logprobs.
- `79be654e3` (absent): stale slices after refit mean the engine serves a weight mix (old/new) until the next refit →
  a systematic train/inference mismatch that vLLM (separate weight sync) would not share.
- `3972b96bb` (absent): changes the *prompt* logprob recorded at a partial chunk boundary and stop-word/logprob
  alignment across eviction checkpoints. With `skip_prompt_log_probs=True` and no stop words, generated-token logprobs
  are unaffected. The golden-value refreshes on 2026-09-14/16 are explained by this commit, i.e. upstream did not
  change generated-token numerics in this window.
- `8da558d5f`/`6ffe9f732` (absent): stale pending async logits after a RECOMPUTE suspend/resume would be consumed by
  the first post-resume step; with `kv_cache_management_mode=persist` the pending forward stays valid. Relevant only if
  recompute-after-refit is enabled.
- `f76d34522`/`268be1327` normalisation of `top_p>=1`: fork already treats `top_p=1.0` as a no-op filter in
  `TorchSampling.filter_logits` (`0.0 < top_p < 1.0`), so the sampled distribution is identical; the fork's dispatch
  flags merely take the slower per-bucket path. Raw logprobs are computed in fp32 (`.float()` before `log_softmax`) in
  both trees.
- Not upstream fixes but fork deltas that affect termination: `8b880867e` (all model EOS tokens terminate) — shortens,
  never lengthens, relative to upstream.

## 5. Refit / resharding fixes that can leave stale weights or state

- `79be654e3` — missing stream ordering for local copies (see above). Applies to the nvshmem service used here
  (`_local_copy_stream`) as well as nccl/gloo.
- `fbbc142b8` — Mamba `_A_neg_exp_cache` refresh: present in fork (post_refit hook). The hook runs after
  `execute_reshard_plan` has synchronised and before NeMo RL resumes/unpauses the engine.
- Prefix cache across refit: fork bumps `_weight_epoch` on `resume()` (`ce0e6bc43`), so blocks cached under old
  weights are not matched by new requests; upstream has the same mechanism. In `persist` mode in-flight requests keep
  their old KV/SSM state by design (in-order lag 1).
- MoE `expert_bias` is a persistent buffer refit alongside parameters (`get_refit_tensor_dict` includes persistent
  buffers) in both trees; the count buffer is non-persistent and not refit (correct).
- NCCL-backend-only hang fixes (`af6d4a985`, `325c9936b`) do not cause staleness but a hang; nvshmem service in the
  fork differs from upstream only by the missing `wait_stream` line.

## 6. `dev` branch

`dev` diverged from `main` on 2026-07-20 (`f41ec5495`) and carries 646 patches with no equivalent on `main`. Of the 271
that touch priority paths and have no same-title counterpart on `main`, the inference/SSM/resharding ones are
Gated-Delta-Net / Qwen3-Next / Ling-V3 / DeepSeek-V4 / mHC features, dev-side main→dev sync repairs
(2026-09-04 `8c00fe023` "fix(inference): restore synchronized async and cache contracts", `324e065a6` "restore async
scheduling token commit", `c896badce`), and a reverted 100-commit sync (`5ccdc5527`/`68447eeaa`). Nothing on `dev` is a
fix for a main-based fork's Mamba2/MINF path; ignore `dev` for cherry-picks.

## 7. Caveats

- GitHub API was rate-limited from this host and `gh` is not installed, so PR descriptions were not available; every
  assessment above is from the diffs themselves.
- The base HF model directory (`/lustre/fsw/portfolios/llmservice/...`) is not mounted on the login node; model facts
  come from the exported `runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf/checkpoints/step_5/hf/config.json`.
- Kernel-level impact of `487bb3948` (which FA variant reads trailing block ids) was not tested; the assessment is from
  the upstream comment and the shared block-table plumbing (`mha_metadata.py` → `attention.py` FA2/FA3/FA4 calls).

## 8. Commands used

```
# fork trees (read-only)
cd .../swe_dump/Megatron-LM && git remote -v && git log --oneline -1 880de0fce && git branch -a
git merge-base 880de0fce upstream/main            # 14346b65a (in-tree upstream/main was at cef2973b3, 2026-09-15)
git log --format='%h %ci %an | %s' 14346b65a..880de0fce
git show <fork sha> --format= -- <paths>          # 43004649e 5f0d342d8 f7f625318 b7638dbbf e6ba7b82c 880de0fce
grep -n ... megatron/core/inference/... megatron/core/resharding/... megatron/core/ssm/... megatron/core/transformer/moe/router.py

# fresh clone for upstream history
git clone --filter=blob:none https://github.com/NVIDIA/Megatron-LM.git /home/rkirby/.claude/jobs/e247f0cd/tmp/mlm_review
git remote rename origin upstream; git remote add fork https://github.com/ArEsKay3/Megatron-LM.git; git fetch --filter=blob:none fork
git remote set-url fork git@github.com:ArEsKay3/Megatron-LM.git   # lazy blob fetch for fork objects needs SSH
git merge-base 880de0fce upstream/main; git rev-list --count 14346b65a..upstream/main
git log --format='%h|%ci|%an|%s' 14346b65a..upstream/main > main_commits.txt
git log --format='COMMIT %h|%ci|%s' --name-only 14346b65a..upstream/main > main_commits_files.txt   # + awk area classifier
git cherry -v upstream/main upstream/dev > dev_cherry.txt; git cherry -v upstream/main 880de0fce 14346b65a
git show <sha> --format= --stat / -- <paths>      # for every candidate listed in section 3
git diff 880de0fce upstream/main -- megatron/core/inference/sampling/torch_sampling.py megatron/core/ssm/{ssm_inference,mamba_mixer}.py \
    megatron/core/inference/moe/vllm_fused_moe.py megatron/core/resharding/{execution,refit}.py
git diff --stat 880de0fce upstream/main -- megatron/core/inference/... megatron/core/resharding/... megatron/core/ssm/...
git log -S'<symbol>' --format='%h %ci %s' 14346b65a..upstream/main -- megatron/core/inference/   # decode_only_cuda_graphs, is_decode_only, termination_id, num_tokens_to_generate, stop_word, generated_length
git show upstream/main:<path> | grep -n ...        # request_metadata keys, is_decode_only, _weight_epoch, newly_paused usage

# NeMo RL side (read-only)
grep -rn "megatron.core.inference\|megatron.training\|megatron.core.resharding" nemo_rl/; sed -n ... nemo_rl/models/generation/megatron/megatron_worker.py
config chain: examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_dump_minf.yaml -> swe_sc_cmh_minf.yaml -> swe_sc_cmh_stream4.yaml -> swe_sc_cmh_common.yaml -> examples/configs/grpo_math_1B.yaml
python3 -c 'json.load(open(".../checkpoints/step_5/hf/config.json"))'
```

# Sampler audit: MINF (Megatron in-engine) vs vLLM rollouts, SWE dump runs

Date: 2026-09-22. Read-only code audit; no files outside this report were touched.

Question: which sampling code path produced the rollout tokens in (1) the MINF runs
(`swe_sc_cmh_dump_minf.yaml`) and (2) the vLLM runs (`swe_sc_cmh_dump_vllm.yaml`), and are the
two paths the same distribution given identical logits. The earlier project note claimed
"samplers identical: fp32 softmax + exponential race, no top-p/top-k, T=1".

Verdict up front: VERIFIED for the runs as launched, with caveats listed in section 6.

## 0. Trees and versions audited

| Component | Location | Version in effect |
|---|---|---|
| NeMo RL | `swe_dump/nemo_rl` | `7f8a2b9d` (`rkirby/swe-v2-dump`) |
| Megatron-LM (mounted into MINF runs) | `swe_dump/Megatron-LM` = `launch_swe_dump.sh:20-21,135` -> `/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM` | fork `880de0fce` (upstream base `14346b65a`) |
| NeMo Gym | `swe_dump/nemo_rl/3rdparty/Gym-workspace/Gym` | `354babf7e` |
| nv-OpenHands (agent) | `Gym/cache/swe_agents/swe_openhands_setup/nv-OpenHands-dd7d06ea/5f0180054732945df08ad2293903e6873f0492b6/OpenHands` | `5f0180054` (pinned in `swebench_openhands_training.yaml`) |
| vLLM, pinned | `nemo_rl/pyproject.toml:147-150`, `nemo_rl/uv.lock:8424-8426` | 0.25.1 |
| vLLM, actually run | 132 `Initializing a V1 LLM engine (v0.25.1)` lines in `runs/nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918/ray_logs/3847610-logs/ray/*/session_*/logs/worker-*.out` | 0.25.1 (container `rl-gym.63635108-zstd.sqsh`, `launch_swe_dump.sh:38`) |
| vLLM source on disk | `~/.cache/uv/archive-v0/gZiZ2202O68pnN97FQqKm/vllm` (also `T_YNyNNqo4mpbOYtAoVbT`); `swe_dump/analysis/vllm_container_src/` is empty | **0.20.0 only** (`vllm-0.20.0.dist-info`) |

All vLLM file:line citations below are from the 0.20.0 copy and are labelled `[vLLM 0.20.0 src]`.
The 0.25.1 code that ran is not on disk; where a 0.25.1 fact is taken from run logs it is labelled
`[0.25.1 log]`. Everything else is from the trees above.

Runtime confirmation of the MINF path (`grep` over the first job of every `minf_dump` run dir):

| MINF run dir (`runs/nano35-swe-v2-stream128-inorder1-cmh-64n-…`) | job | files announcing `[minf-sampler] TorchSampling: gumbel-max exponential race, fp32 (PR#16 PRESENT) \| vocab_size=131072` | FlashInfer sampler markers |
|---|---|---|---|
| `minf_dump-from10-20260918` (chain B) | 3847572 | 128 | 0 |
| `minf_dump-nopg-noprefix-20260919` (chain C) | 3870192 | 128 | 0 |
| `minf_dump-fromvllm10-20260920` (chain F) | 3878147 | 128 | 0 |
| `minf_dump-from0-20260920` (chain E) | 3875886 | 128 | 0 |
| `minf_dump-from0-nopg-noprefix-20260920` (chain G) | 3880119 | 128 | 0 |
| `minf_dump-from0-prefix-nopg-r1-20260921` | 3901530-1 | 128 | 0 |
| `minf_dump-from0-prefix-nopg-r2-20260921` | 3901609-1 | 0 (no engine announcement anywhere in its logs; 1407 worker logs, engine apparently never initialised in the captured logs) | 0 |

128 = one announcement per generation rank (32 nodes x 4 GPUs). The announcement is printed by
`Megatron-LM/megatron/core/inference/sampling/torch_sampling.py:27-34`.

## 1. Configuration actually in effect (deliverable d)

Yaml chain: `swe_sc_cmh_dump_{minf,vllm}.yaml` -> (`swe_sc_cmh_minf.yaml` ->) `swe_sc_cmh_stream4.yaml`
-> `swe_sc_cmh_common.yaml` -> `examples/configs/grpo_math_1B.yaml`. Files under
`swe_dump/nemo_rl/examples/nemo_gym/nemotron-3.5-nano/` unless stated.

Common to both arms:

| Key | Value | Source |
|---|---|---|
| `policy.generation.temperature` | 1.0 | `swe_sc_cmh_common.yaml:315` |
| `policy.generation.top_p` | 1.0 | `swe_sc_cmh_common.yaml:316` |
| `policy.generation.top_k` | null | `swe_sc_cmh_common.yaml:317` |
| `policy.generation.stop_token_ids` / `stop_strings` | null / null | `swe_sc_cmh_common.yaml:318-319` |
| `policy.generation.max_new_tokens` | `${policy.max_total_sequence_length}` = 196608 | `swe_sc_cmh_common.yaml:314,150` |
| `policy.generation.val_temperature/val_top_p/val_top_k` | `${.temperature}` etc. = 1.0 / 1.0 / null | inherited `examples/configs/grpo_math_1B.yaml:375-377` |
| `policy.precision` | bfloat16 | `swe_sc_cmh_common.yaml:151` |
| `grpo.seed` | 42 | `swe_sc_cmh_common.yaml:74` |
| tokenizer/model | `.../akamehra/swe_e2e_corrected/base_model/step_18/hf` (`vocab_size` 131072, `generation_config.json`: `do_sample true, eos_token_id [2, 11], top_p 0.95`) | `launch_swe_dump.sh:35`; model dir |

vLLM arm (`swe_sc_cmh_dump_vllm.yaml`):

| Key | Value | Source |
|---|---|---|
| `policy.generation.backend` | vllm | `swe_sc_cmh_common.yaml:312` |
| `vllm_cfg.logprobs_mode` | processed_logprobs | `swe_sc_cmh_dump_vllm.yaml:9` (also base `grpo_math_1B.yaml:415`) |
| `vllm_cfg.precision` / `kv_cache_dtype` | bfloat16 / auto | `swe_sc_cmh_common.yaml:322-323` |
| `vllm_cfg.enforce_eager` | false | `swe_sc_cmh_common.yaml:329` |
| `vllm_cfg.enable_prefix_caching` | absent -> True (`_resolve_enable_prefix_caching`, `nemo_rl/models/generation/vllm/vllm_worker.py:84-88`) | engine log `enable_prefix_caching=True` [0.25.1 log] |
| `vllm_kwargs` | attention_backend FLASH_ATTN, moe_backend triton, max_num_batched_tokens 8480, mamba_ssm_cache_dtype float32, mamba_cache_mode align, cudagraph_capture_sizes [1,2,4,8,16,32,64] | `swe_sc_cmh_common.yaml:352-361` |
| speculative decoding | none (`speculative_config=None`) | [0.25.1 log] |
| engine `seed` | node_idx*1024 + bundle_id (observed `seed=30720`) | `vllm_worker.py:161-169`; [0.25.1 log] |
| engine dtype / TP | bfloat16 / 4 | [0.25.1 log] |
| launch overrides | none touching sampling (`MINF_OVERRIDES=()`) | `launch_swe_dump.sh:71-74` |

MINF arm (`swe_sc_cmh_dump_minf.yaml` -> `swe_sc_cmh_minf.yaml`):

| Key | Value | Source |
|---|---|---|
| `policy.generation.backend` | megatron | `swe_sc_cmh_minf.yaml:7` |
| `mcore_generation_config.transformer_impl` | inference_optimized | `swe_sc_cmh_minf.yaml:18` |
| `mamba_inference_ssm_states_dtype` | float32 | `swe_sc_cmh_minf.yaml:20` |
| `inference_grouped_gemm_backend` | vllm | `swe_sc_cmh_minf.yaml:21` |
| `cuda_graph_impl` / `inference_cuda_graph_scope` / `num_cuda_graphs` / `use_cuda_graphs_for_non_decode_steps` | local / block / -1 / true | `swe_sc_cmh_minf.yaml:24-27` |
| `materialize_only_last_token_logits` | true | `swe_sc_cmh_minf.yaml:34` |
| `num_speculative_tokens` | 0 | `swe_sc_cmh_minf.yaml:37` |
| `enable_prefix_caching` | false (yaml) and forced false by launcher | `swe_sc_cmh_minf.yaml:32`; `launch_swe_dump.sh:66` |
| `expose_http_server` | true | `swe_sc_cmh_minf.yaml:36` |
| `NRL_MINF_SAMPLING_BACKEND` | **torch** (launcher default; nemo_rl's own default is `flashinfer`) | `launch_swe_dump.sh:62`; `nemo_rl/models/generation/megatron/megatron_worker.py:265-267` |
| `NRL_MINF_LOGPROBS_MODE` | raw_logprobs | `launch_swe_dump.sh:63`; `megatron_worker.py:312` |
| `InferenceConfig.sampling_backend` default / `logprobs_mode` default / `offset_sampling_seed_by_dp_rank` | 'torch' / 'raw_logprobs' / True | `Megatron-LM/megatron/core/inference/config.py:452,468,456` |
| `TransformerConfig.inference_sampling_seed` / `deterministic_mode` | 42 / False (nemo_rl overrides neither; grep found no use of either in `nemo_rl/models/generation/megatron/`) | `megatron/core/transformer/transformer_config.py:1135`; `megatron/core/model_parallel_config.py:229` |
| launch overrides | `overlap_param_gather=false`, `enable_prefix_caching=false` | `launch_swe_dump.sh:64-67` |

Evidence the resolved values reached the runs: vLLM run `checkpoints/step_10/config.yaml`
(`temperature: 1.0`, `top_k: null`, `top_p: 1.0`, `val_*` same, `vllm_cfg.logprobs_mode: processed_logprobs`,
`enforce_eager: false`, `precision: bfloat16`, `grpo.seed: 42` at line 229); chain C `checkpoints/step_16/config.yaml`
(`backend: megatron`, `enable_prefix_caching: false`, `overlap_param_gather: false`, same T/top_p/top_k;
its `step_10` is the copied seed checkpoint and carries the main chain's `enable_prefix_caching: true`).
Both `ray-driver.log` `MasterConfig(...)` lines agree.

## 2. Request path from the agent to the engine (shared by both arms)

The OpenHands agent never calls the engine directly; every request is stamped by NeMo RL and proxied by Gym.

1. NeMo RL builds the Responses request: `nemo_rl/experience/rollout_manager.py:884-893`
   sets `responses_create_params["temperature"] = generation.temperature` (1.0), `["top_p"]` (1.0),
   `["max_output_tokens"] = min(existing, max_new_tokens=196608)`.
2. Gym SWE wrapper maps them to `inference_params`: `Gym/responses_api_agents/swe_agents/app.py:3570-3578`
   (`temperature`, `top_p`, `max_output_tokens -> tokens_to_generate`), then writes them into the OpenHands
   config `[llm.model]` at `app.py:1716-1725` (`oh_config.toml` itself only sets `custom_llm_provider = "openai"`,
   `native_tool_calling = true`).
3. nv-OpenHands builds the request kwargs: `OpenHands/openhands/llm/llm.py:123-133`
   (`temperature`, `max_completion_tokens`, `top_p` if not None); `llm.py:225-228`
   `_nemo_gym_llm_kwargs = kwargs | {model, seed: config.seed}`; `LLMConfig.seed` default None
   (`openhands/core/config/llm_config.py:92`), `top_k` default None (`:73`) so not sent.
   `agenthub/codeact_agent/codeact_agent.py:236` -> `agenthub/nemo_gym_client.py:79-83,105-110`:
   body = `{messages, **_nemo_gym_llm_kwargs, tools}` POSTed to Gym `/v1/chat/completions`.
   Not sent: `tool_choice` (defaults to `auto` in vLLM, `protocol.py:696-699` [vLLM 0.20.0 src]; treated as
   parser-only in MCore, `chat_completions.py:471-473`), `stop` (only in the mock-function-calling branch,
   `llm.py:335-340`, not taken), `response_format`, penalties, `min_p`, `logit_bias`, `logprobs`.
4. Gym `vllm_model` proxy: `Gym/responses_api_models/vllm_model/app.py:477-487` adds
   `logprobs=True, top_logprobs=0, return_tokens_as_token_ids=True`; sets `model`, `chat_template_kwargs`;
   `extra_body` and `sampling_overrides` (`app.py:190,193`) are unset in
   `vllm_model_for_training.yaml` and in the `swe_sc_cmh_common.yaml:432-439` override, so
   `temperature=1.0`, `top_p=1.0`, `seed=null`, `max_completion_tokens` pass through unchanged; `n=1`
   enforced (`app.py:925-927`).
5. Engine-side parsing differs, see 3.1 and 4.1.

NeMo RL's internal (non-HTTP) `generate()` paths are not used by the SWE agent but set the same values:
`nemo_rl/models/generation/vllm/vllm_worker.py:641-668` (`top_k=-1` when null, `logprobs=0`) and
`nemo_rl/models/generation/megatron/megatron_worker.py:641-662` (`top_k=0` when null, `top_p=1.0`,
`return_log_probs=True`, `skip_prompt_log_probs=True`).

## 3. MINF: from request arrival to sampled token

### 3.1 HTTP frontend (Megatron-LM fork)
`megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py`
- `:461` `chat_completions()`; prompt tokenised with `apply_chat_template`.
- `:592-594` `temperature = float(req.temperature or 1.0)`, `top_p = float(req.top_p or 1.0)`,
  `top_k = int(req.top_k or 0)`; `:597-599` T==0 -> greedy.
- `:602-604` `logprobs=True` -> `return_log_probs=True`, `top_n_logprobs=0`, `skip_prompt_log_probs=True`.
- `:620` `max_tokens = max_completion_tokens or max_tokens`.
- `:634-649` `SamplingParams(temperature=1.0, top_k=0, top_p=1.0, return_log_probs=True, ...)`.
- Ignored request fields: `seed` (0 occurrences in the file), `min_p`, `repetition_penalty`,
  `frequency_penalty`, `presence_penalty`, `logit_bias`, `stop` (chat endpoint; only `/v1/completions` parses `stop`).
- `:698` `client.add_request(prompt_tokens, sampling_params, block_hashes)` -> ZMQ -> DP coordinator.

### 3.2 Engine admission
`megatron/core/inference/engines/dynamic_engine.py`
- `:1245` `add_request`; `:1179-1181` `num_tokens_to_generate None -> max_sequence_length - prompt_len`;
  `:1195-1208` clamp to remaining context; `:1183-1193` `termination_id None -> eod`.
- Per-request sampling metadata is pinned CPU tensors: `temperature float32, top_k int32, top_p float32`
  (`megatron/core/inference/inference_request.py:541-547`).
- Multi-EOS: `text_generation_controller.py:371-405` adds `generation_config.eos_token_id` ([2, 11]) to the
  tokenizer `eod`.

### 3.3 Engine construction (NeMo RL side)
`nemo_rl/models/generation/megatron/megatron_worker.py:236-333`: `InferenceConfig(sampling_backend=env
NRL_MINF_SAMPLING_BACKEND -> 'torch' (:265-267), logprobs_mode=env -> 'raw_logprobs' (:312),
materialize_only_last_token_logits=True, async_sched_mode=ASYNC (:298), num_speculative_tokens=0)`;
`DynamicInferenceContext`, `GPTInferenceWrapper`, `TextGenerationController`, `DynamicInferenceEngine`.

`megatron/core/inference/text_generation_controllers/text_generation_controller.py`
- `:253` `self.vocab_size = unwrapped_model.vocab_size` (131072, confirmed by the announcement).
- `:260-268` RNG: `self.sampling_rng = torch.Generator(cuda)`; `seed = model_config.inference_sampling_seed`
  (42) `+ dp_rank` (offset default True, `deterministic_mode` False); `manual_seed(seed)`.
- `:321-330` `_all_logits_cuda` static buffer, `dtype = params_dtype` (bf16), allocated because
  `cuda_graph_impl == "local"`.
- `:353-361` backend selection: not flashinfer -> `TorchSampling(self.sampling_rng, self.vocab_size)`.
- `InferenceConfig.__post_init__` silently falls back to torch if flashinfer is missing (`config.py:528-536`).

### 3.4 Per-step call chain (ASYNC schedule mode)
1. `dynamic_engine.py:2606` `async_step` -> controller `generate_output_tokens_dynamic_batch`
   (`text_generation_controller.py:3428`) -> `:3402-3405` dispatch on `async_sched_mode`
   -> `_run_async_sched_step_overlap` (`:3042`) / `_no_overlap` (`:2945`).
2. Forward: `_run_async_sched_forward` (`:2744`) -> `_dynamic_step_forward_logits` (`:854-904`)
   -> `inference_wrapped_model.run_one_forward_step` (`:870`) -> `HybridModel.forward`
   (`megatron/core/models/hybrid/hybrid_model.py:423`; the NemotronH model is `MambaModel(HybridModel)`,
   `megatron/core/models/mamba/mamba_model.py:12`) with `runtime_gather_output=True`
   (`megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py:126`;
   asserted at `hybrid_model.py:458`).
   - last-token rows selected before the output layer (`hybrid_model.py:601-619`);
   - `self.output_layer(...)` (`hybrid_model.py:621-623`) = `tensor_parallel.ColumnParallelLinear`
     (`hybrid_model.py:316`), bf16 GEMM (`megatron/core/tensor_parallel/layers.py:448-467`, `output_dtype`
     None -> input dtype);
   - TP gather inside the layer, **bf16**: `layers.py:1232-1249` -> NVLS symmetric-memory all-gather with
     `dtype=x.dtype` (`megatron/core/tensor_parallel/inference_layers.py:596-608`) or NCCL
     `gather_from_tensor_model_parallel_region` (`:608`);
   - `_scale_logits` is a no-op (`use_mup` False: `megatron/core/models/common/language_module/language_module.py:339-343`;
     default `transformer_config.py:421`).
   - logits `[1, n, 131072]` bf16 copied into the static bf16 buffer: `text_generation_controller.py:901-902`.
   - CUDA graph: at `inference_cuda_graph_scope=block` the `CudaGraphManager` lives on the HybridModel and
     wraps the whole forward including the output layer and TP gather whenever
     `using_cuda_graph_this_step()` (`hybrid_model.py:387-421`). The sampler is never captured: `TorchSampling`
     has no graph wrapper (`torch_sampling.py:212,218`), the engine warm-up only samples for flashinfer
     (`dynamic_engine.py:544-548`).
3. Sample: `_run_async_sched_sample` (`:2343`) -> `_dynamic_step_sample_logits` (`:1302-1332`):
   `n = active_request_count`, `gather_indices=None` (last-token logits already materialised),
   `no_top_k/no_top_p` flags from `_active_requests_sampling_filter_flags` (`:1388-1411`; note `no_top_p` is
   **False** for `top_p=1.0` because the test is `== 0.0`, `:1410`), then
   `self._sampling.sample_kernel(self._all_logits_cuda.squeeze(0), n, context, ..., output=self._sampled_tokens_cuda[:n])`.
4. `megatron/core/inference/sampling/torch_sampling.py`
   - `:185-264` `sample_kernel`: flags discarded (`:218`), requests bucketed by `(temperature, top_k, top_p)`
     read from pinned CPU metadata (`:221-235`); one bucket here `(1.0, 0, 1.0)`.
   - `:250-257` -> `sample_from_logits(logits[rows], 1.0, 0, 1.0, generator=self._rng, vocab_size=131072)`.
   - `:87-138` `sample_from_logits`: `top_k == 1` argmax branch not taken (`:116`);
     `filter_logits` (`:58-84`): `clone()` of the bf16 rows (`:74`), `if temperature != 1.0: div_` **skipped**
     (`:75-76`), top-k branch skipped (`top_k >= 1` false, `:77`), top-p branch skipped
     (`0.0 < 1.0 < 1.0` false, `:82`);
     `probabilities = filtered.softmax(dim=-1, dtype=torch.float32)` (`:122`) -> fp32 probs from bf16 logits;
     `q = torch.empty_like(probabilities); q.exponential_(generator=generator)` (`:131-132`);
     `sampled = probabilities.div_(q).argmax(dim=-1)` (`:133`) -> exponential race / Gumbel-max;
     `clamp(0, vocab-1)` no-op (`:135-136`).
5. Log-prob of the sampled token (`raw_logprobs`): `_run_async_sched_log_probs` (`:2452`) ->
   `DynamicInferenceContext.calculate_log_probs_tensors` (`megatron/core/inference/contexts/dynamic_context.py:4626`):
   `active_logits = logits_squeezed[:n].float()` (`:4656`) -> `_processed_log_probs` (`:4594`):
   `if logprobs_mode == "raw_logprobs": return F.log_softmax(logits, dim=-1)` (`:4616-4617`) in fp32;
   selected `log_probs[seq_idx, new_tokens]` (`:4660`) -> `request.generated_log_probs`
   (`dynamic_engine.py:1534-1572`) -> `logprobs.content[i].logprob` in the chat response
   (`chat_completions.py:777-807`). The processed branch (`:4624`, `TorchSampling.log_probs_kernel`,
   `torch_sampling.py:140-183`) is not used in these runs.

### 3.5 Fork provenance of the torch sampler
`git log -- megatron/core/inference/sampling/`: `8253359a2 debug(inference): announce the active TorchSampling
once per process`, `2c43cacff fix(inference): sample via Gumbel-max exponential race in fp32`, `111b44de8
fix(inference): skip renorm for unfiltered rows when reporting log-probs`, then upstream commits
(`541d5eef0`, `650b78382`, `a2bb5e543 Add logprobs_mode (raw/processed)`, `878228fd0 FlashInfer sampling`).
The exponential race is therefore a fork-only change ("PR#16"); upstream Megatron-Core's torch backend draws
differently (the fork's own comment at `torch_sampling.py:123-130` states it replaced a cumulative-sum draw,
i.e. `torch.multinomial`).

## 4. vLLM: from request arrival to sampled token

### 4.1 NeMo RL HTTP wrapper
`nemo_rl/models/generation/vllm/vllm_worker_async.py`
- `:731-779` `create_chat_completion`: `assert request.top_k in (None, -1)`; `request.top_k = -1` (`:737-740`);
  `assert request.top_p is not None` (`:753`, comment explains vLLM would otherwise use
  `generation_config.json`); asserts `(temperature, top_p)` equals the train `(1.0, 1.0)` or validation
  `(1.0, 1.0)` profile (`:759-774`); `_clamp_max_tokens` so prompt + output <= max_model_len (`:566-572`).
- Chat serving object: `NeMoRLOpenAIServingChat` with `return_tokens_as_token_ids=True`, `enable_auto_tools`,
  `tool_parser qwen3_coder`, `reasoning_parser nano_v3` (`:652-726`).

### 4.2 Engine construction
`nemo_rl/models/generation/vllm/vllm_worker.py:570-601`: `LLM(model=..., load_format='dummy', dtype=self.precision`
(bfloat16, `:324`), `seed=seed`, `enforce_eager=False`, `enable_prefix_caching=True` (`:84-88`),
`enable_sleep_mode=True`, `logprobs_mode='processed_logprobs'` (`:599-601`), `**vllm_kwargs`). `seed` =
`node_idx*1024 + bundle_id` (`:161-169`); observed `seed=30720` [0.25.1 log]. Not set: `generation_config`
(so vLLM's default `"auto"` imports `{'top_p': 0.95}` from the model's `generation_config.json`; the run logs
carry the warning `Default vLLM sampling parameters have been overridden by the model's generation_config.json:
{'top_p': 0.95}` [0.25.1 log]; harmless here because every request carries an explicit `top_p`),
`VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_USE_V2_MODEL_RUNNER` (no occurrences in `nemo_rl/` or `nano35_launch.sh`;
the run logs reference `[gpu_model_runner.py:…]`, i.e. the classic V1 runner, and no `[model_runner.py` lines).

### 4.3 Request -> SamplingParams  [vLLM 0.20.0 src]
- `vllm/entrypoints/openai/chat_completion/serving.py:154` `default_sampling_params = model_config.get_diff_sampling_param()`
  (`vllm/config/model.py:1415-1426` imports only `repetition_penalty, temperature, top_k, top_p, min_p,
  max_new_tokens` that are present in `generation_config.json`; here `{'top_p': 0.95}`).
- `serving.py:284-302` `max_tokens = get_max_tokens(max_model_len, max_completion_tokens|max_tokens, prompt_len,
  default_sampling_params, override)` = `min(max_model_len - prompt_len, requested, ...)`
  (`vllm/entrypoints/utils.py:174-200`); then `request.to_sampling_params(max_tokens, default_sampling_params)`.
- `vllm/entrypoints/openai/chat_completion/protocol.py:469-560`: each of `repetition_penalty, temperature, top_p,
  top_k, min_p` falls back to `default_sampling_params` then `_DEFAULT_SAMPLING_PARAMS`
  (`:443-449`: 1.0 / 1.0 / 1.0 / 0 / 0.0) only when the request field is None. Here temperature 1.0 and top_p 1.0
  are explicit, top_k is the explicit -1, min_p/repetition_penalty resolve to 0.0 / 1.0. `seed=self.seed` (None);
  `logprobs = self.top_logprobs if self.logprobs else None` -> 0 (`:549`); `tool_choice` defaults to `auto` when
  tools are present (`:696-699`) so no structured-output grammar.
- `vllm/sampling_params.py:476-481` "quietly accept -1 as disabled"; `:611-616` `sampling_type` = RANDOM
  (seed None -> no per-request generator).

### 4.4 Per-step call chain  [vLLM 0.20.0 src]
1. Seeding: `vllm/v1/worker/gpu_worker.py:275` `set_random_seed(model_config.seed)` in `init_device`, and
   `:692` again after warm-up/profiling; `vllm/utils/torch_utils.py:373-380` seeds `random`, `numpy`,
   `torch.manual_seed`, `manual_seed_all`. Per-request generators only for `RANDOM_SEED` requests
   (`vllm/v1/worker/gpu_model_runner.py:1140-1147`) -> none here.
2. Batch metadata: `vllm/v1/worker/gpu_input_batch.py:366-376`: `temperature` fp32; `top_p_reqs` only if
   `top_p < 1`; `top_k_reqs` only if `0 < top_k < vocab_size`, otherwise `top_k` stored as `vocab_size`;
   `SamplingMetadata.top_p/top_k = None` when no request uses them (`:890-891,1064-1069`).
3. Logits: `gpu_model_runner.py:4087` `logits = self.model.compute_logits(sample_hidden_states)` after the
   cudagraph-dispatched `_model_forward`; `vllm/model_executor/models/nemotron_h.py:929-934` ->
   `LogitsProcessor(config.vocab_size)` (`:867`; `scale=1.0`, `soft_cap=None`) ->
   `vllm/model_executor/layers/logits_processor.py:89-104`: bf16 `lm_head` matmul, TP gather/all-gather in
   **bf16** (`:75-87`), slice `[..., :org_vocab_size]` (131072, `:103`); no scaling/soft-cap (`:66-72`).
4. Sample: `gpu_model_runner.py:3329-3343` `_sample` -> `Sampler.forward` (`vllm/v1/sample/sampler.py:68-143`):
   `num_logprobs = 0` (not None) so a logprob tensor will be produced; `logits = logits.to(torch.float32)` (`:91`);
   `apply_logits_processors` (`:357-391`): `allowed_token_ids_mask` None, `bad_words` empty, built-in
   non-argmax-invariant processors are no-ops without requests (`vllm/v1/sample/logits_processor/builtin.py:101-103,
   161-164, 524-526`), `apply_penalties` returns early on `no_penalties` (`:399-400`);
   `sample` (`:232-288`): `all_random` -> `apply_temperature` = `logits.div_(temp.unsqueeze(1))` with fp32 1.0
   (`:217-226`, exact); argmax-invariant processors (min_p) no-op; `TopKTopPSampler.forward`
   (`vllm/v1/sample/ops/topk_topp_sampler.py`): constructor pins `forward_native` whenever `logprobs_mode` is
   `processed_*` (`:35-38, 73-92`), so the FlashInfer sampler is impossible in this configuration;
   `forward_native` (`:94-113`): `apply_top_k_top_p(logits, None, None)` returns logits unchanged
   (`:245-249`), `logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)` (`:110-111`),
   `probs = logits.softmax(dim=-1, dtype=torch.float32)` (`:112`), `random_sample(probs, generators={})`
   (`:325-346`): `q = torch.empty_like(probs); q.exponential_()` (default CUDA generator, `:334-340`),
   `probs.div_(q).argmax(dim=-1)` (`:346`).
5. Log-prob: `raw_logprobs = processed_logprobs` (`sampler.py:97-99`) -> `gather_logprobs(raw_logprobs, 0,
   sampled)` (`:123-125, 295-342`) -> `logprobs.content[i].logprob` -> Gym `_extract_choice_logprobs`
   (`vllm_model/app.py:887-913`).
6. CUDA graphs: `FULL_AND_PIECEWISE`, capture sizes 1..64, `enforce_eager=False` [0.25.1 log]; the graphs wrap
   `_model_forward` only; `compute_logits` (`:4087`) and the sampler (`:3340`) run eagerly afterwards.
   `_dummy_sampler_run` (`gpu_worker.py:690`) is warm-up only and is followed by the re-seed at `:692`.

## 5. Side-by-side

| Item | MINF (Megatron fork 880de0fce, torch backend) | vLLM 0.25.1 (citations from 0.20.0 source) | Same? |
|---|---|---|---|
| Logits dtype entering the sampler | bf16 (static `_all_logits_cuda`, `params_dtype`; controller `:321-330, :902`) | bf16 from `compute_logits`; upcast `logits.to(float32)` at `sampler.py:91` | yes: both bf16-rounded, both upcast to fp32 before softmax |
| TP gather of logits | before sampling, bf16, every TP rank holds full logits (`layers.py:1232-1249`, `inference_layers.py:596-608`) | before sampling, bf16 gather/all-gather (`logits_processor.py:75-87`) | yes (dtype and position) |
| Vocab padding | vocab 131072, no padding, clamp no-op (`torch_sampling.py:135-136`) | slice to `org_vocab_size` 131072 (`logits_processor.py:103`) | yes |
| Temperature 1.0 | division skipped (`torch_sampling.py:75-76`) | `div_` by fp32 1.0, exact (`sampler.py:226`) | yes |
| top-k | request `top_k=0` -> no filter (`:77`) | request `top_k=-1` -> stored as vocab -> `k=None` -> no filter (`gpu_input_batch.py:372-376`, `topk_topp_sampler.py:248`) | yes, no filtering runs |
| top-p | request `top_p=1.0` -> `0.0<1.0<1.0` false -> no filter (`:82`); batch flag `no_top_p=False` but ignored by the torch backend (`:218`) | `top_p=1.0` not `<1` -> `p=None` -> no filter (`gpu_input_batch.py:370-371`) | yes, no filtering runs |
| min-p / repetition / frequency / presence / logit_bias / bad_words / allowed ids / min_tokens | not implemented on the MCore chat path (fields ignored) | implemented; all inactive at defaults and none requested | same in effect |
| softmax dtype | fp32 (`softmax(dim=-1, dtype=float32)`, `:122`) | fp32 (`:112`) | yes |
| Random draw | `q ~ Exp(1)` via `exponential_(generator=rng)`, `argmax(p/q)` (`:131-133`) | `q ~ Exp(1)` via `exponential_()` on the default CUDA generator, `argmax(p/q)` (`:334-346`) | same algorithm (exponential race == Gumbel-max) |
| RNG seeding | dedicated `torch.Generator`, seed 42 + DP rank (controller `:260-268`; defaults `transformer_config.py:1135`, `config.py:456`) | process default CUDA generator, seed `node_idx*1024+bundle_id` (`vllm_worker.py:161-169`; `gpu_worker.py:275,692`); per-request seed None | both seeded; different streams; neither per-request; **neither derives from `grpo.seed`** |
| Returned log-prob | raw mode: fp32 `log_softmax` of `.float()` logits at the sampled token (`dynamic_context.py:4616-4617, 4656, 4660`) | processed mode: fp32 `log_softmax` of the fp32 post-temperature, unfiltered logits at the sampled token (`topk_topp_sampler.py:111`, `sampler.py:97-125`) | same quantity (temperature and filters are identities here) |
| CUDA-graph involvement | block-scope graph wraps the entire model forward incl. output layer and TP gather (`hybrid_model.py:387-421`); logits copied to a static bf16 buffer; sampler and log-probs eager | FULL_AND_PIECEWISE graphs wrap the model forward; `compute_logits` and sampler eager (`gpu_model_runner.py:4087, 3340`) | equivalent: sampling outside graphs in both |
| Speculative decoding | 0 draft tokens | none | yes |
| Termination | EOS set {eod, 2, 11} (controller `:371-405`); length = min(requested, max_seq_len - prompt) (`dynamic_engine.py:1179-1208`) | eos ids [2, 11] from `generation_config.json`; `max_tokens = min(requested, max_model_len - prompt)` (`utils.py:186-200`) plus NeMo RL clamp | yes |
| Prefix caching (affects logits, not the sampler) | off in chains C/G (launcher `:66`; chain C `step_16/config.yaml`) | **on** (`enable_prefix_caching=True` [0.25.1 log]) | differs |

## 6. Verdict

Given identical bf16 logits, the two samplers define the **same categorical distribution**
`softmax(fp32(logits))` (temperature 1, no top-k, no top-p, no min-p, no penalties, no soft-cap or MuP scaling)
and draw from it with the **same estimator** (exponential race: `argmax(p / Exp(1))` in fp32). The reported
per-token log-prob is the same function of the logits in both (fp32 `log_softmax` at the sampled token).
The earlier note is therefore correct for these runs.

Numeric differences that remain, ranked by potential impact on long-generation behaviour:

1. **Upstream of the sampler, not in it.** The logits themselves come from different kernels and memory
   layouts (MCore `inference_optimized` hybrid layers with the vLLM grouped-GEMM backend, NVLS symmetric-memory
   all-gather, fp32 SSM states vs vLLM FLASH_ATTN + Triton MoE with fp32 Mamba cache), different batch
   composition per step, prefix caching on in the vLLM run and off in the MINF chains. Any distribution
   difference between the arms must be attributed here, not to the sampling code.
2. **Backend selection is an environment variable, not config.** The equivalence holds only because
   `launch_swe_dump.sh:62` forces `NRL_MINF_SAMPLING_BACKEND=torch`; NeMo RL's default is `flashinfer`
   (`megatron_worker.py:265-267`). With flashinfer and `top_p=1.0`, `no_top_p` is False (controller `:1410`),
   so sampling goes through `flashinfer.sampling.top_p_sampling_from_probs(probs, 1.0)` (rejection sampling,
   `flashinfer_sampling.py:114-120`): statistically equivalent, but a different kernel and RNG path, and with
   `processed_logprobs` a different log-prob path (`:142-202`). Every logged MINF dump run announced the torch
   sampler (section 0), so this did not affect the runs audited.
3. **Fork dependence.** The exponential race in the torch backend is fork commit `2c43cacff`; an MLM tree
   without it samples with a cumulative-sum method (upstream `torch.multinomial`). The `PR#16 PRESENT`
   announcement is the runtime guard.
4. **vLLM version gap.** vLLM behaviour was verified on the 0.20.0 source on disk; the run used 0.25.1
   (log-confirmed). The 0.25.1 engine-init line is consistent with the analysed path (bf16, no speculative
   config, `processed_logprobs`, no FlashInfer-sampler opt-in, classic `gpu_model_runner`), but the 0.25.1
   sampler source itself was not inspected.
5. **RNG streams.** Different generators and seeds (42+dp_rank vs node*1024+bundle); irrelevant to the
   distribution. `grpo.seed=42` only seeds the driver process (`nemo_rl/algorithms/utils.py:276-281`,
   `grpo.py:542`); MINF's 42 is the Megatron `inference_sampling_seed` default, a coincidence.
6. **Kernel/library versions across the two worker venvs** (`exponential_`, `softmax`, `log_softmax`) were not
   compared; any difference would be rounding-level and distribution-preserving.
7. **Default-handling asymmetry for clients that omit `top_p`.** vLLM would apply `generation_config.json`
   `top_p=0.95` (NeMo RL refuses such requests with an assertion, `vllm_worker_async.py:753`); the MCore endpoint
   silently defaults to 1.0 (`chat_completions.py:593`). Not triggered here (OpenHands always sends `top_p`).

Not found / not verifiable from disk: vLLM 0.25.1 source; torch versions inside the two worker venvs;
the r2 MINF run's engine announcement (its logs contain no engine start).

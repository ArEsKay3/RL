# vllm-numerical-parity: two eval-mode gates break NeMo RL training

Branch `santhnm2/Megatron-LM:vllm-numerical-parity`, tip `026e7078`, forked from
`880de0fce8`. Found while standing up an RL arm (chain V) on the adapter.

Both bugs are the same mistake: **parity paths gated on `not self.training`.**
That is not a safe proxy for "we are doing inference" under NeMo RL, because
NeMo RL runs eval-mode forwards on the *training* model —
`MegatronPolicyWorker.get_logprobs_presharded` calls `self.model.eval()` before
recomputing logprobs for the PPO ratio. `self.training` is then `False` on the
training model, with context parallelism active and sequence packing on.

`hybrid_block.py:311` already uses the correct gate:

```python
if self.config.inference_vllm_parity and InferenceMode.is_active():
```

## 1. `attention.py` — parity QKV in the trainer (fatal)

`SelfAttention.get_query_key_value_tensors`:

```python
if self.config.inference_vllm_parity and not self.training:      # fires in the trainer
    from megatron.core.inference.vllm_parity import attention_qkv
    ...
    return attention_qkv(self, hidden_states)
```

The trainer's logprob pass takes this branch and hands vLLM-layout tensors to
TE's context-parallel flash attention at CP=4. Observed on a 16-node SWE-E2E
smoke (job 4050516), ~29 minutes in, at the first `get_logprobs_presharded`
after step 1's rollouts:

```
ray::MegatronPolicyWorker.get_logprobs_presharded()
  transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py:4723
    out = AttnFuncWithCPAndKVP2P.apply(*args)
SystemError: <class 'UserWarning'> returned a result with an exception set
```

The `SystemError` is CPython's warning machinery firing with an exception
already set, so the underlying error is masked — worth knowing, because the
traceback does not name the real cause.

Fix applied locally:

```python
if self.config.inference_vllm_parity and InferenceMode.is_active():
```

`InferenceMode` is already imported in `attention.py` (line 16) and used at
1371 and 1455. The adapter's whole forward runs inside `InferenceMode`, entered
at `hybrid_block.py:311`, so the parity QKV is still reached on the inference
path.

## 2. `mamba_mixer.py` — compiled norm outside the adapter (silent)

`ExtendedRMSNorm.forward`:

```python
if (
    not self.training                                            # fires in the trainer
    and z is not None
    and not self.norm_before_gate
    and self.group_size is not None
):
    norm = (compiled_grouped_gated_rmsnorm
            if getattr(self, 'inference_vllm_compile_norm', True)   # default True
            else grouped_gated_rmsnorm)
```

Two consequences, neither fatal and both silent:

- The trainer's logprob pass computes its grouped gated norm in vLLM's
  operation order instead of the fused Triton path, changing PPO-ratio
  numerics.
- `getattr(..., True)` means the compiled path is also the default for **plain
  non-parity MINF** — `VllmHybridParity.__init__` (`vllm_parity.py:171`) is the
  only writer of `inference_vllm_compile_norm`, so a run with
  `inference_vllm_parity=False` still gets the compiled norm, plus a
  `torch.compile(fullgraph=True)` on every worker.

Fix applied locally: key on the attribute's presence, which is set only by the
adapter, and read it directly rather than through a defaulted `getattr`.

## Audit of the remaining gates

Every other `inference_vllm_parity` site on the branch is safe. `hybrid_block`
uses `InferenceMode`. `mamba_mixer` 939/942/1031/1100/1239 are inside
`ssm_prefill` / `ssm_decode`, which only exist on the dynamic-inference mixin.
`inference_layers.py` keys on `_vllm_normalized_replicated_input` and
`_vllm_communicator`, both written only by the adapter. `attention.py:1068` is
inside `flash_decode_and_prefill`, which asserts `not self.training` and is
reachable only from the dynamic inference context.

## Two smaller notes

- `transformer_config.py`'s `inference_vllm_parity` docstring still says
  "Requires vLLM 0.25.1 in the same Python environment" at the tip, where
  `requirements/inference-parity.txt` says "No vLLM installation required".
- The SSD kernel edits are unconditional, not gated on the flag: `tl.exp` ->
  `fast_exp` (`exp2(log2e*x)`) in `ssd_chunk_scan.py`, `ssd_chunk_state.py` and
  `ssd_state_passing.py`, which are shared with the training forward, while the
  backward kernels keep `tl.exp`. Stripping `autotune_configs()` from four of
  five call sites is harmless in practice — the wrapper is a no-op unless
  `MAMBA_DETERMINISTIC` or torch deterministic mode is set.

## Deployment note

The audited pin `1541eba45b` cannot run in `rl-gym.63635108-zstd.sqsh`: it does
`from vllm import _custom_ops` and vLLM is not installed in the Megatron
policy-worker venv. The tip vendors those kernels, but still needs `cutlass`
(CUTLASS DSL 4.5.2) and `quack`, neither of which is importable in that venv —
the `nvidia-cutlass-dsl` dist-info is present without the package, and
`quack-kernels` is absent. Both can be taken from
`/opt/gym_venvs/responses_api_models/local_vllm_model/.venv` in the same image,
which matches on Python 3.13.14, torch 2.11.0+cu130 and triton 3.6.0.

## After both fixes

16-node smoke 4051233 completed 2/2 steps, exit 0:0: refits 76.594 / 5.154 /
3.360 s against a five-smoke control of 75.4-91.8 s startup and 1.773-4.196 s
steady; 0 sequences masked on all six logprob passes; reward 0.469 / 0.406;
rollout length mean 28,284 / 21,845 tokens. The 64-node arm's startup sync is
39.896 s against chain P⁗2's 37.488 s, and its first four steady refits are
5.898-7.102 s against chain P⁗2's 4.808-7.083 s over sixteen. No measurable
refit cost from the adapter at either scale.

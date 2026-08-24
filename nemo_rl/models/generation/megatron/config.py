# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Literal, NotRequired, Optional, TypedDict, cast

from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.policy import PolicyConfig


class MCoreGenerationSpecificArgs(TypedDict):
    """Megatron fields related only to inference.

    Any fields not declared here but declared in the training-side config can be overwritten.
    For example, Megatron inference might want `transformer_impl: "inference_optimized"`,
    while Megatron training might want `transformer_impl: "transformer_engine"`.
    """

    expose_http_server: bool
    parsers: list[str]

    buffer_size_gb: int
    block_size_tokens: int
    max_tokens: int
    max_model_len: int

    num_cuda_graphs: int
    use_cuda_graphs_for_non_decode_steps: bool
    cuda_graph_impl: str
    # How captured CUDA-graph token counts are spaced. Defaults to 'hybrid'.
    # - 'hybrid': exponential for prefill/mixed graphs, linear for decode-only graphs. The two
    #   cover ranges differing by orders of magnitude (prefill spans cuda_graph_max_tokens,
    #   decode is capped at max_requests), so a single spacing serves one badly.
    # - 'exponential': halve from cuda_graph_max_tokens down to tp_size for both.
    # - 'linear': linear strides for both; captures far more prefill graphs.
    cuda_graph_sizing_distribution: NotRequired[
        Literal["exponential", "linear", "hybrid"]
    ]
    # Ceiling on the token count of captured prefill/mixed graphs; clamped to
    # [max_requests, max_tokens]. mcore defaults to 512, so longer prefill steps run
    # eager. Raise toward max_tokens to cover them, at the cost of capture time and
    # graph memory. Omit to keep mcore's default.
    cuda_graph_max_tokens: NotRequired[int]
    # Inference CUDA-graph scope. Options:
    # - 'none': inference runs in eager mode (no CUDA graphs).
    # - 'layer': graphs are owned at the per-layer boundary (TransformerLayer / MambaLayer).
    # - 'block': graphs are owned at the enclosing block (TransformerBlock / HybridBlock).
    # Only meaningful when cuda_graph_impl='local'.
    inference_cuda_graph_scope: NotRequired[str]

    materialize_only_last_token_logits: bool
    enable_chunked_prefill: bool
    enable_prefix_caching: bool

    refit_backend: Literal["gloo", "nccl", "nvshmem"]
    num_speculative_tokens: int

    mamba_inference_ssm_states_dtype: NotRequired[str]
    mamba_inference_conv_states_dtype: NotRequired[str]
    prefix_caching_mamba_gb: NotRequired[int]

    # How the DP coordinator picks an engine for each request. Defaults to
    # 'longest_prefix'; forced to 'load_balanced' when enable_prefix_caching is
    # false, since there is then no cache to route on.
    # - 'longest_prefix': cheapest (prefill blocks still to compute) x (load), so
    #   idle ranks are filled first and a rank already holding the prefix wins
    #   thereafter.
    # - 'load_balanced': fewest in-flight requests; ignores prefix affinity.
    # - 'first_prefix_block': route on the first block hash alone.
    prefix_caching_coordinator_policy: NotRequired[
        Literal["load_balanced", "longest_prefix", "first_prefix_block"]
    ]
    # How long the coordinator assumes an engine still holds a block it routed
    # there. Only meaningful under the LRU eviction policy.
    prefix_cache_ttl_seconds: NotRequired[float]
    # How 'longest_prefix' weighs prefix affinity against rank load:
    # - 'load_aware': score = prefix_fraction - beta * (load - mean) / max(1, mean).
    #   Normalized and subtractive, so load still counts on a full cache hit and the
    #   penalty vanishes when ranks are balanced.
    # - 'simple_multiplicative': cost = remaining_blocks * (1 + load). A full hit costs
    #   0 regardless of load, which strands work on a busy rank during the drain phase.
    prefix_caching_cost_policy: NotRequired[
        Literal["simple_multiplicative", "load_aware"]
    ]
    # Weight on the load penalty under 'load_aware', in units of "full cache hits per
    # 100% above mean load". 0 is pure affinity.
    prefix_caching_load_beta: NotRequired[float]

    # KV cache lifecycle across suspend/resume:
    # - "persist": cache stays allocated; CUDA graphs remain valid (default)
    # - "offload": cache is moved off-GPU between iterations
    #
    # The third mcore value, "recompute" (drop + rebuild on resume), must be set via
    # `grpo.async_grpo.recompute_kv_cache_after_weight_updates=true`.
    # TODO: Unify `kv_cache_management_mode` and `recompute_kv_cache_after_weight_updates`.
    kv_cache_management_mode: Literal["persist", "offload"]

    logging_step_interval: NotRequired[int]


class MCoreGenerationConfig(GenerationConfig):
    """Generation config for Megatron Inference."""

    mcore_generation_config: MCoreGenerationSpecificArgs


def merged_inference_megatron_cfg(policy_config: PolicyConfig) -> dict[str, Any]:
    """The `megatron_cfg` a dedicated inference model runs with."""
    generation_config = cast(MCoreGenerationConfig, policy_config["generation"])
    merged: dict[str, Any] = {
        **cast(dict[str, Any], policy_config["megatron_cfg"]),
        **(generation_config.get("mcore_generation_config") or {}),
        "activation_checkpointing": False,
    }
    # inference_optimized layers hard-require SP with TP>1. Raise with the
    # config key: the colocated build bypasses validate_and_set_config, so this
    # merge is the only spot the inference cfg gets a named error instead of a
    # raw MCore assert at model build.
    if (
        merged.get("transformer_impl") == "inference_optimized"
        and merged["tensor_model_parallel_size"] > 1
        and not merged["sequence_parallel"]
    ):
        raise ValueError(
            "transformer_impl=inference_optimized requires sequence parallelism "
            "with TP>1 on the generation model: set "
            "policy.generation.mcore_generation_config.sequence_parallel=true."
        )
    return merged


def dedicated_inference_megatron_cfg(
    policy_config: PolicyConfig,
) -> Optional[dict[str, Any]]:
    """The `megatron_cfg` for a dedicated colocated inference model, or None.

    Colocated Megatron generation shares the training model unless the resolved
    inference layout or `transformer_impl` differs from training; then the worker
    builds a second model and reshards into it on every wake. Inference never
    uses CP, so CP is pinned to 1 (CP>1 training therefore always differs).

    Returns None when the resolved config matches training (reshardless:
    generate directly on the shared training model).
    """
    inference_mcfg = merged_inference_megatron_cfg(policy_config)
    inference_mcfg["context_parallel_size"] = 1

    train_mcfg = cast(dict[str, Any], policy_config["megatron_cfg"])
    layout_keys = (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "context_parallel_size",
    )
    layout_differs = any(inference_mcfg[k] != train_mcfg[k] for k in layout_keys)
    impl_differs = inference_mcfg.get("transformer_impl") != train_mcfg.get(
        "transformer_impl"
    )
    if not (layout_differs or impl_differs):
        return None
    return inference_mcfg

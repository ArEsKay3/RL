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

"""Standalone vLLM post-processing adapters for Megatron's wire interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncIterator, Protocol
from uuid import uuid4

import torch

from nemo_rl._vendor.vllm.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Logprob,
    RequestResponseMetadata,
)
from nemo_rl._vendor.vllm.response import (
    CompletionOutput,
    OpenAIServingChat,
    RequestOutput,
)
from nemo_rl._vendor.vllm.runtime import NativeSampler, get_parser
from nemo_rl._vendor.vllm.sampling import (
    LogitsProcessors,
    SamplingMetadata,
    apply_top_k_top_p,
    apply_top_k_top_p_pytorch,
)
from nemo_rl.utils.vllm_postprocessing_config import VllmPostprocessingConfig


class SamplingContext(Protocol):
    total_request_count: int
    paused_request_count: int
    active_request_metadata: dict[str, torch.Tensor]


@dataclass(frozen=True)
class _Parameters:
    temperature: float
    top_k: int
    top_p: float
    greedy: bool


@lru_cache(maxsize=1024)
def _sampling_params(temperature: float, top_k: int, top_p: float) -> _Parameters:
    # Transport normalization from vLLM SamplingParams.__post_init__ (0.25.1).
    # All actual logit transforms and sampling live in the vendored code.
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be finite and in [0, 2]")
    if top_k < -1 or not math.isfinite(top_p) or not 0 <= top_p <= 1:
        raise ValueError("Invalid top_k or top_p")
    if 0 < temperature < 1e-2:
        temperature = 1e-2
    greedy = temperature < 1e-5
    return _Parameters(
        temperature=temperature,
        top_k=0 if greedy else top_k,
        top_p=1.0 if greedy or top_p == 0 else top_p,
        greedy=greedy,
    )


class VllmSampler:
    """Use vLLM's V1 sampler with Megatron's engine-local seeded generator."""

    def __init__(
        self, rng: torch.Generator, vocab_size: int, logprobs_mode: str
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(
                "vLLM post-processing requires a positive model vocabulary size"
            )
        if logprobs_mode not in ("raw_logprobs", "processed_logprobs"):
            raise ValueError(f"Unsupported vLLM logprobs mode: {logprobs_mode}")
        self._rng = rng
        self._vocab_size = vocab_size
        self._sampler = NativeSampler(logprobs_mode=logprobs_mode)

    def _metadata(
        self,
        context: SamplingContext,
        n: int,
        device: torch.device,
        token_to_request_index: torch.Tensor | None,
    ) -> SamplingMetadata:
        md = context.active_request_metadata
        count = context.total_request_count - context.paused_request_count
        if token_to_request_index is None:
            indices = list(range(n))
        else:
            indices = token_to_request_index[:n].tolist()
        if len(indices) != n or any(i < 0 or i >= count for i in indices):
            raise ValueError("Sampling rows must map to active Megatron requests")
        params = [
            _sampling_params(
                float(md["temperature"][i]), int(md["top_k"][i]), float(md["top_p"][i])
            )
            for i in indices
        ]
        greedy = [p.greedy for p in params]
        top_k = [
            self._vocab_size if p.top_k <= 0 else min(p.top_k, self._vocab_size)
            for p in params
        ]
        top_p = [p.top_p for p in params]
        empty = torch.empty(0, device=device)
        return SamplingMetadata(
            temperature=torch.tensor([p.temperature for p in params], device=device),
            all_greedy=all(greedy),
            all_random=not any(greedy),
            top_k=(
                torch.tensor(top_k, dtype=torch.int32, device=device)
                if any(k != self._vocab_size for k in top_k)
                else None
            ),
            top_p=(
                torch.tensor(top_p, dtype=torch.float32, device=device)
                if any(p != 1.0 for p in top_p)
                else None
            ),
            # vLLM's seeded path avoids process-global RNG state. The engine's
            # generator is already synchronized across TP/PP ranks by Megatron.
            generators={i: self._rng for i in range(n)},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=empty,
            presence_penalties=empty,
            repetition_penalties=empty,
            output_token_ids=[[] for _ in range(n)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

    def _logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] < self._vocab_size:
            raise ValueError("Model logits are narrower than the unpadded vocabulary")
        # vLLM truncates TP padding before sampling. Clone even FP32 inputs:
        # vLLM transforms in place and Megatron still needs the original logits.
        return logits[:, : self._vocab_size].to(dtype=torch.float32, copy=True)

    def sample_kernel(
        self,
        logits: torch.Tensor,
        n: int,
        context: SamplingContext,
        *,
        no_top_k: bool,
        no_top_p: bool,
        gather_indices: torch.Tensor | None = None,
        token_to_request_index: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        eager: bool = False,
        cache_key: Any = None,
    ) -> torch.Tensor:
        """Sample the active rows and preserve the caller's stable output buffer."""
        del no_top_k, no_top_p, eager, cache_key
        rows = logits[:n] if gather_indices is None else logits[gather_indices[:n]]
        if n == 0:
            return (
                output if output is not None else logits.new_empty(0, dtype=torch.long)
            )
        metadata = self._metadata(context, n, logits.device, token_to_request_index)
        sampled = self._sampler.sample(self._logits(rows), metadata)[0]
        if output is not None:
            output.copy_(sampled)
            return output
        return sampled.long()

    def log_probs_kernel(
        self,
        logits: torch.Tensor,
        context: SamplingContext,
        *,
        token_to_request_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute processed logprobs using vLLM operations without another draw."""
        logits = self._logits(logits)
        if logits.shape[0] == 0:
            return logits
        metadata = self._metadata(
            context, logits.shape[0], logits.device, token_to_request_index
        )
        # This is the same path as Sampler.sample/TopKTopPSampler.forward_native,
        # excluding random_sample. In particular greedy logprobs are not a delta
        # distribution, and small positive temperatures are normalized by vLLM.
        if not metadata.all_greedy:
            logits = self._sampler.apply_temperature(
                logits, metadata.temperature, metadata.all_random
            )
            active_count = context.total_request_count - context.paused_request_count
            if logits.is_cuda and active_count < 8 <= logits.shape[0]:
                # vLLM switches to Triton at eight sampled requests. Expanded
                # prefill rows must not switch the logprob path: tied logits
                # can be filtered differently by PyTorch and Triton.
                logits = apply_top_k_top_p_pytorch(
                    logits, metadata.top_k, metadata.top_p
                )
            else:
                logits = apply_top_k_top_p(logits, metadata.top_k, metadata.top_p)
        return self._sampler.compute_logprobs(logits)


class _OutputOnlyServingChat(OpenAIServingChat):
    """Initialize only the dependencies of vLLM's completed-response formatter.

    We never call create_chat_completion or submit to a vLLM engine. Calling
    super().__init__ would require an EngineClient and a model renderer despite
    inference and chat templating being owned by Megatron. The source manifest and
    tests cover this deliberately narrow use of the pinned upstream method.
    """

    def __init__(self, config: VllmPostprocessingConfig) -> None:
        self.response_role = "assistant"
        self.enable_auto_tools = config.enable_auto_tools
        self.parser_cls = get_parser(
            config.reasoning_parser, config.tool_parser, config.enable_auto_tools
        )
        self.enable_prompt_tokens_details = True
        self.enable_per_request_metrics = False
        self.enable_log_outputs = False
        self.request_logger = None
        self.return_tokens_as_token_ids = False
        self.system_fingerprint = None


@dataclass
class VllmResponseFormatter:
    """Picklable frontend callback; parser instances are isolated per request."""

    tokenizer: Any
    config: VllmPostprocessingConfig

    def _serving(self) -> _OutputOnlyServingChat:
        return _OutputOnlyServingChat(self.config)

    def validate(self) -> None:
        """Check parser configuration before launching frontend processes."""
        if self.config.tool_parser and not self.config.enable_auto_tools:
            raise ValueError(
                "vllm_postprocessing.tool_parser requires enable_auto_tools=true"
            )
        serving = self._serving()
        if serving.parser_cls is not None:
            serving.parser_cls(self.tokenizer)

    def validate_request(self, body: dict[str, Any]) -> None:
        """Reject options the Megatron wire protocol cannot represent faithfully."""
        request = ChatCompletionRequest.model_validate(body)
        _sampling_params(
            float(
                body.get("temperature") if body.get("temperature") is not None else 1.0
            ),
            int(body.get("top_k") if body.get("top_k") is not None else 0),
            float(body.get("top_p") if body.get("top_p") is not None else 1.0),
        )
        if request.stream:
            raise ValueError("vLLM post-processing currently requires stream=false")
        if request.top_logprobs not in (None, 0):
            raise ValueError(
                "vLLM post-processing supports sampled-token logprobs (top_logprobs=0); "
                "Megatron's top-N wire format does not retain token IDs"
            )
        unsupported = {
            "frequency_penalty": (None, 0, 0.0),
            "presence_penalty": (None, 0, 0.0),
            "repetition_penalty": (None, 1, 1.0),
            "min_p": (None, 0, 0.0),
            "min_tokens": (None, 0),
            "seed": (None,),
            "logit_bias": (None, {}),
            "allowed_token_ids": (None,),
            "bad_words": (None, []),
            "structured_outputs": (None,),
            "response_format": (None, {"type": "text"}),
            "use_beam_search": (None, False),
            "thinking_token_budget": (None,),
            "logits_processors": (None, []),
            "prompt_logprobs": (None,),
        }
        for name, allowed in unsupported.items():
            if body.get(name) not in allowed:
                raise ValueError(
                    f"Megatron vLLM post-processing does not support {name}"
                )

    async def format_response(
        self,
        body: dict[str, Any],
        prompt_tokens: list[int],
        results: list[dict[str, Any]],
        texts: list[str],
    ) -> dict[str, Any]:
        """Translate engine outputs, then call vLLM's actual OpenAI formatter."""
        request = ChatCompletionRequest.model_validate(body)
        serving = self._serving()
        parser = (
            serving.parser_cls(
                self.tokenizer,
                request.tools,
                chat_template_kwargs=request.chat_template_kwargs or {},
            )
            if serving.parser_cls is not None
            else None
        )
        request_id = f"chatcmpl-{uuid4().hex}"
        outputs = []
        for index, result in enumerate(results):
            tokens = result["generated_tokens"]
            values = result.get("generated_log_probs")
            if request.logprobs and (values is None or len(values) != len(tokens)):
                raise ValueError(
                    "Megatron did not return one log probability per generated token"
                )
            logprobs = (
                [
                    {token: Logprob(logprob=value)}
                    for token, value in zip(tokens, values)
                ]
                if values is not None
                else None
            )
            limit = result["sampling_params"]["num_tokens_to_generate"]
            outputs.append(
                CompletionOutput(
                    index=index,
                    text=texts[index],
                    token_ids=tokens,
                    cumulative_logprob=None,
                    logprobs=logprobs,
                    finish_reason="length"
                    if limit is not None and len(tokens) >= limit
                    else "stop",
                )
            )
        final = RequestOutput(
            request_id=request_id,
            prompt=self.tokenizer.decode(prompt_tokens),
            prompt_token_ids=prompt_tokens,
            prompt_logprobs=None,
            outputs=outputs,
            finished=True,
            num_cached_tokens=max(
                (r.get("num_cached_tokens", 0) for r in results), default=0
            ),
        )

        async def completed() -> AsyncIterator[RequestOutput]:
            yield final

        response = await serving.chat_completion_full_generator(
            request,
            completed(),
            request_id,
            request.model or "EMPTY",
            request.messages,
            self.tokenizer,
            RequestResponseMetadata(request_id=request_id),
            parser=parser,
        )
        if not isinstance(response, ChatCompletionResponse):
            raise RuntimeError(
                f"vLLM could not format the Megatron response: {response}"
            )
        payload = response.model_dump(mode="json")
        # Retain the token data consumed by NeMo-Gym without retokenizing parsed
        # reasoning/tool content. All standard OpenAI fields above come from vLLM.
        for choice, result in zip(payload["choices"], results):
            message = choice["message"]
            if body.get("return_tokenized_data") or body.get(
                "prevent_retokenization", True
            ):
                message["prompt_token_ids"] = prompt_tokens
                message["generation_token_ids"] = result["generated_tokens"]
            message["generation_log_probs"] = result.get("generated_log_probs", [])
            message["policy_epoch"] = result["policy_epoch"]
            message["kv_cache_epoch"] = result["kv_cache_epoch"]
            message["num_evictions"] = sum(
                e.get("type") == "EVICT" for e in result["events"]
            )
            if body.get("return_raw_text"):
                message["raw_text"] = final.prompt + outputs[choice["index"]].text
            if result.get("routing_indices") is not None:
                choice["moe_topk_indices"] = result["routing_indices"]
                choice["prompt_moe_topk_indices"] = result["routing_indices"][
                    : len(prompt_tokens)
                ]
        return payload

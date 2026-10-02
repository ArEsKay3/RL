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

"""Small adapters around the unchanged upstream definitions.

Only the native seeded sampler is exposed: this is the path vLLM selects
when SamplingMetadata contains per-row generators. It uses the upstream
Triton top-k/top-p implementation on CUDA for batches of at least eight.
"""

import torch
import torch.nn as nn

from . import parsers
from .protocol import ChatCompletionRequest
from .sampling import Sampler, TopKTopPSampler


class NativeTopKTopPSampler(TopKTopPSampler):
    def __init__(self, logprobs_mode: str, use_fp64_gumbel: bool) -> None:
        nn.Module.__init__(self)
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel

    def forward(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(logits, generators, k, p)


class NativeSampler(Sampler):
    def __init__(self, logprobs_mode: str) -> None:
        nn.Module.__init__(self)
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = False
        self.topk_topp_sampler = NativeTopKTopPSampler(logprobs_mode, False)


class NanoV3ReasoningParser(parsers.DeepSeekR1ReasoningParser):
    """Same override as NeMo-RL's nano_v3 vLLM parser plugin."""

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)
        if (
            chat_template_kwargs
            and chat_template_kwargs.get("enable_thinking") is False
            and final_content is None
        ):
            return None, reasoning_content
        return reasoning_content, final_content


Qwen3ReasoningAdapter, Qwen3ToolAdapter = parsers.make_adapters(parsers.Qwen3Parser)
NemotronReasoningAdapter, _ = parsers.make_adapters(parsers.NemotronV3Parser)


class Qwen3EngineToolParser(Qwen3ToolAdapter):
    # Identical to vllm/tool_parsers/qwen3_engine_tool_parser.py.
    structural_tag_model = "qwen_3_coder"


def get_parser(
    reasoning_parser: str | None, tool_parser: str | None, enable_auto_tools: bool
) -> type[parsers.DelegatingParser] | None:
    """Compose the supported vendored parsers without vLLM's plugin loader."""
    reasoning_classes = {
        None: None,
        "nano_v3": NanoV3ReasoningParser,
        "deepseek_r1": parsers.DeepSeekR1ReasoningParser,
        "nemotron_v3": NemotronReasoningAdapter,
        "qwen3": Qwen3ReasoningAdapter,
    }
    tool_classes = {
        None: None,
        "qwen3_coder": Qwen3EngineToolParser,
        "qwen3_xml": Qwen3EngineToolParser,
    }
    if reasoning_parser not in reasoning_classes or tool_parser not in tool_classes:
        raise ValueError("Unsupported standalone vLLM parser")
    if tool_parser is not None and not enable_auto_tools:
        raise ValueError("A tool parser requires enable_auto_tools=true")
    if reasoning_parser is None and tool_parser is None:
        return None

    class Parser(parsers.DelegatingParser):
        reasoning_parser_cls = reasoning_classes[reasoning_parser]
        tool_parser_cls = tool_classes[tool_parser]

    return Parser

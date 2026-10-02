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

"""Non-numerical compatibility scaffolding for the vendored vLLM code."""

import logging
import os
from typing import Any
from uuid import uuid4


class VLLMValidationError(ValueError):
    def __init__(
        self, message: str, *, parameter: str | None = None, value: Any = None
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.value = value


def random_uuid() -> str:
    return uuid4().hex


def init_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class _Envs:
    VLLM_ENFORCE_STRICT_TOOL_CALLING = os.getenv(
        "VLLM_ENFORCE_STRICT_TOOL_CALLING", "True"
    ).lower() in ("true", "1")


envs = _Envs()


class ResponsesRequest:
    """Type discriminator: this adapter accepts Chat Completions only."""


class NamespaceTool:
    """Discriminator for unsupported Responses namespaces on older OpenAI SDKs."""


def get_tool_call_id_type(model_config: Any) -> str:
    raise ValueError("Model-specific tool IDs are not supported by this adapter")


def record_tool_parser_invocation(**kwargs: Any) -> None:
    """Do not collect vLLM metrics in Megatron frontend processes."""

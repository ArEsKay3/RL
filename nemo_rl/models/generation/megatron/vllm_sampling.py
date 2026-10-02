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

"""Adapt Megatron's dynamic sampler interface to the vendored vLLM sampler.

This module is imported only when post-processing parity is enabled. All
numerical transformations below are calls into vLLM, not local equivalents.
"""

from typing import Any

import torch
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.sampling.base import Sampling
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo_rl._vendor.vllm.sampling import Sampler
from nemo_rl.utils.vllm_postprocessing import VllmSampler


class VllmSampling(VllmSampler, Sampling):
    """Bind the standalone adapter to Megatron's sampling interface."""


class VllmInferenceContext(DynamicInferenceContext):
    """Route raw logprobs through vLLM as well as processed logprobs."""

    def __init__(
        self,
        model_config: TransformerConfig,
        inference_config: InferenceConfig,
        *,
        unpadded_vocab_size: int,
    ) -> None:
        self._vllm_vocab_size = unpadded_vocab_size
        super().__init__(model_config, inference_config)

    def _processed_log_probs(
        self,
        logits: torch.Tensor,
        n_active: int,
        active_query_lengths: torch.Tensor | None,
        sampling: Sampling | None,
        row_to_request: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Exclude CUDA-graph padding rows when the model materializes every
        # prefill token. They have no corresponding request sampling parameters.
        if active_query_lengths is not None:
            logits = logits[: int(active_query_lengths.sum())]
        if self.config.logprobs_mode == "raw_logprobs":
            return Sampler.compute_logprobs(logits[:, : self._vllm_vocab_size])
        return super()._processed_log_probs(
            logits, n_active, active_query_lengths, sampling, row_to_request
        )


class VllmTextGenerationController(TextGenerationController):
    """Install the adapter before Megatron's engine starts or captures graphs."""

    def __init__(
        self,
        inference_wrapped_model: AbstractModelInferenceWrapper,
        tokenizer: Any,
        *,
        unpadded_vocab_size: int,
    ) -> None:
        config = inference_wrapped_model.inference_context.config
        if config.num_speculative_tokens:
            raise ValueError(
                "vLLM post-processing does not support speculative decoding"
            )
        self._vllm_vocab_size = unpadded_vocab_size
        super().__init__(inference_wrapped_model, tokenizer)

    def _init_dynamic_sampling_tensors(self) -> None:
        super()._init_dynamic_sampling_tensors()
        self._sampling = VllmSampling(
            self.sampling_rng,
            self._vllm_vocab_size,
            self.inference_wrapped_model.inference_context.config.logprobs_mode,
        )
        self._sampling_backend = "vllm"

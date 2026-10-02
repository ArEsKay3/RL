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

"""CUDA/CPU discovery without importing the vLLM runtime."""

import torch

HAS_TRITON = torch.cuda.is_available()


class _Platform:
    @staticmethod
    def is_cpu() -> bool:
        return not torch.cuda.is_available()


current_platform = _Platform()


def next_power_of_2(n: int) -> int:
    return 1 if n < 1 else 1 << (n - 1).bit_length()


def num_compute_units(device: int | torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def apply_top_k_top_p_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    # Triton is optional on CPU; the CUDA path uses the unchanged upstream kernel.
    from .topk_topp_triton import apply_top_k_top_p_triton as apply

    return apply(logits, k, p)

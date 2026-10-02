# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA coverage for vLLM's small-batch PyTorch and larger-batch Triton paths."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl._vendor.vllm.sampling import (
    apply_top_k_top_p,
    apply_top_k_top_p_pytorch,
    random_sample,
)
from nemo_rl.utils.vllm_postprocessing import VllmSampler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_cuda_sampling_and_processed_logprobs(batch_size: int) -> None:
    vocab_size = 131072
    device = torch.device("cuda")
    logits = torch.randn(
        batch_size,
        vocab_size + 128,
        device=device,
        dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(91),
    )
    logits[:, vocab_size:] = 1000  # Padding must never enter the distribution.
    original = logits.clone()
    context = SimpleNamespace(
        total_request_count=batch_size,
        paused_request_count=0,
        active_request_metadata={
            "temperature": torch.full((batch_size,), 0.7),
            "top_k": torch.full((batch_size,), 40),
            "top_p": torch.full((batch_size,), 0.9),
        },
    )
    rng = torch.Generator(device=device).manual_seed(731)
    sampler = VllmSampler(rng, vocab_size, "processed_logprobs")
    sampled = sampler.sample_kernel(
        logits, batch_size, context, no_top_k=False, no_top_p=False
    )
    rng_after = rng.get_state().clone()
    actual = sampler.log_probs_kernel(logits, context)
    assert torch.equal(rng.get_state(), rng_after)
    # Compare against the actual vLLM dispatch. Its two implementations can
    # choose different tokens at a tied top-p boundary on BF16 model logits.
    reference = apply_top_k_top_p(
        logits[:, :vocab_size].float() / torch.tensor(0.7, device=device),
        torch.full((batch_size,), 40, device=device, dtype=torch.int32),
        torch.full((batch_size,), 0.9, device=device),
    )
    torch.testing.assert_close(actual, reference.log_softmax(-1), rtol=1e-6, atol=1e-6)
    reference_rng = torch.Generator(device=device).manual_seed(731)
    expected = random_sample(
        reference.softmax(-1), {i: reference_rng for i in range(batch_size)}
    )
    assert torch.equal(sampled, expected)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)


def test_expanded_prefill_preserves_sampling_filter_dispatch() -> None:
    logits = torch.randn(
        16,
        131072,
        device="cuda",
        dtype=torch.bfloat16,
        generator=torch.Generator(device="cuda").manual_seed(91),
    )
    context = SimpleNamespace(
        total_request_count=1,
        paused_request_count=0,
        active_request_metadata={
            "temperature": torch.tensor([0.7]),
            "top_k": torch.tensor([40]),
            "top_p": torch.tensor([0.9]),
        },
    )
    sampler = VllmSampler(torch.Generator(device="cuda"), 131072, "processed_logprobs")
    actual = sampler.log_probs_kernel(
        logits, context, token_to_request_index=torch.zeros(16, dtype=torch.long)
    )
    reference = torch.cat(
        [
            apply_top_k_top_p_pytorch(
                row[None].float() / torch.tensor(0.7, device="cuda"),
                torch.tensor([40], device="cuda", dtype=torch.int32),
                torch.tensor([0.9], device="cuda"),
            )
            for row in logits
        ]
    )
    torch.testing.assert_close(actual, reference.log_softmax(-1), rtol=0, atol=0)

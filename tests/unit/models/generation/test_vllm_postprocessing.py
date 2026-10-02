"""CPU tests for standalone numerical processing and real vendored parsers.

These tests need torch, pydantic, openai, regex and numpy, but neither vLLM,
Megatron, Ray nor a model download. Run with --noconftest on a CPU-only host.
"""

import asyncio
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nemo_rl._vendor.vllm import sampling
from nemo_rl.utils.vllm_postprocessing import VllmResponseFormatter, VllmSampler
from nemo_rl.utils.vllm_postprocessing_config import VllmPostprocessingConfig


def context(temperatures, top_ks=None, top_ps=None, *, paused=0):
    n = len(temperatures)
    return SimpleNamespace(
        total_request_count=n + paused,
        paused_request_count=paused,
        active_request_metadata={
            "temperature": torch.tensor(temperatures),
            "top_k": torch.tensor(top_ks if top_ks is not None else [0] * n),
            "top_p": torch.tensor(top_ps if top_ps is not None else [0.0] * n),
        },
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_greedy_ignores_padding_and_filters_and_keeps_logits(dtype):
    logits = torch.tensor([[1.0, 3.0, 2.0, 1000.0]], dtype=dtype)
    original = logits.clone()
    adapter = VllmSampler(torch.Generator().manual_seed(17), 3, "processed_logprobs")
    ctx = context([0.0], [1], [0.4])
    output = torch.empty(1, dtype=torch.long)
    result = adapter.sample_kernel(
        logits, 1, ctx, no_top_k=False, no_top_p=False, output=output
    )
    assert result.data_ptr() == output.data_ptr()
    assert result.item() == 1
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    torch.testing.assert_close(
        adapter.log_probs_kernel(logits, ctx),
        logits[:, :3].float().log_softmax(-1),
        rtol=0,
        atol=0,
    )


def test_seeded_sampling_matches_direct_upstream_calls_and_does_not_use_global_rng():
    logits = torch.randn(3, 61, generator=torch.Generator().manual_seed(91))
    ctx = context([1.0, 0.7, 1.2], [0, 7, 9], [0.0, 0.9, 0.7])
    seed = 731
    adapter = VllmSampler(torch.Generator().manual_seed(seed), 61, "processed_logprobs")
    before = torch.random.get_rng_state()
    actual = adapter.sample_kernel(logits, 3, ctx, no_top_k=False, no_top_p=False)
    expected_logits = sampling.Sampler.apply_temperature(
        logits.clone(), torch.tensor([1.0, 0.7, 1.2]), True
    )
    expected_logits = sampling.apply_top_k_top_p(
        expected_logits,
        torch.tensor([61, 7, 9], dtype=torch.int32),
        torch.tensor([1.0, 0.9, 0.7]),
    )
    rng = torch.Generator().manual_seed(seed)
    expected = sampling.random_sample(
        expected_logits.softmax(-1), {i: rng for i in range(3)}
    )
    assert torch.equal(actual, expected)
    assert torch.equal(before, torch.random.get_rng_state())
    after_sample = adapter._rng.get_state().clone()
    torch.testing.assert_close(
        adapter.log_probs_kernel(logits, ctx),
        expected_logits.log_softmax(-1),
        rtol=0,
        atol=0,
    )
    assert torch.equal(after_sample, adapter._rng.get_state())


def test_mixed_greedy_mapping_and_inactive_rows():
    logits = torch.tensor(
        [[0.2, 0.7, 0.4], [1.0, 0.0, -1.0], [-2.0, 2.0, 0.0], [99.0, 99.0, 99.0]]
    )
    ctx = context([0.0, 0.5], [1, 2], [0.0, 0.8], paused=3)
    adapter = VllmSampler(torch.Generator().manual_seed(22), 3, "processed_logprobs")
    mapping = torch.tensor([1, 0, 1])
    actual = adapter.sample_kernel(
        logits,
        3,
        ctx,
        no_top_k=False,
        no_top_p=False,
        gather_indices=torch.tensor([2, 0, 1]),
        token_to_request_index=mapping,
    )
    assert actual.shape == (3,)
    assert actual[1].item() == 1
    selected = logits[[2, 0, 1]]
    logprobs = adapter.log_probs_kernel(selected, ctx, token_to_request_index=mapping)
    torch.testing.assert_close(logprobs[1], selected[1].log_softmax(-1), rtol=0, atol=0)
    assert torch.isfinite(logprobs[torch.arange(3), actual]).all()


def test_tied_top_k_and_small_temperature():
    adapter = VllmSampler(torch.Generator(), 4, "processed_logprobs")
    logits = torch.tensor([[1.0, 1.0, 0.0, -1.0]])
    lp = adapter.log_probs_kernel(logits, context([0.00001], [1], [0.0]))
    assert torch.isfinite(lp).tolist() == [[True, True, False, False]]
    torch.testing.assert_close(lp[0, :2], torch.tensor([-0.69314718, -0.69314718]))


def test_invalid_parameters_and_row_maps_fail():
    adapter = VllmSampler(torch.Generator(), 3, "raw_logprobs")
    logits = torch.ones(1, 3)
    with pytest.raises(ValueError, match="temperature"):
        adapter.sample_kernel(logits, 1, context([-1.0]), no_top_k=True, no_top_p=True)
    with pytest.raises(ValueError, match="map"):
        adapter.sample_kernel(
            logits,
            1,
            context([1.0]),
            no_top_k=True,
            no_top_p=True,
            token_to_request_index=torch.tensor([2]),
        )


class Tokenizer:
    all_special_tokens = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
    all_special_ids = [1, 2, 3, 4]

    def get_vocab(self):
        return dict(zip(self.all_special_tokens, self.all_special_ids))

    def decode(self, ids):
        return "".join(
            {
                1: "<think>",
                2: "</think>",
                3: "<tool_call>",
                4: "</tool_call>",
                8: "hello",
                9: "world",
            }.get(i, "?")
            for i in ids
        )


def formatter():
    value = VllmResponseFormatter(
        Tokenizer(),
        VllmPostprocessingConfig(
            enabled=True,
            reasoning_parser="nano_v3",
            tool_parser="qwen3_coder",
            enable_auto_tools=True,
        ),
    )
    value.validate()
    return value


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "query": {"type": "string"},
                },
            },
        },
    }
]
TOOL_TEXT = "<tool_call>\n<function=lookup>\n<parameter=count>3</parameter>\n<parameter=query>a & b</parameter>\n</function>\n</tool_call>"


def format_text(text, **overrides):
    fmt = formatter()
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": TOOLS,
        "logprobs": True,
        "top_logprobs": 0,
        **overrides,
    }
    fmt.validate_request(body)
    result = {
        "generated_tokens": [8, 9],
        "generated_log_probs": [-0.2, -0.3],
        "sampling_params": {"num_tokens_to_generate": 16},
        "policy_epoch": 5,
        "kv_cache_epoch": 4,
        "events": [],
    }
    return asyncio.run(fmt.format_response(body, [8], [result], [text]))


def test_reasoning_and_real_qwen_tool_parser_and_rl_metadata():
    response = format_text("<think>Consider this.</think>" + TOOL_TEXT)
    choice = response["choices"][0]
    message = choice["message"]
    assert message["reasoning"] == "Consider this."
    assert message["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "count": 3,
        "query": "a & b",
    }
    assert choice["finish_reason"] == "tool_calls"
    assert message["prompt_token_ids"] == [8]
    assert message["generation_token_ids"] == [8, 9]
    assert message["generation_log_probs"] == [-0.2, -0.3]
    assert message["policy_epoch"] == 5
    assert [p["logprob"] for p in choice["logprobs"]["content"]] == [-0.2, -0.3]
    assert response["usage"]["total_tokens"] == 3


def test_disabled_thinking_and_parser_state_isolated():
    response = format_text(
        "Plain answer", chat_template_kwargs={"enable_thinking": False}
    )
    assert response["choices"][0]["message"]["content"] == "Plain answer"
    assert response["choices"][0]["message"]["reasoning"] is None
    response = format_text("<think>Again.</think>Done", tool_choice="none")
    assert response["choices"][0]["message"]["content"] == "Done"
    assert "tool_calls" not in response["choices"][0]["message"]


def test_tool_choice_and_parallel_tools_use_upstream_formatter():
    response = format_text(
        "</think>" + TOOL_TEXT + TOOL_TEXT, parallel_tool_calls=False
    )
    assert len(response["choices"][0]["message"]["tool_calls"]) == 1
    response = format_text(
        "</think>" + TOOL_TEXT,
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"]["content"] == ""


def test_formatter_survives_frontend_spawn_pickle():
    cloned = pickle.loads(pickle.dumps(formatter()))
    cloned.validate()


@pytest.mark.parametrize(
    "extra",
    [
        {"stream": True},
        {"top_logprobs": 2, "logprobs": True},
        {"min_p": 0.1},
        {"seed": 7},
        {"presence_penalty": 0.1},
        {"temperature": -1},
        {"top_k": -2},
        {"top_p": 1.1},
    ],
)
def test_rejects_unsupported_requests_before_generation(extra):
    with pytest.raises(ValueError):
        formatter().validate_request(
            {"messages": [{"role": "user", "content": "x"}], **extra}
        )


def test_vendored_files_match_manifest_and_do_not_import_vllm():
    root = Path(__file__).resolve().parents[4] / "nemo_rl/_vendor/vllm"
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, entry in manifest["generated"].items():
        assert (
            hashlib.sha256((root / filename).read_bytes()).hexdigest()
            == entry["sha256"]
        ), filename
    assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)

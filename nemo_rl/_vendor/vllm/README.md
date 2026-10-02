# Standalone vLLM post-processing

This directory vendors the native sampler, top-k/top-p Triton kernels, reasoning
and tool parsers, and non-streaming Chat Completions response formatter from
[vLLM v0.25.1](https://github.com/vllm-project/vllm/tree/v0.25.1). It does not
import or install the `vllm` package, load its CUDA extensions, or start a vLLM
engine. Existing NeMo-RL dependencies provide PyTorch, Triton, NumPy, Pydantic,
OpenAI protocol types, and regex.

`manifest.json` records source URLs, full upstream file SHA256 values, the line
ranges and hashes of extracted definitions, and hashes of the generated files.
The Apache-2.0 license and upstream copyright notices are retained in this
package. NeMo-RL's handwritten compatibility code is kept in `support.py`,
`sampling_support.py`, and `runtime.py`. Its `nano_v3` override is the existing
NeMo-RL reasoning parser plugin, using the copied DeepSeek R1 base parser.

Reproduce and verify from the repository root:

```bash
uv run --no-project tools/vllm_postprocessing/vendor.py /path/to/vllm-0.25.1
uv run --no-project tools/vllm_postprocessing/verify.py --upstream
```

Generated files are intentionally excluded from Ruff formatting. Update the
generator, regenerate, and review the manifest when changing the pinned source.
The generator selects whole functions and class members without rewriting their
bodies. Two deferred parser imports become references to definitions in the same
module; the unused structural-decoding hook raises `NotImplementedError`.
The Triton file is copied whole with three import substitutions. Request models
retain the fields and validators used by response processing; Megatron owns
request tokenization and scheduling. Parser metrics are a no-op. Plugin loading,
structured decoding, and streaming output are not exposed.
Responses-only request/namespace types are inert discriminators, so importing
the Chat Completions adapter also works with NeMo-RL's pinned OpenAI SDK 2.6.1.

The sampler invokes vLLM's native seeded path, including vLLM's batch-size-based
Triton/PyTorch filtering dispatch on CUDA. CPU tests use its PyTorch filtering
path. FlashInfer, ROCm and XPU sampler implementations are not included. Numerical
parity with a vLLM run therefore requires matching the native sampler path,
precision, logits, parameters and random draws. This code does not make different
request schedules or RNG states produce identical token sequences.

Supported parsers are `nano_v3`, `deepseek_r1`, `nemotron_v3`, and `qwen3` for
reasoning, and `qwen3_coder`/`qwen3_xml` for tools. The latter use vLLM's real Qwen3
parser engine, including schema-based argument coercion. The full response
formatter, tool-choice handling, parallel-call filtering and OpenAI logprob
serialization are copied upstream methods.

# Generation Interface

This document explains the token generation interface and various backends for the NeMo RL framework. The generation system is designed with a unified interface that allows different backends (like VLLM, Megatron, Hugging Face, SGLang, and TRT-LLM) to provide token generation capabilities while adhering to the same API.

## Generation Interface

The core of the generation system is defined in `interfaces.py`, which establishes an abstract interface that all generation backends must implement. This ensures consistency across different implementations and makes it easy to swap backends without changing the calling code.

### Key Components

1. **GenerationConfig**: A TypedDict that defines the configuration for generation:
   ```python
   class GenerationConfig(TypedDict):
       """Configuration for generation."""
       backend: str              # The backend to use (e.g., "vllm", "megatron", "hf")
       max_new_tokens: int       # Maximum number of tokens to generate
       temperature: float        # Sampling temperature
       top_p: float              # Top-p sampling parameter
       top_k: int | None         # Top-k sampling parameter
       model_name: str           # Name or path of the model
   ```

2. **GenerationDatumSpec**: A TypedDict that defines the input data format:
   ```python
   class GenerationDatumSpec(TypedDict):
       input_ids: torch.Tensor         # Input token IDs
       attention_mask: torch.Tensor    # Attention mask
       __extra__: Any                  # Additional data specific to the backend
   ```

3. **GenerationOutputSpec**: A TypedDict that defines output data format:
   ```python
   class GenerationOutputSpec(TypedDict):
       output_ids: torch.Tensor
       generation_lengths: torch.Tensor  # Length of just the generated response part
       unpadded_sequence_lengths: torch.Tensor  # Length of full valid sequence (input + generated response)
       logprobs: torch.Tensor
       __extra__: Any                  # Additional output data specific to the backend
   ```

4. **GenerationInterface**: An abstract base class that all generation backends must implement:
   ```python
   class GenerationInterface(ABC):
       """Abstract base class defining the interface for RL policies."""

       @abstractmethod
       def generate(
           self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool
       ) -> BatchedDataDict["GenerationOutputSpec"]:
           pass

       @abstractmethod
       def prepare_for_generation(self, *args, **kwargs):
           pass

       @abstractmethod
       def finish_generation(self, *args, **kwargs):
           pass
   ```

A key design principle for generation backends is that they process tokens directly, without involving the tokenizer. By ensuring that only tokens are exchanged, we eliminate the risk of inconsistencies arising from different tokenizer versions or specifications between the training and generation frameworks.

## Generation Backends

NeMo RL supports multiple generation backends that implement the {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>` to provide efficient text generation for different use cases.

## VLLM Backend

The VLLM backend (`models/generation/vllm/vllm_generation.py`) implements the {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>` to provide efficient text generation using the VLLM library, which is optimized for large language models.

### VllmGeneration Class

The {py:class}`VllmGeneration <nemo_rl.models.generation.vllm.VllmGeneration>` class is the main implementation of the {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>` for VLLM. It performs the following functions:

1. Sets up VLLM workers in a distributed environment using Ray.
2. Manages the lifecycle of these workers (initialization, generation, shutdown).
3. Distributes inputs to workers and collects outputs.
4. Handles weight updates and synchronization.

### VllmGenerationWorker

The {py:class}`VllmGenerationWorker <nemo_rl.models.generation.vllm.VllmGenerationWorker>` is a Ray actor that:

1. Initializes and manages a VLLM model instance.
2. Performs the actual generation on a GPU.
3. Supports dynamic weight updates through IPC handles.
4. Implements sleep/wake mechanisms for efficient resource utilization.

### Custom VLLM Extensions

The {py:class}`UpdatableVllmInternalWorker <nemo_rl.models.generation.vllm_backend.UpdatableVllmInternalWorker>` class in `vllm_backend.py` extends the VLLM worker with additional capabilities:

1. Reporting device IDs to allow mapping of workers to specific GPUs.
2. Updating weights from IPC handles for efficient weight sharing.
3. Checking if weights have been updated correctly.

## Megatron Backend

The Megatron backend provides native Megatron-Core inference capabilities, eliminating the need for weight conversion between training and generation. This backend is particularly beneficial when using Megatron for training, as it enables seamless integration and optimal performance.

### Key Features

1. **No Weight Conversion**: Uses the same Megatron model format for both training and generation, eliminating conversion overhead and potential inconsistencies.
2. **CUDA Graph Support**: Leverages CUDA graphs for optimized inference performance.
3. **Dynamic Inference Engine**: Utilizes Megatron Core's `DynamicInferenceEngine` for efficient batched generation.
4. **Integrated with Training**: The generation capability is built directly into the `MegatronPolicyWorker`, enabling efficient co-located training and generation.

### MegatronPolicyWorker Generation

The Megatron generation backend is implemented within the {py:class}`MegatronPolicyWorker <nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker>` class. The `generate <nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.generate>` method performs the following:

1. Wraps the Megatron model with `GPTInferenceWrapper` for inference optimization.
2. Creates a `DynamicInferenceContext` to manage inference state and memory.
3. Initializes a `DynamicInferenceEngine` with CUDA graph support enabled.
4. Processes batched requests with proper sampling parameters (temperature, top_k, top_p).
5. Returns outputs conforming to {py:class}`GenerationOutputSpec <nemo_rl.models.generation.interfaces.GenerationOutputSpec>`.

### Configuration

To use the Megatron generation backend, configure your YAML file as follows:

```yaml
policy:
  megatron_cfg:
    enabled: true
  generation:
    backend: megatron
    max_new_tokens: 512
    temperature: 1.0
    top_p: 1.0
    top_k: null
    mcore_generation_config:
      buffer_size_gb: 10               # Memory buffer size for requests, total buffer size is 2x this value (active requests + paused requests)
      num_cuda_graphs: 16              # Number of CUDA graphs to pre-allocate
      max_tokens: 16384                # Maximum number of tokens for inference
```

### Configuration Parameters

The `mcore_generation_config` section controls Megatron Core inference engine behavior:

- **buffer_size_gb**: Buffer size reserved for active requests that live on the GPU. The total buffer size (stored in unified memory) is 2x this value, with the other half of the buffer reserved for paused requests that live on the CPU.
- **num_cuda_graphs**: Number of CUDA graphs to pre-allocate for different batch sizes. More graphs can improve performance by avoiding runtime graph capture, but consume more memory.
- **max_tokens**: Maximum total number of tokens (across all requests) that can be processed simultaneously. This limits the maximum batch size and sequence length combinations. Increasing this might throw OOM depending on vocab size and buffer size allocated. 


## Usage Examples

### Using VLLM Backend

To use the VLLM generation backend:

```python
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration, VllmConfig

# Set up the configuration
config = VllmConfig(
    model_name="Qwen/Qwen2.5-1.5B",
    max_new_tokens=100,
    temperature=0.7,
    top_p=1,
    top_k=None,
    backend="vllm",
    vllm_cfg={
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 2048,
    }
)

# Configure config with tokenizer
tokenizer = get_tokenizer(config["model_name"])
config = configure_generation_config(config, tokenizer)

# Initialize the cluster and generation backend
cluster = RayVirtualCluster(...)
generator = VllmGeneration(cluster, config)

# Prepare input data
input_data = BatchedDataDict(...)

# Generate text
generator.prepare_for_generation()
output = generator.generate(input_data, greedy=False)
generator.finish_generation()
```

### Using Megatron Backend

To use the Megatron generation backend, configure your YAML file:

```yaml
policy:
  model_name: meta-llama/Llama-3.2-1B-Instruct
  megatron_cfg:
    enabled: true
  generation:
    backend: megatron
    max_new_tokens: 512
    temperature: 1.0
    top_p: 1.0
    top_k: null
    mcore_generation_config:
      buffer_size_gb: 10
      num_cuda_graphs: 16
      max_tokens: 16384
```

For a complete example, see:
- **Configuration**: `examples/configs/recipes/llm/grpo-llama3.2-1b-instruct-1n8g-megatron_generation.yaml`
- **Test Script**: `tests/functional/grpo_megatron_generation.sh`

## Extend with New Backends

To add a new generation backend:

1. Create a new class that implements {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>`.
2. Implement the required methods: {py:meth}`generate <nemo_rl.models.generation.interfaces.GenerationInterface.generate>`, {py:meth}`prepare_for_generation <nemo_rl.models.generation.interfaces.GenerationInterface.prepare_for_generation>`, and {py:meth}`finish_generation <nemo_rl.models.generation.interfaces.GenerationInterface.finish_generation>`.
3. Ensure your implementation works with the standard {py:class}`GenerationConfig <nemo_rl.models.generation.interfaces.GenerationConfig>` and {py:class}`GenerationDatumSpec <nemo_rl.models.generation.interfaces.GenerationDatumSpec>` structures.
4. Register your backend with the system (if needed) to make it accessible.

This modular design allows for easy extension with new backends while maintaining a consistent interface for the rest of the system.

### Standalone vLLM post-processing with Megatron

Enable `policy.generation.mcore_generation_config.vllm_postprocessing.enabled`
to run Megatron forward passes with the vendored vLLM **0.25.1** native sampler
and OpenAI Chat Completions response formatter. This is independent of
`inference_vllm_parity`, which selects the forward-pass kernels. Both can be
enabled together. There is no vLLM installation or dependency-extra change.

For the Nemotron SWE recipe:

```yaml
policy:
  megatron_cfg:
    model_overrides:
      inference_vllm_parity: true
  generation:
    backend: megatron
    mcore_generation_config:
      num_speculative_tokens: 0
      expose_http_server: true
      parsers: []
      vllm_postprocessing:
        enabled: true
        reasoning_parser: nano_v3
        tool_parser: qwen3_coder
        enable_auto_tools: true
```

`examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_parity_postprocessing.yaml`
inherits the existing forward-parity recipe and enables these options.

The HTTP frontend needs the companion
[`ArEsKay3/Megatron-LM:ksanthanam/vllm-postprocessing`](https://github.com/ArEsKay3/Megatron-LM/tree/ksanthanam/vllm-postprocessing)
branch (commit `fbe6d22`, based on `d37db1077`). It adds a picklable response-formatter callback to
each frontend process. Mount that checkout in the same place as the existing
forward-parity fork. Startup fails with a named error if the callback is absent
or legacy `parsers` are still configured. The companion branch also allows the
adapter's derived inference context through Megatron's CUDA-graph type check.
Eager sampler-only use does not need the HTTP hook. The forward-parity kernels and their staging requirements
are unchanged.

`NRL_MINF_LOGPROBS_MODE=raw_logprobs` returns unmodified-model log probabilities;
`processed_logprobs` returns probabilities after temperature and top-k/top-p.
Set the same mode in the vLLM comparison arm. The enabled mode supersedes
`NRL_MINF_SAMPLING_BACKEND` and prints its selection at startup. It removes
Megatron's vocabulary padding before sampling and computing log probabilities.
It supports joint top-k/top-p, greedy and mixed batches, gathered prefill rows,
and materializing either all logits or only last-token logits.

Megatron continues to own scheduling, KV caches, detokenization, chat templates,
and stop handling. The formatter calls the copied vLLM parsing and response code,
and preserves NeMo-Gym's token IDs, generation log probabilities and epoch fields.
The standard response uses vLLM's `reasoning` field. Parser state is created per
request. The supported parser names are listed in the exemplar configuration.

This mode currently supports **non-streaming `/v1/chat/completions`**, ordinary
sampling, and sampled-token log probabilities (`top_logprobs: 0`). It rejects
streaming, speculative decoding, top-N HTTP log probabilities (Megatron's current
wire format loses their token IDs), per-request seeds, penalties, and constrained
logit processing instead of silently substituting Megatron behavior. It does not
add guided decoding for required/named tool choices. Parsed tool choices use the
same vLLM code on the text the model actually emitted.

The copied native sampler uses Megatron's engine generator through vLLM's seeded
sampling interface. Same-seed end-to-end token identity across different batching
and request schedules is not guaranteed. For comparisons, align logits,
parameters, RNG state, and the native vLLM sampling path. CPU adapter and parser
tests do not replace a GPU end-to-end parity run.

See [the vendored source manifest and regeneration instructions](../../nemo_rl/_vendor/vllm/README.md)
for the upstream hashes and the boundary between copied code and compatibility
code. Run the CPU checks with:

```bash
uv run --group test python -m pytest --noconftest -q tests/unit/models/generation/test_vllm_postprocessing.py
```

On a CUDA host, `tests/unit/models/generation/test_vllm_postprocessing_cuda.py`
checks both filtering dispatch paths at batch sizes 1, 8 and 16 with a 131072-token
vocabulary. For an inference-only SWE-bench trial, start a Megatron generation
server with the recipe above and replay recorded conversation histories:

```bash
uv run --no-sync python tools/vllm_postprocessing/replay_swe.py \
  --url http://HOST:PORT/v1 --traces /path/to/minimax_traces \
  --sessions 0 1 32 --output /path/to/results --concurrency 8
```

The client saves requests, responses, trace hashes and a validation summary. It
replays early, middle and final turns with recorded and stochastic sampling
parameters, checking token/logprob lengths, finite sampled log probabilities,
usage and parsed messages. Later turns retain their recorded history; generated
commands are not executed. This is an inference smoke test, not a SWE-bench score.

Hardware validation on 2026-10-02 (Slurm job `4176866`, completed with exit code
0) used four GB300 GPUs, the existing step-18 checkpoint, TP4/EP1/ETP4, forward
parity, CUDA graphs, and prefix caching. All 36 selected SWE-bench requests passed:

| Logprob mode | Requests | Generated tokens | Prompt length range |
| --- | ---: | ---: | ---: |
| raw_logprobs | 18 | 6,389 | 8,477–40,954 |
| processed_logprobs | 18 | 9,213 | 8,477–40,954 |

Every prompt token count matched its recorded trace. All responses had aligned,
finite sampled-token log probabilities and parsed reasoning. The traces had no
API tool schemas; a separate request derived from the same SWE case supplied an
`execute_bash` schema and passed tool-call JSON, finish-reason and logprob checks.
The four CUDA sampler tests also passed, including expanded-prefill dispatch.

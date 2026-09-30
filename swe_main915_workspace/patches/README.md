# Megatron-LM patches carried by this branch (for rebuilding the stack elsewhere)

Two patch series, both `git am`-able onto public Megatron-LM commits, so a clone
of this NeMo RL branch alone is enough to reconstruct the Megatron-LM trees the
swe_main915 arms run, without access to the ArEsKay3/Megatron-LM fork.

| directory | applies onto | produces | used by |
|---|---|---|---|
| `megatron-lm-main915-latest/` (5 patches) | upstream `NVIDIA/Megatron-LM` `6a3660905` (the commit NeMo RL main's Megatron-Bridge pin `1f8873bb0` resolves to) | branch `mlm-main915-latest` = `475167fa4` | chain U (MINF), chain W mounts nothing |
| `megatron-lm-parity/` (4 patches) | `mlm-main915-latest` `475167fa4` | branch `mlm-main915-parity` = `f28af974d` | chain AD, chain AE (vLLM numerical parity) |

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM && cd Megatron-LM
git checkout -b mlm-main915-latest 6a3660905
git am ../nemo_rl/swe_main915_workspace/patches/megatron-lm-main915-latest/*.patch
cd .. && git clone Megatron-LM Megatron-LM-parity && cd Megatron-LM-parity
git checkout -b mlm-main915-parity mlm-main915-latest
git am ../nemo_rl/swe_main915_workspace/patches/megatron-lm-parity/*.patch
```

`megatron-lm-parity/0001` is the adapter commit `cfa2b0b48` from
`github.com/santhnm2/Megatron-LM` branch `vllm-numerical-parity-main` (rebased
onto our base; only `uv.lock` differs from the original, kept as ours), `0002`
its docs commit, `0003`/`0004` rkirby's two eval-mode gate fixes (`517934242`,
`d37db1077` on `rkirby/vllm-parity-armV`). Megatron-Bridge needs no patch:
`mlm-bridge-main915-latest` is upstream `1f8873bb0`. The parity runtime also
needs the staged CUTLASS-DSL/quack packages: run `tools/stage_parity_site.sbatch`
against the target container (see HANDOFF.md section 10).

# Arm V — SWE-E2E v2 on the Megatron vLLM numerical-parity adapter

chain V (MINF from scratch, vLLM numerical parity, prefix cache kept across refits).

Workspace: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_vllm_parity/`

| piece | where |
|---|---|
| launcher | `launch_swe_vllm_parity.sh` |
| config | `nemo_rl/examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_parity_minf.yaml` |
| nemo_rl | branch `rkirby/swe-v2-vllm-parity` (from `swe_replay_splice` 4f779abc) |
| Megatron-LM | branch `rkirby/vllm-parity-armV` |
| staged deps | `parity_site/`, injected by `parity_paths.pth` |
| JIT cache | `torch_extensions/` |

## Submitting

```bash
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_vllm_parity
SMOKE=1 bash launch_swe_vllm_parity.sh                      # 16 nodes, 2 steps
SLURM_PARTITION=batch_long SLURM_QOS=hero-res \
  SLURM_RESERVATION=sla_res_nemotron_sw_post WALLTIME=8:00:00 \
  bash launch_swe_vllm_parity.sh                            # 64 nodes, full arm
```

The Gym checkout under `nemo_rl/3rdparty/Gym-workspace/Gym` is untracked by the
nemo_rl repo. A fresh `git clone` of the workspace leaves it empty and the run
dies with `ModuleNotFoundError: No module named 'nemo_gym'` after the engines
come up. Clone it from a working tree and check out the same commit (354babf7).

## Why the branch tip, not the audited pin

The post-mortem audits `1541eba45b`, which does `from vllm import _custom_ops`.
vLLM is not installed in the Megatron policy-worker venv, so that commit cannot
run in `rl-gym.63635108-zstd.sqsh` at all. The tip `026e7078` vendors those
kernels. Chain V therefore runs code the post-mortem does not cover.

## Staged dependencies

The worker venv has `nvidia-cutlass-dsl`'s dist-info but not the package, and no
`quack`. Both are copied out of `/opt/gym_venvs/responses_api_models/local_vllm_model/.venv`
in the same image (py 3.13.14, torch 2.11.0+cu130, triton 3.6.0 all match; quack
there is 0.5.0 where the branch asks for 0.4.1 and imports fine). See
`tools/stage_parity_site.sbatch`.

`parity_paths.pth` is bind-mounted to `…/site-packages/zz_parity_paths.pth`;
pyxis creates the missing target file. The `.pth` also sets
`TRITON_PTXAS_BLACKWELL_PATH` and `TORCH_EXTENSIONS_DIR`, which a plain export
can lose to Ray's runtime_env.

## Fixes carried on top of the parity branch

| commit | what |
|---|---|
| `87869b3a2` | prefix-cache keep knob ported onto the parity branch |
| `517934242` | `ExtendedRMSNorm.forward` keyed on the parity adapter, not eval mode |
| `d37db1077` | parity QKV path gated on `InferenceMode`, not eval mode |

The last two are the same defect: the branch gates parity paths on
`not self.training`, but NeMo RL runs eval-mode forwards on the *training*
model (`get_logprobs_presharded` calls `model.eval()`). Smoke 4050516 died on
it — the trainer's logprob pass handed vLLM-layout tensors to TE's
context-parallel flash attention and raised
`SystemError: <class 'UserWarning'> returned a result with an exception set`.
Report both upstream.

## Validation, smoke 4051233 (2026-09-27, COMPLETED 0:0, 2/2 steps)

`_sync_weights`: 76.594 s startup, then 5.154 s and 3.360 s.

Against eight mid-run refits from five 16-node MINF smokes
(1.773 1.905 2.092 2.194 2.346 2.902 2.965 4.196): parity ranks 8th and 10th of
ten — upper-end, overlapping. Mean 4.257 s vs 2.547 s, about +1.7 s per refit,
~1.7 min over an arm's ~60 refits. Startup 76.594 s is mid-pack against
75.4/76.1/76.2/78.8/91.8, so there is no startup regression.

Compare startup sync only within one node count: chain P⁗2's 64-node startup is
37.5 s, *faster* than every 16-node figure, so the metric is not monotonic in
node count.

Step metrics: reward 0.469 / 0.406, rollout length mean 28,284 / 21,845 tokens,
128 valid samples both steps, `token_mult_prob_error` 1.0139,
`gen_kl_error` 0.00126 / 0.00149, 0 sequences masked on all 6 masking passes,
0 tracebacks.

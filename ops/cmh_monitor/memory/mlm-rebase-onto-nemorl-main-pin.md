---
name: mlm-rebase-onto-nemorl-main-pin
description: "2026-09-25 hail-mary: the MINF fork's 29 MLM patches reduce to 1 real patch + 3 import shims on top of MLM 6a3660905a27 (the commit NeMo RL main pins); candidate branch built and validated"
metadata:
  type: project
---

GOAL: take the newest Megatron-LM fixes that NeMo RL main points at, WITHOUT meaningfully changing NeMo RL.

THE PIN CHAIN (measured 2026-09-25):
- our nemo_rl (rkirby/swe-v2-dump) -> Bridge `8c46dc4259080c510b7455f43e836fdff222c5d3` (2026-08-14) -> MLM `14346b65a2d0`
- NeMo RL main `4aaa48fab` (2026-09-24) -> Bridge `1f8873bb00a8` (2026-09-23) -> **MLM `6a3660905a2736b5670baed1ca5954372937918b` = THE TARGET**
- Bridge main `eb96fea38` also pins MLM 6a366090. MLM upstream/main was `8c487209a`.
- Our fork `880de0fce` sits on exactly `14346b65a` — the same commit our Bridge pins, so the fork point is consistent.
- Gap 14346b65a -> target = **482 commits**; target IS an ancestor of upstream/main.

THE TARGET ALREADY CONTAINS every fix the 2026-09-20 review flagged as absent:
487bb3948 (#6481 -1 sentinels into inference kernels, the token-affecting one), 79be654e3 (#6823 order local refit
copies after source updates), 94939b85d (#6442 hidden chunked prefill during async overlap), 8dc7ed381 (#7081 exact
expert token counts), AND 3972b96bb (#7256) — the guard we currently run as an UNCOMMITTED local edit, so moving
makes that hack unnecessary.

IT ALSO CONTAINS the upstream equivalents of most fork patches, which therefore get dropped: 268be1327 (Gumbel-max
fp32) = fork 2c43cacff; 6e5a8c11f (Mamba selective_state_update) = 5f0d342d8; fbbc142b8 (Mamba decode cache after
refit) = f7f625318; 47f079ae2 (Mamba prefix-skip clamp) = 43004649e; 55b2e6ceb (vLLM fused-MoE buffers) = d276ccdfa;
d0612089d (prefix-cache dummy redirect) = b7638dbbf+e6ba7b82c; 5f4c9ac90 (#7320 Qwen3-Coder parser) = fcc4d283c+
367b7b53f; the renorm fix 111b44de8 is upstream verbatim (target flashinfer_sampling.py:188); the weight-scoped
prefix-cache salt (ce0e6bc43) and the DP-coordinator prefix-affinity routing (1ae2bb73d, 19ea4ab52 + the 6 wire/
socket commits) are upstream too — PrefixCachingCostPolicy now lives in the coordinator module, not inference/config.py.
NOTE `git cherry` is USELESS here: 482 commits of drift means zero patch-id matches even for verbatim-upstreamed fixes.
Classify semantically (does upstream commit X exist in the target) instead.

RESULT: the 29 fork patches reduce to **1 real patch + 3 import shims**. Branch `mlm-target-minf` built at
/home/rkirby/.claude/jobs/51d77432/tmp/mlmwork/mlm (local --shared clone of swe_dump/Megatron-LM; no live tree touched):
  ffce55f17 Honor all model-declared EOS tokens during generation  (cherry-pick of fork 8b880867e; 1 add-conflict in
            text_generation_controller.py, HEAD side empty, resolved by keeping the added _build_eos_token_ids)
  02cb1305f Re-add get_async_strategy                              (reused from the Sep-20 attempt, c67ca576c)
  64fa628db Re-export PrefixCachingCostPolicy from inference/config.py
  6fe80df3d Re-export AsyncCallsQueue/AsyncRequest at strategies/async_utils.py
Total +271/-9 over 6 files, vs the fork's +2133/-456 over 38 files. Byte-compiles clean.

VALIDATION: all 45 megatron.core modules NeMo RL imports resolve; all three named symbols resolve. Only three NeMo RL
call sites touch the removed APIs and two were already defensive:
  - models/policy/workers/megatron_policy_worker.py:40 `get_async_strategy` — MODULE-LEVEL import, the one hard break.
  - models/megatron/setup.py:285 `AsyncCallsQueue` — inside try/except ImportError in destroy_parallel_state().
  - models/generation/megatron/megatron_worker.py:43 `PrefixCachingCostPolicy` — already try/except with a helpful message.
Upstream deleted the vendored strategies/async_utils.py and now sources AsyncCallsQueue/AsyncRequest from
nvidia_resiliency_ext.checkpointing.async_ckpt.core; megatron/training/async_utils.py made the queue LAZY
(`_async_calls_queue = None` + `_get_async_calls_queue()`). `AsyncCallsQueue(persistent: bool = False)` still has a
default, so NeMo RL's no-arg construction is fine.

NOT YET DONE: no runtime test (no container run), Bridge is still pinned at 8c46dc425 in our nemo_rl — moving MLM
without moving Bridge may or may not be consistent; the DP-coordinator upstreaming means our `longest_prefix` routing
config keys need checking against the target's coordinator API.

**Why:** the fork was 482 commits stale and carrying ~28 patches that upstream had already absorbed.

**How to apply:** validate the config-key mapping and a smoke run before using it for any chain. Related:
[[mlm-upstream-fixes-review]], [[swe-splice-experiments]].

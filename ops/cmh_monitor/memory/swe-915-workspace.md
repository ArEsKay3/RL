---
name: swe-915-workspace
description: "swe_915 = clean clone of the 2026-09-15 SWE stack (vLLM + MINF, dumps on) with Gym env-flagged sample masking turned off"
metadata: 
  node_type: memory
  type: project
  originSessionId: 51d77432-fbb8-4536-b679-a1714b74af54
  modified: 2026-09-25T19:10:09.025Z
---

`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_915`
— created 2026-09-25 so experiments run from the 9-15 code state instead of the
newer dump/splice lineages.

- `nemo_rl` branch `rkirby/swe-915-nomask`, base `952eaf85b` (tip of
  `rkirby/swe-v2-minf`; the two 9-15 commits are ports of upstream #3864 and
  #3727), then `e5189f1f0` (`env.should_mask_flagged_samples: false`),
  `6a848309c` + `3a7fde92c` (cherry-picks of `f3717359`/`33a9bf6a`, the offline
  rollout + token-level dumps). **Committed locally, NOT pushed.**
- `Megatron-LM` branch `fork-915`, `880de0fce` + `8f745b0e9` (the PR #7256
  prefill-logprob guard, byte-identical to upstream `3972b96bb`) + `a012970be`
  (partial port of PR #7598 `8381dbb97`: the `prevent_retokenization` EOS
  detection + splice fix only, not the PromptPreparer refactor). The nano35
  model declares `generation_config.eos_token_id = [2, 11]` (`</s>`,
  `<|im_end|>`) while `tokenizer.eos_id` resolves to **11** — so the old
  single-id splice matched the template terminator and did not corrupt the
  common path, but a turn that terminates on `</s>`(2) would have been spliced
  into a doubled terminator. Multi-EOS termination IS active on this fork
  (`_build_eos_token_ids` returns a tensor when >1 id), so that case is
  reachable. That guard had
  been living as an *uncommitted* edit on swe_dump / swe_replay_splice;
  `dynamic_engine.py` md5 is now `b9529594` on all three.
- Container `.../users/akamehra/containers/rl-gym.63635108-zstd.sqsh` — bakes
  `megatron_core 0.20.0+14346b65a` and `megatron_bridge 0.7.0+8c46dc42`, and the
  fork MLM is `14346b65a` + 29 patches, so the mount matches the baked Bridge.
- `launch_swe_915.sh` is a port of `swe_dump/launch_swe_dump.sh`
  (`ENGINE=vllm|minf`, dump configs, `SEED_CHECKPOINT` guard) with one change:
  `overlap_param_gather=false` is in **both** engine branches, not just MINF
  (swe_dump leaves it empty on vLLM, so vLLM runs there needed it by hand).
  MINF additionally gets `enable_prefix_caching=${ENABLE_PREFIX_CACHING:-true}`.
- Wipe mitigation applied: `chmod -R a-w` on `nemo_rl/nemo_rl` and
  `nemo_rl/examples/nemo_gym/nemotron-3.5-nano`; Gym left writable. Editing
  those dirs now needs `chmod +w` first.
- Smoke `nano35-swe-915-minf-smoke-16n` **PASSED** 2026-09-25 (job 4005333,
  COMPLETED 52:20, 2 steps, reward 0.398 -> 0.328, 6.4 GB rollout dumps + 6
  token-level chunks, zero tracebacks). Masking verified in the dumped data:
  48/48 rows carry `full_result.instance_config`, 0 carry `mask_sample`.
  First attempt 4005092 FAILED at +384s — a fresh `git clone` leaves
  `3rdparty/Gym-workspace/Gym` **empty**, and `nano35_launch.sh:627` guards only
  on `[[ -d ]]`, so it bind-mounts the empty dir over the container's Gym and
  hides it (`ModuleNotFoundError: No module named 'nemo_gym'`). Fix: clone Gym
  from swe_minf at the pinned `354babf7e`. Always populate that submodule in a
  new clone. Automodel and Megatron-Bridge are NOT overlay-mounted, so their
  empty submodule dirs are harmless.

Reservation arms submitted 2026-09-25 10:16 (both `hero-res`, `batch_long`,
8:00:00, 64 nodes, reservation `sla_res_nemotron_sw_post`, from scratch off
`base_model/step_18/hf`). Agreed arm letters across all three sessions:
- **chain S (swe_915 vLLM from scratch, masking off)** = 4006271
  `nano35-swe-915-64n-vllm-20260925`, started 10:16:57, wall 18:16:57.
- **chain T (swe_915 MINF from scratch, masking off)** = 4006272
  `nano35-swe-915-64n-minf-20260925`, Slurm StartTime 14:52:46 (inherits chain
  R's nodes at its wall). The user held chain P″ (4004676) himself to clear the slot. Launch with
`RESERVATION=1 ENGINE=vllm|minf bash launch_swe_915.sh` — `RESERVATION=1` sets
partition, QOS, reservation and walltime together.

**Why:** the user wants vLLM and MINF runs from the 9-15 stack, without the
changes that mask sequences Gym labels as bad.

**How to apply:** three gotchas worth keeping.
1. `should_mask_flagged_samples` defaults to **true** when absent
   (`rollouts.py:1834`, `env_config.get(...) is not False`) so it must be set
   explicitly. On this lineage SingleController never consumed `mask_sample` at
   all (upstream #3766 is not an ancestor of `952eaf85b`;
   `single_controller.py` has zero references), so the key is belt-and-braces —
   it also stops `AsyncNemoGymRolloutImpl` (`rollout_manager.py:1079`) carrying
   the flag out of `instance_config`.
2. `enable_prefix_caching` — the *committed* value at `952eaf85b` is already
   `true`. swe_dump's `e7aa15ab` set it false and an uncommitted edit flips it
   back, so "skip e7aa15ab" lands on true here, not on the old default.
3. "Force the inference worker to load the checkpoint" is **not a flag**:
   `skip_weight_load=False` is unconditional at `grpo.py:1230` and
   `single_controller_utils/setup.py:268` (commit `d6e88f241`).

Related: [[swe-e2e-base-vs-minf-fork]], [[mlm-rebase-onto-nemorl-main-pin]],
[[feedback-minf-defaults-noprefix-nopg]], [[mounted-tree-deletion-incidents]].

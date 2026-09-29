---
name: swe-main915-penalties
description: "SWE recipe penalties: the v2/old single-controller path never applied the -5 advantage penalties or reward_penalties (dead code), NeMo RL main does; swe_main915 recipe disables both explicitly to reproduce the v2 arms"
metadata:
  node_type: memory
  type: project
  originSessionId: 51d77432-fbb8-4536-b679-a1714b74af54
  modified: 2026-09-28T15:23:47.412Z
---

Asked by rkirby 2026-09-28 ("I think there was a bug in our old checkout I want to make sure
we're reproducing"). Verified from code and dumps:

**Old checkout (swe_915 nemo_rl 3a7fde92c / the v2 arms incl. chain V):** the recipe set
`grpo.penalize_invalid_tool_call/penalize_malformed_thinking: true`,
`invalid_tool_call_advantage/malformed_thinking_advantage: -5.0` and all four
`reward_penalties.penalize_*: true`, but the single-controller path applied NONE of it:
`_apply_configured_message_level_advantage_penalties` was called only from the non-SC
`grpo.py` train loops, `apply_reward_penalties` only from `rollouts.run_async_nemo_gym_rollout`
(async GRPO), and old `single_controller.py` / SC `rollout_manager.py` had no penalty code.
Chain V evidence (`nano35-swe-v2-from0-parity-minf-20260927`): 0 penalty-rate metrics in the
driver log; 192/192 `resolved=True` samples of step 0 keep reward 1.0; the 3 sequences with
flagged assistant turns carry ordinary group-normalized advantages (-0.379 / 0.0), not -5.
`env.should_mask_flagged_samples: false` in both stacks (env-flag sample masking off).

**NeMo RL main (swe_main915):** both are wired into the SC path --
`rollout_manager.py:1419 apply_reward_penalties` sets `full_result["reward"] = 0.0` when
triggered (so the dumped Gym reward is zeroed too; detect via `full_result.resolved=True`
with reward 0), and `single_controller.py:5327 apply_message_level_advantage_penalties`
overwrites flagged tokens with -5.0 (gated on the two `*_advantage` values being non-null;
masks come from `payload._add_message_violation_masks`; `mask = token_mask * final_sample_mask`,
so logprob-masked samples never show the -5 -- which is why the buggy smoke's 3 flagged
sequences, all masked, were a confounded check). Smokes 4065840 / 4065835 / 4057393: 10 / 13 /
7 resolved samples per run had reward zeroed; `malformed_think_tag_rate` 0.086-0.125,
`empty_final_answer_rate` 0.008-0.031.

**Decision:** `swe_sc_cmh_common.yaml` (nemo_rl `rkirby/swe-main915-latest`) now sets the two
advantages to `null`, `penalize_invalid_tool_call/penalize_malformed_thinking: false` and all
four `reward_penalties.penalize_*: false`, so chain U reproduces the v2 arms' effective
behaviour by configuration. Re-enabling penalties later is a deliberate training change, not
a bug fix. Validation smoke: `nano35-swe-main915-minf-smoke-16n-nopenalty` (expect no
penalty-rate metrics, `resolved == reward` on every sample). Tooling:
`/home/rkirby/.claude/jobs/51d77432/tmp/{flag_advantages.py,reward_vs_gym_reward.py}`
(copy to `swe_main915/analysis/prefix_skip_bug/` if needed long-term).

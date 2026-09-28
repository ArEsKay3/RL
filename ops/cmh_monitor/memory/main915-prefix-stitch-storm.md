---
name: main915-prefix-stitch-storm
description: "chain U (main@9-15 MINF) dies of a superlinear ValueError: Prefix stitching requires compact_prompt_token_ids storm while still closing steps - main@9-15 only, invisible to smokes"
metadata:
  type: project
---

2026-09-27: chain U (main@9-15 MINF from scratch, masking off) job 4050528 cancelled 13:43 PDT by rkirby after 34 min.

`ValueError: Prefix stitching requires compact_prompt_token_ids from the previous Megatron-Inference response.` count in `ray_logs/4050528-logs/ray-driver.log`:
51 (20 min) -> 109 (32 min) -> 3,865 (34 min) -> 3,869 (35 min). Superlinear, not a fixed background rate. Lines carry Ray "[repeated Nx across cluster]" suffixes so true per-request failures are higher. Ranks 12, 23, 36, 76, 100, 112, 120, 124+.

The job never died: it closed train step 6, saved rungs step_5 and step_6, kept dumps and gym bars advancing, guard counters 0/0.

**THE ARM PRODUCED NOTHING TRAINABLE.** Measured from `dumps/rollouts/target_step_0000N.jsonl` (found by the Log Analysis session, verified independently):

| run | assistant msgs/episode (median) | generated tokens (median) | reward mean | nonzero rewards |
|---|---|---|---|---|
| chain U 4050528 steps 1/3/5 | **1** (1 on 512/512) | 91 / 109 / 92 | **0.0000** | **0 of 512** |
| smoke 4051227 (post-fix) steps 0/1 | **1** | 81 / 88 | **0.0000** | **0 of 128** |
| chain P⁗2 (clean MINF, v2 stack) steps 22/23 | 39.5 / 44.5 | 8,179 / 9,867 | 0.77 / 1.00 | 37/48, 16/16 |

Every episode dies after the first assistant message, ending mid tool call at `<parameter=command`. The 256 stitch failures are what kills turn 2: the agent gets `Invalid 'messages'` and the episode ends. Batches are FULL (`num_valid_samples: 512.0`, 512 dump lines) - they are full of dead stubs. GRPO on all-zero rewards gives zero advantage, so chain U's step_5/step_6 rungs are effectively the base model and are worthless as eval targets.

**Two of my own calls were wrong here.** First I said the batches were "silently degraded" without measuring. Then I corrected that to "no samples lost, rungs comparable" on the strength of `num_valid_samples` and dump line counts - which are exactly the two checks that cannot see this. Cancelling chain U was right for a reason better than the one given at the time.

**ROOT CAUSE** (Get Latest Main? session, 2026-09-27, confirmed byte-identical at the 1e7598cb base so not introduced by the 5 MLM patches): `TokenIDLogProbMixin` / `TokenIDLogProbTypedDictMixin` in `nemo_gym/openai_utils.py` declare `prompt_token_ids`, `generation_token_ids`, `generation_log_probs` and `routed_experts` but NOT `compact_prompt_token_ids`, so Gym silently drops that field on every round-trip through its own typed models. Gym-side gap, not a main@9-15 code bug.

**But that was not the operative mechanism.** `openhands/agenthub/nemo_gym_client.py` in the vendored OpenHands build never uses Gym's Pydantic models for this path. It has its own hardcoded 3-name allowlist: `fields_to_remove = ["prompt_token_ids", "generation_token_ids", "generation_log_probs"]` at lines 86-90 (stripped from all but the most recent message before the next request) and the same three names copied by plain `dict.get()` into `response._provider_specific_fields` at lines 149-156. `compact_prompt_token_ids` is in neither list and appears nowhere in the OpenHands tree, so it can never survive into turn 2 regardless of what Gym sends.

**Localised:** `grep -rn compact_prompt_token_ids Megatron-LM` = **0 hits** on the v2 stack (MLM 880de0fce), **1 file** on main@9-15 (`chat_completions.py:953` demands it, `:1370` sets it). Both stacks default `prevent_retokenization=True` and neither Gym overrides it, which is why every v2 arm works. The newer MLM added a required round-trip field that OpenHands' allowlist never learned.

**Fix path (rkirby 2026-09-27):** `prevent_retokenization=false` is RULED OUT - "you absolutely cannot use prevent_retokenization false". Remaining option is the 2-line OpenHands field-list patch, in a lockdir-protected cached build artifact pinned to commit 2e96cb6b8. Open question is whether a patch there survives a fresh `swe_openhands_setup` rebuild or the pin has to move.

Never seen on any v2-stack arm (P⁗, P⁗2, J, G, I, F, B all clean).

**It is NOT scale-gated** - it was already present in the "clean" 16-node smoke 4024035 (256 occurrences in 630 lines), just not grepped for. What gated it was *bug ordering*: with `aws_region_name` unfixed almost no conversation reached a second turn, so `prevent_retokenization`'s "previous assistant message" precondition was rarely exercised. Fixing bugs #1/#2 is what exposed #3. A 16-node smoke reproduces it cheaply and reliably.

Third main@9-15 failure in a row, after `aws_region_name` and the `checkpointing.save_data_plane=true` ValueError at setup.py:1033 that killed 4023058 (that one WAS smoke-invisible: SMOKE=1 sets `checkpointing.enabled=false`).

**Why:** the real hazard was not "smokes are too small" - it was that nobody was grepping for unfamiliar exception strings, so a smoke that already contained 256 instances was called clean.

**How to apply:** `num_valid_samples` and dump line counts are NOT health checks - they count that a row exists, not that it holds an episode. The real check is three numbers from the rollout dump: median assistant messages per episode, median generated tokens, and mean reward. On this workload a healthy MINF arm is ~40 messages / ~8-10 k tokens / reward 0.7-1.0; anything at 1 message / ~90 tokens / reward 0.0 is a dead agent loop no matter what the job state says. A smoke whose pass criterion is "N/N valid samples" cannot see this and will go green on a run that trains on nothing. Also diff distinct exception strings against the known-noise list and track the rate, not the count. Owner session is "Get Latest Main?". Related: [[swe-main915-workspace]], [[megatron-prefix-splice-7598]], [[feedback-cancel-relaunch-on-hang]].

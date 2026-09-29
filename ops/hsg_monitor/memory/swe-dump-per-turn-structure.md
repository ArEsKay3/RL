---
name: swe-dump-per-turn-structure
description: "Per-turn prompt/generation tokens are fully recoverable from the existing SWE dumps (.pt flat row + .jsonl n_tokens); the rollout .jsonl carries no token IDs; NeMo RL asserts per-turn prefix identity; turn_alignment_check.py verified 512/512 rows on chains G and M at steps 1 and 10 with flat, engine-identical per-turn logprob mismatch"
metadata: 
  node_type: memory
  type: project
  originSessionId: 824af663-9c1e-4cb4-9c9e-fd1feb14c5fa
  modified: 2026-09-25T19:40:59.180Z
---

Verified 2026-09-25 (user asked whether the .jsonl carries Gym's per-turn "prompt_tokens"/"generated_tokens" and whether we only know the final turn).

- Rollout `.jsonl` (`rollout_dump.py::_message_summary`) keeps per message only `role`, `n_tokens` (= len of the message's `token_ids`), and for assistant turns `is_invalid_tool_call`/`has_malformed_thinking`. No content, no IDs. Per-turn *counts* also exist from OpenHands under `full_result.per_turn_metrics.token_usages[k]` (`prompt_tokens`, `completion_tokens`, `per_turn_token`). Gym `token_id_capture` is False in these runs; `gym_results/` per-instance dirs hold only timings/patch/scripts.
- Gym's actual annotation is `prompt_token_ids`/`generation_token_ids`/`generation_log_probs` on each assistant turn's chat completion (`nemo_gym/openai_utils.py TokenIDLogProbMixin`). `nemo_rl/environments/nemo_gym.py::_postprocess_nemo_gym_to_nemo_rl_result` (~line 840-965) turns them into the flat row: per turn, `prompt_token_ids[len(seen):]` -> user message, `generation_token_ids` -> assistant message, and asserts `seen == prompt_token_ids[:len(seen)]` ("Non-contiguous messages found!"). Failure = exception -> group dropped (rollout_manager ~1368) and counted by exception name. Zero hits in all Ray logs of chain G (73k files) and chain M (76k files).
- token_level `.pt` is PACKED 1-D: `input_ids`/`token_mask` length sum(input_lengths); `generation_logprobs`/`prev_logprobs` length sum(input_lengths-1), index j <-> token j+1. Keys: step, chunk_index, trainer_version, sample_ids, tags, input_lengths, input_ids, token_mask, logprob_offset, generation_logprobs, prev_logprobs, advantages, rewards, sample_mask_before/after, seq_mult_prob_error.
- Tool: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/engine_loop_test/analysis/optim/turn_alignment_check.py` (+ `.sbatch`, cpu partition, container import ~14 min cold / ~3 min warm on cpu-0004). Log: `engine_loop_test/logs/turn_alignment_4007998.out`. Result: c1 sum(n_tokens)==length, c2 mask==assistant spans, c3 turns==runs all 512/512 for G and M at steps 1 and 10. c4 (OpenHands usage list) 482-491/511: the misses are extra OpenHands LLM-call entries (retried/discarded calls), not data problems -> the script's RESULT line says FAIL only because of c4.
- Per-turn |generation_logprobs - prev_logprobs|: mean 1.1-1.8e-2, p50 ~1e-6, p99 0.20-0.29, bit-eq 19-32 %, flat across turn index and the same for MINF (G) and vLLM (M). Includes lag-1 staleness + kernel numerics; no engine-specific later-turn excess.
- Both engines splice the previous turn's real tokens onto a re-rendered tail (MINF: Megatron `_replace_prefix_tokens`, pre-#7598; vLLM: NeMo RL `replace_prefix_tokens`, fixed), so engine input == training row by construction; splice correctness was separately covered by the junction scan in [[megatron-prefix-splice-7598]] (15.16 G tokens clean). Scanner scripts preserved at `engine_loop_test/analysis/optim/junction_scan/`.

**Why:** the user considered adding a per-turn token dump to compare "what we reinforce" with "what the engine sees"; that comparison is already implied by construction + assertion and reconstructible from existing dumps, so no new dump is needed.

**How to apply:** for any per-turn question, segment the flat row with the .jsonl `n_tokens` (cumsum) and use `token_mask`; use the alignment script as the template. Related: [[swe-splice-experiments]], [[swe-dumpbrowse-server]], [[swe-dump-runs]].

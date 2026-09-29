---
name: megatron-prefix-splice-7598
description: "Megatron PR #7598 (EOS-counting prefix stitching) vs NeMo RL's monotonic-prompt assertion: the assertion cannot catch it, MINF runs the unfixed splice, but 15.16 G tokens of dumps show zero corruption because truncate_history_thinking is false"
metadata:
  type: project
---

Investigated 2026-09-24. Two separate copies of the same multi-turn prefix-stitching algorithm exist:

- NeMo RL `nemo_rl/models/generation/openai_server_utils.py:28` `replace_prefix_tokens` — ALREADY implements Megatron PR #7598's fix (counts EOS in the template prefix, cuts at the N-th EOS in the current render). Imported ONLY by `vllm/vllm_worker_async.py` and `trtllm/trtllm_http_server.py`.
- Megatron-LM fork 880de0fce `megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py:339` `_replace_prefix_tokens` — the PRE-#7598 version: strips <=1 trailing EOS from the previous turn, then takes the LAST EOS in `current[:min(len(retok), len(current))]` as the splice point. MINF arms use this, not NeMo RL's, because the MINF path exposes Megatron's own HTTP server (`mcore_generation_config.expose_http_server`). Confirmed live: "Avoiding prefix retokenization" appears in every MINF driver log.

THE ASSERTION CANNOT CATCH IT. `nemo_rl/environments/nemo_gym.py:852` asserts `seen_token_ids == prompt_token_ids[:len(seen_token_ids)]` — prefix containment, not equality. The splice always returns `previous_turn_tokens[:maybe -1] + current[boundary:]`, so a mis-chosen boundary only corrupts at/after index `len(seen)-1`. Overlap with the asserted region is at most ONE token. Doubled-EOS and duplicated-history outcomes land entirely in the unconstrained new-observation delta and pass silently; only a dropped terminator at the junction fires it.

NO EOS-ID MISMATCH: `config.json` says eos_token_id 2 but the tokenizer's `eos_token` is `<|im_end|>` = 11, which is what both sides resolve to and what every generation terminates on. `</s>` = 2 is only ever file content.

WHY IT DOESN'T BITE: the heuristic breaks only when the chat template re-renders history at a different length. `chat_template.jinja:14,114,149` does strip <think> from assistant turns before the last user message under `truncate_history_thinking` (default True) — but the runs set `chat_template_kwargs: {enable_thinking: true, truncate_history_thinking: false}` (see any checkpoint config.yaml ~line 133). With it false the current render is a clean extension of the prefix render, so `scan_len` never overshoots.

EVIDENCE (full sweep of every token_level dump, 887 files, 15.16 G tokens, 167,168 rollouts, 15,087,105 turn junctions): 0 doubled `<|im_end|>`, 0 dropped terminators, 0 duplicated tails, 0 load errors. The only 11 deviations are turns where the model emitted `</s>` copied out of repo file content (a stop token per generation_config `eos_token_id: [2, 11]`); `J_DOUBLE == gen_term_2 == 11` exactly, and the rate is slightly HIGHER in vLLM (6/6.09 M) than MINF (5/9.00 M), so it is engine-independent content, not stitching. Separately, 185,229 observation deltas across both engines open with exactly `[1010, 10, 3263]` (`\n<|im_start|>`) 100% of the time.

**Why:** this was the first plausible mechanism for a MINF-vs-vLLM divergence that sits below the training loop; ruling it out with a number matters as much as finding it would have.

**How to apply:** the MINF arms are safe ONLY because of that one flag. Setting `truncate_history_thinking: true`, adopting a template that re-renders history at a different length, or any history reasoning-stripping would expose MINF (not vLLM) immediately — re-run the scan then. Scanner + positive control: `/home/rkirby/.claude/jobs/e247f0cd/tmp/eos_scan_all.py`, `poscontrol.py`, `delta_openers.py`, `perrow.py`. Related: [[swe-length-growth-investigation]], [[mlm-upstream-fixes-review]], [[swe-dump-runs-nopg]].

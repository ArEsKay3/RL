---
name: swe-dump-analysis-2026-09-19
description: 2026-09-19 comparison of the vLLM (run A, steps 0-9) and MINF (run B, steps 10-17) rollout+token dumps; reward attribution identical, parser strictness identical, notable engine differences and the analysis tool paths
metadata:
  type: project
---

Analysis of the first segments of the SWE dump runs (see [[swe-dump-runs]]), tools under
`users/rkirby/workspaces/swe_dump/tools/` (rollout_summary.py, token_level_summary.py,
aggregate_dumps.py, runaway_detail.py, inspect_row.py, pt_numpy.py) and outputs under
`swe_dump/analysis/` (aggregate_report.md, rollouts/*.summary.jsonl, tokens/*.rows.csv,
runaway_detail.csv, wandb_step_metrics.json).

Findings (4096 rollouts per engine, disjoint prompt sets, different policy stages):
- Reward == resolved exactly in both engines; Gym mask_sample (67 MINF / 79 vLLM rows) is
  dropped by SC in both (sample_mask_before == 1 everywhere); only masking is the seq
  logprob-error mask (4 MINF / 6 vLLM rows). Advantage sign matches group-relative reward
  in 100% of rows; ~53% rows have zero advantage (uniform groups).
- Parser: the mounted fork Megatron-LM 880de0fce predates #6411 (f7b220977, 2026-08-13,
  "Fix tool call reasoning boundary" = implicit <tool_call> reasoning end) AND my opt-in
  commit c09b3fe68/38bb3fd9f (2026-09-14). So MINF is strict (tool call before </think>
  stays in reasoning) which equals vLLM nano_v3 + qwen3_coder. Apparent "function_call
  without </think>" rows are parallel tool calls whose earlier items carry an empty
  generation_str, in both engines.
- Runaway generations: no per-turn cap in either engine (single turns up to 141k/153k
  tokens); rows with a >16k-token turn score ~0 reward in both; MINF tail is heavier
  (p99 single-turn 24.0k vs 15.6k tokens, 39 vs 31 rows >32k) but prompts differ.
- Overflow surfacing differs: MINF HTTP 400 -> Gym empty completion -> OpenHands
  context_window; vLLM raises 500 (ValueError) -> some rows land as agent_error_kind
  "other" (4 truncated vLLM rows). Reward 0 either way.
- Cosmetic: MINF returns content "\n" on every tool-call turn (fork chat_completions keeps
  the parser leftover), so Gym emits a blank message item per turn; vLLM does not.
  Rendered next-turn prompt is identical (`</think>\n<tool_call>`), verified on token ids.
- Same-prompt reward check via W&B (nvidia/ultra-v3-swe-e2e-convergence, per-segment runs):
  steps 1-8 main MINF chain 0.358 vs run A vLLM 0.361; steps 11-17 main MINF 0.275 vs run B
  MINF 0.276; per-step values track within noise, so no engine-level reward gap so far.
- Blank-content mechanics (traced 2026-09-19): both engines leave "\n" before <tool_call>
  after the reasoning split; vLLM engine/serving.py nulls whitespace-only content, the fork
  chat_completions.py returns it. Gym vllm_model returns content = <think>reasoning</think> +
  leftover to OpenHands, so OpenHands sees a trailing "\n" only; it lands in action.thought,
  conversation_memory keeps content (strip check passes via the think block), Gym strips the
  think tags on the way back and the chat template `| trim`s it. vLLM nulls whitespace-only
  content ONLY when a tool call was parsed; with no tool call "\n" survives on both engines.
  Only divergence: a turn with NO content at all (`</think><|im_end|>` or an immediate
  `<|im_end|>`): vLLM returns content None -> OpenHands response_to_actions raises
  LLMContextWindowExceedError -> AgentState.ERROR -> Gym classifies context_window; the fork
  returns content "" -> MessageAction('') -> codeact_user_response nudge -> agent continues
  (3 identical empty turns -> monologue stuck -> stuck_in_loop). 0 such turns seen in 4
  scanned step files. Parity fix =
  mirror vLLM's whitespace check in the fork endpoint or `if content.strip()` in Gym's
  postprocess_assistant_message_dict. nv-OpenHands checkout: Gym/cache/swe_agents/
  swe_openhands_setup/nv-OpenHands-dd7d06ea/5f0180054.../OpenHands.
- MINF reports cache_read_tokens (~5.6M/rollout); vLLM reports 0. Per-call latency p50 23 s
  vs 16 s. Logprob mismatch distributions are near identical (mean |dlp| 0.0155 vs 0.0140).

**Why:** the user asked whether the rollout objects attribute reward or handle runaways
differently between engines, and whether the MLM carries the implicit-reasoning-end fix.
**How to apply:** rerun `aggregate_dumps.py swe_dump/analysis <tokenizer.json>` after new
segments land; the W&B fetch lives in the job tmp dir (step_rewards via GraphQL sampledHistory);
akamehra's baseline runs are under entity joc and were not readable with this key.

**Notebook (2026-09-20):** `swe_dump/notebooks/browse_dumps.ipynb` browses every dump (inventory, per-step aggregates, group view, streamed raw rollouts with turn text, token-level sequences with gen/trainer logprobs, hist.json, cached W&B, Gym per-instance results, early-step comparison). Needs only pandas+numpy (matplotlib/tokenizers optional); all cells verified headless. No Jupyter on the login node; the user declined the background venv install (`uv venv .venv-nb` + jupyterlab) on 2026-09-20, so run it in their own environment.

**Summaries (2026-09-20 06:2x):** `analysis/logs/summaries_all.sbatch` (cpu partition, ~2 min) builds `analysis/rollouts/*.summary.jsonl` + `analysis/tokens/*.rows.csv` for every dump file whose summary is missing or older than the dump (job 3875785 built 34+97 files; a second run refreshes 4 stale ones). Gotchas learned: a resumed segment APPENDS regenerated rollouts to the existing `target_step_*.jsonl` (same target step, new sample_ids) and REWRITES `step_*_chunk_001.pt` onward (chunk numbering restarts), so summaries made before a resume go stale; a segment that dies mid-step leaves untrained committed rollouts in the rollout dump; a timed-out segment leaves a partial step in both dumps. The notebook marks `R.trained` / `T.committed` (present on both sides) and `per_step` / `groups` use only those; `per_step_tokens(T)` is the authoritative trained view.

**Refresh pipeline (2026-09-20 08:2x):** `sbatch analysis/logs/refresh_all.sbatch` (cpu, ~5 min) = summaries (missing/stale only) + tools/build_wv_map.py + think_stop_audit + token_bias_by_id + think_close_by_step (missing/stale chunks only) + the three aggregators -> analysis/think_stop_report.md, token_bias_report.md, think_close_by_step.md. Covers run A, run B, chain C, chain D, chain E. The chain D and chain E crons run it when new train steps land and report chain D/E rows next to run B/run A (metrics: geo-mean P(</think>) = exp(mean engine logprob of sampled closes), 10th-pct P, share of closes with engine logprob < -0.1, think-length p95).

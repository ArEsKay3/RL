---
name: swe-dump-inventory-and-vllm-rerun-plan
description: "Inventory (2026-09-18) of trajectory / token-logprob dump facilities in the swe_minf nemo_rl tree (rkirby/swe-v2-minf 952eaf85b) and the proposed vLLM rerun continuing from the MINF chain's step_10 with full dumps; user asked for the review before any change, no code changed yet"
metadata:
  type: project
---

User request 2026-09-18 ~11:35 PDT: review what dump/debug code is enabled and what data exists, then propose
what to add for a rerun of the SWE-E2E v2 experiment on vLLM, continuing from step_10 of the MINF chain
([[swe-minf-64n-run]]), dumping all trajectory-level and token/logprob-level data. Review delivered; NO changes made.

**Enabled today (SC path):** only `NRL_LOGPROB_DUMP_DIR` (grpo.py `_dump_logprob_pairs`, called from
`compute_and_apply_seq_logprob_error_masking`, reached from `SingleControllerActor._advantage_stage`,
single_controller.py ~2046). Payload per call: generation_logprobs, prev_logprobs, lp_error, seq_token_counts,
seq_mult_prob_error, sample_mask, rewards, minf_logprobs_mode; masked-in tokens only, fp32. Not set in the chain's
launcher, so nothing is dumped now. Bugs for SC: no `step=` passed -> always logprobs_step00000.pt; called once per
streaming chunk -> overwrites; no sample ids / input_ids -> cannot join to prompts.
**Persisted today:** checkpoint replay_buffer.pt (only untrained buffered groups at save time; fields input_ids,
input_lengths, generation_logprobs, token_mask, sample_mask, prompt_ids_for_adv, total_reward + tags weight_version,
rollout_environment/generation_length/reward/truncated, numeric env extras; no prev_logprobs/advantages/text/task id),
pending_rollouts.pt (prompt journal), replacement_reserve.pt. Gym per-instance dirs land on the host because the
launcher mounts the workspace Gym checkout: swe_minf/nemo_rl/3rdparty/Gym-workspace/Gym/results/swebench_results_<session>/
<instance>_<ts>_<uuid>/ (nemo_gym_metrics.json, eval_results/report.json, patch.diff, agent/eval apptainer logs,
container scripts, instance jsonl; no token ids/logprobs; 18 sessions, 5000+ instance dirs for segment 9). W&B: scalars only.
**Exists but not wired on SC:** train_data_step{N}.jsonl (grpo.py ~3717, gate env.should_log_nemo_gym_responses is
inverted), W&B full-result tables (logger.wandb.log_nemo_gym_full_result_tables), print_message_log_samples, token
mult-prob plot; SC TODO at single_controller.py ~1478. `TQReplayBuffer.commit` (async_utils/replay_buffer.py ~817)
drops Completion.env_extras (= Gym full_result incl. decoded text, _ng_task_index, per-turn structure, tool-call flags)
and PromptGroupRecord.rollout_metrics (incl. a per-group W&B table built at rollout_manager.py ~1159).
Appears (verify): apply_reward_penalties is not reachable from rollout_manager, so grpo.penalize_invalid_tool_call /
penalize_malformed_thinking (-5) are inert on SC in both arms.
**On other fork branches:** d63cb42c3 `NRL_TRAJECTORY_DUMP=1` (trimmed input_ids + bit-packed token mask into the
logprob dump; applies cleanly on HEAD); 41eae1a03 / 9ebf12ec3 advantage dumps (adv_stepNNNNN.pt, NRL_ADV_DUMP_MESSAGE_LOG,
NRL_ADV_DUMP_MAX_FLAGGED; do not apply cleanly, manual port to _advantage_stage needed).
**Gym-side unused knobs, forwardable via env.nemo_gym with no nemo_rl change:** results_dir, observability_enabled +
model_call_capture_dir, token_id_capture {enabled, dir, sink} (records token ids + logprobs per model call; dir is
meant to be node-local, use a shared path or sink). Gym checkout 354babf7e has them.
**Reference vLLM run:** akamehra/runs/nano35-swe-v2-stream128-inorder1-cmh-64n (job 3541201; vLLM TP4 EP1,
logprobs_mode processed_logprobs, 32+32 non-colocated, in_order lag 1, GBS 512, seed 42, rungs step_5..step_73).
akamehra's telemetry branch (bundle nemo-rl-swe-v2.bundle, PR #4068 + #3766 backport) = W&B metric distributions only.

**Proposed rerun (not started):** new EXP e.g. nano35-swe-v2-stream128-inorder1-cmh-64n-vllm-from10; copy (not symlink)
minf checkpoints/step_10 (369 GB) into <new>/checkpoints/; CONFIG_PATH swe_sc_cmh_stream4.yaml (vLLM base), same
SHAPE_OVERRIDES, `+checkpointing.load_replay_buffer=false` (journal regenerates the same in_order prompts); resume picks
the highest step_* dir. Additions: (A) fix/extend the advantage-stage dump (step + chunk in filename, sample/group/journal
ids, weight_version, input_ids trimmed, token_mask packed, prompt_ids_for_adv, total_reward, advantages written after
computation), ~0.8 GB/step; (B) per-rollout jsonl at commit (task identity, reward, per-turn roles/token counts, tool-call
flags, truncated, timings) with optional decoded text / full_result JSON; (C) env.nemo_gym.results_dir under the run dir,
optional token_id_capture to a shared dir. Decision points for the user: vllm logprobs_mode (processed = reference
parity, raw = like MINF), include text/full_result or not, schedule vs the running MINF chain, run name.

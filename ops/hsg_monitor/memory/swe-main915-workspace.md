---
name: swe-main915-workspace
description: SWE-E2E stack on nemo_rl main (swe_main915); chain U (MINF) idle at step_15 and chain W (vLLM) idle at step_6 since the 09-28 outage, followers cancelled 09-29; load_replay_buffer is inert on main's SC resume (what it restores, decoded per checkpoint, port design)
metadata: 
  node_type: memory
  type: project
  originSessionId: 51d77432-fbb8-4536-b679-a1714b74af54
  modified: 2026-09-29T11:55:26.860Z
---

Workspace: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_main915/`.

Base is nemo_rl **main** as of 2026-09-15 (`b03da0f46`) -- not [[swe-915-workspace]]'s branch tip
(`952eaf85b`, dated 9-15 but forked off main at `b59165262` on 2026-08-21 and never resynced, 172
commits of drift). Rkirby's correction: "when I say the commit in nemo_rl on 9_15 I mean nemo_rl
main." Megatron-Bridge's pin off main@9-15 resolves to Megatron-LM `1e7598cb`, 273 commits ahead of
swe_915's `14346b65a` base.

**Branches** (all local, pushed nowhere):
- nemo_rl `rkirby/swe-main915`, top `de12a8e36`, base `b03da0f46` -- recipe tree + dump feature +
  MINF fixes hand-ported from swe_915's lineage, plus 2 genuine nemo_rl-main bugs found and fixed
  along the way (`NATIVE_MULTIMODAL_KEYS` stale import; `nemo.lens` hard-imports with no container
  guard).
- Megatron-LM `mlm-main915`, top `c04ebc896`, base `1e7598cb` -- 5 patches (multi-EOS termination,
  Gumbel-max fp32 sampling, tokenizer coercion, dynamic-engine prefill logprobs + EOS-set replay)
  hand-adapted or cherry-picked from the fork line onto the main-tracked tree.
- Gym `rkirby/gym-main915`, top `13ef93615`, checked out at the pin main@9-15 declares (`267305e2a`)
  -- 2 fixes, both the same root cause, see below.

**Gym fixes (the openai==2.44.0 strictness incident):** Gym's `openai==2.44.0` pin (commit
`65129dd49`, 2026-08-26, deliberate: "unknown fields rejected instead of silently removed")
post-dates swe_915's older Gym pin (`354babf7e`) and broke two independent things the SWE-agent
sandbox's bundled OpenHands/litellm client sends that were never part of the OpenAI schema:
1. `aws_access_key_id`/`aws_secret_access_key`/`aws_session_token`/`aws_region_name` attached to
   every completion call regardless of target -- rejected by `NeMoGymResponseCreateParamsNonStreaming`
   / `NeMoGymChatCompletionCreateParamsNonStreaming`'s `extra="forbid"`. Fixed with a
   `model_validator(mode="before")` on both models (commit `81cea49a3`), same pattern
   `normalize_output_items_for_replay` already used for a different quirk.
2. A redundant `name` field on `tool`-role chat messages -- `ChatCompletionToolMessageParam` has
   never declared `name` at any openai SDK version (confirmed directly against two installed
   `openai==2.44.0` venvs, `/tmp/gymtest-venv` and `/tmp/pr3670-review-venv`: only
   `content`/`role`/`tool_call_id`). `NeMoGymChatCompletionMessageParam` is a flat, non-discriminated
   `Union` of 6 message types, so one bad field fails every branch and the whole `/chat/completions`
   request 422s. Gym already had `_strip_outer_tool_call_names` compensating for the same client's
   redundant `name` on the assistant's `tool_calls[]`; added the matching `BeforeValidator` for the
   tool-response side (commit `13ef93615`).

Checked the full 94-commit range to Gym's current main for an upstream fix for either -- none exists
(one adjacent precedent, `dbd26a976`, patches a different server for different quirks). Both bugs
were silent in the old `swe_915` stack not because the code differed (the tool-message class body is
byte-identical between the two Gym pins) but because its older openai SDK pin predates 65129dd49
entirely.

Verification took 3 smoke cycles because the first "clean" result (job 4022822: 2/2 steps, exit 0,
128/128 valid samples, zero `aws_region_name`/`extra_forbidden`) still had 256 silent
`Hit validation exception!` occurrences (8% of completions) that I initially mis-read as pre-existing
benign noise -- only caught by diffing occurrence counts against the old swe_915 stack's logs (0 there
vs 256 here), which is what prompted fix #2. The next attempt (job 4023471) then failed for an
unrelated, self-inflicted reason: a blanket `chmod -R a-w` re-lock of the *entire* Gym tree (following
the [[mounted-tree-deletion-incidents]] wipe-mitigation pattern used for `nemo_rl/nemo_rl`) caught
Gym's own runtime cache dir (`Gym/cache/`, 11 GB, untracked except `cache/.gitignore` -- OpenHands
setup locks, sandbox bootstrap state), causing a `PermissionError` on a lockdir `mkdir`. **Lesson:**
that blanket-lock pattern is only safe for pure-source trees with no runtime-writable subdirectories;
Gym is a live application and needs `cache/` (at least) left writable. Fixed by `chmod -R u+w` on
`cache/` only, leaving the actual `.py` source locked. Job 4024035 confirmed genuinely clean: exit 0,
128/128 valid samples both steps, zero occurrences of either validation exception.

**Launcher:** `launch_swe_main915.sh` in the workspace root (not git-tracked). `SMOKE=1 ENGINE=vllm|minf`
for the 16-node wiring test; `ENGINE=vllm|minf` alone for the full 32+32 shape. `RESERVATION=1` sets
`batch_long`/`hero-res`/`sla_res_nemotron_sw_post`, walltime 8:00:00.

**Chain U** (first real 64-node run on this stack): job 4023058, MINF, `nano35-swe-main915-64n-minf-20260926`,
submitted 2026-09-26 after the smoke passed. Immediately put on a user hold (`scontrol hold`, still
held as of submission) -- per "Manage Ongoing Runs," it queued behind chain P⁗ (highest-priority arm,
targeting step_50) and chain Q⁗, both occupying the reservation's two 64-node halves via multi-segment
singleton chains at identical priority 86845; there is a real (small, unquantified) race window at
each 8h wall where chain U could slip into a just-freed slot ahead of the next singleton segment
during its COMPLETING transition. Holding was my call, not rkirby's instruction -- release when a
chain genuinely finishes its queued segments, not mid-transition, or if rkirby weighs in directly.

**2026-09-27 12:56 — chain U released.** rkirby (via "Manage Ongoing Runs"): put chain P⁗2 on hold and get chain U running; the manager held P⁗2's queued segments, released 4023058 (PENDING Resources at 12:57, hero-res / batch_long / sla_res_nemotron_sw_post, 8 h, 64 nodes), and cancelled P⁗2's running segment. Chain U has never run before this; its submit-time provenance is nemo_rl de12a8e36 with the Gym submodule dirty. Post-submit drift check (Cross Train Experiment session, 13:00 on 09-27): nemo_rl HEAD still de12a8e36 (no reflog moves since 09-25 23:07), Megatron-LM still c04ebc896, Gym now 13ef93615 (committed 09:27:48 on 09-26, i.e. 25 min AFTER submit — the tool-message `name` fix that smoke 4024035 later validated); the only source file newer than the submit is Gym/nemo_gym/openai_utils.py (09:27). Because USE_SNAPSHOT=0 the job will run with the fixed Gym, which is the validated state, not a regression. Chain U's overrides carry no `+checkpointing.load_replay_buffer=false` and no `grpo.seed`, and it is a single job with no follow-on segments queued.

**2026-09-27 13:02 — chain U FAILED at startup (first 64-node attempt, job 4023058, 4 min 20 s).** `ValueError: SingleController checkpointing with a replay-checkpoint-capable sampler requires checkpointing.save_data_plane=true so completed, unconsumed rollouts are recoverable.` raised in `nemo_rl/algorithms/single_controller_utils/setup.py:1033` (and mirrored in single_controller.py ~479). On main@9-15 `checkpointing.save_data_plane` (NotRequired bool, "include the native TQ snapshot and replay-buffer metadata; simple data-plane backend only") is mandatory whenever checkpointing is enabled and the sampler supports buffer checkpointing (in_order does); upstream's `grpo_math_1B_megatron_single_controller.yaml` sets it true, but the ported recipe `swe_sc_cmh_common.yaml` never did. The smoke (4024035) passed only because `SMOKE=1` sets `checkpointing.enabled=false`, so the whole checkpoint path on this stack — including the step_5 data-plane snapshot save — is still unexercised. Fix path actually taken: at 13:06:49 on 09-27 swe_sc_cmh_common.yaml was edited (uncommitted, live-mounted tree) to add `save_data_plane: true` and `load_replay_buffer: false` under `checkpointing:`; with the key in the yaml, a `+checkpointing.save_data_plane=true` command-line override would COLLIDE (Hydra `+` refuses an existing key), so the relaunch must be the plain launch with no extra override. My earlier DRY_RUN render with the `+` token (swe_prefix_keep/logs/dry_run_render_U_savedataplane.out) is therefore obsolete. Relaunch is the manager's/rkirby's call (chain U is not this session's job). Also note the 64 nodes chain U held (nvl72d056/167/180/205 racks) sit idle until it is relaunched; Q⁗2's and P⁗2's queued segments cannot take them (singleton / held).

**2026-09-27 13:43 — chain U relaunch cancelled by the manager, not relaunched again** (reported by the manager 14:3x): its rollouts were 1-turn ~90-token stubs with reward 0.0000 on 512/512 — a dead agent loop from a `compact_prompt_token_ids` round-trip bug in the vendored OpenHands on that stack (first spotted by the "Log Analysis" session in the token-level dumps: num_assistant_messages=1 everywhere, episodes ending mid tool call at `<parameter=command`). Its 64-node slot went to chain V (MINF from scratch, vLLM numerical parity, prefix cache kept across refits; job 4052187, running since 14:30, single 8 h segment, not this session's). Letters H (never-launched max_tokens plan) and W (planned main@9-15 vLLM control) are reserved per the manager's registry.

**2026-09-27/28 — stack moved to the actual latest commits, long-episode defect root-caused and fixed.**
Branches now: nemo_rl `rkirby/swe-main915-latest` (base `4aaa48fab`, top `c0f9d3fc4`), Megatron-LM
`mlm-main915-latest` (base `6a366090`, top **`475167fa4`** = the prefix-skip fix, see
[[swe-main915-mamba-length-defect]]), Megatron-Bridge `mlm-bridge-main915-latest` at `1f8873bb`
(new checkout in the workspace, mounted by the launcher because the container's baked-in
`megatron.bridge` cannot import the newer `megatron.core`). Gym unchanged. All local, nothing pushed.
Smoke 4065835 validated the fix (memory-only regime, exit 0, 57:56); smoke 4065840 = fix +
`prefix_caching_mamba_gb=50` (chain V's regime, run dir `nano35-swe-main915-minf-smoke-16n-mambacache50`,
exit 0, 37:40) was fully clean too (256/256 unmasked, max err 1.052, 0 errors). The recipe
`swe_sc_cmh_minf.yaml` now declares `prefix_caching_mamba_gb: 50` (v2's NeMo RL worker defaulted to
it; main's does not) so chain U runs the same Mamba-cache regime as chain V. Both smoke run dirs
(`.../runs/nano35-swe-main915-minf-smoke-16n{,-mambacache50}`) still hold gym_results (~40 GB each)
and dumps; the buggy-run dumps are preserved as `dumps.4057393-prefix-skip-bug`. Penalties: `b97225bb6` disables the -5 advantage penalties and all `reward_penalties` (see
[[swe-main915-penalties]]); smoke 4068305 (`.../runs/nano35-swe-main915-minf-smoke-16n-nopenalty`)
validated the final recipe (exit 0, 40:32, 0 penalty metrics, resolved==reward, 128/128, max err 1.046).

**2026-09-28 11:27 PDT — chain U SUBMITTED on the regular batch queue** (rkirby: "clear to submit a
full scale run to the regular batch queue not using the reservation"): jobs **4072307 / 4072308 /
4072310**, singleton-chained 4 h segments, partition `batch`, QOS `normal`, 64 nodes (32+32), no
reservation; `EXP_NAME=nano35-swe-main915-64n-minf-20260928`, run dir
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-main915-64n-minf-20260928`
(cold start, dumps on, W&B ultra-v3-swe-e2e-convergence). Stack at submit: nemo_rl `b97225bb6`,
Megatron-LM `475167fa4`, Bridge `1f8873bb`, Gym `d54e6374e`. Manage Ongoing Runs was messaged with
the health signals (masked-seq count ~0, no penalty metrics, 0 stitch/non-contiguous). More
segments = rerun `ENGINE=minf bash launch_swe_main915.sh` the same day (RUN_DATE fixes the name;
pass `RUN_DATE=20260928` on a later day).

**2026-09-29 — post-outage state + a rule gap.** chain U 4072307/4072308 TIMEOUT (seg 2 died ~20:49
09-28 on the data-plane checkpoint save, `TQ_STORAGE ... Resource temporarily unavailable`, the start
of the Lustre degradation; step_15 complete, tmp_step_16 incomplete and auto-removed on next save);
chain W 4077153 TIMEOUT (step_6 complete 20:24, then SC ping failures). Followers 4072310 / 4077154 /
4077156 are PENDING JobHeldUser (monitor re-held everything after the admins' mass release). **NeMo RL
main's single-controller path ignores `checkpointing.load_replay_buffer`** -- the key is consumed only
by async-GRPO (`grpo.py:195-216`); `single_controller_utils/setup.py` restores the native TQ data plane
whenever `replay_buffer_metadata.pt` exists and `single_controller.py` restores the replay metadata
unconditionally. Chain U seg 2 logged "Restoring native TQ checkpoint … step_9/data_plane" and
"restored and validated: groups=30" with no "Skipping replay buffer restore" line, so rkirby's
skip-restore rule ([[feedback-always-skip-replay-buffer-restore]]) is NOT honored on this stack even
though the yaml sets false; the v2 stack had the gate (`_maybe_restore_replay_buffer` +
`_load_pending_rollouts_for_regeneration`). Do not release U/W followers until rkirby decides (port
the gate to main, or accept restore for these arms).

**2026-09-28 14:44 — vLLM counterpart submitted** (rkirby: "launch another run on batch, outside the
reservation, that uses VLLM but has the same setup otherwise"): jobs **4077153 / 4077154 / 4077156**
(batch / normal, 4 h singleton segments, 64 nodes), `EXP_NAME=nano35-swe-main915-64n-vllm-20260928`,
run dir `.../runs/nano35-swe-main915-64n-vllm-20260928`, config `swe_sc_cmh_dump_vllm.yaml`, nemo_rl
`6a6f38b8` (same recipe as chain U minus the engine; proposed label chain W). The vLLM path was never
smoke-tested on this stack, so smoke **4077148** (`nano35-swe-main915-vllm-smoke-16n`, reservation
queue) ran first: **PASSED** 15:30 (exit 0, 40:47, 256/256 unmasked, max err 1.028, 0 errors, no
penalty metrics, resolved==reward) -> chain W stays queued (PENDING Resources on batch at 15:30).
Chain U segment 1 (4072307) RUNNING on batch since ~13:17; **validated at full scale 14:50**: steps
1-2 + partial 3 (1152 seqs) masked 0.4% (pre-fix 15-20%), err>2 5/1152, 150-200-turn bucket 1/177,
spikes 0.00-0.03/1e4 everywhere, 113679/113794 turns hit the Mamba cache, ~33 min/step, reward 0.43.
Both arms: sampler `in_order` (max_lookahead_versions 1), min_groups_for_streaming_train 8,
max_buffered_rollouts 96, max_inflight_prompts 64, GBS 512, 32 prompts x 16, seed 42 -- identical to
chain V's resolved config; only `data.train.split_validation_size/seed` differ (0/0 vs None/None,
main's schema, inert).

**2026-09-28 12:07 — cluster-move handoff prepared (manager relayed rkirby: document + check code
into his forks).** nemo_rl `6a6f38b8` adds `swe_main915_workspace/` (launcher, HANDOFF.md,
analysis/prefix_skip_bug tooling, tools/) and moves the Gym submodule pointer to `d54e6374e`;
HANDOFF.md also at the workspace top. nemo_rl and Megatron-LM were repacked to drop
`.git/objects/info/alternates` -> `workspaces/pipeline_B` (Megatron-LM had no packs of its own);
none of the four clones is shallow. **Pushes are NOT done: my session's permission classifier
blocks `git push` (Data Exfiltration); rkirby must run the four commands in HANDOFF.md section 2**
(targets: ArEsKay3/RL origin, ArEsKay3/Megatron-LM, ArEsKay3/Megatron-Bridge, ArEsKay3/Gym -- all
forks exist; `gh` is not installed on the login node). Do not ask a peer session to push instead.

**2026-09-29 04:35-05:00 — post-outage state + what a resume on main's SC path really does.**
The U/W followers 4072310 / 4077154 / 4077156 were CANCELLED at 04:32 (rkirby, via the monitor
session: "cancel all held jobs that are not follow ups of running jobs"), so no chain U / chain W job
exists; any resume is a fresh `launch_swe_main915.sh` submission (same EXP_NAME, picks
`get_latest_checkpoint_path` = highest step_N) and needs rkirby's word. Resume points: chain U step_15
(19:51 09-28, `tmp_step_16` is a 78 KB stub), chain W step_6 (20:24). No rollout snapshots exist
(`rollout_checkpointing` unset); seg 2 had resumed from a step_9 pre-timeout save
(`checkpoint_must_save_by`) that ft_keep_latest_k=1 later removed.
Decoded with the torch-free reader `/home/rkirby/.claude/jobs/51d77432/tmp/inspect_resume_state.py`
(`replay|ledger|reserve <file.pt>`): U step_15 = 0 canonical groups, 96 in-flight ledger groups
(targets 15/16/null, 0 of 1536 siblings sealed), 32 spares, dispatch_index 16; W step_6 = 24
canonical (target 6, weights v5) + 72 in-flight (0 sealed); W step_5 = 29 canonical; U step_10 = 32;
U step_5 = 32. `token_capture.enabled` is False on this recipe, so siblings never seal, and every
in-flight group is regenerated in full at restart (the persisted `sibling` granularity is moot).
Main's SC resume = restore canonical buffered groups from the TQ data plane (unconditionally; the
`load_replay_buffer` flag is read only by async GRPO) + regenerate in-flight groups from
`rollout_recovery.pt` (prompt idx -> dataset rehydration) + restore 32 spares + live dataloader
cursor. Seg 2's actual restart: "Restored 30 replay group(s)", "Loaded 66 unfinished rollout
group(s)", "Redispatched 66": its step 9 trained 30 early-finishers + 2 fresh groups, the
short-biased mix the rule targets. Consequence: a chain U resume from step_15 is rule-compliant in
effect (0 retained, all 96 regenerated); a chain W resume from step_6 would reuse 24/32 groups of
step 6. Port design if rkirby wants strict empty-buffer resumes on main: in
`single_controller.py::_maybe_restore_replay_buffer`, under `load_replay_buffer=false`, read the
replay metadata, take `meta.tags[i]["prompt_idx"]` / `target_step` / `start_weight` per canonical
group, `clear_samples` those sample ids from the canonical partition, inject equivalent
ADMITTED/GENERATING ledger groups (fresh uuids, all attempts `reserved`) before
`recovery_ledger.load_state_dict`, and let `_rehydrate_rollout_recovery_prompts` +
`_redispatch_restored_rollouts` do the rest; validate with a 16-node save-then-resume smoke (a
step-1 save usually has most target-1 groups complete, so the path is exercised).

**2026-09-29 04:55 — rkirby: "Resume U and W" (as is; no port, W accepts its 24 retained groups).**
Resubmitted with the original recipe (`RUN_DATE=20260928 ENGINE=minf|vllm bash launch_swe_main915.sh`,
one 64-node / 4 h / batch / normal singleton segment per invocation, same run dirs and overrides as
the 09-28 jobs; dry runs matched the old `driver_command.sh` line for line): chain U (MINF) **4090292 ->
4090293 -> 4090295** resuming from step_15; chain W (vLLM twin) **4090297 -> 4090298 -> 4090299**
resuming from step_6. Expected restore lines in `ray_logs/<jobid>-logs/ray-driver.log`: U "Restored 0
replay group(s)" + "Loaded 96 unfinished rollout group(s)" + "Restored 32 pooled spare prompt(s)"; W
"Restored 24 replay group(s)" + "Loaded 72 unfinished" + 32 spares. Health check after the first new
step: dumps for steps 16+ (U) / 7+ (W), masked fraction ~0.4%, no penalty metrics. The Megatron-LM
tree's only dirty entry is the untracked build dir `megatron/core/datasets/helpers_cpp` (harmless).


**Pushed 2026-09-29 08:25 PDT by the run manager on rkirby's order ("Push the U stack"), verified by ls-remote:** ArEsKay3/RL rkirby/swe-main915-latest @ 6a6f38b8; ArEsKay3/Megatron-LM mlm-main915-latest @ 475167fa4; ArEsKay3/Megatron-Bridge mlm-bridge-main915-latest @ 1f8873bb0; ArEsKay3/Gym rkirby/gym-main915 @ d54e6374e. .git dirs re-locked (chmod a-w) afterwards.

**HSG update 2026-10-01 16:5x CDT:** a new peer session, "Latest Main Runner" (forked from the Run Manager), owns the NeMo RL main stack on HSG at rkirby's direction (its account). Workspace /lustre/fsw/portfolios/llmservice/users/rkirby/workspaces/swe_main915 (nemo_rl b99d1f848, Gym d54e6374e, Bridge 1f8873bb0, Megatron-LM 475167fa4), container rl-gym.69725534.sqsh, launcher launch_swe_main915_hsg.sh. First jobs: 16-node smokes 7600771 (vLLM, nano35-swe-main915-vllm-smoke-hsg-16n) and 7600779 (MINF, ...-minf-smoke-hsg-16n), batch/short, 2 steps, checkpointing off, no followers; that session monitors them itself. The HSG launcher passes +checkpointing.load_replay_buffer=false on the command line as well as in the yaml (still inert on this stack). The run manager's audit reads the recipe yaml for main915 names.

---
name: swe-main915-workspace
description: SWE-E2E v2+MINF stack rebuilt on nemo_rl actual main@9-15 (not the swe_915 branch tip); Gym aws_region_name fix; chain U held
metadata: 
  node_type: memory
  type: project
  originSessionId: 51d77432-fbb8-4536-b679-a1714b74af54
  modified: 2026-09-28T18:28:13.476Z
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

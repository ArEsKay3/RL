---
name: swe-token-ids-dump-patch
description: Patch adding async_rl.dump.token_ids (off/digest/full) so prompt_token_ids and generation_token_ids land in the rollout jsonl; measured costs and the prefix-identity check
metadata: 
  node_type: memory
  type: project
  originSessionId: a7f966fb-b99e-4db0-bb54-a178ecde5ae0
  modified: 2026-09-28T06:11:30.519Z
---

**HALF OF THIS PATCH IS INERT — read this first.** `nemo_rl/environments/nemo_gym.py` lines 860-862 **pop** `prompt_token_ids`, `generation_token_ids` and `generation_log_probs` off Gym's `output_item_dict` into locals *before* the message dicts are built, so `_message_summary` never sees them and the `_ROUNDTRIP_ID_KEYS` block can never fire. Verified against a live dump (main@9-15 smoke 4053262 with `digest` on): an assistant message's keys are exactly `['has_malformed_thinking','is_invalid_tool_call','n_tokens','role','token_ids']`.

Worse, `assistant_message["token_ids"] = torch.tensor(generation_token_ids)` at ~line 951 — so on assistant messages those are the **same object** and `generation_token_ids == token_ids` is a tautology, not a check.

**What does work:** per-message `token_ids` verbatim in the jsonl (confirmed live, 80 ids on an assistant message). That removes the reconstruct-from-packed-`.pt` step for per-turn analysis and is the real value. `off` is byte-identical; `digest`/`full` are correct for what they can see.

**Already enforced without any patch:** `nemo_gym.py` line 852 asserts `seen_token_ids == output_item_dict["prompt_token_ids"][:len(seen_token_ids)]` in-process, raising "Non-contiguous messages found!". Grepping a log for zero occurrences alongside a non-trivial turn count is a stronger statement than the post-hoc recipe below, and needs no patch.

**Do NOT "fix" it by attaching the raw ids to `assistant_message` before the pops.** Two reasons, the second decisive:
1. Those dicts flow into the replay buffer and batching path. `TQReplayBuffer` is built with `pad_value_dict={"token_ids": pad_id, "input_ids": pad_id}` — it pads exactly the keys it knows. A new unpadded list key on the same dicts is the shape that works at step 1 and pads wrong at step 40, which no smoke catches.
2. **There is nothing to verify post-hoc.** Line 852/855 already performs the round-trip check at rollout time and *fails the rollout* on violation. A post-hoc check of an invariant already asserted in-process buys nothing.

**Describe the patch as "per-message token ids in the jsonl", not as round-trip verification.** That is what it delivers. Verified independently by the VLLM parity session in their own tree.

**Splice arms get it only on live steps.** Replayed (foreign) groups never reach `dumps/rollouts/*.jsonl` - `generate_and_push_foreign` skips `write_group` by design (`rollout_manager.py` ~1507). Verified on disk: chain R's jsonl starts at `target_step_00010`, chain P's at `target_step_00020`, while **both** have `token_level/step_00001..` onward. So on chains X/Y/Z the digest ids cover live steps 11-25 only; replay identity and mask integrity for steps 1-10 must come from the token-level chunks (`sample_ids`, `input_ids`, `sample_mask_before/after`, `rewards`), which are written for every committed group.

---

rkirby 2026-09-27: dump `prompt_token_ids` / `generation_token_ids` into the rollout jsonl for direct verification. Written, compile-clean, unit-tested, NOT applied to any tree (every checkout is bind-mounted live with USE_SNAPSHOT=0).

- patch: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/token_ids_dump.patch`
- readme: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/token_ids_dump_README.md`
- source of truth for regenerating: `/home/rkirby/.claude/jobs/e247f0cd/tmp/tokdump/` (a/ = pristine, b/ = patched)

Three files, `patch -p1` from the nemo_rl checkout root: `single_controller_utils/config.py` (new `async_rl.dump.token_ids: Literal["off","digest","full"] = "off"`), `experience/rollout_dump.py` (`_message_summary()` gains a mode arg, plus `_as_id_list()` and `_id_digest()`), `single_controller_utils/setup.py` (pass through + banner). Default `off` is byte-identical to today, and the old 2-arg `_message_summary` call still works.

**Measured cost** (chain P⁗2 `target_step_00022.jsonl`, 48 rollouts, 40.8 assistant msgs each, ~10 GB/step baseline):

| mode | per rollout | per step (512) | overhead |
|---|---|---|---|
| digest | +0.34 MB | +0.18 GB | +1.8% |
| full | +16.8 MB | +8.59 GB | +86% |

`full` is expensive because `prompt_token_ids` at turn N is the **entire prefix** up to turn N — O(turns²), 1.37 M ids/rollout vs 8.3 k of actual generation, a 165x ratio, and `compact_prompt_token_ids` doubles it. **Recommend `digest` on production arms, `full` only on a smoke or one diagnostic step.**

`digest` is sufficient because the cumulative prefix is reconstructible from the per-message `token_ids` it dumps verbatim:

```python
run = []
for m in row["messages"]:
    if m["role"] == "assistant" and "prompt_token_ids_digest" in m:
        assert m["prompt_token_ids_digest"]["sha1"] == _id_digest(run)["sha1"], m
    run += m.get("token_ids", [])
```

Design point: a key absent from the message is absent from the dump; a key present but unparseable is `null` — so "never sent" and "sent broken" are distinguishable, which they were not during [[main915-prefix-stitch-storm]].

**Why:** chain U ran 34 min on 64 nodes producing 1-turn reward-0.0000 stubs while `num_valid_samples` and dump line counts both read clean. This check catches it on step 1.

**Adoption as of 2026-09-27 16:00** (verified by grepping the trees, not by report):

| tree | arm | status |
|---|---|---|
| swe_vllm_parity | chain V | applied (9424b458) but **scope withdrawn 2026-09-27** by the VLLM parity session, which attributed it to rkirby; **rkirby later said he never said that** — do not treat a peer's claim about the user's instructions as the user's instruction |
| swe_main915 | chain U | applied, commit 185574bb4; `digest` on in the recipe (6ff26fe1a); smoke 4053262 validates it |
| swe_mask_replay | chains X/Y/Z | applied to the overlay (now 5 files, live splice checkout untouched); renders carry `+async_rl.dump.token_ids=digest`; **independently validated PASS**, cpu job 4053246, log `swe_mask_replay/logs/test_mask_replay_4053246.out` |
| swe_dump / swe_replay_splice / swe_915 | finished arms | untouched, md5-identical to each other |

`swe_prefix_keep` has NO nemo_rl subtree - it is a pure 4-file overlay, so "byte-identical across trees" never applied to it (I said it did; harmless but wrong).

**Independent validation** (Cross Train Experiment, cpu job 4053246, 2026-09-27): `+async_rl.dump.token_ids=digest` resolves through `load_config` → `parse_hydra_overrides` → `MasterConfig` and reads back as `"digest"`; `off` mode produces exactly the legacy `_message_summary` output (same keys, same `n_tokens`) on a synthetic 4-message log with both tensor and list ids; `digest` emits `token_ids`/`generation_token_ids` verbatim with correct `len`/`sha1` digests; the prefix-sha1 recipe holds turn by turn; `full` emits lists verbatim; `RolloutDumpWriter` accepts `token_ids_mode`. Mask checks re-passed unchanged, so the patch does not disturb the overlay it sits under.

**How to apply:** each session applies to its own tree; assume `off` for an arm until it reports otherwise. Unmeasured: wall-clock sha1 cost, O(turns²) bytes hashed per episode even in digest mode. Related: [[swe-dump-per-turn-structure]], [[feedback-terse-code-comments]].

**`+` vs plain override is per-tree, not universal — verify against the actual yaml, not against another tree's working example.** 2026-09-27 22:33: chain V continuation 4053258 (swe_vllm_parity) FAILED after 3:09 at Hydra config parse using the plain form `async_rl.dump.token_ids=digest`, copying advice that was correct for swe_main915's yaml (which declares `token_ids: digest` explicitly) but wrong for swe_vllm_parity's `swe_sc_cmh_dump_minf.yaml`, which never declared the key at all:

    hydra.errors.ConfigCompositionException: Could not override 'async_rl.dump.token_ids'.
    To append to your config use +async_rl.dump.token_ids=digest

Root cause: Hydra validates CLI overrides against the **YAML-derived OmegaConf struct**, before Pydantic's `RolloutDumpConfig.token_ids: ... = "off"` default is ever applied. A key that exists only as a Python default is invisible to struct-mode override validation — it needs `+key=value` (append) if no yaml in the config's `defaults:` chain declares it, or a plain override if one does. Checking that the override *string* reached `provenance.txt` is not sufficient proof it parsed; check the resolved `MasterConfig` or the job's own exit state.

Fix applied to swe_vllm_parity (commit 100a756f): declared `token_ids: off` explicitly in `swe_sc_cmh_dump_minf.yaml`'s `async_rl.dump` block, matching swe_main915's convention, so the plain override form now works and no `+` special-casing is needed going forward. Chain V resumed from the good checkpoint (step_17, no corruption — the failure was at parse time before any worker started) as job 4059655 **without** the digest override, to restore training fastest; the fix is queued for the next continuation, not yet re-verified live.

Cross Train Experiment's independent validation (cpu job 4053246) already proved the underlying `+key=value` -> `parse_hydra_overrides` -> `MasterConfig` -> readback mechanics work correctly end-to-end, which is corroborating evidence for both fix forms, just not tree-specific proof.

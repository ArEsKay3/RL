---
name: swe-mask-replay-xyz-arms
description: "chains X/Y/Z masked-replay arms: chain X FINAL step_30 (stopped 04:38 09-28); chain AA ON HOLD at step_10 since 08:46 09-28 (4061468 held, resumable); AB gated; Y/Z held; job ids, Slurm gating chain, per-step expected mask totals, and the three gating traps"
metadata:
  type: project
---

Launched 2026-09-27 18:53 on rkirby's word ("let X,Y,Z run sequentially"), from `workspaces/swe_mask_replay/launch_swe_mask_replay.sh`. All 64 n, 8 h, hero-res / batch_long / `sla_res_nemotron_sw_post`. Target **step_30** each (10 replayed + 20 live) - **extended from 25 to 30 by rkirby 2026-09-28 00:0x**, who also said X/Y/Z run before any chain AA. No config change needed: `max_num_steps` is 1156, so only the monitor's stop trigger moved. **Timeline (Cross Train's arithmetic, 2026-09-28 00:10, supersedes my earlier ~03:00 estimate):** seg 1 (4055486) walls 02:53:29 with chain X at step 27-29, not 30. A segment resume regenerates every untrained prompt of the in-flight step with the usual long tail — chain Q⁗2's seg 2 took **~60 min from job start to first close** (15:31 → 16:3x) — and splice arms add the foreign-index startup ladder. So on 4055489 the first close lands ~60-70 min after 02:56, then 20-25 min/step: **chain X step_30 ≈ 04:05-04:50**. chain Y then needs ~5 min startup + ~40 min replay + 20 live steps at 20-30 min = 7.5-11 h, walls its seg 1 at step ~26-30 and likely needs its seg 2 with the same ~60 min resume cost: **chain Y done ~12:30-16:30 09-28**, **chain Z ~21:00 09-28 to ~03:00 09-29**. **AA is a 09-29 decision** unless Z is skipped.


**chain X (masked replay, test) FINAL = step_30, stopped 2026-09-28 04:38:09 PDT** by `scancel 4055489` after the file check (step_30: 140 files, 0 tmp, config.yaml/policy/training_info.json, status `last_checkpoint_step: 30`, training_info current_step 30, 369 G). seg 1 4055486 walled 02:53:29 (TIMEOUT) at step 27 (rolling step_27); seg 2 4055489 started 02:56:39, `Skipping replay buffer restore ... regenerating 64`, first close 03:51 (step 28), 29 at 03:55, 30 at 04:35. Live-step closes on seg 2: 28 = 0.2148 / 24,540 / masked 0 / gen_kl 0.00164; 29 = 0.3496 / 23,822 / 0 / 0.00179; 30 = 0.2676 / 26,927 / 0 / 0.00176 (trunc 0). Rungs on disk: step_5 10 15 20 25 30 (rolling 29 retired). No eval or export ordered. The cancel satisfied chain AA 4061467's `afterany:4055489` gate (its 4055486 term had already cleared) - AA is the next arm; Y/Z 4055490-93 stay JobHeldUser until rkirby says release.

**Rule of thumb for any segment-rollover estimate on these arms: add ~60 min for the resume, not ~5.** Logs `swe_mask_replay/logs/launch_chain{X,Y,Z}_seg{1,2}_20260927_185320.out`.

| arm | name | seg 1 | seg 2 |
|---|---|---|---|
| chain X (masked replay, test) | nano35-swe-v2-splice-vllm-runGdata-to10-maskX-rewarded-deep-20260927 | **4055486** (no gate) | 4055489 (singleton + afterany:4055486) |
| chain Y (masked replay, control) | ...-maskY-random-rewarded-20260927 | 4055490 (afterany:4055486 AND :4055489) | 4055491 (singleton + afterany:4055490) |
| chain Z (masked replay, mirror) | ...runMdata-to10-maskZ-punished-deep-20260927 | 4055492 (afterany:4055490 AND :4055491) | 4055493 (singleton + afterany:4055492) |

Strictly sequential by Slurm dependency, not by anyone watching.

**REORDERED 2026-09-28 01:0x by rkirby: "Let's let AA and AB slip ahead of Y and Z."** New order **X -> AA -> AB -> Y -> Z**; AC unlaunched. I held all four Y/Z ids (4055490/4055491/4055492/4055493 -> `JobHeldUser`) so the step_30 stop on chain X does not release chain Y. Held, NOT cancelled - cancelling Y would satisfy Z's afterany gate. Their gates survive under the hold; releasing them after AB restores Y -> Z. Release needs rkirby's word. Submitted 2026-09-28 01:00 (launch logs `swe_range_replay/logs/launch_chain{AA,AB}_seg{1,2}_20260928_010044.out`):

| arm | name | seg 1 | seg 2 |
|---|---|---|---|
| chain AA (G steps 8-10 replayed, live 1-7 and 11-25) | nano35-swe-v2-splice-vllm-runGdata-8to10-20260928 | **4061467** (afterany:4055486 AND :4055489) | 4061468 (singleton + afterany:4061467) |
| chain AB (G steps 1-7 replayed, live 8-25) | nano35-swe-v2-splice-vllm-runGdata-1to7-20260928 | 4061469 (afterany:4061467 AND :4061468) | 4061470 (singleton + afterany:4061469) |

Recipe: plain `launch_swe_splice.sh`, `+foreign_rollout.steps=8:10|1:7` from chain G, `+checkpointing.load_replay_buffer=false`, `overlap_param_gather=false`, no mask, no digest. Target step_25 (spec) unless rkirby moves it; my `*-8to10-*|*-1to7-*` trigger is at 25.


**chain AA (step-range replay: chain G steps 8-10 only) ON HOLD since 08:46 PDT 2026-09-28 (rkirby: "OK let's put AA on hold").** Sequence used (gating trap #1): `scontrol hold 4061468` first (seg 2, now JobHeldUser), then `scancel 4061467` (seg 1, CANCELLED 08:46:21 after 4:05:39, mid live step 11). State on disk: rungs step_5 and step_10 (step_10 verified 140 files), 10 steps trained = live 1-7 + replayed 8-10 (replayed steps 8/9/10 matched chain R's replayed 8/9/10 exactly: 0.2852/32,926, 0.3145/25,987, 0.3457/25,462); live closes 1 0.3691, 2 0.5293, 3 0.2461, 4 0.4336, 5 0.2441, 6 0.3770, 7 0.4199; step 11's in-flight rollouts lost. Resume = `scontrol release 4061468` (singleton + afterany:4061467 already satisfied; it restarts from step_10 and regenerates step 11, ~60 min to first close; NO +checkpointing.load_replay_buffer=false was baked into the AA render - it was submitted with load_replay_buffer=false already, see the README). Consequence for chain AB: 4061469 is gated on afterany:4061467 (now satisfied) AND afterany:4061468 (held, unfulfilled) - AB stays blocked while 4061468 sits held; to run AB before AA resumes: hold 4061469+4061470 first, then cancel 4061468 (which also drops AA's resume slot - re-submit AA later with the launcher). Y/Z 4055490-93 unchanged (JobHeldUser).

**First-tick shape differs between them:** chain AA runs LIVE steps 1-7 first (first jsonl = `target_step_00000`, first live step ~60 min incl. startup), replays 8-10 in minutes (token_level `step_00008..10`, no jsonl for target steps 7-9), then live 11-25. chain AB replays 1-7 first (no jsonl until `target_step_00007`), then live from 8. Both print `foreign_rollout enabled ... steps=8:10|1:7`, no `mask_file` line, zero `foreign_rollout: masked` lines - a masked line on either would be wrong.

**Expected `foreign_rollout` per-target-step mask totals** (flag immediately on divergence — the replayed steps are the cheapest place to catch a knob masking the wrong rows):
- chain X: 4, 10, 9, 23, 12, 6, 23, 18, 11, 11 (127 rewarded-deep rows, chain G data)
- chain Y: 8, 18, 8, 19, 6, 23, 15, 8, 11, 11 (127 random rewarded rows, chain G data)
- chain Z: 11, 9, 3, 28, 25, 31, 28, 48, 14, 16 (213 punished-deep rows, chain M data)

**Three gating traps:**
1. Hang-cancelling a running seg 1 makes that arm's seg 2 eligible **at once** — it becomes the relaunch. No manual resubmit needed.
2. A third segment for any arm requires `scontrol hold` on the NEXT arm's seg-1 gate job first, or that gate fires when seg 2 ends and two arms overlap.
3. **Never cancel a pending seg-1 gate job alone** — cancelling 4055490 by itself frees 4055491 to start immediately. Stopping an arm means cancelling BOTH its ids, which the `*mask[XYZ]*` **step_30** trigger does because it resolves by name - and that cancel is also the handoff, since the next arm's seg-1 gate is `afterany` on both ids.

Startup is expected to show the splice `SingleController ping failed` ladder while `ForeignRolloutSource` indexes chain G's ten steps (see [[splice-foreign-index-startup-cost]]), then the `foreign_rollout: mask_file=... sample_ids=127 expected per target_step={...}` init line and the `Rollout dump enabled ... token_ids=digest` banner.

**No `dumps/rollouts/*.jsonl` during steps 1-10** — replayed groups skip the jsonl writer; `token_level/step_00001..` is written. First jsonl is `target_step_00010`. See [[swe-token-ids-dump-patch]].

Related: [[swe-splice-experiments]], [[feedback-arm-labels]].

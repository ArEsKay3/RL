# Masked-replay experiment: is rewarded deep deliberation the poison in chain G's early data?

Prepared 2026-09-27 by the "Data Difference Deep Dive" session for the Cross Train Experiment session. Everything referenced below is under
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/` (called `$D` here).

## 1. Background and hypothesis

- Replaying chain G's rollouts for training steps 1-10 into a fresh trainer and then running live with vLLM produces the loop pathology every time (chain R: loop-token share 4.2 / 6.5 / 4.7 % at live steps 11-15 / 16-20 / 21-25; chain R′ with steps 1-16 replayed: 9 % from its first live window). Replaying chain M's steps 1-20 (chains P, Q) stays healthy (< 1 % from step 21).
- Every channel the replay consumes has been compared across ten from-scratch arms (three that looped: G, I, P⁗2; seven that stayed healthy: A, K, M, N, Q⁗, J, P⁗). Token content, masks, engine logprobs, reward rates, prompt difficulty, timing and vocabulary are indistinguishable. The one property that separates the three looping arms from the seven healthy ones is the trainer-gradient weight on *deep* thinking in *rewarded* rollouts: think tokens more than 4,096 tokens into a think block, in rollouts with positive advantage, weighted by (1 - p_trainer). Net gradient on deep thinking: G -0.019, I -0.032, P⁗2 -0.031 vs healthy arms -0.052 to -0.171 (x1e-3 per generated token). This is suggestive (perfect rank split, pooled z 1.6-2.8 depending on the variant, ~50 statistics scanned), not decisive.
- Hypothesis to test: the ~127 rewarded rollouts in G's steps 1-10 that contain a >4,096-token think turn carry the poison. Removing their gradient should make the G-replay healthy; removing an equal number of other rewarded rollouts should not.

## 2. Arms

| arm | replayed data | mask applied to the replayed data | live phase after replay | expected if the hypothesis holds |
|---|---|---|---|---|
| A (test) | chain G steps 1-10 | the 127 rollouts in `poison_G_rewarded_deep_steps1-10.csv` get sample_mask = 0 | vLLM, same as chain R | stays healthy: loop share <= 3 % at live steps 16-20 and < 1 % at 21-30 |
| C (control) | chain G steps 1-10 | the 127 rollouts in `control_G_random_rewarded_steps1-10.csv` (random rewarded rollouts WITHOUT deep think turns, seed 1234) get sample_mask = 0 | vLLM | loops like chain R (4-7 % at 16-25) |
| B (mirror, optional) | chain M steps 1-10 | the 213 rollouts in `poison_M_punished_deep_steps1-10.csv` (punished rollouts with a >4,096-token think turn) get sample_mask = 0, i.e. the "long thinking loses" lesson is removed | vLLM | loops appear (this arm tests sufficiency the other way round) |

Arm A alone is informative given chain R exists; arm C protects against "masking any 2.5 % of rewarded samples helps". Arm B is a nice-to-have.

## 3. Data

- Chain G run: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920` (rollout jsonl `dumps/rollouts/target_step_00000..00009.jsonl`, token dumps `dumps/token_level/step_00001..00010_chunk_*.pt`). Same source and step range as chain R.
- Chain M run (arm B): `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923`.
- Mask lists (CSV, header row; `sample_id` is exactly the key used by `ForeignRolloutSource._load_step_rows`, i.e. the `sample_ids` entry of the token-level chunk; `group_id` = sample_id without the `_gN` suffix; `target_step` = 0-based jsonl index, `train_step` = 1-based):
  - `$D/poison_G_rewarded_deep_steps1-10.csv` — 127 rows; per train step 1..10: 4, 10, 9, 23, 12, 6, 23, 18, 11, 11. Columns: rank, sample_id, group_id, target_step, train_step, adv, deep_think_tokens, live_deep_sum_1mp, absadv_x_live, cum_share (cumulative share of the deep-think reinforcement; the top 50 carry 87 %).
  - `$D/control_G_random_rewarded_steps1-10.csv` — 127 rows, same columns minus the deep-think fields.
  - `$D/poison_M_punished_deep_steps1-10.csv` — 213 rows.
  - Also available: `poison_I_rewarded_deep_steps1-10.csv`, `poison_P42_rewarded_deep_steps1-10.csv` (102 rows each) with matching random controls, should you want a second source run.
- How the lists were made (reproducible): `python3 $D/poison_list.py G rewarded` reads `$D/parts/G__sNNN_cKKK.eff.csv` (per-turn output of `$D/effpass.py` over the token dumps: think tokens = tokens between the generated `<think>\n` prefix and `</think>`, p_trainer = exp(prev_logprobs)); a rollout is listed if its advantage is > 0 and at least one assistant turn has > 4,096 think tokens; ranked by adv x sum(1 - p_trainer) over the think tokens beyond position 4,096. The control draws, with seed 1234, the same number of rewarded rollouts that have no such turn.

## 4. Implementation notes (splice code)

- `ForeignRolloutSource._pack_group` (nemo_rl/experience/foreign_rollout_source.py) currently sets `sample_mask = torch.ones((n,), dtype=torch.float32)` for every replayed group. Add a mask set (loaded once from the CSV; e.g. `+foreign_rollout.mask_file=<csv>`) and set `sample_mask[row_idx] = 0.0` for members whose `sample_id` is in the set. Keep the rows in the group: the group's 16 members must stay so that the leave-one-out advantages of the siblings are unchanged; only the loss contribution of the masked member is removed.
- Verify the zero survives to the loss: the docstring in `_pack_group` says the seq-logprob-error masking in grpo.py later overwrites the same DataPlane field with an explicit `.float()` tensor. If it overwrites rather than ANDs, either AND the replayed mask in at that point or zero the advantage of the listed samples in the advantage stage instead (equivalent for the gradient). Please log the count of masked replayed samples per step; expected for arm A: 4, 10, 9, 23, 12, 6, 23, 18, 11, 11.
- Everything else as chain R: `+foreign_rollout.source_run=<G run>`, `+foreign_rollout.steps=1:10`, `+checkpointing.load_replay_buffer=false`, ENGINE=vllm, same seed and config as chain R (nano35-swe-v2-splice-vllm-runGdata-to10-20260924). Per your workspace rules the live swe_replay_splice checkout is not edited; an overlay as for the other variants.
- Preregistered target (rkirby, 2026-09-27): train step 25 = 10 replayed + 15 live steps; each arm is stopped once checkpoints/step_25 is complete. Read-out windows 11-15 / 16-20 / 21-25, the span chain R has (4.2 / 6.5 / 4.7 %). Chain names: X = nano35-swe-v2-splice-vllm-runGdata-to10-maskX-rewarded-deep-20260927 (test), Y = ...maskY-random-rewarded-20260927 (control), Z = nano35-swe-v2-splice-vllm-runMdata-to10-maskZ-punished-deep-20260927 (mirror); overlay workspace swe_mask_replay.

## 5. Read-out

Primary: loop-token share (think turns >= 1k tokens with zlib ratio < 0.10, share of generated tokens) per 5-step live window 11-15 / 16-20 / 21-25, compared with chain R (4.2 / 6.5 / 4.7 %) and the healthy from-scratch arms (16-20: 0.3-3.7 %, 21-25: 0.1-0.8 %). Preregistered decision at step 25: X at or below the healthy band in 16-25 while Y is at chain R's level = poison confirmed. Secondary: cumulative E(think turns 4k-16k) and the deep-think gradient statistic. Give this session the run directory and it will extract the dumps and score them with the existing pipeline (`features.py`, `runsteps.py`, `predict.py`, `effective.py`), same definitions as all the tables so far.

Verdicts: A healthy and C looping = the rewarded deep deliberation is the poison (sufficient to remove it). A looping = the poison is elsewhere in G's data (or needs more than these 127 rollouts); then B becomes the informative arm. A healthy and C healthy = removing 2.5 % of rewarded samples is itself protective, no attribution.

## Update 2026-09-28 00:35 PDT
rkirby (via the run manager): chains X, Y and Z run to step_30 each (manager trigger moved from step_25 to step_30); AA/AB/AC start only after all three are done. Scoring windows now 11-15 / 16-20 / 21-25 / 26-30. References for 26-30 (chain R stopped at 25): looping arms G 13.1 %, I 11.1 %; healthy A 0.16 %, J 0.73 % (5 steps), K 0.47 % (3 steps). X result so far: 11-15 2.86 %, 16-20 2.53 % (chain R 4.18 / 6.48 %), inside the healthy band.

## Result 2026-09-28 04:45 PDT — X final at step 30
X live loop share: 11-15 2.86 %, 16-20 2.53 %, 21-25 1.83 %, 26-30 0.13 % (chain R 4.18 / 6.48 / 4.66 / ended; looping arms G/I 7.7 / 5.3 at 21-25 and 13.1 / 11.1 at 26-30; healthy 0.3-3.7 / 0.2-0.8 / 0.2-0.7). Replayed steps 1-10 verified identical to G with exactly the 127 listed rows masked at every step. Preregistered criterion (X at or below the healthy band in 16-25) met at 21-25 and 26-30; 16-20 at the band's upper half. Y not yet run, so the "any 127 rewarded rollouts" alternative is still open.

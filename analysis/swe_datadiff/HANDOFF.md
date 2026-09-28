# HANDOFF — SWE data-difference deep dive (chain G vs chain M replay data)

Written 2026-09-28 12:10 PDT for resumption on another cluster. Owner: rkirby. Author: the "Data Difference Deep Dive" Claude session.
Workspace on CMH: /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff
Git copy: git@github.com:ArEsKay3/RL.git branch rkirby/swe-analysis-datadiff, directory analysis/swe_datadiff/ (everything except parts/ and browser/*.csv, which are regenerated from the dumps).

## 1. The question
Replaying chain G's first ten training steps (MINF from scratch, no prefix cache) into a fresh vLLM trainer makes the model loop every time (chain R); replaying chain M's (vLLM from scratch) does not (chain Q). The replay consumes the token-level .pt dumps (input_ids, token_mask, generation_logprobs, rewards); the jsonl is only the readable record. What in G's data does this, and is it fixable?

## 2. Definitions used everywhere
- Turn = one assistant generation; think block = generated tokens up to `</think>`; deep think tokens = think tokens beyond position 4096 of the block.
- Loop turn = think >= 1000 tokens with zlib(6) ratio of the reasoning bytes < 0.10; near-repetitive = 0.10-0.25; normal long = >= 0.25; runaway = unclosed think block at truncation. Loop share = loop-turn tokens / generated tokens of a step.
- Trainer advantage = leave-one-out within the 16-rollout group: (r_i - mean(others)) / std_ddof1(others), unnormalised if the std is 0.
- E(class) = 1000 x sum over turns in the class of adv x turn tokens / all generated tokens (equal token weight).
- R_deep = 1000 x sum over deep think tokens of adv x (1 - p_trainer) / all tokens; R_deep+ / R_deep- = rewarded (adv > 0) / punished halves; S+ / S- = the same without the advantage (live deep tokens per 1k). Loop tokens have p_trainer ~ 0.997 and are gradient-inert.
- Masking metric for the X/Z lists: absadv_x_live = |adv| x sum over a rollout's deep think tokens of (1 - p_trainer).

## 3. Arms (run directories under /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/)
Calibration set for steps 1-10, "bad" = loops later, "good" = healthy to >= step 30:
- G bad  Minf from scratch, no prefix cache: nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920
- I bad  Minf, prefix cache, replica 1: ...-minf_dump-from0-prefix-nopg-r1-20260921
- P⁗2 bad Minf, prefix cache kept across refits, seed 1234: nano35-swe-v2-from0-nvshmem-keepprefix-minf-seed1234-20260926 (on hold at step 22)
- A good vLLM from scratch: ...-vllm_dump-20260918;  K good vLLM seed 1234: ...-vllm_dump-from0-seed1234-20260922
- M good vLLM no-opg r1: ...-vllm_dump-from0-nopg-r1-20260923;  N good vLLM no-opg r2: ...-vllm_dump-from0-nopg-r2-20260923
- J good Minf prefix cache r2: ...-minf_dump-from0-prefix-nopg-r2-20260921;  P⁗ good Minf keep-prefix: nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925
- Q⁗ good vLLM no prefix caching: nano35-swe-v2-from0-noprefix-vllm-20260926;  Q⁗2 (held at 20, healthy so far) ...-noprefix-vllm-seed1234-20260927;  V (Minf parity, running) nano35-swe-v2-from0-parity-minf-20260927
Replay arms (vLLM trainer, splice of foreign token-level dumps):
- R  vLLM <- G steps 1-10, unmasked: nano35-swe-v2-splice-vllm-runGdata-to10-20260924 (loops: 4.2 / 6.5 / 4.7 % at 11-15 / 16-20 / 21-25)
- X  vLLM <- G 1-10 with 127 rewarded deep-thinking rollouts sample-masked: nano35-swe-v2-splice-vllm-runGdata-to10-maskX-rewarded-deep-20260927 (FINAL step 30: 2.86 / 2.53 / 1.83 / 0.13 %)
- Y  control: G 1-10 with 127 random rewarded rollouts without deep turns masked: ...-maskY-random-rewarded-20260927 (HELD, jobs 4055490/4055491)
- Z  mirror: M 1-10 with 213 punished deep-thinking rollouts masked: nano35-swe-v2-splice-vllm-runMdata-to10-maskZ-punished-deep-20260927 (HELD, 4055492/4055493)
- AA ablation: G steps 8-10 only replayed, live 1-7 and 11-25: nano35-swe-v2-splice-vllm-runGdata-8to10-20260928 (seg 1 4061467 cancelled 08:46 09-28 after the replayed steps, seg 2 4061468 held; verified replay, no post-replay live step yet)
- AB ablation: G steps 1-7 only: nano35-swe-v2-splice-vllm-runGdata-1to7-20260928 (4061469/4061470 gated behind AA, not started); AC (M 8-10) never launched.
Overlays that ran them (owned by the Cross Train Experiment session): workspaces/swe_mask_replay (adds +foreign_rollout.mask_file; sample_mask = 0 for listed replayed rows, rows kept, token_ids=digest dump), workspaces/swe_range_replay (plain splice launcher, +foreign_rollout.steps=a:b). Splice mechanics: ForeignRolloutSource indexes (target_step, prompt_idx) -> group, misses fall through to live generation, replayed rows never appear in the rollout jsonl, sample/group ids are re-minted (verify by content, not by id).

## 4. Results (all steps 1-10 unless stated; numbers per 1k generated tokens)
Ruled out across 10 arms: engine-logprob fidelity by depth, engine-specific tokens, text statistics, reward-side statistics (rewarded loops / runaways / truncations), refit timing, per-update "cliff", context-limit artefacts, prefix-cache retention (P⁗ healthy with retained cache; Q⁗ healthy without), frozen-weight engine swap (P′/Q′ null).
The one separator (3 bad vs 7 good, chance 1/120): the trainer-gradient weight on deep thinking, driven by the rewarded side.
| arm | R_deep | R+ | R- | R-/R+ | S+ |
| G -0.019 0.227 -0.246 1.09 0.202 | I -0.032 0.235 -0.266 1.14 0.184 | P⁗2 -0.031 0.241 -0.272 1.13 0.172 |
| K -0.052 | P⁗ -0.099 | Q⁗ -0.100 | M -0.129 | J -0.129 | Q⁗2 -0.141 | N -0.155 | A -0.171 (ratios 1.29-2.12, S+ 0.122-0.156) |
Decomposition: per-token (1-p) within a class is identical across arms (near-repetitive deep tokens 0.09-0.11 everywhere, loops ~0.002); the mean advantage of rewarded deep rollouts is identical; what differs is the VOLUME of rewarded non-loop deep thinking (1.99 / 1.89 / 1.59 per 1k vs healthy 1.17-1.50). ~87 % of rewarded deep tokens are near-repetitive (zlib 0.10-0.25) in every arm. Unweighted: mean advantage per 4k-16k think turn is +0.003 / -0.024 / -0.015 in the bad arms vs -0.038..-0.087 in the good ones, with the same number of such turns (~900/arm); no other length bucket, no rollout-length bin and no count of low-zlib trajectories separates (total count of zlib<0.25 rollouts: no; rewarded zlib<0.25 rollouts: yes but 157 vs 152). Token-weighted total E separates by 1.6 units (thin); the plain adv-length correlation does not.
Group structure: the surplus sits in "poison groups" where punished deep-thinking siblings cancel only 80-91 % of the reinforcement; diffuse (~67 groups with a rewarded deep rollout in G vs 41 in A; top-5 share 14-30 %). Emerges over steps 4-10; no separation at k <= 7. Early-warning calibration (predict.py): cumulative E(4k-16k) or R_deep over steps 1..k separates at k = 7-10; nothing separates after k = 15; at 21-30 the late marker is the deep-token share itself (G 7.7 %, I 3.8 % vs healthy 0.9-2.3 %).
Causal test: chain X (127 rows = 1.2 % of rows, 5.4 % of tokens, sample_mask 0, everything else bit-identical to chain R) does NOT loop: 2.86 / 2.53 / 1.83 / 0.13 % vs chain R 4.18 / 6.48 / 4.66 / -, healthy band 0.3-3.7 / 0.2-0.8 / 0.2-0.7 %, looping arms G/I 13.1 / 11.1 % at 26-30. Trained-row statistics of X: R_deep -0.246 = R+ 0.000 + R- -0.246, S+ 0.000. Open: Y (would any 127 rewarded rollouts do?), AA/AB (which steps carry it), Z (does the mirror hurt M?).
Predictions logged before outcomes: Q⁗ healthy (correct), Q⁗2 healthy (healthy to 20), V healthy (running), P⁗2 at risk from R_deep (looped at 21-22; the equal-weight E missed it).

## 5. How to re-run elsewhere (inputs = run dirs above; each step has dumps/token_level/step_NNNNN_chunk_KKK.pt and dumps/rollouts/target_step_NNNNN.jsonl, train step = target_step + 1)
All scripts are stdlib + numpy/pandas; CPU only. Paths at the top of each script point at the CMH workspace: change D (analysis dir) and R (runs root).
1. features.py <arm> <run_name> <chunk.pt> <outdir>  -> parts/<arm>__sNNN_cKKK.turns.csv + .samples.csv (per-turn tokens, think tokens, zlib, closes, engine/trainer logprob stats, adv, reward; per-rollout loop stats, masks). Batch: a jobs file of "<arm> <run> <chunk> <outdir>" lines + features.sbatch (xargs -P 44, 48-CPU node, ~2 min per 50 chunks).
2. effpass.py <arm> <run_name> <chunk.pt> <outdir> -> parts/*.eff.csv (per turn: cls, s1mp_* = sums of (1-p_trainer) over think / entry / deep / answer tokens). Batch: eff.sbatch.
3. Tables: deep.py [lo hi] (R_deep with group bootstrap), deep2.py (S+, concentration, prompt-paired), deep_robust.py (exclusions), groups.py / groupmask.py (group-level), effective.py (depth curves), tables.py -> tables.md (treatment tables), runsteps.py -> runsteps.md (per-step E), predict.py -> predict_report.txt (early warning at k), xlive.py <arms> <lo> <hi> (rollout-level per-step comparison), deeplive.py <arms> <lo> <hi> (gradient metrics per step).
4. Mask lists: poison_list.py <arm> rewarded|punished -> poison_<arm>_<sign>_deep_steps1-10.csv (+ control_<arm>_random_rewarded_steps1-10.csv). Consumed lists: poison_G_rewarded_deep_steps1-10.csv (X), control_G_random_rewarded_steps1-10.csv (Y), poison_M_punished_deep_steps1-10.csv (Z); nomask.csv for the range ablations.
5. Replay verification: verify_replay.py <replay_run> <source_run> <mask.csv> [lo hi] — content join on sha1(input_ids) per step; reports rows matched, content / mask / reward identical, listed rows found and masked (matched by count per content hash), extra sample_mask_after zeros (seq-logprob-error filter).
6. Scoring loop for the replay arms: score_xyz.sh (resolves run dirs by glob, extracts new live steps, verifies replayed ranges, prints per-step loop share and windows vs chain R / G / I / healthy band).
7. Browser export: browser_export.py -> browser/rollout_metrics_<arm>.csv, long_turns_<arm>.csv (per-rollout and per-turn masking metrics, group net / cancel ratio, mask ranks); column guide in browser/BROWSER_HANDOFF.md; consumed by swe_dump/tools/dumpbrowse (Data Browser session).
Records: LOG.md (chronological, every result with timestamps), SUMMARY.md (consolidated findings), EXPERIMENTS.md (designs 1-6b), EXPERIMENT_masked_replay.md (X/Y/Z spec + result), tables.md, runsteps.md, predict_report.txt.

## 6. Open items
- Y control (held): decides whether the X result is specific to the rewarded-deep selection. Z mirror (held). AA (stopped after its replayed steps; resume = release 4061468, restarts from step_10 rung) and AB (gated) locate the effect within steps 1-10.
- Candidate fixes (not tested): repetition-aware shaping on rewarded near-repetitive deep thinking; std floor / LOO normalisation change; data filter by the masking metric at train time (the X list is 1.2 % of rows).
- The separating statistics come from 10 arms and ~50 scanned statistics; the perfect 3-vs-7 splits are suggestive, X is the causal evidence.

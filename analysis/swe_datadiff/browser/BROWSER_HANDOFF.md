# Deep-thinking gradient metrics for the dump browser

Written 2026-09-27 23:58 PDT by the Data Difference Deep Dive session for the Data Browser session.
Purpose: let rkirby sort rollouts (and turns) in dumpbrowse by the metric that decided which
rollouts chains X and Z mask, and by the signed per-rollout contribution to R_deep.

## Files (regenerate with `python3 /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/browser_export.py`)

Per rollout, one row per trained rollout (512 per train step), one file per arm plus a concatenation:

    /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/browser/rollout_metrics_<ARM>.csv
    /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/browser/rollout_metrics_all_arms.csv

Per long turn (assistant turns whose reasoning is >= 1000 tokens), one file per arm plus a concatenation:

    /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/browser/long_turns_<ARM>.csv
    /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/browser/long_turns_all_arms.csv

ARM in {A, G, I, J, K, M, N, P4, P42, Q4, Q42, R, V, X}. The `run` column is the run directory name under
/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/ (the browser's run list), e.g.
G = nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920,
M = nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923.

The three mask lists the X/Y/Z runs actually consumed:

    /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/poison_G_rewarded_deep_steps1-10.csv   (X, 127 rows)
    /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/control_G_random_rewarded_steps1-10.csv (Y, 127 rows)
    /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/poison_M_punished_deep_steps1-10.csv    (Z, 213 rows)

## Join keys

* `sample_id` is the jsonl top-level `sample_id` (same string, e.g. `e8348aab-...-6110ead85bd1_g13`); `group_id` = the part before `_g`.
* `train_step` = the token-level dump step; `target_step` = train_step - 1 = the jsonl file index
  (`dumps/rollouts/target_step_%05d.jsonl`). Join on (run, target_step, sample_id) or simply on sample_id (uuids, unique).
* Restart duplicates (G target_step 6 and 7 have two attempts in the jsonl): only the attempt whose sample_ids appear in
  these files was trained; jsonl rows with no match are the untrained attempt.
* `turn` in the long-turn file is the 0-based index of the assistant generation within the rollout (the k-th reasoning +
  tool-call/message pair in `full_result.response.output`, counting only assistant generations).

## The masking metric

For every assistant turn, `deep_tokens` = max(0, reasoning tokens - 4096), i.e. the thinking beyond position 4096 of the
turn's think block. `live_deep_1mp` = sum over those deep tokens of (1 - p_trainer), where p_trainer is the trainer's
probability of the sampled token (tokens the trainer already predicts with p ~ 1 contribute ~0; loops sit at p ~ 0.995 and are
nearly inert). It is the size of the gradient the turn's deep thinking receives, before the advantage.

* `absadv_x_live` = |adv| x live_deep_1mp (rollout level = sum over its turns). THIS is the ranking key of the mask lists.
* `mask_metric_X` = absadv_x_live for rewarded rollouts (adv > 0) with at least one deep turn, NaN otherwise. Sorting a G
  step range 1-10 by it descending reproduces the X list exactly (`rank_X_metric_steps1_10` == `in_X_mask_rank` for all 127).
* `mask_metric_Z` = absadv_x_live for punished rollouts (adv < 0) with at least one deep turn. Sorting M steps 1-10 by it
  reproduces the Z list (`rank_Z_metric_steps1_10` == `in_Z_mask_rank` for all 213).
* `r_deep_contrib` = adv x live_deep_1mp, signed: the rollout's contribution to R_deep. Positive = reinforces deep thinking,
  negative = suppresses it. `r_deep_contrib_per1k_step` = 1000 x r_deep_contrib / (all generated tokens of that arm-step);
  summing it over a step gives that step's R_deep exactly (G steps 1-10 pooled: -0.019 = R+ 0.227 + R- -0.246, matching the
  tables in LOG.md / SUMMARY.md).
* `in_X_mask_rank`, `in_Y_mask_rank`, `in_Z_mask_rank`: rank in the consumed mask list (NaN = not masked). Only G rows carry
  X/Y ranks, only M rows carry Z ranks. Y is a random control: rewarded rollouts WITHOUT a deep turn, so its rows have
  mask_metric_X = NaN by construction.
* `rank_*_all_steps` are the same rankings over every dumped step (for browsing late steps, e.g. G 12-19 hold the largest
  rewarded deep rollouts of the run: f62d25f9..._g2 at step 13, b15ebc2c..._g11 at step 12).

## Other columns

Rollout file: `reward` (0/1), `adv` (trainer leave-one-out advantage), `n_gen` generated tokens, `n_turns`, `max_turn`
(largest turn in tokens), `think_tok_total`, `max_think_turn`, `n_long_turns` (>= 1000 think tokens), `n_deep_turns`
(> 4096), `deep_think_tokens`, `mean_1mp_deep` (= live_deep_1mp / deep tokens; ~0.005 means a loop, 0.10-0.15 means live
reasoning), group aggregates over the 16 rollouts of the same prompt at that step: `group_net_contrib`, `group_pos_contrib`,
`group_neg_contrib`, `group_cancel_ratio` (= |neg| / pos; < 1 means the punished siblings do not cancel the reinforcement,
the "poison group" pattern), `group_n_deep_rollouts`, `group_rank_net_steps1_10`, `group_mean_reward`; loop stats
`loop_turns`, `loop_tok`, `loop_share_pct` (think >= 1000 tokens with zlib ratio < 0.10), `near_loop_turns`, `near_loop_tok`
(zlib 0.10-0.25), `truncated`, `ends_seq_noclose` (runaway: last turn never closed its think block), `mask_before`,
`mask_after` (trainer sample mask before/after the seq-logprob-error filter), `seq_mult_prob_error`.

Turn file: `n_tok` (turn generated tokens), `think_tok`, `deep_tokens`, `live_deep_1mp`, `mean_1mp_deep`, `turn_contrib`
(= adv x live_deep_1mp for this turn), `zlib` (zlib-6 ratio of the reasoning bytes), `cls` in {normal_long, near, loop}
(short turns are excluded from this file), `s1mp_think_tr` (sum of 1-p over the whole think block), `s1mp_entry_tr`
(first 256 think tokens), `close_p_tr` (trainer probability of the `</think>` token), `mean_ptr_think`.

## What rkirby wants to see

1. In the rollout table of a G or M step: sort by `mask_metric_X` / `mask_metric_Z` / `r_deep_contrib`, with the
   `in_X_mask_rank` / `in_Z_mask_rank` badge, so the top-ranked rollouts can be opened and read.
2. Inside a rollout: highlight the turn(s) with deep_tokens > 0 and show turn_contrib / mean_1mp_deep next to the existing
   zlib column, so the reasoning that carries the gradient can be found without scrolling 80 turns.
3. The group view: the 16 siblings with their signed contributions and the group net / cancel ratio.

Both metrics are arm-local (live_deep_1mp uses that arm's trainer logprobs), so compare rollouts within an arm; across arms
compare ranks or per-1k values.

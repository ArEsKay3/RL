# Data difference deep dive — summary as of 2026-09-25 13:00 PDT

Question: chain G's (MINF from scratch, no prefix cache) rollout data for training steps 1-10 poisons a fresh trainer (replay chains R, R′ go into loops) while chain M's (vLLM from scratch, replica 1) does not. What differs in the data?

## What the replay consumes
`ForeignRolloutSource` uses the jsonl only as an index; the tensors come from `dumps/token_level/step_*.pt`: input_ids, token_mask, generation_logprobs, rewards. sample_mask is reset to 1; prev_logprobs and advantages are recomputed. Verified byte-exact agreement between token dumps and jsonl text, per-turn token counts, totals and rewards in both arms.

## Ruled out (identical in G and M, and across all seven from-scratch arms)
- Engine-logprob channel: trainer-minus-engine mismatch (mean, tails, TIS clipping ~1e-5), at `</think>`, `<|im_end|>`, first tokens, by context position and turn length; seq_mult_prob_error; masked samples.
- Content marginals: lengths, turns, truncation, runaway, loop share, near-repetitive share (steps 1-3 identical, +0.5 pt in MINF arms from step 4), zlib/dist8/sentence structure of long thinks, vocabulary frequencies, tool-output context, within-rollout regime jumps, aftermath of long thinks (next-turn length, later loops, runaway, success).
- Reward bookkeeping: reward == resolved always; error kinds, timeouts, OOM comparable; prompt difficulty (group success counts) identical.
- Timing/refit spanning: whole-step property, never within a group.
- Restart duplicates in G's jsonl: only the post-resume attempt is in the token dumps.

## What differs
Token-weighted advantage exposure E(class) = 1000 x sum(adv x turn tokens in class) / all generated tokens, steps 1-10:

| class | G (bad) | I (bad) | J | A | K | M | N |
|---|---|---|---|---|---|---|---|
| think turns 4k-16k tokens | -0.09 | -0.82 | -3.44 | -4.20 | -2.33 | -3.88 | -4.77 |
| near-repetitive think (zlib 0.10-0.25, >=1k) | -0.92 | -1.34 | -5.82 | -5.36 | -3.64 | -4.06 | -6.35 |
| full loops (zlib < 0.10) | -7.68 | -5.28 | -7.61 | -5.67 | -5.62 | -7.84 | -7.08 |

- Bad arms rank 1-2 of 7 in 38 of 51 class definitions; present in steps 1-5 and 1-3 alone (weights ~ base model). Arm-level p = 1/21 per statistic; z vs the good arms' spread +3 to +4.
- E_g = 16 x within-group covariance(advantage, long-think tokens). The shift is broad (bad > good in 101 prompts, < in 63), not an outlier. Source: more rewarded long thinks on hard prompts (k<=4: G 50, I 53 vs 31-45; adv +1.7..+3.9 each) and fewer/lighter punished long thinks on easy prompts (k>=12).
- Holds with reward alone (share of near-repetitive think tokens in successful rollouts: G/I 0.116 vs 0.081-0.103), but the pure-reward CIs are wide; the LOO std-normalized advantage sharpens the gap ~3x.
- Not lexical: the extra reinforcement sits on ordinary prose tokens inside long turns.
- Propagates: chain R (vLLM live after replaying G 1-10) shows E(long) -1.6 and loops 4-6 % in steps 11-20. Precedes: chain L shows +0.46 at steps 21-25 before 12.9 % loops at 26-30. Not a symptom marker for already-sick weights (chains B, D strongly negative while loops grow).

## Added 13:45 PDT
- Frozen-weight engine test: P′ (MINF) vs Q′ (vLLM) on the same chain P step-20 weights, 15 steps each: reward 0.292 vs 0.295, loop share 1.3 vs 0.9 %, E(long) -2.7 vs -1.6, E(near-rep) -11.2 vs -5.9, reward share of long tokens 0.172 vs 0.187; paired by prompt (480 prompts) all differences null (sign tests p 0.57-0.82; per-prompt success counts correlate 0.967). The engine does not set the coupling at fixed weights.
- Refit timing: steps straddling a weight refit have less negative E(long) (-1.4 vs -3.1) but are the harder steps; within-group rank correlation of reward with duration is identical (-0.073 vs -0.078); healthy arms straddle refits more often than bad ones. Not the mechanism.
- Reward-only: the separation holds with pure reward (share of near-repetitive think tokens in successful rollouts G/I 0.116 vs good 0.081-0.103) but with wide CIs; the LOO advantage sharpens it ~3x.
- Actions after long thinks identical (execute_bash ~60 %, str_replace_editor ~25 %, think ~8 %) in G and M for rewarded and punished rollouts.


## Added 2026-09-26 09:10 PDT — prefix cache across refits (rkirby's observation) and chain P⁗
- Code: vLLM arms ran with prefix caching on (default on capability >= 8) and never reset it across refits (reset only via sleep_async in the colocated path or recompute_kv_cache_after_weight_updates=true, both off); MINF re-salts its prefix cache at every resume() after a refit. Knob added on the MINF side: mcore_generation_config.invalidate_prefix_cache_on_weight_update=false (chains P⁗ from scratch, P‴ from chain P step 20).
- Signature of a retained cache: the trainer-engine logprob mismatch on short turns grows with steps since the engine started and resets at every Slurm segment restart (fresh engine). Chain P‴ first-turn |d| 12.5 -> 23.3 x1e-3 over its 16-step first segment, back to 14.1 at the restart; chain P (same weights, cache invalidated) flat. vLLM arms: A +9.9 %, K +9.9 %, M +4.9 %, N +6.5 % over steps 1-5 -> 16-20; MINF invalidating G +2.1 %, I +2.6 %, J +6.4 %; chain P⁗ (MINF, cache kept) +9.0 % [5.9, 12.1], signed drift identical to chain K. The vLLM arms' growing mismatch is the retained cache.
- Chain P⁗ advantage pressure, steps 1-10: E(long 4k-16k) -3.18 [-5.45,-0.86] (rank 4 of 8; z +0.6 vs the good arms; G/I +4.0/+3.2), E(near-rep) -10.92, reward share of near-rep tokens 0.097; steps 1-5 already in the good range (-2.77 / -6.84); steps 11-20: -4.45 / -5.47 with the lowest loop share of any arm at that age (0.62 % vs 1.75-5.03 %). Prompt-paired: more suppressive than chain G by 2.3 se, indistinguishable from the good arms. Chain P‴: healthy through step 50 (loop share <= 0.7 % every window).
- Tally, healthy-start lineages: cache retained across refits 6 of 6 healthy so far (A, K, M, N, P‴ to 50, P⁗ to 20); cache invalidated at refits 2 of 3 bad (G, I bad; J good) plus chain L bad. The from-scratch verdict window for P⁗ is steps 20-30 (lands 2026-09-26 afternoon). rkirby is queuing vLLM arms with prefix caching disabled: the mirror prediction is a flat mismatch and, if the cache is what protects, a G-like early coupling and later loops.

## Interpretation
In the bad arms' first ten steps, long / near-repetitive deliberation was not the thing that lost inside a prompt group; in every good arm it was. GRPO on the bad data never learns "stop thinking", the style gets net reinforcement, and by steps 20-30 it turns into loops. The data cannot say whether MINF causes the coupling or two MINF runs were unlucky (2 of 3 MINF arms, 0 of 4 vLLM arms; chain J, MINF, looks like the vLLM arms). No generation-only feature separates the engines at matched weights.

## Advantage formula (verified exactly)
adv_i = (r_i - mean of the other 15) / std_ddof1 of the other 15; unnormalized when that std is 0. A lone success gets +1.0, two successes +3.6 each.

## Proposals
1. Discriminating replays via the splice (`_pack_group`, sample_mask): (A) chain G 1-10 with sample_mask = 0 for rollouts containing a >=4k generated segment; (B) chain M 1-10 with sample_mask = 0 for failed rollouts containing such a segment.
2. Training-signal fix independent of engine: a small penalty on near-repetitive long thinking (must include the 0.10-0.25 zlib band; full loops alone do not equalize). Gives the ~105/320 zero-variance groups per step a gradient. With LOO-std normalization use a std floor or an unnormalized term (continuous penalties otherwise explode advantages).
3. Frozen base-model pair (MINF and vLLM, lr 0, 10 steps of 512) to test the engine in the near-base regime where G/I had their episodes; expected resolution ~2-3 sigma for a gap the size of bad-vs-good.
4. Early-warning metric for any run: E(think turns 1k-16k) or the pure-reward share above, computed from the token dumps at step 10.

## Files
All under /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/: LOG.md (chronological), features.py + parts/ (per-turn/per-sample CSVs for arms A,B,C,D,F,G,I,J,K,L,M,N,P,Q,R,Rp), compare*_report.txt, arms_exposure*_report.txt, decompose_report.txt, trace_report.txt, groupcov_report.txt, robust_report.txt, mass_report.txt, dig2_report.txt, temporal_report.txt, after_report.txt, enginescan_report.txt, fingerprint_report.txt, rewardonly_report.txt, tokexp_report.txt, shaping.py/shaping2.py.

# Proposed experiments and how to score them (2026-09-25)

Scoring for every variant below: extract per-turn features from the run's token dumps with
`python3 features.py <arm> <EXP> <chunk.pt> parts` (or a jobs file + features.sbatch, cpu partition, ~2 min per 50 chunks),
then compute on steps of interest:
- E(long) = 1000 x sum over 4k-16k-token think turns of (advantage x turn tokens) / all generated tokens
- E(near-rep) = same for turns with >= 1k think tokens and zlib(think) < 0.25
- reward share of near-repetitive think tokens = sum(reward x tokens) / sum(tokens) over those turns
- loop-token share (turns with >= 1k think and zlib < 0.10)
Reference values, steps 1-10 (bad G/I vs good J/A/K/M/N): E(long) -0.1/-0.8 vs -2.3..-4.8; E(near-rep) -0.9/-1.3 vs -3.6..-6.4; reward share of near-rep tokens 0.116 vs 0.081-0.103. Per-arm bootstrap se on E ~1.1-1.4 (10 steps x 512 rollouts).

## 1. Frozen base-model pair (engine test in the regime where G/I had their episodes)
Two runs from the base checkpoint with lr 0 (the swe_lr0 overlay used for P′/Q′: +policy.megatron_cfg.checkpoint.load_optim=false, scheduler override), ENGINE=minf and ENGINE=vllm, same grpo.seed, >= 10 steps of 512. Weights never change, so the two arms sample the same policy on the same prompts.
Read-out: E(long), E(near-rep), reward share per arm with group bootstrap; paired by prompt hash (prompt_ids.py) for a sign test over 320 prompts.
Expected if the engine is innocent: both arms equal (either both around the good arms' values or both anywhere else). Expected if the engine drives it: MINF arm 3+ units less negative than vLLM arm. P′/Q′ (frozen step-20 weights) gave equal values (diff -1.1 +/- 1.8 for E(long); sign test p 0.82).

## 2. Masked replays (mechanism test, independent of engine)
The splice packs each replayed group in ForeignRolloutSource._pack_group with sample_mask = 1. Add a rule there:
- Variant A: source = chain G steps 1-10; sample_mask = 0 for any rollout whose token_mask has a generated segment >= 4,000 tokens (~16 % of rollouts). Then run live (either engine).
- Variant B: source = chain M steps 1-10; sample_mask = 0 only for rollouts with reward 0 that contain such a segment (removes the "long thinking loses" lesson). Then run live.
Read-out: loop-token share at live steps 15-30 vs chains R (G data, unmasked: 4-9 %) and Q (M data: < 1 %). A healthy A and a sick B would pin the poison to the long-think advantage weighting; a sick A says the poison is elsewhere in the same data.

## 3. Repetition-aware reward term (fix candidate)
r' = r - lam x (tokens in think turns with >= 1k think tokens and zlib < 0.25) / 10,000, lam ~ 0.05, applied before the group advantage. Include the 0.10-0.25 band (penalizing only full loops does not equalize the arms). With the leave-one-out std normalization a continuous term inflates advantages when the other 15 rewards are nearly equal: use a std floor (e.g. 0.1) or add the penalty after normalization. On the existing data this gives ~105 of 320 groups per step a gradient (today zero-variance) and makes E(long) strongly negative in every arm.

## 4. Early-warning metric for running arms
Compute E(long) and E(near-rep) over steps 1-10 (or any 10-step window while long-think tokens are >= 5 % of generated tokens). Values above about -1.5 on E(long) or -2 on E(near-rep) put a run with the two arms that later looped.

## 5. Cache-retention arms (added 2026-09-26)
- Done/running: chain P⁗ (MINF from scratch, invalidate_prefix_cache_on_weight_update=false) and chain P‴ (same knob from chain P step 20). Scoring as above plus the retained-cache signature: per-step short-turn |d| and first-turn |d| against steps since the last engine start (segments.py; slope > +0.15 x1e-3/step on turn 0 and a drop at each segment restart = cache retained).
- To run (rkirby): vLLM from scratch with vllm_cfg.enable_prefix_caching=false (or grpo.async_grpo.recompute_kv_cache_after_weight_updates=true). Predictions if the retained cache is what protected the vLLM arms: flat |d| over training (like chain G), E(long)/E(near-rep) in steps 1-10 drifting toward the G/I values in some replicas, loops by steps 20-30. If it stays healthy with a flat |d|, the cache is not the protective factor and the engine difference lies elsewhere.
- Two replicas per treatment are needed to beat the 1-in-3 base rate of a healthy MINF-from-scratch arm.

## 6. Step-range ablation of chain G's data (final design 2026-09-27 21:30; build now, launch on rkirby's word after chain X's 16-20 window)
Purpose: localize in time which of chain G's replayed steps carry the poison, with no hypothesis about which rollouts do. LR warmup (lr_warmup_iters 10, 3e-7 -> 3e-6 linear, constant after) puts ~70 % of the first ten steps' update magnitude in steps 6-10 and ~47 % in steps 8-10.

Design choice: REPLACE the excluded steps with the arm's own live vLLM generation rather than masking them out. Reasons: (1) it is what the splice does today for any step outside the replayed range (single_controller.py: `+foreign_rollout.steps=a:b` -> ForeignRolloutSource(range(a, b+1)); a lookup miss falls through to generate_and_push, no code change); (2) masking an entire step leaves the optimizer with a zero gradient but a live Adam momentum step and a loss normalized over zero unmasked tokens (NaN hazard), plus iteration/LR bookkeeping with nothing trained; (3) the natural control for "live steps + G's 6-10" is an ordinary vLLM-from-scratch run (A, K, M, N, Q⁗, all healthy), so G's 6-10 data is the only difference. The replayed steps stay at chain G's LR because iteration numbers are unchanged. Off-policy note: the arm's weights at step 6 come from its own five live steps, not chain G's; at warmup LRs the two policies are within a few updates of the base model, so G's step-6..10 rollouts are only slightly off-policy for it (TIS ratios handle it, as they did for chain R itself).

Arms (rkirby's split, 21:15: first round 1:7 vs 8:10; everything else as chain R / X: same yaml, seed 42, 64 nodes, +checkpointing.load_replay_buffer=false, dumps on, target train step 25, stop at step_25):
- G8-10: `+foreign_rollout.source_run=<chain G run> +foreign_rollout.steps=8:10`. Live vLLM steps 1-7, chain G's data at 8-10 (about 47 % of the first ten steps' LR mass), live 11-25. Prediction if the poison rides the high-LR steps: loops like chain R (4-7 % at 16-25).
- G1-7:  `+foreign_rollout.steps=1:7`. Chain G's data at 1-7 (about 53 % of the LR mass), live 8-25.
- Next round depends on the outcome: if only G8-10 loops, split 8:8 / 9:9 / 10:10 (or 8:9 vs 10:10); if only G1-7 loops, split 1:4 vs 5:7; if both loop, the poison is spread (consistent with the diffuse-tilt reading) and the localization ends there.
- Optional control: M8-10 (`source_run=<chain M run> +foreign_rollout.steps=8:10`), healthy data at the same range.
Off-trajectory note (from the cross-train session): the arm's policy at the replayed steps has been trained on its own live steps, so chain G's rows are slightly off-trajectory and the 2.0 seq-logprob-error threshold may zero more rows than chain R's two; sample_mask_after in the token-level chunks gives the count and it is part of the read-out.
Chain G run: /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920. Chain M run: .../nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923.

Verification and read-out: verify_replay.py on the replayed range only (content identity 512/512 per replayed step; no mask file, so expect zero masked rows apart from the usual seq-error ones); the live steps 1-5 of G6-10 should score like a normal vLLM start (loop share 1-4 %, cumulative E(4k-16k) -1 to -5 by step 5). Loop share per 5-step window at 11-15 / 16-20 / 21-25 vs chain R (4.2 / 6.5 / 4.7 %) and the healthy band. Priority vs Y/Z: if chain X's 16-20 window is at chain R's level, run G6-10 and G1-5 in Y's and Z's slots; if X is healthy, keep Y (X's control) and run G6-10 in Z's slot.

### 6b. Launch record (2026-09-28 01:00 PDT)
AA = jobs 4061467/4061468 (run nano35-swe-v2-splice-vllm-runGdata-8to10-20260928), AB = jobs 4061469/4061470 (run nano35-swe-v2-splice-vllm-runGdata-1to7-20260928), both to step_25, chained X -> AA -> AB by Slurm gates; Y/Z held for release after AB; AC not launched. Scoring by score_xyz.sh (empty mask verification of the replayed range, live-step windows).

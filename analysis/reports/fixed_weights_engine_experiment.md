# Fixed-weights engine comparison: runaway-reasoning-loop propensity, MINF vs vLLM

Written 2026-09-22 from the SWE-E2E v2 dump chains (chain G = MINF from scratch, no prefix cache; chain I = MINF from scratch, prefix cache on; run A = vLLM from scratch; run B = MINF from MINF step 10; chain D = vLLM from MINF step 10; chain F = MINF from vLLM step 10; chain K = vLLM from scratch seed 1234, queued as job 3925311).

## 1. Question

MINF-trained weights acquire a propensity for runaway reasoning loops (single `<think>` blocks of 20k-180k tokens that repeat a 300-500-token cycle until the context window closes the rollout) by step ~10-15; vLLM-trained weights do not. Is this caused by a systematic difference in what the two engines generate, or in the training signal computed from those generations (structural), or by chance reinforcement of rare loop events early in training (luck)?

## 2. What the dump chains already establish

Methodology (all implemented under `analysis/loopfeedback/`):

- Reasoning block = tokens from the start of a model turn to `</think>`. Loop block = block >= 1,000 tokens whose text has zlib compression ratio < 0.10 (normal long blocks sit at 0.25-0.40; there is almost nothing between 0.10 and 0.20). Self-exited loop = closed with `</think>`; runaway = still repeating when the context window truncated the rollout.
- Loop-token share = share of a batch's reasoning tokens inside loop blocks. Chain G: 1-4 % through step 14, 37-42 % at steps 35-36. Run A: ~0 % after step 19.
- Weights carry the propensity: run B and chain D (both started from the MINF step-10 weights, one served by MINF, one by vLLM) loop; chain F (MINF engine on the vLLM step-10 weights) does not; run A does not.
- In steps 1-10, paired by (step, SWE instance) across chains G, I and run A (identical base weights at step 1, identical prompt stream every step, 320 instances x 16 generations per arm), NO metric separates the arms: loop-token share 0.0065 / 0.0081 / 0.0057 (p 0.53, 0.07, 0.36), blocks >= 1,000 tokens per rollout 2.42 / 2.44 / 2.45, longest block 2,722 / 2,882 / 2,546 tokens, sentences per normal block 8.57 / 8.64 / 8.71, reward 0.359 / 0.358 / 0.355, trainer-vs-engine logprob error 0.01412 / 0.01421 / 0.01418 nats/token. The one marginal signal - MINF-sampled tokens more probable by 0.0014 nats/token over all generated tokens (gen_lp_mean G-A, p = 0.013) - disappears once loop tokens (p ~ 1) are excluded: on normal blocks G-A is +0.0015 (p 0.12), I-A -0.0004 (p 0.69), and at step 1 (identical weights) the sign is reversed (G-A -0.0066). Sampling-sharpness metrics must therefore be computed on normal blocks only. In steps 11-20 the same test separates the arms on every metric (loop share G 0.0121, I 0.0094, A 0.0046, p < 0.001).
- There is no phrase-level feedback loop: sentence-opener usage drift is uncorrelated with the gradient push on the opener (rank corr ~0 in every arm); loop bodies are gradient-inert (tokens at p > 0.99).
- Exit gate: `</think>` follows a bare `.` (19.5 % of the time in normal blocks), never `.\n\n` (0 of 2.79 M); inside loops both punctuation choices sit at p ~0.998, so exits are rare-sample events (31 % of chain G loop exits are "fork" exits with the exit token sampled at p < 0.05, vs 1.2 % in normal blocks).

So the difference is entrenched in the weights by step 10 while the rollouts and the training signal in steps 1-10 look statistically identical at n = 5,120 rollouts per arm. The experiment below measures the engine effect at frozen weights with seed replicates, i.e. with the training dynamics removed.

## 3. Design

Each run = fixed weights W x engine E x seed s, producing N generation rounds of 32 prompts x 16 generations on the identical in-order prompt stream, with the learning rate set to 0 so the weights never change. Everything else is the dump-chain recipe (rollout + token-level dumps on; MINF with prefix caching off and overlap_param_gather off; the same 64-node shape).

Factors

- W0 = base checkpoint (the starting point of every chain). Primary.
- W_G10 = chain G `checkpoints/step_10` (MINF-trained carrier), W_A10 = run A `checkpoints/step_10` (vLLM-trained non-carrier). Secondary: measures the propensity as a property of the weights and the engine x weights interaction.
- E in {MINF, vLLM}.
- s in {42, 1234}: two seeds per engine so the seed-to-seed spread is measured with the same n as the engine difference.
- N = 10 rounds (5,120 rollouts per run) in tier 1; 20 if tier-1 intervals do not separate.

Launch (from `swe_dump/`, same launcher as the chains; overrides are appended to `launch_swe_dump.sh`):

```
ENGINE=minf|vllm EXP_NAME=<name> bash ./launch_swe_dump.sh \
  policy.megatron_cfg.optimizer.lr=0 policy.megatron_cfg.optimizer.min_lr=0 \
  policy.megatron_cfg.scheduler.lr_warmup_iters=0 \
  grpo.seed=<s> grpo.max_num_steps=<N> +checkpointing.load_replay_buffer=false
```
weight_decay is already 0 in `swe_sc_cmh_common.yaml`. For W_G10 / W_A10 copy the step_10 checkpoint into `runs/<EXP_NAME>/checkpoints/step_10` and pass `SEED_CHECKPOINT=<path>` (the launcher refuses to submit unless it is complete). One MINF run and one vLLM run fit the standing 64-node reservation side by side (32 + 32 nodes each in the dump layout).

Sanity checks built into the design

- With lr = 0 the step-5 / step-10 checkpoints must be bit-identical to the start weights (compare `policy/weights` tensors). Any drift of generation statistics across rounds at frozen weights is therefore an engine or refit artifact - the MINF refit path gets tested for free.
- Round 1 of every run is generated before any refit; rounds 2..N after refits. Compare round 1 vs rounds 2..N within each run.
- The training step still executes (forward/backward with zero update), so the trainer-side logprobs, the mismatch metrics and the usual W&B metrics are produced for every round.

## 4. Metrics

Per rollout, then averaged per (round, SWE instance) unit (16 generations), then paired across runs.

Generation

1. Loop-token share; loop blocks (self-exited, runaway); blocks >= 1,000 tokens by zlib bin (< 0.10 loop, 0.10-0.20 semi-repetitive, >= 0.20 ordinary); longest block; reasoning tokens per rollout; sentences per normal block; context truncations; reward.
2. Sampling sharpness, on NORMAL blocks only (loop tokens sit at p ~ 1 and dominate any all-token mean): token-weighted mean log p of generated tokens under the engine and under the trainer; share of tokens at p >= 0.99; the engine-vs-engine difference at identical weights is an effective-temperature estimate.
3. Exit gate: share of `.` vs `.\n\n` at sentence ends; probability histograms of `.`, `.\n\n` and `</think>` (bins >= .99, .9-.99, .7-.9, .5-.7, .3-.5, .1-.3, .03-.1, .01-.03, < .01); `.` -> `</think>` rate; fork-exit share among loop exits.

Training signal (from the same rollouts)

4. Trainer-vs-engine logprob error per token (mean, p99, count > 1 nat), per-sequence multiplicative probability error, token-mask drop share (`sample_mask_after`).
5. Advantage-weighted gradient mass on loop-entry regions; share of context-truncated rollouts that carry positive advantage (the "post-solution loop" channel).

## 5. Statistics

- Unit = (round, SWE instance). Runs are paired because the prompt stream is identical. Two-sided sign-flip permutation test (20,000 draws) on paired differences plus a bootstrap 95 % interval; `analysis/loopfeedback/early_compare.py` already does this for the chains and only needs the arm list changed.
- Report every engine contrast next to the seed contrast of the same engine. "Structural" requires the engine effect to exceed the seed effect with non-overlapping intervals on the primary endpoints.
- Primary endpoints: loop-token share; blocks >= 1,000 tokens with zlib < 0.20 per rollout; mean log p of sampled tokens; sentences per normal block. Everything else secondary.
- Power, from the steps 1-10 comparison (320 units): standard error of a paired difference ~0.0015 for loop-token share (baseline 0.006), ~0.0006 nats for mean log p, ~0.09 for sentences per normal block. Ten rounds detect a 50 % relative change in loop share, a 0.0012-nat sharpness shift, or 0.2 sentences per block at 2 SE; twenty rounds shrink these by 0.7x.

## 6. Cost

Per run: N rounds x ~12 min on 64 nodes plus ~15 min startup; dumps ~10 GB per round (rollout JSONL ~9 GB, token level ~1 GB). Token dumps alone carry every metric above except the phrase scans, so the seed-replicate runs can drop the rollout JSONL.

- Tier 1: W0 x 2 engines x 2 seeds x 10 rounds = 4 runs, ~9 h of 64-node time, <= 400 GB.
- Tier 2: + W_G10 and W_A10 x 2 engines x 1 seed = 4 runs, ~9 h.
- Tier 3 (propensity probe along the chains): vLLM only, 2 rounds per checkpoint at chain G rungs 5..35 and run A rungs 5..30 (13 checkpoints, ~26 rounds, ~5.5 h). Gives loop propensity as a function of training step, decoupled from the engine that produced the weights.

## 7. Decision rules

- Engine effect at W0 exceeds seed effect on a primary endpoint -> structural in generation. Follow with a logits-level comparison (same prompt prefix, both engines, full distributions) to locate the numeric source (sampling sharpness or the punctuation gate).
- No engine effect at W0, W_G10 loops under both engines, W_A10 under neither, and chain K stays loop-free -> the divergence arises in training-time dynamics on statistically identical samples: rare-event reinforcement plus the update path. Follow with the gradient-side comparison (channel-b share, refit/optimizer path).
- Chain K develops loops -> "luck" is live; the rate needs more replicates before any engine claim.

## 8. Code to reuse

- `analysis/loopfeedback/lf_worker.py` - loop labels, per-block items (zlib ratio, tokens, sentences, mean log p, class).
- `analysis/loopfeedback/punct_worker.py`, `exit_worker.py` - sentence-final punctuation, exit structure.
- `analysis/loopfeedback/sfp_worker.py` - sentence-final probability histograms (int64 JSON bug fixed 2026-09-22; not yet rerun).
- `analysis/loopfeedback/early_compare.py` - per-step tables and the paired sign-flip test.
- `analysis/loopfeedback/report1.py`, `punct_report.py`, `exit_report.py`, `sfp_report.py` - report generators.
- Job lists are built from `runs/<EXP>/dumps/token_level/*.pt`; a full pass over ~400 chunks runs in ~3 min on partition `cpu` (48 CPUs, `xargs -P 40`).

## Appendix: paired comparison of the dump chains (2026-09-22)

```
PAIRED BY (step, SWE instance), steps 1-10: 320 instances x 16 gens per arm; p = two-sided sign-flip permutation test
metric                                              G          I          A |        G-A       p |        I-A       p |        G-I       p
loop token share per rollout                   0.0065     0.0081     0.0057 |     0.0009   0.534 |     0.0024   0.071 |    -0.0016   0.356
rollouts with >=1 loop block                   0.0145     0.0162     0.0129 |     0.0016   0.570 |     0.0033   0.175 |    -0.0018   0.566
runaway blocks per rollout                     0.0109     0.0125     0.0098 |     0.0012   0.629 |     0.0027   0.249 |    -0.0016   0.605
blocks >=1000 tok per rollout                  2.4156     2.4381     2.4527 |    -0.0371   0.401 |    -0.0146   0.750 |    -0.0225   0.611
blocks >=1000 tok & zlib<0.20 per rollout      0.0469     0.0520     0.0398 |     0.0070   0.206 |     0.0121   0.052 |    -0.0051   0.549
longest block per rollout (tok)                  2722       2882       2546 |        175   0.217 |        336   0.062 |       -161   0.443
reasoning tokens per rollout                    13678      13945      13622 |         55   0.791 |        323   0.157 |       -267   0.297
sentences per normal block                     8.5717     8.6439     8.7103 |    -0.1386   0.140 |    -0.0664   0.437 |    -0.0722   0.384
mean log p of reasoning tokens                -0.3155    -0.3171    -0.3173 |     0.0018   0.081 |     0.0002   0.813 |     0.0016   0.147
reward                                         0.3594     0.3576     0.3551 |     0.0043   0.527 |     0.0025   0.647 |     0.0018   0.800
context-truncated rollouts                     0.0143     0.0141     0.0123 |     0.0020   0.397 |     0.0018   0.526 |     0.0002   1.000
trainer-vs-engine |log p| error, mean         0.01412    0.01421    0.01418 |   -0.00006   0.215 |    0.00003   0.535 |   -0.00009   0.073
gen_lp_mean (all generated tokens)           -0.16241   -0.16319   -0.16383 |    0.00142   0.013 |    0.00064   0.263 |    0.00078   0.171
prev_lp_mean (trainer log p, same tokens)    -0.16383   -0.16463   -0.16525 |    0.00142   0.014 |    0.00062   0.286 |    0.00080   0.166

PAIRED BY (step, SWE instance), steps 11-20: 320 instances x 16 gens per arm
loop token share per rollout                   0.0121     0.0094     0.0046 |     0.0075   0.000 |     0.0048   0.001 |     0.0027   0.075
rollouts with >=1 loop block                   0.0187     0.0189     0.0102 |     0.0086   0.001 |     0.0088   0.001 |    -0.0002   1.000
blocks >=1000 tok per rollout                  2.7340     2.2719     3.0418 |    -0.3078   0.000 |    -0.7699   0.000 |     0.4621   0.000
blocks >=1000 tok & zlib<0.20 per rollout      0.0607     0.0627     0.0398 |     0.0209   0.002 |     0.0229   0.002 |    -0.0020   0.804
longest block per rollout (tok)                  3466       2716       2722 |        744   0.000 |         -6   0.966 |        751   0.000
sentences per normal block                     9.3097     8.1882     9.6502 |    -0.3405   0.000 |    -1.4620   0.000 |     1.1215   0.000
mean log p of reasoning tokens                -0.2972    -0.3395    -0.3139 |     0.0167   0.000 |    -0.0256   0.000 |     0.0423   0.000
trainer-vs-engine |log p| error, mean         0.01481    0.01448    0.01589 |   -0.00109   0.000 |   -0.00141   0.000 |    0.00032   0.000
```
Full per-step tables: `analysis/loopfeedback/early_compare.txt`.

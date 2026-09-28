# First ten steps, five arms: run A (vLLM from scratch), chain G (MINF from scratch, no prefix cache), chain I (MINF from scratch, prefix cache on, replica 1), chain J (MINF from scratch, prefix cache on, replica 2), chain K (vLLM from scratch, seed 1234)

All arms start from the same base model and see the same 32 prompts x 16 rollouts per step (in_order sampler); every paired comparison below pairs prompts by SWE instance id within the same step. Rollout counts: run A (vLLM from scratch) 5120, chain G (MINF from scratch, no prefix cache) 5120, chain I (MINF from scratch, prefix cache on, replica 1) 5120, chain J (MINF from scratch, prefix cache on, replica 2) 5120, chain K (vLLM from scratch, seed 1234) 5120. Bootstrap draws: 2000.

Pairing choice: run A (vLLM from scratch) is the reference; every paired statistic is reported as (other arm minus run A (vLLM from scratch)), one block per other arm. Distribution panels show all five arms. Histograms of per-rollout or per-group quantities are shares (normalized), not counts.

Caveats. (1) A two-line guard was applied live to the mounted Megatron-LM dynamic_engine.py on 2026-09-22 07:47: chain I (MINF from scratch, prefix cache on, replica 1) ran unguarded for steps 1-15 and guarded from step 16, so its steps 1-10 are unguarded like run A (vLLM from scratch) and chain G (MINF from scratch, no prefix cache) and this window is internally consistent; chain J (MINF from scratch, prefix cache on, replica 2) is guarded throughout. (2) Step accounting uses the SingleController worker log (worker-*.out containing SingleControllerActor); the driver log is unreliable for this family. (3) chain K (vLLM from scratch, seed 1234) is the only arm not on grpo.seed 42.

## Per-step trainer metrics (controller logs)

Cell order in every column: run A (vLLM from scratch) / chain G (MINF from scratch, no prefix cache) / chain I (MINF from scratch, prefix cache on, replica 1) / chain J (MINF from scratch, prefix cache on, replica 2) / chain K (vLLM from scratch, seed 1234).

| step | reward | gen tokens (mean) | truncation rate | gen_kl_error | approx_entropy |
|---|---|---|---|---|---|
| 1 | 0.344 / 0.359 / 0.357 / 0.381 / 0.359 | 26493 / 25741 / 25539 / 25775 / 26711 | 0.014 / 0.012 / 0.006 / 0.014 / 0.006 | 0.00128 / 0.00128 / 0.00129 / 0.00126 / 0.00127 | 0.1550 / 0.1567 / 0.1564 / 0.1536 / 0.1529 |
| 2 | 0.541 / 0.543 / 0.533 / 0.510 / 0.539 | 20568 / 21319 / 21682 / 20447 / 21139 | 0.006 / 0.004 / 0.010 / 0.006 / 0.012 | 0.00143 / 0.00146 / 0.00146 / 0.00143 / 0.00144 | 0.1629 / 0.1618 / 0.1634 / 0.1613 / 0.1598 |
| 3 | 0.256 / 0.262 / 0.252 / 0.244 / 0.254 | 26381 / 26084 / 25979 / 27229 / 26494 | 0.000 / 0.008 / 0.006 / 0.012 / 0.006 | 0.00139 / 0.00137 / 0.00143 / 0.00137 / 0.00137 | 0.1548 / 0.1529 / 0.1562 / 0.1492 / 0.1523 |
| 4 | 0.402 / 0.428 / 0.406 / 0.406 / 0.430 | 27127 / 28073 / 28620 / 28373 / 27216 | 0.018 / 0.020 / 0.010 / 0.021 / 0.020 | 0.00154 / 0.00154 / 0.00152 / 0.00149 / 0.00150 | 0.1550 / 0.1555 / 0.1504 / 0.1489 / 0.1510 |
| 5 | 0.254 / 0.266 / 0.232 / 0.248 / 0.221 | 33050 / 33630 / 34727 / 33767 / 35077 | 0.021 / 0.008 / 0.021 / 0.018 / 0.021 | 0.00153 / 0.00158 / 0.00158 / 0.00153 / 0.00146 | 0.1540 / 0.1526 / 0.1511 / 0.1531 / 0.1483 |
| 6 | 0.371 / 0.357 / 0.414 / 0.379 / 0.363 | 25578 / 25823 / 25329 / 25602 / 25822 | 0.012 / 0.014 / 0.008 / 0.006 / 0.020 | 0.00143 / 0.00143 / 0.00144 / 0.00146 / 0.00140 | 0.1466 / 0.1451 / 0.1485 / 0.1473 / 0.1424 |
| 7 | 0.432 / 0.434 / 0.398 / 0.434 / 0.418 | 26319 / 25500 / 25229 / 25148 / 26102 | 0.012 / 0.016 / 0.014 / 0.014 / 0.020 | 0.00151 / 0.00143 / 0.00155 / 0.00149 / 0.00149 | 0.1592 / 0.1535 / 0.1568 / 0.1562 / 0.1568 |
| 8 | 0.289 / 0.285 / 0.312 / 0.289 / 0.285 | 33225 / 32926 / 34993 / 36071 / 33633 | 0.027 / 0.039 / 0.041 / 0.049 / 0.041 | 0.00146 / 0.00144 / 0.00136 / 0.00136 / 0.00142 | 0.1404 / 0.1369 / 0.1329 / 0.1291 / 0.1346 |
| 9 | 0.314 / 0.314 / 0.305 / 0.303 / 0.340 | 24983 / 25987 / 25156 / 26815 / 26007 | 0.004 / 0.012 / 0.012 / 0.018 / 0.006 | 0.00153 / 0.00153 / 0.00160 / 0.00155 / 0.00154 | 0.1658 / 0.1596 / 0.1622 / 0.1571 / 0.1595 |
| 10 | 0.348 / 0.346 / 0.365 / 0.348 / 0.367 | 24892 / 25462 / 24906 / 24489 / 24985 | 0.010 / 0.012 / 0.014 / 0.010 / 0.002 | 0.00163 / 0.00160 / 0.00157 / 0.00164 / 0.00166 | 0.1617 / 0.1562 / 0.1603 / 0.1591 / 0.1603 |

## Generation length per rollout (steps 1-10 pooled; run A (vLLM from scratch) 5,120, chain G (MINF from scratch, no prefix cache) 5,120, chain I (MINF from scratch, prefix cache on, replica 1) 5,120, chain J (MINF from scratch, prefix cache on, replica 2) 5,120, chain K (vLLM from scratch, seed 1234) 5,120 rollouts)

| arm | mean | p50 | p90 | p95 | p99 | max | share > 50k | share > 100k | share > 150k | truncated (context window) |
|---|---|---|---|---|---|---|---|---|---|---|
| run A (vLLM from scratch) | 26862 | 22405 | 50416 | 59274 | 78753 | 172600 | 0.103 | 0.005 | 0.001 | 0.012 |
| chain G (MINF from scratch, no prefix cache) | 27055 | 22904 | 51212 | 59443 | 78883 | 181841 | 0.109 | 0.006 | 0.002 | 0.014 |
| chain I (MINF from scratch, prefix cache on, replica 1) | 27216 | 23056 | 51464 | 59198 | 82707 | 172093 | 0.112 | 0.007 | 0.003 | 0.014 |
| chain J (MINF from scratch, prefix cache on, replica 2) | 27372 | 23142 | 51196 | 60485 | 81493 | 185819 | 0.110 | 0.007 | 0.003 | 0.017 |
| chain K (vLLM from scratch, seed 1234) | 27319 | 23122 | 51553 | 59494 | 87798 | 178120 | 0.111 | 0.008 | 0.003 | 0.015 |

Paired by prompt, other arm minus run A (vLLM from scratch) (n = matched prompt-steps):

- chain G (MINF from scratch, no prefix cache): mean length +193 tokens, 95% CI [-307, +673] (n=320; longer in 167, shorter in 153); per step: s1 -752, s2 +751, s3 -297, s4 +946, s5 +580, s6 +244, s7 -819, s8 -300, s9 +1004, s10 +570; per-prompt MAX length +810 [-1758, +3132]; KS distance 0.014, paired-permutation p = 0.521.
- chain I (MINF from scratch, prefix cache on, replica 1): mean length +354 tokens, 95% CI [-194, +911] (n=320; longer in 161, shorter in 159); per step: s1 -954, s2 +1114, s3 -402, s4 +1492, s5 +1678, s6 -249, s7 -1091, s8 +1767, s9 +173, s10 +14; per-prompt MAX length -97 [-2573, +2420]; KS distance 0.016, paired-permutation p = 0.278.
- chain J (MINF from scratch, prefix cache on, replica 2): mean length +510 tokens, 95% CI [-102, +1187] (n=320; longer in 164, shorter in 156); per step: s1 -718, s2 -121, s3 +848, s4 +1246, s5 +717, s6 +23, s7 -1171, s8 +2846, s9 +1832, s10 -404; per-prompt MAX length +15 [-2690, +2740]; KS distance 0.019, paired-permutation p = 0.150.
- chain K (vLLM from scratch, seed 1234): mean length +457 tokens, 95% CI [-82, +1045] (n=320; longer in 157, shorter in 163); per step: s1 +218, s2 +571, s3 +113, s4 +89, s5 +2028, s6 +243, s7 -217, s8 +408, s9 +1024, s10 +93; per-prompt MAX length +2064 [-497, +4848]; KS distance 0.018, paired-permutation p = 0.188.

## Reward and group structure

| arm | mean reward | mean within-group std | groups all-fail | groups all-pass | mixed groups | mean adv. (trainer) |
|---|---|---|---|---|---|---|
| run A (vLLM from scratch) | 0.355 | 0.175 | 0.419 | 0.119 | 0.463 | -0.0397 |
| chain G (MINF from scratch, no prefix cache) | 0.359 | 0.181 | 0.406 | 0.109 | 0.484 | -0.0371 |
| chain I (MINF from scratch, prefix cache on, replica 1) | 0.358 | 0.177 | 0.403 | 0.125 | 0.472 | -0.0274 |
| chain J (MINF from scratch, prefix cache on, replica 2) | 0.354 | 0.170 | 0.406 | 0.141 | 0.453 | -0.0470 |
| chain K (vLLM from scratch, seed 1234) | 0.358 | 0.179 | 0.397 | 0.119 | 0.484 | -0.0440 |

Paired pass-rate and within-group-std differences, other arm minus run A (vLLM from scratch):

- chain G (MINF from scratch, no prefix cache): pass rate +0.0043, 95% CI [-0.0088, +0.0170] (higher in 73, lower in 63, equal in 184 of 320 prompt-steps); within-group reward std +0.0065, 95% CI [-0.0067, +0.0196].
- chain I (MINF from scratch, prefix cache on, replica 1): pass rate +0.0025, 95% CI [-0.0078, +0.0131] (higher in 76, lower in 68, equal in 176 of 320 prompt-steps); within-group reward std +0.0025, 95% CI [-0.0096, +0.0149].
- chain J (MINF from scratch, prefix cache on, replica 2): pass rate -0.0010, 95% CI [-0.0119, +0.0096] (higher in 72, lower in 63, equal in 185 of 320 prompt-steps); within-group reward std -0.0046, 95% CI [-0.0177, +0.0097].
- chain K (vLLM from scratch, seed 1234): pass rate +0.0025, 95% CI [-0.0098, +0.0150] (higher in 79, lower in 66, equal in 175 of 320 prompt-steps); within-group reward std +0.0040, 95% CI [-0.0109, +0.0184].

## `</think>` close probability (engine logprob of the sampled close)

| arm | closes | geo-mean P | 10th-pct P | 1st-pct P | share P<0.905 | share P<0.5 | share P<0.1 | turns without </think> |
|---|---|---|---|---|---|---|---|---|
| run A (vLLM from scratch) | 479710 | 0.9330 | 0.883 | 0.246 | 0.1120 | 0.0274 | 0.00373 | 0.00018 |
| chain G (MINF from scratch, no prefix cache) | 483468 | 0.9332 | 0.885 | 0.249 | 0.1112 | 0.0273 | 0.00360 | 0.00019 |
| chain I (MINF from scratch, prefix cache on, replica 1) | 481231 | 0.9333 | 0.885 | 0.243 | 0.1110 | 0.0271 | 0.00359 | 0.00021 |
| chain J (MINF from scratch, prefix cache on, replica 2) | 484717 | 0.9338 | 0.887 | 0.244 | 0.1098 | 0.0272 | 0.00362 | 0.00022 |
| chain K (vLLM from scratch, seed 1234) | 484423 | 0.9331 | 0.885 | 0.241 | 0.1105 | 0.0276 | 0.00365 | 0.00021 |

Paired per-prompt mean close logprob, other arm minus run A (vLLM from scratch):

- chain G (MINF from scratch, no prefix cache): +0.0002 (multiplicative P ratio 1.0002), 95% CI [-0.0012, +0.0016]; higher in 163, lower in 157 of 320 prompt-steps; per step: s1 -0.0018, s2 -0.0033, s3 +0.0048, s4 +0.0003, s5 -0.0012, s6 -0.0003, s7 -0.0054, s8 +0.0029, s9 +0.0022, s10 +0.0039.
- chain I (MINF from scratch, prefix cache on, replica 1): -0.0000 (multiplicative P ratio 1.0000), 95% CI [-0.0016, +0.0014]; higher in 154, lower in 166 of 320 prompt-steps; per step: s1 -0.0029, s2 -0.0082, s3 +0.0026, s4 +0.0020, s5 -0.0014, s6 +0.0019, s7 +0.0011, s8 +0.0009, s9 +0.0001, s10 +0.0036.
- chain J (MINF from scratch, prefix cache on, replica 2): +0.0004 (multiplicative P ratio 1.0004), 95% CI [-0.0011, +0.0019]; higher in 178, lower in 142 of 320 prompt-steps; per step: s1 -0.0006, s2 -0.0049, s3 +0.0022, s4 +0.0009, s5 +0.0031, s6 +0.0004, s7 +0.0007, s8 +0.0002, s9 +0.0005, s10 +0.0015.
- chain K (vLLM from scratch, seed 1234): +0.0003 (multiplicative P ratio 1.0003), 95% CI [-0.0011, +0.0018]; higher in 165, lower in 155 of 320 prompt-steps; per step: s1 -0.0051, s2 -0.0028, s3 +0.0044, s4 +0.0029, s5 +0.0013, s6 -0.0001, s7 -0.0009, s8 +0.0013, s9 +0.0001, s10 +0.0023.

## Thinking tokens per assistant turn

| arm | turns | mean | p50 | p90 | p95 | p99 | max | share > 1k | share > 2k | share > 5k | turn tokens p50 / p99 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run A (vLLM from scratch) | 479710 | 142 | 25 | 341 | 642 | 1836 | 97721 | 0.0261 | 0.0086 | 0.00123 | 106 / 2228 / 152727 |
| chain G (MINF from scratch, no prefix cache) | 483468 | 139 | 25 | 333 | 633 | 1821 | 123816 | 0.0255 | 0.0085 | 0.00121 | 106 / 2237 / 177644 |
| chain I (MINF from scratch, prefix cache on, replica 1) | 481231 | 141 | 25 | 338 | 638 | 1833 | 119747 | 0.0258 | 0.0086 | 0.00125 | 106 / 2224 / 147347 |
| chain J (MINF from scratch, prefix cache on, replica 2) | 484717 | 141 | 25 | 334 | 632 | 1837 | 113259 | 0.0254 | 0.0085 | 0.00120 | 105 / 2251 / 184380 |
| chain K (vLLM from scratch, seed 1234) | 484423 | 140 | 25 | 337 | 635 | 1809 | 114653 | 0.0253 | 0.0084 | 0.00112 | 106 / 2198 / 169118 |

Paired per-prompt thinking length, other arm minus run A (vLLM from scratch):

- chain G (MINF from scratch, no prefix cache): mean thinking tokens per turn -2.0, 95% CI [-5.7, +1.8] (higher in 152/320); per-rollout longest thinking span +10, 95% CI [-114, +134].
- chain I (MINF from scratch, prefix cache on, replica 1): mean thinking tokens per turn -0.1, 95% CI [-2.8, +2.6] (higher in 154/320); per-rollout longest thinking span +20, 95% CI [-67, +112].
- chain J (MINF from scratch, prefix cache on, replica 2): mean thinking tokens per turn +2.3, 95% CI [-3.7, +11.9] (higher in 155/320); per-rollout longest thinking span +87, 95% CI [-64, +275].
- chain K (vLLM from scratch, seed 1234): mean thinking tokens per turn -1.1, 95% CI [-4.9, +3.1] (higher in 156/320); per-rollout longest thinking span -22, 95% CI [-160, +119].

## Engine logprob of sampled tokens (all generated tokens, steps 1-10)

| arm | tokens | mean logprob | share < -1 | share < -5 | per-million < -10 | per-million < -15 | per-million < -20 |
|---|---|---|---|---|---|---|---|
| run A (vLLM from scratch) | 137,532,468 | -0.1548 | 0.0485 | 0.00169 | 48.5 | 2.14 | 0.00 |
| chain G (MINF from scratch, no prefix cache) | 138,519,061 | -0.1525 | 0.0477 | 0.00167 | 48.2 | 2.28 | 0.00 |
| chain I (MINF from scratch, prefix cache on, replica 1) | 139,345,719 | -0.1525 | 0.0478 | 0.00166 | 49.1 | 2.06 | 0.00 |
| chain J (MINF from scratch, prefix cache on, replica 2) | 140,142,140 | -0.1504 | 0.0471 | 0.00163 | 47.9 | 2.15 | 0.00 |
| chain K (vLLM from scratch, seed 1234) | 139,871,722 | -0.1511 | 0.0473 | 0.00165 | 48.3 | 2.09 | 0.00 |

Paired per-prompt engine-logprob statistics, other arm minus run A (vLLM from scratch):

- chain G (MINF from scratch, no prefix cache): mean logprob +0.0014, 95% CI [+0.0004, +0.0025]; per-rollout minimum logprob -0.01, CI [-0.08, +0.06]; tokens below -10 per rollout +0.00, CI [-0.04, +0.05].
- chain I (MINF from scratch, prefix cache on, replica 1): mean logprob +0.0006, 95% CI [-0.0004, +0.0018]; per-rollout minimum logprob -0.01, CI [-0.09, +0.06]; tokens below -10 per rollout +0.03, CI [-0.02, +0.09].
- chain J (MINF from scratch, prefix cache on, replica 2): mean logprob +0.0025, 95% CI [+0.0012, +0.0039]; per-rollout minimum logprob -0.01, CI [-0.09, +0.07]; tokens below -10 per rollout +0.01, CI [-0.04, +0.06].
- chain K (vLLM from scratch, seed 1234): mean logprob +0.0016, 95% CI [+0.0005, +0.0028]; per-rollout minimum logprob +0.03, CI [-0.04, +0.11]; tokens below -10 per rollout +0.02, CI [-0.03, +0.07].

## Trainer-minus-engine logprob mismatch d (all generated tokens)

| arm | mean d | mean abs d | share abs d > 0.5 | share > 1 | per-million > 5 | per-million > 10 | rollouts masked (seq error > 2) | seq error p99 |
|---|---|---|---|---|---|---|---|---|
| run A (vLLM from scratch) | -0.00147 | 0.01420 | 0.00146 | 0.00010 | 0.31 | 0.04 | 0.0012 | 1.026 |
| chain G (MINF from scratch, no prefix cache) | -0.00147 | 0.01407 | 0.00147 | 0.00010 | 0.29 | 0.03 | 0.0008 | 1.025 |
| chain I (MINF from scratch, prefix cache on, replica 1) | -0.00148 | 0.01413 | 0.00148 | 0.00010 | 0.34 | 0.04 | 0.0010 | 1.026 |
| chain J (MINF from scratch, prefix cache on, replica 2) | -0.00146 | 0.01394 | 0.00146 | 0.00010 | 0.34 | 0.03 | 0.0008 | 1.026 |
| chain K (vLLM from scratch, seed 1234) | -0.00145 | 0.01395 | 0.00146 | 0.00010 | 0.31 | 0.04 | 0.0012 | 1.026 |

Paired per-prompt mean |d|, other arm minus run A (vLLM from scratch):

- chain G (MINF from scratch, no prefix cache): -0.00006, 95% CI [-0.00015, +0.00004]; higher in 153/320 prompt-steps.
- chain I (MINF from scratch, prefix cache on, replica 1): +0.00003, 95% CI [-0.00006, +0.00013]; higher in 165/320 prompt-steps.
- chain J (MINF from scratch, prefix cache on, replica 2): -0.00008, 95% CI [-0.00020, +0.00003]; higher in 159/320 prompt-steps.
- chain K (vLLM from scratch, seed 1234): -0.00004, 95% CI [-0.00014, +0.00006]; higher in 146/320 prompt-steps.

## Failure modes and generation limits (share of rollouts, 95% Wilson CI)

| metric | run A (vLLM from scratch) | chain G (MINF from scratch, no prefix cache) | chain I (MINF from scratch, prefix cache on, replica 1) | chain J (MINF from scratch, prefix cache on, replica 2) | chain K (vLLM from scratch, seed 1234) | paired diff chain G (MINF from scratch, no prefix cache) minus run A (vLLM from scratch) (95% CI) | paired diff chain I (MINF from scratch, prefix cache on, replica 1) minus run A (vLLM from scratch) (95% CI) | paired diff chain J (MINF from scratch, prefix cache on, replica 2) minus run A (vLLM from scratch) (95% CI) | paired diff chain K (vLLM from scratch, seed 1234) minus run A (vLLM from scratch) (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| context-window truncation (trainer flag) | 0.0123 [0.0096, 0.0157] (n=63) | 0.0143 [0.0114, 0.0179] (n=73) | 0.0141 [0.0112, 0.0177] (n=72) | 0.0166 [0.0134, 0.0205] (n=85) | 0.0152 [0.0122, 0.0190] (n=78) | +0.0020 [-0.0021, +0.0061] | +0.0018 [-0.0027, +0.0066] | +0.0043 [-0.0002, +0.0094] | +0.0029 [-0.0016, +0.0074] |
| context_window error kind (harness) | 0.0281 [0.0239, 0.0330] (n=144) | 0.0365 [0.0317, 0.0420] (n=187) | 0.0332 [0.0286, 0.0385] (n=170) | 0.0346 [0.0299, 0.0399] (n=177) | 0.0318 [0.0274, 0.0370] (n=163) | +0.0084 [+0.0027, +0.0145] | +0.0051 [-0.0006, +0.0111] | +0.0064 [+0.0008, +0.0127] | +0.0037 [-0.0020, +0.0098] |
| hit max turns (200) | 0.0490 [0.0434, 0.0553] (n=251) | 0.0471 [0.0416, 0.0532] (n=241) | 0.0473 [0.0418, 0.0534] (n=242) | 0.0508 [0.0451, 0.0571] (n=260) | 0.0518 [0.0460, 0.0582] (n=265) | -0.0020 [-0.0092, +0.0053] | -0.0018 [-0.0107, +0.0066] | +0.0018 [-0.0074, +0.0102] | +0.0027 [-0.0053, +0.0107] |
| >=1 invalid tool call | 0.0074 [0.0054, 0.0102] (n=38) | 0.0061 [0.0043, 0.0086] (n=31) | 0.0055 [0.0038, 0.0079] (n=28) | 0.0076 [0.0056, 0.0104] (n=39) | 0.0076 [0.0056, 0.0104] (n=39) | -0.0014 [-0.0045, +0.0018] | -0.0020 [-0.0051, +0.0010] | +0.0002 [-0.0029, +0.0031] | +0.0002 [-0.0027, +0.0035] |
| >=1 malformed thinking | 0.0000 [-0.0000, 0.0007] (n=0) | 0.0000 [-0.0000, 0.0007] (n=0) | 0.0000 [-0.0000, 0.0007] (n=0) | 0.0000 [-0.0000, 0.0007] (n=0) | 0.0000 [-0.0000, 0.0007] (n=0) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| >=1 turn without </think> | 0.0059 [0.0041, 0.0084] (n=30) | 0.0064 [0.0046, 0.0090] (n=33) | 0.0074 [0.0054, 0.0102] (n=38) | 0.0064 [0.0046, 0.0090] (n=33) | 0.0082 [0.0061, 0.0111] (n=42) | +0.0006 [-0.0023, +0.0033] | +0.0016 [-0.0012, +0.0045] | +0.0006 [-0.0023, +0.0035] | +0.0023 [-0.0008, +0.0057] |
| >=1 turn without <|im_end|> | 0.0082 [0.0061, 0.0111] (n=42) | 0.0096 [0.0072, 0.0126] (n=49) | 0.0088 [0.0066, 0.0117] (n=45) | 0.0090 [0.0067, 0.0120] (n=46) | 0.0113 [0.0088, 0.0146] (n=58) | +0.0014 [-0.0020, +0.0047] | +0.0006 [-0.0025, +0.0037] | +0.0008 [-0.0027, +0.0045] | +0.0031 [-0.0006, +0.0070] |
| incomplete response | 0.0000 [-0.0000, 0.0007] (n=0) | 0.0000 [-0.0000, 0.0007] (n=0) | 0.0000 [-0.0000, 0.0007] (n=0) | 0.0000 [-0.0000, 0.0007] (n=0) | 0.0000 [-0.0000, 0.0007] (n=0) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

| termination kind | run A (vLLM from scratch) | chain G (MINF from scratch, no prefix cache) | chain I (MINF from scratch, prefix cache on, replica 1) | chain J (MINF from scratch, prefix cache on, replica 2) | chain K (vLLM from scratch, seed 1234) |
|---|---|---|---|---|---|
| context_window | 144 (0.0281) | 187 (0.0365) | 170 (0.0332) | 177 (0.0346) | 163 (0.0318) |
| max_iteration | 254 (0.0496) | 249 (0.0486) | 246 (0.0480) | 261 (0.0510) | 276 (0.0539) |
| none | 4682 (0.9145) | 4659 (0.9100) | 4689 (0.9158) | 4658 (0.9098) | 4648 (0.9078) |
| oom | 12 (0.0023) | 12 (0.0023) | 11 (0.0021) | 13 (0.0025) | 14 (0.0027) |
| other | 10 (0.0020) | 4 (0.0008) | 0 (0.0000) | 2 (0.0004) | 8 (0.0016) |
| stuck_in_loop | 18 (0.0035) | 9 (0.0018) | 4 (0.0008) | 9 (0.0018) | 11 (0.0021) |

- run A (vLLM from scratch): 39 invalid tool calls over 479798 assistant turns (0.08 per 1,000 turns); 47 generations without `</think>` (0.10 per 1,000 turns).
- chain G (MINF from scratch, no prefix cache): 36 invalid tool calls over 483561 assistant turns (0.07 per 1,000 turns); 54 generations without `</think>` (0.11 per 1,000 turns).
- chain I (MINF from scratch, prefix cache on, replica 1): 32 invalid tool calls over 481334 assistant turns (0.07 per 1,000 turns); 64 generations without `</think>` (0.13 per 1,000 turns).
- chain J (MINF from scratch, prefix cache on, replica 2): 39 invalid tool calls over 484824 assistant turns (0.08 per 1,000 turns); 48 generations without `</think>` (0.10 per 1,000 turns).
- chain K (vLLM from scratch, seed 1234): 49 invalid tool calls over 484526 assistant turns (0.10 per 1,000 turns); 56 generations without `</think>` (0.12 per 1,000 turns).


# Engine-pooled view: MINF engine (chain G + chain I + chain J) vs vLLM engine (run A + chain K), steps 1-10

The same eight pages with the from-scratch arms pooled by serving engine: MINF engine = chain G (MINF from scratch, no prefix cache) + chain I (MINF from scratch, prefix cache on, replica 1) + chain J (MINF from scratch, prefix cache on, replica 2), 15,360 rollouts; vLLM engine = run A (vLLM from scratch) + chain K (vLLM from scratch, seed 1234), 10,240 rollouts. Every per-rollout, per-group or per-token histogram is a share (normalized within the pool), so the unequal pool sizes do not affect the shapes; paired statistics pair by (step, SWE instance) with all pooled rollouts of that prompt on each side and are reported as MINF minus vLLM. Trainer metrics from the controller logs are averaged over the pooled arms per step. The per-prompt MAX-length comparison is biased toward the larger pool (48 vs 32 rollouts per prompt-step) and is descriptive only.

## Per-step trainer metrics (controller logs) (mean over the pooled arms)

Cell order in every column: vLLM engine (run A + chain K, both from scratch) / MINF engine (chain G + chain I + chain J, all from scratch).

| step | reward | gen tokens (mean) | truncation rate | gen_kl_error | approx_entropy |
|---|---|---|---|---|---|
| 1 | 0.352 / 0.366 | 26602 / 25685 | 0.010 / 0.010 | 0.00128 / 0.00128 | 0.1539 / 0.1556 |
| 2 | 0.540 / 0.529 | 20854 / 21149 | 0.009 / 0.007 | 0.00143 / 0.00145 | 0.1614 / 0.1622 |
| 3 | 0.255 / 0.253 | 26438 / 26431 | 0.003 / 0.008 | 0.00138 / 0.00139 | 0.1536 / 0.1528 |
| 4 | 0.416 / 0.413 | 27172 / 28355 | 0.019 / 0.017 | 0.00152 / 0.00152 | 0.1530 / 0.1516 |
| 5 | 0.237 / 0.249 | 34063 / 34041 | 0.021 / 0.016 | 0.00149 / 0.00157 | 0.1511 / 0.1523 |
| 6 | 0.367 / 0.383 | 25700 / 25585 | 0.016 / 0.009 | 0.00142 / 0.00144 | 0.1445 / 0.1470 |
| 7 | 0.425 / 0.422 | 26211 / 25292 | 0.016 / 0.014 | 0.00150 / 0.00149 | 0.1580 / 0.1555 |
| 8 | 0.287 / 0.296 | 33429 / 34663 | 0.034 / 0.043 | 0.00144 / 0.00139 | 0.1375 / 0.1329 |
| 9 | 0.327 / 0.307 | 25495 / 25986 | 0.005 / 0.014 | 0.00153 / 0.00156 | 0.1626 / 0.1596 |
| 10 | 0.357 / 0.353 | 24939 / 24952 | 0.006 / 0.012 | 0.00165 / 0.00160 | 0.1610 / 0.1585 |

## Generation length per rollout (steps 1-10 pooled; vLLM engine (run A + chain K, both from scratch) 10,240, MINF engine (chain G + chain I + chain J, all from scratch) 15,360 rollouts)

| arm | mean | p50 | p90 | p95 | p99 | max | share > 50k | share > 100k | share > 150k | truncated (context window) |
|---|---|---|---|---|---|---|---|---|---|---|
| vLLM engine (run A + chain K, both from scratch) | 27090 | 22735 | 50964 | 59444 | 82630 | 178120 | 0.107 | 0.007 | 0.002 | 0.014 |
| MINF engine (chain G + chain I + chain J, all from scratch) | 27214 | 23022 | 51259 | 59677 | 81027 | 185819 | 0.110 | 0.007 | 0.002 | 0.015 |

Paired by prompt, other arm minus vLLM engine (run A + chain K, both from scratch) (n = matched prompt-steps):

- MINF engine (chain G + chain I + chain J, all from scratch): mean length +124 tokens, 95% CI [-182, +460] (n=320; longer in 168, shorter in 152); per step: s1 -917, s2 +296, s3 -7, s4 +1183, s5 -22, s6 -116, s7 -919, s8 +1234, s9 +491, s10 +14; per-prompt MAX length +3449 [+378, +6753]; KS distance 0.007, paired-permutation p = 0.936.

## Reward and group structure

| arm | mean reward | mean within-group std | groups all-fail | groups all-pass | mixed groups | mean adv. (trainer) |
|---|---|---|---|---|---|---|
| vLLM engine (run A + chain K, both from scratch) | 0.356 | 0.177 | 0.408 | 0.119 | 0.473 | -0.0419 |
| MINF engine (chain G + chain I + chain J, all from scratch) | 0.357 | 0.176 | 0.405 | 0.125 | 0.470 | -0.0372 |

Paired pass-rate and within-group-std differences, other arm minus vLLM engine (run A + chain K, both from scratch):

- MINF engine (chain G + chain I + chain J, all from scratch): pass rate +0.0007, 95% CI [-0.0057, +0.0072] (higher in 90, lower in 93, equal in 137 of 320 prompt-steps); within-group reward std +0.0019, 95% CI [-0.0066, +0.0107].

## `</think>` close probability (engine logprob of the sampled close)

| arm | closes | geo-mean P | 10th-pct P | 1st-pct P | share P<0.905 | share P<0.5 | share P<0.1 | turns without </think> |
|---|---|---|---|---|---|---|---|---|
| vLLM engine (run A + chain K, both from scratch) | 964133 | 0.9330 | 0.884 | 0.244 | 0.1112 | 0.0275 | 0.00369 | 0.00020 |
| MINF engine (chain G + chain I + chain J, all from scratch) | 1449416 | 0.9335 | 0.886 | 0.245 | 0.1106 | 0.0272 | 0.00360 | 0.00021 |

Paired per-prompt mean close logprob, other arm minus vLLM engine (run A + chain K, both from scratch):

- MINF engine (chain G + chain I + chain J, all from scratch): +0.0000 (multiplicative P ratio 1.0000), 95% CI [-0.0009, +0.0010]; higher in 153, lower in 167 of 320 prompt-steps; per step: s1 +0.0008, s2 -0.0041, s3 +0.0010, s4 -0.0004, s5 -0.0005, s6 +0.0007, s7 -0.0008, s8 +0.0007, s9 +0.0009, s10 +0.0019.

## Thinking tokens per assistant turn

| arm | turns | mean | p50 | p90 | p95 | p99 | max | share > 1k | share > 2k | share > 5k | turn tokens p50 / p99 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vLLM engine (run A + chain K, both from scratch) | 964133 | 141 | 25 | 339 | 639 | 1821 | 114653 | 0.0257 | 0.0085 | 0.00117 | 106 / 2214 / 169118 |
| MINF engine (chain G + chain I + chain J, all from scratch) | 1449416 | 140 | 25 | 335 | 634 | 1831 | 123816 | 0.0256 | 0.0086 | 0.00122 | 106 / 2237 / 184380 |

Paired per-prompt thinking length, other arm minus vLLM engine (run A + chain K, both from scratch):

- MINF engine (chain G + chain I + chain J, all from scratch): mean thinking tokens per turn +0.6, 95% CI [-1.8, +3.5] (higher in 154/320); per-rollout longest thinking span +50, 95% CI [-9, +118].

## Engine logprob of sampled tokens (all generated tokens, steps 1-10)

| arm | tokens | mean logprob | share < -1 | share < -5 | per-million < -10 | per-million < -15 | per-million < -20 |
|---|---|---|---|---|---|---|---|
| vLLM engine (run A + chain K, both from scratch) | 277,404,190 | -0.1529 | 0.0479 | 0.00167 | 48.4 | 2.12 | 0.00 |
| MINF engine (chain G + chain I + chain J, all from scratch) | 418,006,920 | -0.1518 | 0.0475 | 0.00166 | 48.4 | 2.16 | 0.00 |

Paired per-prompt engine-logprob statistics, other arm minus vLLM engine (run A + chain K, both from scratch):

- MINF engine (chain G + chain I + chain J, all from scratch): mean logprob +0.0007, 95% CI [-0.0001, +0.0014]; per-rollout minimum logprob -0.03, CI [-0.07, +0.02]; tokens below -10 per rollout +0.01, CI [-0.02, +0.04].

## Trainer-minus-engine logprob mismatch d (all generated tokens)

| arm | mean d | mean abs d | share abs d > 0.5 | share > 1 | per-million > 5 | per-million > 10 | rollouts masked (seq error > 2) | seq error p99 |
|---|---|---|---|---|---|---|---|---|
| vLLM engine (run A + chain K, both from scratch) | -0.00146 | 0.01407 | 0.00146 | 0.00010 | 0.31 | 0.04 | 0.0012 | 1.026 |
| MINF engine (chain G + chain I + chain J, all from scratch) | -0.00147 | 0.01405 | 0.00147 | 0.00010 | 0.33 | 0.03 | 0.0008 | 1.026 |

Paired per-prompt mean |d|, other arm minus vLLM engine (run A + chain K, both from scratch):

- MINF engine (chain G + chain I + chain J, all from scratch): -0.00002, 95% CI [-0.00009, +0.00005]; higher in 160/320 prompt-steps.

## Failure modes and generation limits (share of rollouts, 95% Wilson CI)

| metric | vLLM engine (run A + chain K, both from scratch) | MINF engine (chain G + chain I + chain J, all from scratch) | paired diff MINF engine (chain G + chain I + chain J, all from scratch) minus vLLM engine (run A + chain K, both from scratch) (95% CI) |
|---|---|---|---|
| context-window truncation (trainer flag) | 0.0138 [0.0117, 0.0162] (n=141) | 0.0150 [0.0132, 0.0170] (n=230) | +0.0012 [-0.0015, +0.0040] |
| context_window error kind (harness) | 0.0300 [0.0268, 0.0335] (n=307) | 0.0348 [0.0320, 0.0378] (n=534) | +0.0048 [+0.0009, +0.0087] |
| hit max turns (200) | 0.0504 [0.0463, 0.0548] (n=516) | 0.0484 [0.0451, 0.0519] (n=743) | -0.0020 [-0.0076, +0.0035] |
| >=1 invalid tool call | 0.0075 [0.0060, 0.0094] (n=77) | 0.0064 [0.0052, 0.0078] (n=98) | -0.0011 [-0.0031, +0.0008] |
| >=1 malformed thinking | 0.0000 [-0.0000, 0.0004] (n=0) | 0.0000 [-0.0000, 0.0003] (n=0) | +0.0000 [+0.0000, +0.0000] |
| >=1 turn without </think> | 0.0070 [0.0056, 0.0088] (n=72) | 0.0068 [0.0056, 0.0082] (n=104) | -0.0003 [-0.0023, +0.0017] |
| >=1 turn without <|im_end|> | 0.0098 [0.0080, 0.0119] (n=100) | 0.0091 [0.0077, 0.0107] (n=140) | -0.0007 [-0.0029, +0.0016] |
| incomplete response | 0.0000 [-0.0000, 0.0004] (n=0) | 0.0000 [-0.0000, 0.0003] (n=0) | +0.0000 [+0.0000, +0.0000] |

| termination kind | vLLM engine (run A + chain K, both from scratch) | MINF engine (chain G + chain I + chain J, all from scratch) |
|---|---|---|
| context_window | 307 (0.0300) | 534 (0.0348) |
| max_iteration | 530 (0.0518) | 756 (0.0492) |
| none | 9330 (0.9111) | 14006 (0.9118) |
| oom | 26 (0.0025) | 36 (0.0023) |
| other | 18 (0.0018) | 6 (0.0004) |
| stuck_in_loop | 29 (0.0028) | 22 (0.0014) |

- vLLM engine (run A + chain K, both from scratch): 88 invalid tool calls over 964324 assistant turns (0.09 per 1,000 turns); 103 generations without `</think>` (0.11 per 1,000 turns).
- MINF engine (chain G + chain I + chain J, all from scratch): 107 invalid tool calls over 1449719 assistant turns (0.07 per 1,000 turns); 166 generations without `</think>` (0.11 per 1,000 turns).


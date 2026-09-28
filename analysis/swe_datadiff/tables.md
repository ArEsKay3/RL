# Advantage exposure tables (2026-09-26)

Runs: VLLM = the four vLLM-generation runs from scratch plus, at the user's request, the one MINF-generation run from scratch that stayed healthy (row flagged); Minf = the two MINF-generation runs from scratch that later looped (prefix cache invalidated at every weight update, or no prefix cache); Minf (no cache invalidation) = one MINF run from scratch with the prefix cache kept across weight updates. Same prompts per step in every run (320 prompt groups of 16 per 10 steps).
E(class) = 1000 x sum over turns in the class of (advantage x turn tokens) / all generated tokens in the window. Negative = the batch gradient suppresses that class of tokens on net. Brackets = 95 % group-bootstrap interval. Classes: think turns of 4k-16k tokens; near-repetitive think turns (>= 1k think tokens, zlib ratio of the think text < 0.25); long turns (>= 4k tokens) that start 16k-32k tokens into the context.

## Table 1. Advantage exposure, training steps 1-10

| run | E(think turns 4k-16k) | E(near-repetitive think) | E(long turns starting 16k-32k into context) |
|---|---|---|---|
| VLLM | -4.20 [-7.11, -1.47] | -11.03 [-16.93, -5.95] | +0.10 [-0.74, +0.95] |
| VLLM | -2.33 [-4.84, -0.05] | -9.26 [-14.19, -4.85] | -0.18 [-1.02, +0.66] |
| VLLM | -3.88 [-6.85, -1.12] | -11.90 [-19.32, -5.37] | -0.35 [-0.90, +0.17] |
| VLLM | -4.77 [-7.72, -2.17] | -13.43 [-22.13, -6.90] | -0.54 [-1.32, +0.14] |
| Minf | -0.09 [-2.23, +2.13] | -8.60 [-14.96, -3.15] | +0.65 [+0.02, +1.41] |
| Minf | -0.82 [-3.72, +2.12] | -6.62 [-12.36, -1.34] | -0.56 [-1.43, +0.22] |
| VLLM (the healthy Minf run) | -3.44 [-5.82, -1.22] | -13.44 [-23.30, -6.42] | -0.21 [-0.92, +0.42] |
| Minf (no cache invalidation), seed 42 | -3.18 [-5.57, -0.88] | -10.92 [-17.80, -5.15] | +0.11 [-0.75, +0.85] |
| VLLM (no prefix caching) | -2.47 [-4.92, +0.24] | -10.84 [-19.52, -4.40] | -0.83 [-1.72, -0.06] |
| Minf (no cache invalidation), seed 1234 | -3.48 [-6.92, -0.27] | -10.42 [-16.02, -5.79] | -0.62 [-1.50, +0.16] |
| | | | |
| **VLLM, pooled** | -3.72 [-4.92, -2.49] | -11.81 [-15.09, -8.94] | -0.24 [-0.57, +0.09] |
| **Minf, pooled** | -0.46 [-2.18, +1.40] | -7.61 [-11.91, -3.79] | +0.05 [-0.50, +0.59] |
| **Minf (no cache invalidation), pooled** | -3.33 [-5.30, -1.29] | -10.67 [-15.04, -6.79] | -0.25 [-0.83, +0.30] |
| **VLLM (no prefix caching), pooled** | -2.47 [-5.01, +0.17] | -10.84 [-19.97, -4.14] | -0.83 [-1.66, -0.10] |

Separation, steps 1-10 (each pooled treatment minus pooled VLLM; z = difference / bootstrap se):

| class | Minf - VLLM | z | Minf (no cache invalidation) - VLLM | z | VLLM (no prefix caching) - VLLM | z |
|---|---|---|---|---|---|---|
| E(think turns 4k-16k) | +3.26 | +2.9 | +0.39 | +0.3 | +1.25 | +0.8 |
| E(near-repetitive think) | +4.20 | +1.6 | +1.14 | +0.4 | +0.97 | +0.2 |
| E(long turns starting 16k-32k into context) | +0.28 | +0.9 | -0.02 | -0.0 | -0.59 | -1.3 |

## Table 3. The same classes under three weightings, steps 1-10

Advantage = what the trainer uses: leave-one-out group normalization, (r_i - mean of the other 15) / std of the other 15. Centered reward = r_i - group mean, no normalization. Raw reward = share of the class's tokens that sit in rewarded (r = 1) rollouts, no group information at all. Relative gap = (Minf - VLLM) / |VLLM|.

### E(think turns 4k-16k)

| run | advantage-weighted E | centered-reward E | raw reward share |
|---|---|---|---|
| VLLM | -4.20 [-7.11, -1.47] | -1.68 [-2.78, -0.69] | 0.133 [0.090, 0.183] |
| VLLM | -2.33 [-4.84, -0.05] | -0.59 [-1.59, +0.33] | 0.156 [0.110, 0.210] |
| VLLM | -3.88 [-6.85, -1.12] | -1.64 [-2.88, -0.52] | 0.169 [0.121, 0.225] |
| VLLM | -4.77 [-7.72, -2.17] | -1.64 [-2.88, -0.59] | 0.146 [0.106, 0.193] |
| Minf | -0.09 [-2.23, +2.13] | +0.05 [-0.79, +0.88] | 0.179 [0.131, 0.233] |
| Minf | -0.82 [-3.72, +2.12] | -0.49 [-1.52, +0.53] | 0.170 [0.119, 0.234] |
| VLLM (the healthy Minf run) | -3.44 [-5.82, -1.22] | -1.15 [-2.09, -0.33] | 0.143 [0.105, 0.189] |
| Minf (no cache invalidation), seed 42 | -3.18 [-5.57, -0.88] | -0.98 [-1.99, -0.05] | 0.160 [0.112, 0.216] |
| VLLM (no prefix caching) | -2.47 [-4.92, +0.24] | -0.92 [-1.77, +0.02] | 0.136 [0.095, 0.189] |
| Minf (no cache invalidation), seed 1234 | -3.48 [-6.92, -0.27] | -1.45 [-2.77, -0.33] | 0.150 [0.109, 0.200] |
| | | | |
| **VLLM, pooled** | -3.72 [-4.92, -2.49] | -1.34 [-1.84, -0.86] | 0.149 [0.129, 0.170] |
| **Minf, pooled** | -0.46 [-2.18, +1.40] | -0.22 [-0.88, +0.47] | 0.174 [0.137, 0.215] |
| **Minf (no cache invalidation), pooled** | -3.33 [-5.30, -1.29] | -1.21 [-2.02, -0.45] | 0.155 [0.122, 0.191] |
| **VLLM (no prefix caching), pooled** | -2.47 [-5.01, +0.17] | -0.92 [-1.79, -0.03] | 0.136 [0.095, 0.190] |
| | | | |
| **Minf - VLLM by weighting** | difference | z | relative gap |
| advantage | +3.262 | +2.9 | +88 % |
| centered reward | +1.119 | +2.7 | +84 % |
| raw reward share | +0.025 | +1.1 | +17 % |
| | | | |
| **Minf (no cache invalidation) - VLLM by weighting** | difference | z | relative gap |
| advantage | +0.391 | +0.3 | +11 % |
| centered reward | +0.129 | +0.3 | +10 % |
| raw reward share | +0.006 | +0.3 | +4 % |
| | | | |
| **VLLM (no prefix caching) - VLLM by weighting** | difference | z | relative gap |
| advantage | +1.247 | +0.8 | +34 % |
| centered reward | +0.423 | +0.8 | +32 % |
| raw reward share | -0.013 | -0.5 | -9 % |

### E(near-repetitive think)

| run | advantage-weighted E | centered-reward E | raw reward share |
|---|---|---|---|
| VLLM | -11.03 [-16.93, -5.95] | -4.60 [-6.86, -2.62] | 0.086 [0.058, 0.121] |
| VLLM | -9.26 [-14.19, -4.85] | -3.70 [-6.27, -1.59] | 0.096 [0.062, 0.144] |
| VLLM | -11.90 [-19.32, -5.37] | -5.06 [-8.22, -2.33] | 0.103 [0.067, 0.150] |
| VLLM | -13.43 [-22.13, -6.90] | -5.41 [-9.51, -2.45] | 0.081 [0.054, 0.118] |
| Minf | -8.60 [-14.96, -3.15] | -3.53 [-6.23, -1.19] | 0.115 [0.082, 0.161] |
| Minf | -6.62 [-12.36, -1.34] | -2.77 [-4.96, -0.75] | 0.116 [0.072, 0.175] |
| VLLM (the healthy Minf run) | -13.44 [-23.30, -6.42] | -5.72 [-10.28, -2.45] | 0.095 [0.062, 0.141] |
| Minf (no cache invalidation), seed 42 | -10.92 [-17.80, -5.15] | -4.31 [-7.46, -1.82] | 0.097 [0.063, 0.140] |
| VLLM (no prefix caching) | -10.84 [-19.52, -4.40] | -4.26 [-7.98, -1.58] | 0.084 [0.055, 0.127] |
| Minf (no cache invalidation), seed 1234 | -10.42 [-16.02, -5.79] | -4.01 [-6.37, -2.11] | 0.098 [0.066, 0.142] |
| | | | |
| **VLLM, pooled** | -11.81 [-15.09, -8.94] | -4.90 [-6.41, -3.61] | 0.093 [0.077, 0.111] |
| **Minf, pooled** | -7.61 [-11.91, -3.79] | -3.15 [-4.92, -1.60] | 0.116 [0.087, 0.153] |
| **Minf (no cache invalidation), pooled** | -10.67 [-15.04, -6.79] | -4.16 [-6.08, -2.53] | 0.098 [0.073, 0.128] |
| **VLLM (no prefix caching), pooled** | -10.84 [-19.97, -4.14] | -4.26 [-8.03, -1.51] | 0.084 [0.054, 0.127] |
| | | | |
| **Minf - VLLM by weighting** | difference | z | relative gap |
| advantage | +4.204 | +1.6 | +36 % |
| centered reward | +1.750 | +1.6 | +36 % |
| raw reward share | +0.023 | +1.2 | +25 % |
| | | | |
| **Minf (no cache invalidation) - VLLM by weighting** | difference | z | relative gap |
| advantage | +1.145 | +0.4 | +10 % |
| centered reward | +0.737 | +0.6 | +15 % |
| raw reward share | +0.005 | +0.3 | +5 % |
| | | | |
| **VLLM (no prefix caching) - VLLM by weighting** | difference | z | relative gap |
| advantage | +0.968 | +0.2 | +8 % |
| centered reward | +0.638 | +0.3 | +13 % |
| raw reward share | -0.009 | -0.4 | -10 % |

### E(long turns starting 16k-32k into context)

| run | advantage-weighted E | centered-reward E | raw reward share |
|---|---|---|---|
| VLLM | +0.10 [-0.74, +0.95] | +0.09 [-0.25, +0.44] | 0.255 [0.155, 0.376] |
| VLLM | -0.18 [-1.02, +0.66] | -0.03 [-0.41, +0.33] | 0.200 [0.101, 0.346] |
| VLLM | -0.35 [-0.90, +0.17] | -0.12 [-0.35, +0.10] | 0.229 [0.139, 0.339] |
| VLLM | -0.54 [-1.32, +0.14] | -0.15 [-0.41, +0.07] | 0.106 [0.051, 0.225] |
| Minf | +0.65 [+0.02, +1.41] | +0.20 [-0.02, +0.44] | 0.232 [0.124, 0.411] |
| Minf | -0.56 [-1.43, +0.22] | -0.23 [-0.61, +0.13] | 0.262 [0.171, 0.370] |
| VLLM (the healthy Minf run) | -0.21 [-0.92, +0.42] | +0.05 [-0.23, +0.35] | 0.232 [0.144, 0.347] |
| Minf (no cache invalidation), seed 42 | +0.11 [-0.75, +0.85] | +0.05 [-0.34, +0.38] | 0.179 [0.093, 0.338] |
| VLLM (no prefix caching) | -0.83 [-1.72, -0.06] | -0.33 [-0.72, +0.01] | 0.182 [0.107, 0.283] |
| Minf (no cache invalidation), seed 1234 | -0.62 [-1.50, +0.16] | -0.22 [-0.58, +0.09] | 0.209 [0.126, 0.316] |
| | | | |
| **VLLM, pooled** | -0.24 [-0.57, +0.09] | -0.03 [-0.16, +0.10] | 0.195 [0.146, 0.251] |
| **Minf, pooled** | +0.05 [-0.50, +0.59] | -0.01 [-0.24, +0.20] | 0.246 [0.164, 0.344] |
| **Minf (no cache invalidation), pooled** | -0.25 [-0.83, +0.30] | -0.09 [-0.34, +0.15] | 0.192 [0.125, 0.287] |
| **VLLM (no prefix caching), pooled** | -0.83 [-1.66, -0.10] | -0.33 [-0.71, +0.01] | 0.182 [0.109, 0.278] |
| | | | |
| **Minf - VLLM by weighting** | difference | z | relative gap |
| advantage | +0.282 | +0.9 | +119 % |
| centered reward | +0.019 | +0.1 | +61 % |
| raw reward share | +0.052 | +0.9 | +27 % |
| | | | |
| **Minf (no cache invalidation) - VLLM by weighting** | difference | z | relative gap |
| advantage | -0.016 | -0.0 | -7 % |
| centered reward | -0.055 | -0.4 | -178 % |
| raw reward share | -0.002 | -0.0 | -1 % |
| | | | |
| **VLLM (no prefix caching) - VLLM by weighting** | difference | z | relative gap |
| advantage | -0.594 | -1.3 | -251 % |
| centered reward | -0.294 | -1.5 | -955 % |
| raw reward share | -0.012 | -0.2 | -6 % |

## Table 4. Is the suppression concentrated in steps whose rollouts ran past a weight update? (steps 1-10, E(think turns 4k-16k))

For each run, steps are split by that run's own rollouts: 'straddled' = the step's rollouts started under one weight version and finished under the next (their later turns were generated after the refit; in the VLLM and no-invalidation runs those turns reuse prefix KV from the older weights). Because the prompt set differs by step, each run's value is paired against the pooled Minf runs at the same steps (Minf recomputes the prefix after every update, so straddling has no cache effect there).

| run | straddled steps | E, straddled | E, not straddled | Minf at same steps: straddled / not | run - Minf, straddled | run - Minf, not straddled |
|---|---|---|---|---|---|---|
| VLLM | 3, 5, 7, 9 | -3.60 | -4.49 | +2.50 / -2.12 | -6.10 | -2.37 |
| VLLM | 3, 5, 7, 9 | +1.36 | -4.68 | +2.50 / -2.12 | -1.14 | -2.56 |
| VLLM | 3, 5, 7, 9 | -2.29 | -4.56 | +2.50 / -2.12 | -4.79 | -2.44 |
| VLLM | 3, 5, 7, 9 | -2.65 | -6.12 | +2.50 / -2.12 | -5.15 | -4.00 |
| VLLM (the healthy Minf run) | 3, 5, 7, 9 | -1.82 | -4.92 | +2.50 / -2.12 | -4.32 | -2.80 |
| Minf (no cache invalidation), seed 42 | 3, 6 | -5.28 | -2.85 | -1.62 / +0.06 | -3.67 | -2.91 |
| VLLM (no prefix caching) | 3, 5, 7, 9 | -1.92 | -2.79 | +2.50 / -2.12 | -4.42 | -0.67 |
| Minf (no cache invalidation), seed 1234 | 3, 7, 9 | -3.41 | -3.85 | +2.88 / -1.63 | -6.29 | -2.22 |
| **VLLM, mean over runs** | | | | | -4.30 | -2.83 |
| **Minf (no cache invalidation), mean over runs** | | | | | -4.98 | -2.57 |
| **VLLM (no prefix caching), mean over runs** | | | | | -4.42 | -0.67 |


## Table 5. Turns generated before vs after the weight update inside straddled steps (steps 1-10; seed-42 runs only)

| treatment | mismatch |d|, pre | mismatch |d|, post | mismatch |d|, steps with no refit | E(think 4k-16k), pre | E(think 4k-16k), post | E(near-rep), pre | E(near-rep), post |
|---|---|---|---|---|---|---|---|
| VLLM | 13.27 | 12.08 | 12.38 | -1.06 | -1.12 | -1.37 | -2.91 |
| Minf | 12.68 | 12.18 | 12.35 | -0.50 | +0.58 | -1.46 | +0.20 |
| Minf (no cache invalidation) | 12.80 | 11.90 | 12.49 | -1.73 | -0.86 | -1.16 | -1.21 |
| VLLM (no prefix caching) | 12.60 | 11.99 | 12.47 | -1.34 | -0.18 | -1.32 | -1.33 |

See refit_split_report.txt; no per-update cliff in any treatment (post-update mismatch identical with and without a prefix cache).

## Table 2. Advantage exposure, training steps 11-20

| run | E(think turns 4k-16k) | E(near-repetitive think) | E(long turns starting 16k-32k into context) |
|---|---|---|---|
| VLLM | -3.31 [-6.11, -0.59] | -8.85 [-13.77, -4.56] | +0.47 [-0.27, +1.31] |
| VLLM | -2.92 [-5.74, -0.37] | -5.83 [-9.75, -2.22] | -0.45 [-1.02, +0.02] |
| VLLM | -1.22 [-4.12, +1.91] | -6.75 [-11.25, -3.07] | -0.36 [-1.31, +0.39] |
| VLLM | -1.20 [-4.13, +1.74] | -3.87 [-7.58, -0.38] | -0.61 [-1.68, +0.27] |
| Minf | -0.98 [-4.29, +2.12] | -16.83 [-24.61, -9.43] | -0.61 [-1.50, +0.16] |
| Minf | -5.29 [-8.62, -1.96] | -12.73 [-19.34, -6.82] | +1.56 [+0.37, +3.06] |
| VLLM (the healthy Minf run) | -3.96 [-6.00, -1.96] | -10.18 [-15.54, -5.55] | +0.22 [-0.18, +0.68] |
| Minf (no cache invalidation), seed 42 | -4.20 [-7.04, -1.41] | -5.19 [-8.17, -2.47] | -0.54 [-1.33, +0.23] |
| VLLM (no prefix caching) | -0.26 [-3.11, +2.55] | -12.30 [-21.45, -4.72] | -0.25 [-0.95, +0.41] |
| Minf (no cache invalidation), seed 1234 | -2.60 [-5.80, +0.34] | -4.77 [-9.38, -1.03] | -0.34 [-1.29, +0.51] |
| | | | |
| **VLLM, pooled** | -2.56 [-3.81, -1.32] | -7.17 [-9.16, -5.36] | -0.14 [-0.47, +0.17] |
| **Minf, pooled** | -3.13 [-5.37, -0.94] | -14.78 [-19.86, -9.96] | +0.47 [-0.26, +1.29] |
| **Minf (no cache invalidation), pooled** | -3.82 [-6.09, -1.63] | -5.09 [-7.50, -2.82] | -0.49 [-1.14, +0.14] |
| **VLLM (no prefix caching), pooled** | -0.26 [-2.98, +2.54] | -12.30 [-21.16, -4.96] | -0.25 [-0.95, +0.36] |


# Uncertainty of the sampled-</think> probability statistics

Bootstrap draws: 4000. geo-mean P = exp(mean engine logprob of the sampled </think> tokens in the step's trained rollouts). low share = fraction of those tokens with engine logprob < -0.1 (P < 0.905). SE naive = token-level sd/sqrt(n) treating every close as independent. SE rollout = cluster bootstrap over rollouts (about 512). SE prompt = cluster bootstrap over prompts (about 32; 16 rollouts each). Design effect = (SE prompt / SE naive)^2. Paired differences match prompts across arms by SWE instance id at the same train step and bootstrap over matched prompts.

## Per run and step

| run | step | prompts | rollouts | closes | geo-mean P | SE naive (P) | SE rollout (P) | SE prompt (P) | 95% CI prompt | design effect | low share | SE prompt (low) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run A (vLLM from scratch) | 11 | 32 | 512 | 48502 | 0.9302 | 0.0013 | 0.0014 | 0.0021 | 0.9261-0.9343 | 3 | 11.49% | 0.31 |
| run A (vLLM from scratch) | 12 | 32 | 512 | 46185 | 0.9328 | 0.0013 | 0.0015 | 0.0022 | 0.9284-0.9370 | 3 | 11.21% | 0.38 |
| run A (vLLM from scratch) | 13 | 32 | 512 | 55632 | 0.9326 | 0.0011 | 0.0014 | 0.0024 | 0.9277-0.9372 | 4 | 11.31% | 0.38 |
| run A (vLLM from scratch) | 14 | 32 | 512 | 42395 | 0.9295 | 0.0013 | 0.0015 | 0.0020 | 0.9254-0.9332 | 2 | 11.67% | 0.35 |
| run A (vLLM from scratch) | 15 | 32 | 512 | 51338 | 0.9315 | 0.0012 | 0.0014 | 0.0021 | 0.9274-0.9354 | 3 | 11.30% | 0.36 |
| run A (vLLM from scratch) | 16 | 32 | 512 | 51226 | 0.9302 | 0.0012 | 0.0014 | 0.0023 | 0.9257-0.9347 | 4 | 11.65% | 0.36 |
| run A (vLLM from scratch) | 17 | 32 | 512 | 41809 | 0.9235 | 0.0014 | 0.0015 | 0.0024 | 0.9188-0.9282 | 3 | 12.99% | 0.41 |
| run A (vLLM from scratch) | 18 | 32 | 512 | 44959 | 0.9305 | 0.0012 | 0.0015 | 0.0027 | 0.9252-0.9356 | 5 | 11.74% | 0.40 |
| run A (vLLM from scratch) | 19 | 32 | 512 | 46057 | 0.9241 | 0.0013 | 0.0015 | 0.0020 | 0.9202-0.9282 | 2 | 12.78% | 0.31 |
| run A (vLLM from scratch) | 20 | 32 | 512 | 45245 | 0.9246 | 0.0013 | 0.0015 | 0.0019 | 0.9208-0.9283 | 2 | 12.72% | 0.29 |
| run A (vLLM from scratch) | 21 | 32 | 512 | 47511 | 0.9215 | 0.0014 | 0.0015 | 0.0028 | 0.9161-0.9272 | 4 | 12.65% | 0.44 |
| run A (vLLM from scratch) | 22 | 32 | 512 | 39622 | 0.9214 | 0.0015 | 0.0017 | 0.0020 | 0.9175-0.9253 | 2 | 13.05% | 0.30 |
| run A (vLLM from scratch) | 23 | 32 | 512 | 44187 | 0.9204 | 0.0014 | 0.0016 | 0.0024 | 0.9157-0.9249 | 3 | 13.26% | 0.36 |
| run A (vLLM from scratch) | 24 | 31 | 496 | 40922 | 0.9177 | 0.0015 | 0.0017 | 0.0031 | 0.9110-0.9232 | 4 | 13.54% | 0.47 |
| run B (MINF from MINF step 10) | 11 | 32 | 512 | 51132 | 0.9273 | 0.0012 | 0.0015 | 0.0025 | 0.9224-0.9321 | 4 | 12.24% | 0.42 |
| run B (MINF from MINF step 10) | 12 | 32 | 512 | 48540 | 0.9319 | 0.0012 | 0.0014 | 0.0023 | 0.9274-0.9365 | 4 | 11.40% | 0.34 |
| run B (MINF from MINF step 10) | 13 | 32 | 512 | 58262 | 0.9303 | 0.0011 | 0.0014 | 0.0020 | 0.9266-0.9342 | 3 | 11.64% | 0.33 |
| run B (MINF from MINF step 10) | 14 | 32 | 512 | 45497 | 0.9293 | 0.0013 | 0.0015 | 0.0025 | 0.9244-0.9339 | 3 | 11.82% | 0.37 |
| run B (MINF from MINF step 10) | 15 | 32 | 512 | 53951 | 0.9253 | 0.0012 | 0.0014 | 0.0023 | 0.9207-0.9297 | 4 | 12.24% | 0.40 |
| run B (MINF from MINF step 10) | 16 | 32 | 512 | 55002 | 0.9288 | 0.0012 | 0.0015 | 0.0029 | 0.9232-0.9349 | 6 | 11.91% | 0.44 |
| run B (MINF from MINF step 10) | 17 | 32 | 512 | 45868 | 0.9159 | 0.0014 | 0.0017 | 0.0029 | 0.9103-0.9215 | 4 | 13.74% | 0.37 |
| run B (MINF from MINF step 10) | 18 | 32 | 512 | 47257 | 0.9204 | 0.0014 | 0.0016 | 0.0025 | 0.9154-0.9251 | 3 | 13.01% | 0.36 |
| run B (MINF from MINF step 10) | 19 | 32 | 512 | 49821 | 0.9155 | 0.0014 | 0.0016 | 0.0023 | 0.9109-0.9201 | 3 | 13.93% | 0.32 |
| run B (MINF from MINF step 10) | 20 | 32 | 512 | 51123 | 0.9149 | 0.0013 | 0.0017 | 0.0025 | 0.9103-0.9202 | 3 | 14.03% | 0.40 |
| run B (MINF from MINF step 10) | 21 | 32 | 512 | 52834 | 0.9201 | 0.0013 | 0.0016 | 0.0029 | 0.9141-0.9255 | 5 | 13.43% | 0.48 |
| run B (MINF from MINF step 10) | 22 | 32 | 512 | 43391 | 0.9114 | 0.0015 | 0.0017 | 0.0023 | 0.9068-0.9158 | 2 | 14.80% | 0.36 |
| run B (MINF from MINF step 10) | 23 | 32 | 512 | 47982 | 0.9144 | 0.0014 | 0.0016 | 0.0023 | 0.9099-0.9190 | 3 | 14.13% | 0.37 |
| run B (MINF from MINF step 10) | 24 | 32 | 512 | 46322 | 0.9123 | 0.0014 | 0.0018 | 0.0028 | 0.9065-0.9177 | 4 | 14.56% | 0.39 |
| run B (MINF from MINF step 10) | 25 | 32 | 512 | 44055 | 0.9058 | 0.0015 | 0.0018 | 0.0028 | 0.9001-0.9110 | 3 | 15.59% | 0.44 |
| run B (MINF from MINF step 10) | 26 | 32 | 512 | 43233 | 0.9022 | 0.0016 | 0.0018 | 0.0023 | 0.8977-0.9066 | 2 | 15.91% | 0.35 |
| run B (MINF from MINF step 10) | 27 | 32 | 512 | 47536 | 0.9040 | 0.0015 | 0.0018 | 0.0026 | 0.8990-0.9088 | 3 | 15.86% | 0.42 |
| run B (MINF from MINF step 10) | 28 | 32 | 512 | 44477 | 0.9006 | 0.0016 | 0.0019 | 0.0036 | 0.8937-0.9075 | 5 | 16.44% | 0.56 |
| run B (MINF from MINF step 10) | 29 | 16 | 256 | 18473 | 0.9005 | 0.0023 | 0.0027 | 0.0046 | 0.8918-0.9098 | 4 | 16.38% | 0.77 |
| chain D (vLLM from MINF step 10) | 11 | 32 | 512 | 51998 | 0.9237 | 0.0013 | 0.0014 | 0.0021 | 0.9195-0.9280 | 3 | 12.51% | 0.36 |
| chain D (vLLM from MINF step 10) | 12 | 32 | 512 | 46783 | 0.9329 | 0.0013 | 0.0015 | 0.0022 | 0.9285-0.9373 | 3 | 11.33% | 0.36 |
| chain D (vLLM from MINF step 10) | 13 | 32 | 512 | 55434 | 0.9288 | 0.0012 | 0.0015 | 0.0021 | 0.9247-0.9329 | 3 | 11.71% | 0.35 |
| chain D (vLLM from MINF step 10) | 14 | 32 | 512 | 46431 | 0.9259 | 0.0013 | 0.0017 | 0.0025 | 0.9211-0.9309 | 4 | 11.97% | 0.33 |
| chain D (vLLM from MINF step 10) | 15 | 32 | 512 | 53431 | 0.9250 | 0.0013 | 0.0015 | 0.0021 | 0.9207-0.9286 | 3 | 12.40% | 0.33 |
| chain D (vLLM from MINF step 10) | 16 | 32 | 512 | 56276 | 0.9286 | 0.0012 | 0.0015 | 0.0026 | 0.9237-0.9336 | 5 | 12.02% | 0.39 |
| chain D (vLLM from MINF step 10) | 17 | 32 | 512 | 46438 | 0.9200 | 0.0013 | 0.0016 | 0.0027 | 0.9144-0.9251 | 4 | 13.40% | 0.42 |
| chain D (vLLM from MINF step 10) | 18 | 32 | 512 | 50183 | 0.9287 | 0.0012 | 0.0015 | 0.0025 | 0.9237-0.9336 | 4 | 11.92% | 0.39 |
| chain D (vLLM from MINF step 10) | 19 | 32 | 512 | 54373 | 0.9229 | 0.0013 | 0.0015 | 0.0019 | 0.9192-0.9266 | 2 | 12.79% | 0.30 |
| chain D (vLLM from MINF step 10) | 20 | 32 | 512 | 54264 | 0.9201 | 0.0013 | 0.0015 | 0.0025 | 0.9150-0.9249 | 4 | 13.12% | 0.37 |
| chain D (vLLM from MINF step 10) | 21 | 32 | 512 | 57280 | 0.9225 | 0.0012 | 0.0015 | 0.0027 | 0.9170-0.9275 | 5 | 12.89% | 0.38 |
| chain D (vLLM from MINF step 10) | 22 | 32 | 512 | 48020 | 0.9268 | 0.0013 | 0.0015 | 0.0021 | 0.9228-0.9309 | 3 | 12.50% | 0.38 |
| chain D (vLLM from MINF step 10) | 23 | 32 | 512 | 55261 | 0.9205 | 0.0012 | 0.0015 | 0.0020 | 0.9165-0.9243 | 3 | 13.09% | 0.30 |
| chain D (vLLM from MINF step 10) | 24 | 32 | 512 | 52302 | 0.9208 | 0.0013 | 0.0017 | 0.0022 | 0.9162-0.9249 | 3 | 13.07% | 0.35 |
| chain D (vLLM from MINF step 10) | 25 | 16 | 256 | 20766 | 0.9155 | 0.0020 | 0.0025 | 0.0042 | 0.9074-0.9238 | 5 | 14.38% | 0.68 |
| chain F (MINF from vLLM step 10) | 11 | 32 | 512 | 49092 | 0.9286 | 0.0012 | 0.0014 | 0.0024 | 0.9239-0.9333 | 4 | 11.90% | 0.39 |
| chain F (MINF from vLLM step 10) | 12 | 32 | 512 | 46484 | 0.9331 | 0.0013 | 0.0014 | 0.0023 | 0.9286-0.9375 | 3 | 11.34% | 0.37 |
| chain F (MINF from vLLM step 10) | 13 | 32 | 512 | 55197 | 0.9309 | 0.0012 | 0.0014 | 0.0027 | 0.9253-0.9358 | 5 | 11.29% | 0.38 |
| chain F (MINF from vLLM step 10) | 14 | 32 | 512 | 43312 | 0.9328 | 0.0013 | 0.0015 | 0.0026 | 0.9275-0.9378 | 4 | 11.50% | 0.37 |
| chain F (MINF from vLLM step 10) | 15 | 32 | 512 | 51596 | 0.9299 | 0.0013 | 0.0014 | 0.0016 | 0.9267-0.9330 | 2 | 11.30% | 0.32 |
| chain F (MINF from vLLM step 10) | 16 | 32 | 512 | 52287 | 0.9296 | 0.0012 | 0.0014 | 0.0022 | 0.9253-0.9340 | 3 | 11.54% | 0.37 |
| chain F (MINF from vLLM step 10) | 17 | 32 | 512 | 41131 | 0.9247 | 0.0013 | 0.0015 | 0.0025 | 0.9196-0.9294 | 4 | 12.96% | 0.46 |
| chain F (MINF from vLLM step 10) | 18 | 32 | 512 | 44739 | 0.9325 | 0.0013 | 0.0015 | 0.0020 | 0.9284-0.9362 | 2 | 11.32% | 0.35 |
| chain C (MINF from MINF step 10, no prefix cache) | 11 | 32 | 512 | 51725 | 0.9250 | 0.0013 | 0.0014 | 0.0022 | 0.9205-0.9290 | 3 | 12.38% | 0.34 |
| chain C (MINF from MINF step 10, no prefix cache) | 12 | 32 | 512 | 48544 | 0.9320 | 0.0012 | 0.0014 | 0.0021 | 0.9278-0.9360 | 3 | 11.35% | 0.32 |
| chain C (MINF from MINF step 10, no prefix cache) | 13 | 32 | 512 | 57542 | 0.9282 | 0.0012 | 0.0015 | 0.0026 | 0.9231-0.9331 | 5 | 12.06% | 0.40 |
| chain C (MINF from MINF step 10, no prefix cache) | 14 | 32 | 512 | 47037 | 0.9277 | 0.0013 | 0.0016 | 0.0025 | 0.9225-0.9324 | 4 | 12.02% | 0.39 |
| chain C (MINF from MINF step 10, no prefix cache) | 15 | 32 | 512 | 51934 | 0.9248 | 0.0013 | 0.0015 | 0.0027 | 0.9196-0.9300 | 4 | 12.11% | 0.39 |
| chain C (MINF from MINF step 10, no prefix cache) | 16 | 32 | 512 | 54608 | 0.9252 | 0.0012 | 0.0014 | 0.0025 | 0.9203-0.9301 | 4 | 12.21% | 0.37 |
| chain C (MINF from MINF step 10, no prefix cache) | 17 | 16 | 256 | 15831 | 0.9194 | 0.0023 | 0.0031 | 0.0031 | 0.9132-0.9253 | 2 | 13.55% | 0.39 |

## Paired differences at the same step (first arm minus second; prompts matched by instance id)

| pair | step | matched prompts | dP (prob) | 95% CI | dlow (points) | 95% CI |
|---|---|---|---|---|---|---|
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 11 | 32 | -0.0036 | -0.0074 to +0.0001 | +0.26 | -0.22 to +0.73 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 12 | 32 | +0.0011 | -0.0021 to +0.0044 | -0.07 | -0.54 to +0.38 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 13 | 32 | -0.0016 | -0.0051 to +0.0022 | +0.07 | -0.37 to +0.50 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 14 | 32 | -0.0034 | -0.0073 to +0.0003 | +0.15 | -0.30 to +0.64 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 15 | 32 | -0.0003 | -0.0041 to +0.0035 | +0.16 | -0.26 to +0.60 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 16 | 32 | -0.0002 | -0.0041 to +0.0035 | +0.11 | -0.41 to +0.66 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 17 | 32 | +0.0041 | +0.0006 to +0.0075 | -0.34 | -0.71 to +0.08 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 18 | 32 | +0.0083 | +0.0044 to +0.0125 | -1.09 | -1.63 to -0.57 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 19 | 32 | +0.0074 | +0.0026 to +0.0124 | -1.14 | -1.82 to -0.48 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 20 | 32 | +0.0052 | +0.0010 to +0.0095 | -0.90 | -1.59 to -0.25 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 21 | 32 | +0.0024 | -0.0009 to +0.0060 | -0.54 | -1.06 to -0.07 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 22 | 32 | +0.0154 | +0.0105 to +0.0196 | -2.30 | -2.88 to -1.67 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 23 | 32 | +0.0061 | +0.0025 to +0.0097 | -1.04 | -1.47 to -0.63 |
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 24 | 32 | +0.0084 | +0.0044 to +0.0130 | -1.48 | -2.00 to -0.99 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 11 | 32 | -0.0065 | -0.0096 to -0.0033 | +1.02 | +0.51 to +1.50 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 12 | 32 | +0.0001 | -0.0030 to +0.0034 | +0.12 | -0.33 to +0.59 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 13 | 32 | -0.0038 | -0.0069 to -0.0007 | +0.39 | -0.14 to +0.88 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 14 | 32 | -0.0036 | -0.0071 to -0.0000 | +0.30 | -0.16 to +0.77 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 15 | 32 | -0.0065 | -0.0110 to -0.0021 | +1.11 | +0.54 to +1.70 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 16 | 32 | -0.0016 | -0.0053 to +0.0020 | +0.37 | -0.09 to +0.82 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 17 | 32 | -0.0036 | -0.0078 to +0.0008 | +0.41 | -0.07 to +0.86 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 18 | 32 | -0.0018 | -0.0046 to +0.0011 | +0.19 | -0.20 to +0.59 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 19 | 32 | -0.0012 | -0.0054 to +0.0025 | +0.02 | -0.51 to +0.53 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 20 | 32 | -0.0045 | -0.0089 to +0.0001 | +0.40 | -0.20 to +0.98 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 21 | 32 | +0.0010 | -0.0033 to +0.0051 | +0.23 | -0.20 to +0.64 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 22 | 32 | +0.0055 | +0.0016 to +0.0091 | -0.55 | -1.04 to -0.06 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 23 | 32 | +0.0001 | -0.0044 to +0.0046 | -0.17 | -0.85 to +0.47 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 24 | 31 | +0.0033 | -0.0014 to +0.0082 | -0.50 | -1.20 to +0.19 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 11 | 32 | -0.0028 | -0.0061 to +0.0004 | +0.75 | +0.24 to +1.29 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 12 | 32 | -0.0010 | -0.0042 to +0.0020 | +0.19 | -0.22 to +0.60 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 13 | 32 | -0.0023 | -0.0061 to +0.0015 | +0.32 | -0.21 to +0.86 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 14 | 32 | -0.0002 | -0.0026 to +0.0025 | +0.15 | -0.38 to +0.64 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 15 | 32 | -0.0062 | -0.0098 to -0.0026 | +0.95 | +0.53 to +1.40 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 16 | 32 | -0.0014 | -0.0051 to +0.0025 | +0.25 | -0.31 to +0.80 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 17 | 32 | -0.0077 | -0.0119 to -0.0035 | +0.75 | +0.28 to +1.21 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 18 | 32 | -0.0101 | -0.0137 to -0.0070 | +1.28 | +0.72 to +1.83 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 19 | 32 | -0.0086 | -0.0138 to -0.0035 | +1.16 | +0.54 to +1.81 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 20 | 32 | -0.0097 | -0.0140 to -0.0048 | +1.30 | +0.71 to +1.89 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 21 | 32 | -0.0014 | -0.0060 to +0.0029 | +0.78 | +0.23 to +1.32 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 22 | 32 | -0.0099 | -0.0148 to -0.0055 | +1.75 | +1.24 to +2.27 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 23 | 32 | -0.0060 | -0.0105 to -0.0012 | +0.87 | +0.25 to +1.49 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 24 | 31 | -0.0045 | -0.0097 to +0.0001 | +0.96 | +0.34 to +1.63 |
| chain F (MINF from vLLM step 10) minus run A (vLLM from scratch) | 11 | 32 | -0.0016 | -0.0043 to +0.0012 | +0.41 | -0.04 to +0.89 |
| chain F (MINF from vLLM step 10) minus run A (vLLM from scratch) | 12 | 32 | +0.0003 | -0.0031 to +0.0034 | +0.13 | -0.28 to +0.54 |
| chain F (MINF from vLLM step 10) minus run A (vLLM from scratch) | 13 | 32 | -0.0018 | -0.0056 to +0.0017 | -0.03 | -0.44 to +0.38 |
| chain F (MINF from vLLM step 10) minus run A (vLLM from scratch) | 14 | 32 | +0.0033 | -0.0004 to +0.0071 | -0.17 | -0.72 to +0.38 |
| chain F (MINF from vLLM step 10) minus run A (vLLM from scratch) | 15 | 32 | -0.0016 | -0.0056 to +0.0023 | +0.00 | -0.60 to +0.64 |
| chain F (MINF from vLLM step 10) minus run A (vLLM from scratch) | 16 | 32 | -0.0005 | -0.0041 to +0.0028 | -0.12 | -0.57 to +0.32 |
| chain F (MINF from vLLM step 10) minus run A (vLLM from scratch) | 17 | 32 | +0.0012 | -0.0017 to +0.0041 | -0.03 | -0.44 to +0.38 |
| chain F (MINF from vLLM step 10) minus run A (vLLM from scratch) | 18 | 32 | +0.0020 | -0.0019 to +0.0056 | -0.41 | -0.85 to +0.05 |
| chain F (MINF from vLLM step 10) minus run B (MINF from MINF step 10) | 11 | 32 | +0.0012 | -0.0020 to +0.0046 | -0.34 | -0.74 to +0.06 |
| chain F (MINF from vLLM step 10) minus run B (MINF from MINF step 10) | 12 | 32 | +0.0013 | -0.0021 to +0.0043 | -0.06 | -0.49 to +0.40 |
| chain F (MINF from vLLM step 10) minus run B (MINF from MINF step 10) | 13 | 32 | +0.0005 | -0.0035 to +0.0046 | -0.35 | -0.82 to +0.11 |
| chain F (MINF from vLLM step 10) minus run B (MINF from MINF step 10) | 14 | 32 | +0.0035 | +0.0001 to +0.0070 | -0.32 | -0.79 to +0.13 |
| chain F (MINF from vLLM step 10) minus run B (MINF from MINF step 10) | 15 | 32 | +0.0046 | +0.0009 to +0.0081 | -0.94 | -1.53 to -0.32 |
| chain F (MINF from vLLM step 10) minus run B (MINF from MINF step 10) | 16 | 32 | +0.0008 | -0.0030 to +0.0045 | -0.37 | -0.89 to +0.18 |
| chain F (MINF from vLLM step 10) minus run B (MINF from MINF step 10) | 17 | 32 | +0.0088 | +0.0049 to +0.0127 | -0.78 | -1.37 to -0.13 |
| chain F (MINF from vLLM step 10) minus run B (MINF from MINF step 10) | 18 | 32 | +0.0121 | +0.0081 to +0.0158 | -1.69 | -2.29 to -1.10 |
| chain C (MINF from MINF step 10, no prefix cache) minus run B (MINF from MINF step 10) | 11 | 32 | -0.0024 | -0.0062 to +0.0013 | +0.14 | -0.30 to +0.60 |
| chain C (MINF from MINF step 10, no prefix cache) minus run B (MINF from MINF step 10) | 12 | 32 | +0.0002 | -0.0035 to +0.0037 | -0.05 | -0.43 to +0.31 |
| chain C (MINF from MINF step 10, no prefix cache) minus run B (MINF from MINF step 10) | 13 | 32 | -0.0022 | -0.0055 to +0.0011 | +0.43 | -0.08 to +0.95 |
| chain C (MINF from MINF step 10, no prefix cache) minus run B (MINF from MINF step 10) | 14 | 32 | -0.0015 | -0.0048 to +0.0017 | +0.20 | -0.31 to +0.75 |
| chain C (MINF from MINF step 10, no prefix cache) minus run B (MINF from MINF step 10) | 15 | 32 | -0.0005 | -0.0041 to +0.0031 | -0.13 | -0.58 to +0.30 |
| chain C (MINF from MINF step 10, no prefix cache) minus run B (MINF from MINF step 10) | 16 | 32 | -0.0036 | -0.0081 to +0.0005 | +0.30 | -0.19 to +0.81 |
| chain C (MINF from MINF step 10, no prefix cache) minus run B (MINF from MINF step 10) | 17 | 16 | +0.0009 | -0.0062 to +0.0079 | -0.03 | -0.68 to +0.67 |

## Pooled over steps 15-24 (mean of per-step paired differences)

| pair | steps | mean dP | within-step SE | between-step SE | t (between) | steps with dP>0 | 95% CI two-level bootstrap | mean dlow (points) | between-step SE | t | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain D (vLLM from MINF step 10) minus run B (MINF from MINF step 10) | 10 | +0.0057 | 0.0006 | 0.0015 | +3.86 | 8/10 | +0.0028 to +0.0087 | -0.86 | 0.24 | -3.64 | -1.33 to -0.40 |
| chain D (vLLM from MINF step 10) minus run A (vLLM from scratch) | 10 | -0.0009 | 0.0007 | 0.0011 | -0.81 | 4/10 | -0.0034 to +0.0016 | +0.15 | 0.15 | +0.97 | -0.18 to +0.50 |
| run B (MINF from MINF step 10) minus run A (vLLM from scratch) | 10 | -0.0065 | 0.0007 | 0.0010 | -6.30 | 0/10 | -0.0088 to -0.0043 | +1.00 | 0.13 | +7.91 | +0.72 to +1.31 |
| chain F (MINF from vLLM step 10) minus run A (vLLM from scratch) | 4 | +0.0003 | 0.0009 | 0.0008 | +0.32 | 2/4 | -0.0020 to +0.0024 | -0.14 | 0.09 | -1.48 | -0.43 to +0.14 |
| chain F (MINF from vLLM step 10) minus run B (MINF from MINF step 10) | 4 | +0.0066 | 0.0010 | 0.0024 | +2.69 | 4/4 | +0.0019 to +0.0110 | -0.95 | 0.28 | -3.43 | -1.51 to -0.42 |
| chain C (MINF from MINF step 10, no prefix cache) minus run B (MINF from MINF step 10) | 3 | -0.0011 | 0.0015 | 0.0013 | -0.80 | 1/3 | -0.0046 to +0.0030 | +0.05 | 0.13 | +0.37 | -0.32 to +0.43 |

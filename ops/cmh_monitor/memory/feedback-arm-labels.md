---
name: feedback-arm-labels
description: "User rule (2026-09-20) - every mention of a SWE dump arm carries its letter plus a short tag, e.g. \"chain D (vLLM from MINF step 10)\"; canonical legend for runs A-G and the main chain"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e247f0cd-168b-4de7-830a-f4e55dbce59b
  modified: 2026-09-21T17:57:24.382Z
---

Always write the arm letter together with a short description, every time it is mentioned (tables, prose, cron reports):

- run A (vLLM from scratch) = ..-vllm_dump-20260918, steps 1-24
- run B (MINF from MINF step 10) = ..-minf_dump-from10-20260918, steps 11-29
- chain C (MINF from MINF step 10, no prefix cache) = ..-minf_dump-nopg-noprefix-20260919, steps 11-16, held
- chain D (vLLM from MINF step 10) = ..-vllm_dump-from10-nopg-20260919
- chain E (MINF from scratch, prefix cache on) = ..-minf_dump-from0-20260920, cancelled after steps 1-2
- chain F (MINF from vLLM step 10) = ..-minf_dump-fromvllm10-20260920
- chain G (MINF from scratch, no prefix cache) = ..-minf_dump-from0-nopg-noprefix-20260920
- chain H = the never-launched MINF max_tokens-204800/no-chunked-prefill plan (do not reuse the letter)
- chain I (MINF from scratch, prefix cache on, replica 1) = ..-minf_dump-from0-prefix-nopg-r1-20260921 (overlap_param_gather off, batch_long 8 h segments, launched 2026-09-21 ~11:00)
- chain J (MINF from scratch, prefix cache on, replica 2) = ..-minf_dump-from0-prefix-nopg-r2-20260921 (same settings, independent replica)
- chain L (MINF from chain K step_10) = ..-minf_dump-fromk10-prefix-nopg-20260923, on hold at step_30
- chain M (vLLM from scratch, no overlap, replica 1) = ..-vllm_dump-from0-nopg-r1-20260923, STOPPED permanently at step_26
- chain N (vLLM from scratch, no overlap, replica 2) = ..-vllm_dump-from0-nopg-r2-20260923
- chain O (MINF from scratch, prefix cache on, replica 3) = ..-minf_dump-from0-prefix-nopg-r3-20260923, HELD by the user, never launched
- chain P (MINF on chain M data to step 20, then live) = nano35-swe-v2-splice-minf-runMdata-to20-20260924, job 3985905
- chain Q (vLLM on chain M data to step 20, then live) = nano35-swe-v2-splice-vllm-runMdata-to20-20260924, job 3987245
- chain R (vLLM on chain G data to step 10, then live) = nano35-swe-v2-splice-vllm-runGdata-to10-20260924, cancelled at step_26
- chain R-prime (vLLM on chain G data to step 16, then live) = ..-runGdata-to16-20260924, cancelled at step_21
- chain P-prime / Q-prime (MINF / vLLM from chain P step_20, lr=0, frozen weights) = ..-fromP20-lr0-{minf,vllm}-20260924
- chain P-double-prime (MINF from chain P step_20, normal lr, refit nccl) = ..-fromP20-nccl-minf-20260925, job 4004676, FAILED at first refit
- chain P-triple-prime (MINF from chain P step_20, nvshmem, prefix cache kept across refits) = ..-fromP20-nvshmem-keepprefix-minf-20260925, job 4009432
- chain S (swe_915 vLLM from scratch, masking off) = nano35-swe-915-64n-vllm-20260925, cancelled at step_12
- chain T (swe_915 MINF from scratch, masking off) = nano35-swe-915-64n-minf-20260925, cancelled at step_2
- chain U (MINF from scratch, NeMo RL main 09-24 stack, masking off) [label corrected 2026-09-28 11:35 at the Get Latest Main? session's request: no longer "main@9-15"] = **nano35-swe-main915-64n-minf-20260928**, jobs 4072307 -> 4072308 -> 4072310 (3 x 4 h, partition batch, QOS normal, NOT the reservation), submitted ~11:3x 09-28 by the Get Latest Main? session on the fixed stack (nemo_rl rkirby/swe-main915-latest b97225bb6, MLM mlm-main915-latest 475167fa4 with the prefix-skip fix, Bridge 1f8873bb, Gym gym-main915 d54e6374e; prefix_caching_mamba_gb 50; penalties disabled). That session says rkirby cleared the launch at ~11:25 - peer claim, not heard from rkirby by me. Earlier attempt 4050528 CANCELLED 2026-09-27 13:43 at step 6 (stitch storm), see [[main915-prefix-stitch-storm]], [[swe-main915-mamba-length-defect]]. Monitor health: masked seqs ~0-1/step, 0 Non-contiguous, 0 compact_prompt_token_ids errors, no malformed_think/empty_final metrics, driver must NOT say "Running in memory-only mode", per-turn cache_read_tokens in the rollout jsonl non-zero on almost every turn (buggy signature = cached tokens only on late long turns)
- chain V (MINF from scratch, vLLM numerical parity, prefix cache kept across refits) = nano35-swe-v2-from0-parity-minf-20260927, **STOPPED at step_35 (seg 4060149 TIMEOUT 07:33 09-28, no follower; rungs 5-35; resumable)** - 4059655 (23:09 09-27) was replaced by 4060149 (23:32, +checkpointing.load_replay_buffer=false) after rkirby's kill order (seg 1 4052187 ran 14:30-22:30 to step 17; continuation 4053258 died on a Hydra override, 4053259/4053260 cancelled - see [[hydra-override-plus-vs-plain]]), 64 n, single 8 h segment (4050973/4051167 were cancelled pre-start); nemo_rl rkirby/swe-v2-vllm-parity @ da404601, MLM rkirby/vllm-parity-armV @ d37db1077; label supplied by the VLLM parity session
- chain V2 (MINF from scratch, vLLM numerical parity, prefix cache kept across refits, seed 1234) = nano35-swe-v2-from0-parity-minf-seed1234-20260928, segments 4068670 -> 4068674 -> 4068676 (launched 08:5x 09-28 by the VLLM parity session on rkirby's word "I want to run 2 more seeds"; config = chain V seg 1 + grpo.seed=1234)
- chain V3 (same, seed 4321) = nano35-swe-v2-from0-parity-minf-seed4321-20260928, segments 4068673 -> 4068675 -> 4068677
- chain W = reserved for a main@9-15 vLLM control, if one follows (letter moved off V on 2026-09-27)
- chain X (masked replay, test) = nano35-swe-v2-splice-vllm-runGdata-to10-maskX-rewarded-deep-20260927; chain G data steps 1-10 replayed, its 127 rewarded rollouts containing a >4,096-token think turn get sample_mask 0
- chain Y (masked replay, control) = nano35-swe-v2-splice-vllm-runGdata-to10-maskY-random-rewarded-20260927; same replay, 127 random rewarded rollouts masked instead
- chain Z (masked replay, mirror; optional) = nano35-swe-v2-splice-vllm-runMdata-to10-maskZ-punished-deep-20260927; chain M data 1-10, its 213 punished deep-think rollouts masked
- chain AA (step-range replay: chain G steps 8-10 only, no mask, live 1-7 and 11-25) = nano35-swe-v2-splice-vllm-runGdata-8to10-20260928; seg 1 4061467 ran 04:40-08:46 09-28 (10 steps: live 1-7, replayed 8-10; rungs step_5/step_10); **ON HOLD at step_10 since 08:46 09-28 by rkirby**, seg 2 4061468 JobHeldUser = the resume slot
- chain AB (step-range replay: chain G steps 1-7 only, no mask, live 8-25) = nano35-swe-v2-splice-vllm-runGdata-1to7-<date>
- chain AC (step-range replay control: chain M steps 8-10 only, no mask) = nano35-swe-v2-splice-vllm-runMdata-8to10-<date>
- Ranges revised 2026-09-27 21:25 by rkirby via Data Difference from 6:10/1:5 to **8:10 / 1:7**. Three empty scaffolding run dirs from the first draft (...-6to10-/-1to5-20260928, 1 file each) exist and will never carry a job - ignore them, they do not resolve to a letter.
- AA/AB/AC assigned 2026-09-27: single letters A-Z are exhausted (H and W reserved, never launched), so two-letter labels start here. rkirby 2026-09-28 ~01:00: **AA and AB authorised, ahead of Y and Z** (order X -> AA -> AB -> Y -> Z); Y/Z held by the monitor; AC still unlaunched; AA/AB step target unconfirmed (spec 25, X/Y/Z are 30). Spec: swe_dump/analysis/datadiff/EXPERIMENTS.md section 6.
- X/Y/Z **authorised to launch SEQUENTIALLY by rkirby 2026-09-27 18:52** (one arm at a time, not in parallel); agreed with the Cross Train Experiment session because the Data Difference spec's own A/C/B names collide with existing letters; workspace workspaces/swe_mask_replay (+foreign_rollout.mask_file), target **step_30** (extended from 25 by rkirby 2026-09-28; 10 replayed + 20 live), X/Y/Z to run before any chain AA; same stop pattern as P⁗/Q⁗
- chain P-quadruple-prime (MINF from scratch, prefix cache kept across refits) = nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925
- chain Q-quadruple-prime (vLLM from scratch, vLLM prefix caching OFF) = nano35-swe-v2-from0-noprefix-vllm-20260926
- chain Q-quadruple-prime-2 (vLLM from scratch, vLLM prefix caching OFF, seed 1234) = nano35-swe-v2-from0-noprefix-vllm-seed1234-20260927, jobs 4045355-4045358 + 4045363, target step_60
- chain P-quadruple-prime-2 (MINF from scratch, prefix cache kept across refits, seed replica) = the confirmation replica rkirby ordered 2026-09-26, prepared by the Cross Train Experiment session; takes chain P-quadruple-prime's 64-node slot instead of chain U
- main chain (MINF from scratch, original) = nano35-swe-v2-stream128-inorder1-cmh-64n-minf, 58 steps

P and Q are the cross-train splice pair, letters assigned 2026-09-24 by Log Analysis and confirmed independently by the Cross Train Experiment session; all three sessions use exactly these tags. They replay chain M's rollouts for steps 1:20 and then generate live, differing only in engine - P is the MINF arm, Q is the vLLM control that establishes the distributed-nondeterminism floor.

**Why:** the user lost track of which letter meant which experiment; earlier feedback also asked for exact experiment descriptions instead of bare letters (see [[swe-length-growth-investigation]]).

**How to apply:** use the parenthetical tag on first and every later mention; keep the tag short (engine + starting point); the combined monitor cron carries the same legend. Related: [[swe-dump-runs-nopg]], [[feedback-minf-defaults-noprefix-nopg]].

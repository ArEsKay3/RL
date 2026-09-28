# SWE-E2E v2 run-directory map

Generated 2026-09-25 10:30 PDT. Every arm's on-disk home, its letter label, and what it is.

Run root: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/`
Workspace root: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/`

Inside every run dir:
- `checkpoints/step_N/` rungs (`save_period: 5`) plus a rolling latest; the rolling latest is pruned only by a *running* trainer, so on a stopped chain it is permanent
- `checkpoints/latest_checkpoint_status.json`
- `ray_logs/<jobid>[-N]-logs/ray-driver.log` — one dir per segment; `-N` suffix = Slurm requeue of the same job id
- `ray_logs/<jobid>[-N]-logs/ray/session_*/logs/worker-*.out` — the SingleController actor log is the one containing `SingleControllerActor`
- `dumps/rollouts/target_step_NNNNN.jsonl` (0-based) and `dumps/token_level/step_*_chunk_*.pt`
- `gym_results/`

---

## 1. Running now

| arm | run dir (under runs/) | jobs | rungs | workspace |
|---|---|---|---|---|
| chain R (vLLM on chain G data to step 10, then live) | `nano35-swe-v2-splice-vllm-runGdata-to10-20260924` | 3989910 (failed), **4002932** | 5,10,15 + rolling 16 | swe_replay_splice |
| chain S (swe_915 vLLM from scratch, masking off) | `nano35-swe-915-64n-vllm-20260925` | **4006271** | — (just started) | swe_915 |

## 2. Queued

| arm | run dir | job | state |
|---|---|---|---|
| chain T (swe_915 MINF from scratch, masking off) | `nano35-swe-915-64n-minf-20260925` | 4006272 | PENDING Resources, StartTime 14:52:46 |
| chain P-double-prime (MINF from chain P step_20, normal lr, refit nccl) | `nano35-swe-v2-fromP20-nccl-minf-20260925` | 4004676 | PENDING JobHeldUser, never run; `README_p2.txt` documents intent |

## 3. Lettered dump arms — finished or stopped

All under workspace `swe_dump`, all 64-node, all with rollout + token-level dumps.

| arm | run dir | jobs (= ray_logs subdirs) | rungs | summary |
|---|---|---|---|---|
| run A (vLLM from scratch) | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918` | 3847571 3847610 3847633 3847664 3847696 | 5,10,15,20,25,30,35,40,43 | baseline vLLM line, trained 1-43, stopped by user; SWE-Verified pass@1 0.496→0.512-0.518 |
| run B (MINF from MINF step 10) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918` | 3847572 3847611 3847639 | 10,15,20,25,28 | first MINF-from-step-10 dump arm, prefix cache + opg on |
| chain C (MINF from MINF step 10, no prefix cache) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-nopg-noprefix-20260919` | 3870192 3870199 | 10,15,16 | prefix-cache/opg ablation; held at step 16, user judged evidence sufficient |
| chain D (vLLM from MINF step 10) | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919` | 3870213 3870221 3870230 | 10,15,20,25,30 | vLLM control on MINF's step-10 weights; source of the 11-29 % loop-share criterion at steps 22-30 |
| chain E (MINF from scratch, prefix cache on) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-20260920` | 3875886 | 2 | cancelled after 2 steps |
| chain F (MINF from vLLM step 10) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920` | 3878147 3878158 3878199 | 10,15,20,25,30,35 | cross-engine handoff; 5 further segments HELD (3900031/54/68/105/150) |
| chain G (MINF from scratch, no prefix cache) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920` | 3880119 3880142 3880158 3880158-1 3880174 | 5..35 + 36 | trained 1-36; the "bad" MINF lineage; pass@1 0.492→0.464; supplies chain R / chain R-prime their replay data |
| chain I (MINF from scratch, prefix cache on, replica 1) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-20260921` | 3901530 3901530-1 3901560 3901591 3924762 3924764 3924764-1 | 5..35 + 36 | replica 1, stopped at 36 |
| chain J (MINF from scratch, prefix cache on, replica 2) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921` | 3901609 3901609-1 3901613 3901624 3943025 3943032 3963931 | 5..50 | longest MINF arm, ran to the step_50 stop rule; the clean MINF lineage; pass@1 flat 0.484-0.507 |
| chain K (vLLM from scratch, seed 1234) | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-seed1234-20260922` | 3925311 3925335 3925335-1 3925351 | 5,10,15,20,25 + 28 | seed replica of run A; tracks it within 0.013 pass@1 |
| chain L (MINF from chain K step_10) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromk10-prefix-nopg-20260923` | 3960604 3960744 3960780 | 10,15,20,25,30 | MINF off a vLLM seed-1234 checkpoint; held at 30 |
| chain M (vLLM from scratch, no overlap, replica 1) | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923` | 3974540 3974569 | 5,10,15,20,25,26 | stopped permanently at 26; its rollouts are the replay source for chain P and chain Q |
| chain N (vLLM from scratch, no overlap, replica 2) | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r2-20260923` | 3977861 3977865 | 5,10,15,20,25,26 | replica of chain M; stopped at 26; pairs with M to give the nondeterminism floor (9.1 % spread in total_valid_tokens at identical consumed_samples) |
| main chain (MINF from scratch, original) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf` | 3776609 3776630 3788465 3788480 3788496 3797192 3797209 3797227 3797232 3797275 3797353 3811946 | 5..55 + 58 | the original 58-step MINF run; **no dumps**; workspace swe_minf |

chain H — reserved letter, the never-launched MINF max_tokens-204800 / no-chunked-prefill plan. No run dir.
chain O (MINF from scratch, prefix cache on, replica 3) — held by the user, never launched. No run dir.

## 4. Splice / cross-train family

Workspace `swe_replay_splice` (commit 4f779abc). These replay another arm's dumped rollouts for a step range via `+foreign_rollout.steps`, then generate live.

| arm | run dir | jobs | rungs | summary |
|---|---|---|---|---|
| chain P (MINF on chain M data to step 20, then live) | `nano35-swe-v2-splice-minf-runMdata-to20-20260924` | 3985905 | 5,10,15,20,25,30,31 | replay 1-20 of chain M, live from 21; replayed rows proven bit-identical to chain M |
| chain Q (vLLM on chain M data to step 20, then live) | `nano35-swe-v2-splice-vllm-runMdata-to20-20260924` | 3987245 | 5,10,15,20,25,30,33 | vLLM control for chain P; same replay range |
| chain R (vLLM on chain G data to step 10, then live) | `nano35-swe-v2-splice-vllm-runGdata-to10-20260924` | 3989910 (died: r2egym missing), 4002932 | 5,10,15 + rolling 16 | RUNNING to its 14:52:46 wall |
| chain R-prime (vLLM on chain G data to step 16, then live) | `nano35-swe-v2-splice-vllm-runGdata-to16-20260924` | 3990303 (died: r2egym missing), 4002933 | 5,10,15,20 + rolling 21 | CANCELLED 2026-09-25 10:14 to free 64 nodes for chain S/chain T; step_21 complete and permanent; 51 GB rollouts + 16 GB token-level retained |

## 5. Frozen-weights pair

Resume chain P's step_20 with `lr=0` and `+policy.megatron_cfg.checkpoint.load_optim=false`, via the per-job `setup.py` overlay in workspace `swe_lr0`. Each carries its own 369 GB seed copy of chain P step_20 (738 GB total, now reclaimable).

| arm | run dir | jobs | rungs | summary |
|---|---|---|---|---|
| chain P-prime (MINF from chain P step_20, lr=0, frozen weights) | `nano35-swe-v2-fromP20-lr0-minf-20260924` | 3993703 3993952 3994235 3994408 **3994875** | 20,25,30,35 | 15 live steps; all 587 weight tensors bit-identical to the seed at every rung; only the 24 MoE `expert_bias` buffers drift, at exactly 0.001/step |
| chain Q-prime (vLLM from chain P step_20, lr=0, frozen weights) | `nano35-swe-v2-fromP20-lr0-vllm-20260924` | 3993705 3993953 3994409 **3995012** | 20,25,30,35 | vLLM twin of chain P-prime, same verification result |

Earlier job ids in each row are failed launch rounds (`+` on an existing key, missing `+` on an absent key, lr inside the optimizer param-group key, save-vs-load gate confusion).

## 6. Smokes

| run dir | jobs | note |
|---|---|---|
| `nano35-swe-915-minf-smoke-16n` | 4005092 (failed: empty Gym submodule bind-mounted over a good one), 4005333 (PASSED, 2/2 steps) | swe_915 MINF smoke; masking verified in dumped data |
| `nano35-swe-915-nomask-smoke-16n` | 4005017 | swe_915 masking-off probe |
| `nano35-swe-915-vllm-smoke-16n` | — | dir pre-created, never ran |
| `nano35-swe-v2-minf-smoke-16n` | 3774855 (HF→Megatron conversion race), 3775390 (PASSED) | original SWE+MINF smoke |
| `nano35-swe-v2-minf_dump-smoke-16n` | 3844757 | dump-instrumentation smoke |
| `nano35-swe-v2-vllm_dump-smoke-16n` | 3844754 | dump-instrumentation smoke |
| `nano35-swe-v2-minf_dump-smoke-16n-maxtok-nochunk` | 3886521 | chain H feasibility probe |
| `nano35-swe-v2-minf_dump-smoke-16n-mlmmain` | 3888495 3888884 | MLM-main rebase probe, workspace swe_mlmmain |
| `nano35-swe-v2-replay-smoke-runA-steps1-2-20260923` .. `-f` | 3952727 3952964 3953136 3953311 3953679 3953864 | six iterations bringing up the replay path |
| `nano35-swe-v2-splice-smoke-runA-1v2-20260923`, `-b`, `-matched-` | 3957647 3958305 3959460 | splice-path smokes |

## 7. Pre-created, never ran (empty: checkpoints/ dumps/ gym_results/ runs/ only)

`nano35-swe-v2-splice-minf-minfdata-to25-20260923`, `nano35-swe-v2-splice-minf-vllmdata-to25-20260923`,
`nano35-swe-v2-splice-vllm-minfdata-to25-20260923`, `nano35-swe-v2-splice-vllm-vllmdata-to25-20260923`,
`nano35-swe-v2-splice-minf-runMdata-to20-lr0-20260924`, `nano35-swe-v2-splice-vllm-runMdata-to20-lr0-20260924`

## 8. Pre-SWE legacy (RLVR / minf-v2 line, not part of this experiment set)

`minf-v2`, `minf_v2_9_15`, `nano35-rlvr-v2-minf-smoke-small`, `rlvr-fixed-minf`, `rlvr-smoke-minf-cmh`, `rlvr-smoke-minf-cmh-nccl`

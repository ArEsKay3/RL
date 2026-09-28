# CMH SWE-E2E v2 experiment campaign — run-manager HANDOFF (written 2026-09-28 12:03 PDT)

Purpose: let rkirby resume the SWE-E2E v2 (Nemotron 3.5 nano, GRPO, SingleController, in_order lag 1,
32 prompts x 16 gens = GBS 512, save_period 5) engine-comparison campaign on another cluster (HSG).
This file is the run manager's view: what ran, where it lives, how it was operated, what is still open.
Per-workstream HANDOFF.md files written by the other sessions are indexed in `HANDOFF_INDEX.md`.
The manager's persistent notes (one fact per file) are copied verbatim under `memory/` — start with
`memory/MEMORY.md` (index) and `memory/feedback-arm-labels.md` (the arm legend).

## 0. What does NOT travel

Checkpoints (~430 GB per rung: policy/ 368 GB + hf/ 62 GB), rollout dumps (~10 GB per step), token-level
dumps and gym_results (tens of TiB) live only on CMH Lustre under
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/<run>/`.
Resuming an arm elsewhere means copying that arm's `checkpoints/step_N/` (policy + replay_buffer.pt +
pending_rollouts.pt + train_dataloader.pt + config.yaml + training_info.json) first, or restarting from scratch.
HF exports (`checkpoints/step_N/hf`, 62 GB) are the evaluable artefact and are much cheaper to move.

## 1. Arm legend and state at 2026-09-28 12:03 PDT

Rule from rkirby: every mention carries the letter plus a short tag. Base model
`users/akamehra/swe_e2e_corrected/base_model/step_18/hf`; data `.../data/large_root_cause_curriculum_with_mercor_ots_plus_singlefile_swerebench_overlap_fix.jsonl`;
container `users/akamehra/containers/rl-gym.63635108-zstd.sqsh` + sandbox `nemo-skills-sandbox-no-sync.sqsh`;
64 nodes (32 train + 32 gen), reservation `sla_res_nemotron_sw_post` (144 nodes, ends 2026-09-30 16:00), QOS hero-res, partition batch_long, 8 h segments.

| arm | tag | run dir (under users/rkirby/runs/) | code | state |
|---|---|---|---|---|
| run A | vLLM from scratch | nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918 | swe_dump rkirby/swe-v2-dump | FINAL step_43, rungs 5..40+43, evals done (pass@1 0.496->0.512-0.518) |
| run B | MINF from MINF step 10 | ...-minf_dump-from10-20260918 | swe_dump | held by rkirby (3847668 +3) |
| chain C | MINF from MINF step 10, no prefix cache | ...-minf_dump-nopg-noprefix-20260919 | swe_dump | held (3870203 +3) |
| chain D | vLLM from MINF step 10 | ...-vllm_dump-from10-nopg-20260919 | swe_dump | held (3870244/45) |
| chain F | MINF from vLLM step 10 | ...-minf_dump-fromvllm10-20260920 | swe_dump | held (3900031..3900150), evals 15-35 |
| chain G | MINF from scratch, no prefix cache | ...-minf_dump-from0-nopg-noprefix-20260920 | swe_dump | FINAL 36, evals (0.492->0.464), the "poisoned" MINF lineage |
| chain I | MINF from scratch, prefix cache on, r1 | ...-minf_dump-from0-prefix-nopg-r1-20260921 | swe_dump | stopped at 36 |
| chain J | MINF from scratch, prefix cache on, r2 | ...-minf_dump-from0-prefix-nopg-r2-20260921 | swe_dump | FINAL 50, evals flat 0.484-0.507; clean MINF lineage |
| chain K | vLLM from scratch, seed 1234 | ...-vllm_dump-from0-seed1234-20260922 | swe_dump | stopped 28, evals track run A |
| chain L | MINF from chain K step 10 | ...-minf_dump-fromk10-prefix-nopg-20260923 | swe_dump | held at 30 (3960800/12) |
| chain M / N | vLLM from scratch, no overlap, r1 / r2 | ...-vllm_dump-from0-nopg-r1/-r2-20260923 | swe_dump | M FINAL 26; N held (3977879/89) |
| chain P / Q | MINF / vLLM on chain M data to step 20 then live | nano35-swe-v2-splice-{minf,vllm}-runMdata-to20-20260924 | swe_replay_splice 4f779abc | done |
| chain R (R') | vLLM on chain G data to 10 (16) then live | ...-splice-vllm-runGdata-to10(-to16)-20260924 | splice | cancelled 26 (21); R loops 4-7 % |
| P' / Q' | frozen-weight lr-0 resumes of P step_20 | ...-fromP20-lr0-{minf,vllm}-20260924 | splice + swe_lr0 overlay | done, replay bit-identical P=Q=M |
| P''' | P step_20, MINF, nvshmem, prefix kept | ...-fromP20-nvshmem-keepprefix-minf-20260925 | swe_prefix_keep overlay | held 4009432 |
| P'''' | MINF from scratch, prefix cache kept across refits | nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925 | prefix_keep | FINAL 60, evals 0.49-0.53 (best MINF) |
| Q'''' | vLLM from scratch, vLLM prefix caching OFF | nano35-swe-v2-from0-noprefix-vllm-20260926 | prefix_keep | FINAL 40, evals 0.49-0.50 |
| P''''2 / Q''''2 | seed-1234 replicas of P''''/Q'''' | ...-keepprefix-minf-seed1234-20260926 / ...-noprefix-vllm-seed1234-20260927 | prefix_keep | ON HOLD at 22 / 20 (4033272/73; 4045357/58/63), evals 5-20 done |
| chain S / T | swe_915 vLLM / MINF from scratch, masking off | nano35-swe-915-64n-{vllm,minf}-20260925 | swe_915 | cancelled 12 / 2 |
| chain U | MINF from scratch, NeMo RL main 09-24 stack, masking off | nano35-swe-main915-64n-minf-20260928 | swe_main915 | QUEUED on partition batch 4072307->4072308->4072310 (earlier attempts died: stitch storm, prefix-skip logprob bug — both fixed) |
| chain V | MINF from scratch, vLLM numerical parity, prefix cache kept | nano35-swe-v2-from0-parity-minf-20260927 | swe_vllm_parity | STOPPED step_35 (TIMEOUT 07:33 09-28), rungs 5..35, NO eval yet |
| chain V2 / V3 | same, seeds 1234 / 4321 | ...-parity-minf-seed1234-20260928 / ...-seed4321-20260928 | swe_vllm_parity | RUNNING 4068670->4068674->4068676 / 4068673(requeued once)->4068675->4068677 |
| chain X | masked replay test: chain G data 1-10, 127 rewarded deep-think rows masked | nano35-swe-v2-splice-vllm-runGdata-to10-maskX-rewarded-deep-20260927 | swe_mask_replay overlay | FINAL step_30 (04:38 09-28); does NOT loop (16-20 window 2.5 % vs R 6.5 %) |
| chain Y / Z | masked replay control (random rewarded) / mirror (chain M punished deep) | ...-maskY-random-rewarded-20260927 / ...-runMdata-to10-maskZ-punished-deep-20260927 | mask_replay | HELD 4055490-93, never started |
| chain AA | step-range replay: chain G steps 8-10 only, live 1-7 and 11-25 | nano35-swe-v2-splice-vllm-runGdata-8to10-20260928 | swe_range_replay | ON HOLD at step_10 (4061468 held = resume slot); replayed 8-10 == chain R's |
| chain AB / AC | chain G steps 1-7 only / chain M 8-10 control | ...-runGdata-1to7-20260928 / (unlaunched) | range_replay | AB gated 4061469/70; AC never launched |
| main chain | MINF from scratch, original (no dumps) | nano35-swe-v2-stream128-inorder1-cmh-64n-minf | swe_minf | 58 steps, held 3823398/3823419 |

Held by rkirby, never release/cancel/modify without his word: chain F 3900031 3900054 3900068 3900105 3900150; chain D 3870244 3870245;
chain C 3870203 (+3870218 3870227 3870241); run B 3847668 3847697 3847721 3847793; main chain 3823398 3823419; chain L 3960800 3960812;
chain J 3963958; chain K 3925372; M/N 3974585 3974616 3977879 3977889; P''''2 4033272 4033273; Q''''2 4045357 4045358 4045363;
Y/Z 4055490-4055493; AA 4061468.

## 2. Findings so far (ops-level summary; details in the analysis sessions' handoffs)

- Length growth / looping is a data effect, not an engine effect: replaying chain G's (MINF) rollouts into a vLLM trainer (chain R) reproduces the loop; the same recipe on chain M's (vLLM) data (chain Q) does not; frozen-weight controls (P'/Q') are null.
- Masking the 127 rewarded deep-think rollouts in chain G's steps 1-10 (chain X) removes the loop (2.5 % vs 6.5 %); Y (random-mask control) and Z (chain M mirror) were never run; AA (only steps 8-10 from G) is on hold at step_10.
- MINF with prefix cache kept across refits (P'''') is the best MINF arm on SWE-Bench Verified (0.49-0.53), level with run A at 20-25; seed replicas P''''2/Q''''2 sit within seed noise of their twins.
- chain V (Megatron vLLM-numerical-parity adapter) trained to step_35 without incident; rkirby judged it converged and ordered two more seeds (V2/V3). Eval of V/V2/V3 is on hold by rkirby.
- chain U on NeMo RL main: two real bugs found and fixed on the way (main915 sessions' handoff): prefix-stitch storm in OpenHands' nemo_gym_client allowlist; Megatron `_compute_prefix_match` applying the KV-only prefix skip on block-aligned continuation chunks of the hybrid model (48 % of 150-200-turn episodes mis-scored; fix 475167fa4). Main also applies the -5 advantage / reward penalties that the v2 SC path never applied; the U recipe disables them.

## 3. How the monitor works (tools in `ops/cmh_monitor/`)

- `tick.sh` — one squeue + per-arm health for every RUNNING lettered arm: label, checkpoint step, driver/SC log mtimes, guard counters
  (`Coordinator: removed engine`, `post_process_requests` — must stay 0), engine step lines (MINF), gym progress bars, dump growth,
  token-level chunk count, last `train step N/`, rung list, stop-rule triggers. Labels come from the run-name `case` at the top; add new arms there.
  Cache dir = the script's own directory. It resolves SC logs by job id via `sclog.sh` — never via `runs/latest` (that symlink is re-pointed at submit time).
- `sclog.sh <jobid>` — finds the SingleControllerActor worker log under `ray_logs/<jobid>[-N]-logs/ray/session_*/logs/worker-*.out`.
- Metrics per step: the dict on the lines just before `train step N/` in the SC log — keys `reward`, `rollout_length/swe_agents_train/mean`,
  `.../truncation_rate`, `gen_kl_error`, `num_masked_seqs_by_logprob_error`, `num_valid_samples`.
- Rung verification: `checkpoints/step_N` has 140 files, 0 tmp, `config.yaml` + `policy/` + `training_info.json`; `latest_checkpoint_status.json` says N.
- Stop rule: when the target step's rung is verified, `scancel` every job of that run by name (the follower is `afterany` on the running one, and
  afterany fires on CANCELLATION too — so hold the next arm's gate job first if it must not start).
- Hang rule (MINF): only when the SC log is silent 20+ min AND no engine step line for 20+ min; 20 min = investigate (Ray state API on the head
  node), not cancel; an engine gap while the SC writes token-dump chunks is the colocated training phase; the engine step lines come from a few
  ranks only and go quiet when the last agents of a step sit in sandbox test execution — SC token-dump chunk lines are the reliable liveness signal.
- Segment rollover costs ~60 min to the first close (the in-flight step is regenerated). Resume with `+checkpointing.load_replay_buffer=false`
  only when the saved buffer is suspect (rkirby ordered it for chain V after two crashed resumes); default restore is standard.
- Routine noise: MaxSequenceLengthOverflowError / "fills or exceeds max_model_len" admissions, nemo_gym health-check warnings,
  wandb monotonic-step warnings, dynamic_engine.py UserWarnings.
- `strip_policy.sh dry|go` — deletes `checkpoints/step_N/policy` (368 GB) where `step_N/hf` exists and N is not the arm's highest checkpoint; only for finished arms (rkirby: never prune a live arm).
- `tokdump/` — the async_rl.dump.token_ids patch (off|digest|full; default off = byte-identical dumps). Caveat: nemo_gym.py pops the round-trip token-id keys before the dump sees them, so it is not a prompt/generation round-trip check.

## 4. Traps that cost time on CMH

- Hydra resolves overrides against the YAML, not the Pydantic schema: key absent from the yaml -> `+key=value`; present -> plain form. Wrong choice kills the job at startup.
- YAML 1.1: unquoted `off`/`on`/`yes`/`no` become booleans (`token_ids: "off"` must be quoted).
- `afterany` dependencies fire on cancellation; cancelling a pending gate job alone releases the next arm. Hold, then cancel.
- `runs/latest` is stale for running segments; resolve logs by job id.
- CMH nodes have 140 CPUs (launch scripts that assume 144 stall Ray head start); worker `apt-get` hangs without egress; 90-min GPU-idle reaper (uid 146504) kills jobs whose Ray steps never launched.
- One generation rank stuck in `prepare_for_generation` can hold 64 nodes for an hour with no log anywhere — Ray state API finds it.
- `num_valid_samples` counts rows not episodes; health = median assistant turns, gen tokens, mean reward from the dumps.
- Live-mounted checkouts (swe_replay_splice, swe_mask_replay, swe_range_replay, swe_vllm_parity, swe_main915) are bind-mounted read-write into running jobs: never checkout/stash/reset there while a job runs; two unattributed wipes of mounted trees happened (swe_dump/nemo_rl 09-19, splice tree 09-24) — keep everything pushed.
- Slurm may REQUEUE a job (site lua policy) with no node failure visible; it restarts under the same id from the rolling checkpoint.

## 5. Code map (GitHub = github.com/ArEsKay3)

| workspace | repo | branch @ commit | on GitHub |
|---|---|---|---|
| swe_dump | RL | rkirby/swe-v2-dump @ 7f8a2b9d | yes |
| swe_dump / swe_minf / swe_replay_splice | Megatron-LM | 880de0fce (tip of rkirby/rlvr-nolap-repro) + the live dynamic_engine guard | guard pushed as rkirby/mlm-880de0fce-dynengine-logprob-guard @ 91cb08ea7 |
| swe_minf | RL | rkirby/swe-v2-minf @ 952eaf85b | yes |
| engine_loop_test | RL | rkirby/engine-loop-test @ 561ccba2 | yes |
| swe_vllm_parity | RL / Megatron-LM | rkirby/swe-v2-vllm-parity @ f910fe13 / rkirby/vllm-parity-armV @ d37db1077 | yes (VLLM parity session, 12:0x 09-28) |
| swe_replay_splice (+ swe_prefix_keep, swe_lr0, swe_mask_replay, swe_range_replay overlays) | RL | local rkirby/swe-v2-dump @ 4f779abc = 7f8a2b9d + splice | PENDING (Cross Train Experiment session asked to push as a new branch) |
| swe_915 | RL / Megatron-LM | rkirby/swe-915-nomask @ 3a7fde92c / fork-915 @ a012970be | PENDING (Cross Train) |
| swe_main915 | RL / Megatron-LM / Megatron-Bridge / Gym | rkirby/swe-main915-latest @ b97225bb6 / mlm-main915-latest @ 475167fa4 / mlm-bridge-main915-latest @ 1f8873bb0 / rkirby/gym-main915 @ d54e6374e | PENDING (Get Latest Main? session) |
| swe_dump/analysis, tools, notebooks | none | — | PENDING (Log Analysis, Data Difference, Data Browser sessions); copies of run_dir_map.md, first10 report and swe_dump tools are under ops/ here |
| evaluation/jobs (SWE-Bench Verified campaign records) | none | — | copied under ops/evaluation/jobs (runs.json, results.md, recipes, per-step logs) |
| this monitor | RL | rkirby/cmh-ops-monitor-20260928 | this branch |

## 6. SWE-Bench Verified evaluation (job dirs under ops/evaluation/jobs/<arm>/, results.md = record)

Pinned NEL recipe replayed from rkirby's account (local sbatch, derived config, shared env file). Completed campaigns: run A, chains F (15-35), G, J, K, P'''', Q'''', P''''2 (5-20), Q''''2 (5-20).
Headline pass@1: run A 0.496 (5) -> 0.512-0.518 (15-35); chain G 0.492 -> 0.464; chain J flat 0.484-0.507; chain K tracks A within 0.013;
P'''' 0.493/0.484/0.502/0.508/0.511/0.507/0.508/0.513/0.506/0.526/0.506/0.508 at 5..60; Q'''' 0.490-0.503 (5-40);
P''''2 0.4848/0.4792/0.4896/0.4988 vs P'''' 0.4932/0.4840/0.5020/0.5080; Q''''2 0.5044/0.4972/0.4944/0.5052 vs Q'''' 0.4904/0.4940/0.4920/0.4924.
Never evaluate the rolling checkpoint; re-queue retry-limit shards until .shard_done (rkirby: "submit until you get it done").

## 7. Open decisions for rkirby (as of 2026-09-28 12:03 PDT)

1. chain V/V2/V3 eval campaign (on hold since 09-27 17:35).
2. chain AA resume (release 4061468) vs. run chain AB (hold 4061469/70 first, then cancel 4061468) vs. release Y/Z.
3. AA/AB step target (25 assumed, never confirmed).
4. gym_results reclaim (~43 TiB) on finished arms; policy/ strip on more exported rungs.
5. chain U start on the regular batch queue (estimate drifting past 14:39).

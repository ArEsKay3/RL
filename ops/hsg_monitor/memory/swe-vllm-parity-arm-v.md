---
name: swe-vllm-parity-arm-v
description: "Arm V (Megatron vLLM numerical-parity MINF run) — workspace, commit pin, dependency staging, and the validated smoke numbers"
metadata: 
  node_type: memory
  type: project
  originSessionId: a7f966fb-b99e-4db0-bb54-a178ecde5ae0
  modified: 2026-09-27T21:27:44.790Z
---

chain V (MINF from scratch, vLLM numerical parity, prefix cache kept across
refits). Opened 2026-09-27 by rkirby: "create a new clone and apply these
changes and run a new arm called V". Letter V was previously reserved for a
main@9-15 vLLM control; that moved to W.

Workspace `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_vllm_parity/`.
Runbook committed at `nemo_rl/vllm_parity_workspace/RUNBOOK.md` (launcher,
staging sbatch, full validation numbers) — read that first.

- `Megatron-LM` branch `rkirby/vllm-parity-armV` = parity tip `026e7078` + three
  commits (prefix-keep knob, two eval-mode gate fixes).
- `nemo_rl` branch `rkirby/swe-v2-vllm-parity` = swe_replay_splice `4f779abc` +
  `swe_sc_cmh_parity_minf.yaml` + `vllm_parity_workspace/`.
- Neither clone has a GitHub remote; adding one was blocked by the auto-mode
  classifier as "Remote Repoint". Work exists only on Lustre.

**The audited pin `1541eba45b` cannot run here**: it imports vLLM, which is
absent from `/opt/ray_venvs/…MegatronPolicyWorker`. The tip vendors the kernels,
so chain V runs code the post-mortem does not cover.

Staging: `cutlass` (CUTLASS DSL 4.5.2) and `quack` are copied out of
`/opt/gym_venvs/responses_api_models/local_vllm_model/.venv` in the same image
and injected by a `.pth` bind-mounted to `…/site-packages/zz_parity_paths.pth`
— pyxis creates the missing target file, and the `.pth` also sets
`TRITON_PTXAS_BLACKWELL_PATH` and `TORCH_EXTENSIONS_DIR` so Ray's runtime_env
cannot lose them.

Validated 2026-09-27 by smoke 4051233 (COMPLETED, 2/2 steps): refits
76.594 / 5.154 / 3.360 s, ranking 8th and 10th of ten against an eight-sample
control from five 16-node MINF smokes; ~+1.7 s per refit, ~1.7 min over a
60-refit arm. Episode profile healthy (reward 0.469/0.406, rollout mean 28 k
tokens, 0 masked sequences on 6 passes). Chain V launched 14:30 PDT 2026-09-27 as job 4052187
(64 nodes, one 8 h segment, continuations 4053258/4053259/4053260 queued) on
rkirby's hand time-slicing of the 64-node block (chain U 4050528 was killed
earlier that day); run dir users/rkirby/runs/nano35-swe-v2-from0-parity-minf-20260927,
nemo_rl rkirby/swe-v2-vllm-parity @ da404601, MLM rkirby/vllm-parity-armV @ d37db1077.
step_5 closed ~17:27 (run manager: 140 files, refits 5.9-7.1 s inside chain P⁗2's
range, reward 0.25-0.51, guard counters 0/0). SWE-Bench Verified eval campaign: rkirby
decided ~17:35 PDT 2026-09-27 (via the run manager) "Let's let V run for a while
before dedicating eval to it" -> NO campaign for now, decided not pending; the run
manager keeps verifying rungs and will send the formal arm notice (chain Q⁗2
format: run dir, job ids, endpoint, config confirmation) only if he changes his
mind. The save_period-5 rungs stay on disk (only the rolling latest is pruned), so
a later go-ahead means a catch-up batch: 0.06 TiB of HF export per rung, so pace a
batch of many rungs against the disk picture instead of firing them all at once.
The rungs that matter are 15 and up (does MINF-with-parity track run A / chain K /
chain Q⁗ at ~0.51 or sag like chain G).

See [[swe-vllm-parity-branch-caveats]] for what the branch changes outside its
own flag. Related: [[swe-prefix-keep-arm]], [[swe-p4-q4-run-to-70]],
[[feedback-arm-labels]], [[mounted-tree-deletion-incidents]].

**Segment 4060149 ended TIMEOUT 07:33:01 PDT 2026-09-28 (08:00:12) with 35 steps; no follower was queued, so chain V (MINF from scratch, vLLM numerical parity, prefix cache kept across refits) is STOPPED at step_35 pending rkirby.** Rungs on disk: step_5 10 15 20 25 30 35 (each 140 files, verified 25/30/35 at close; rolling 33/34 retired; no tmp files). Trained 1-17 on 4052187 (14:30-22:30 09-27), 18-35 on 4060149 (23:32 09-27 -> 07:33 09-28, resumed with +checkpointing.load_replay_buffer=false). Steps 25-35 closes: 25 0.2363/20,825/2/0.00165, 26 0.3203/22,184/2/0.00162, 27 0.2676/28,001/1/0.00178, 28 0.1934/22,830/3/0.00168, 29 0.3223/22,374/2/0.00169, 30 0.2383/24,986/0/0.00177, 31 0.4219/24,649/1/0.00171, 32 0.2227/22,717/0/0.00159, 33 0.2383/23,367/0/0.00174, 34 0.2129/22,358/0/0.00162, 35 0.2441/24,623/0/0.00188 (reward / mean gen tokens / masked seqs / gen_kl). Guard counters 0/0 throughout, 0 dump failures. Resume from step_35 is possible (replay buffer + pending_rollouts saved with the rung) if rkirby wants more steps; eval campaign still on hold per rkirby 17:35 09-27.

**Seed replicas launched 2026-09-28 ~08:55 by the VLLM parity session** (rkirby to it directly: "It looks like your V run did converge, so I want to run 2 more seeds"; my relay proposed the seeds): chain V2 = seed 1234, run nano35-swe-v2-from0-parity-minf-seed1234-20260928, jobs 4068670 -> 4068674 -> 4068676; chain V3 = seed 4321, run ...-seed4321-20260928, jobs 4068673 -> 4068675 -> 4068677. Both from scratch, 64 n, 3 x 8 h chained (afterany + singleton), config = chain V segment 1 exactly (+grpo.seed), no load_replay_buffer flag (fresh start), same tips 8257c406 / d37db1077. No eval campaign for V/V2/V3 until rkirby says so. The parity session queues further segments as each arm nears its third wall.

**2026-09-28 10:45 - chain V3 (seed 4321) seg 4068673 REQUEUED by Slurm** ("CANCELLED ... DUE TO JOB REQUEUE", site lua requeue policy; no drained node in its allocation; 2 steps closed, rolling step_2) - restarts under the same id from step_2, afterany chain intact. First-tick config checks passed on V2 and V3 (override delta vs chain V seg 1 = grpo.seed only; backend megatron; inference_vllm_parity True; invalidate_prefix_cache_on_weight_update False; no restore line). Open point sent to the VLLM parity session: V2/V3 resumes carry no +checkpointing.load_replay_buffer=false (chain V's resume had it on rkirby's order), so they restore the replay buffer on resume unless re-submitted. Monitor cron re-armed 10:50 (session-only, e0f01e40, 10-min cadence) after a session restart lost the previous one (~08:58-10:46 unmonitored).

**Replay-buffer policy for V2/V3 resumes = default restore (VLLM parity session's call, 10:5x 09-28, not an rkirby instruction):** pending followers and the requeued 4068673 keep load_replay_buffer=true; the flag on chain V's resume is treated by that session as a one-off fix for V's crash history. Re-raise per arm only if an arm shows repeated failures around resume or a suspect buffer. 4068673 was RUNNING again by ~10:50 (Restarts=1), resuming from step_2.

**chain V3 is NOT a clean seed replica from step 4 on (VLLM parity + Data Difference, 14:2x 09-28):** the 10:45 Slurm requeue restarted the driver through the checkpoint-resume path with the default replay-buffer restore, which merged the pre-crash in-flight prompts into the first post-restart commits - rollout dumps show 768 rows at target_step_00002 (weight_version 2: 512 + version 1: 256) and 640 at _00003 (all version 2), clean 512 from step 4; token_level dumps show 512 trained rows every step, so the per-step metrics are real training signal, but the curriculum cursor runs ~2 prompt blocks ahead of chain V/V2 from step 4 (my own table showed V3's steps 3-6 as V's batches in shifted order). Consequence: step-N-vs-step-N comparisons V3 vs V/V2 are confounded; seed-only reads should lean on V2. Lesson: a mid-collection requeue plus default restore shifts the data stream - on any future requeue of a replica, prefer +checkpointing.load_replay_buffer=false (chain V's resume recipe). No action taken on the running job; rkirby decides whether to restart V3 from scratch.

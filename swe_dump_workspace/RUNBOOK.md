# SWE-E2E v2 on CMH: MINF chain and dump runs (terse runbook)

Prereqs: CMH login, Slurm account `nemotron_sw_post`, membership in group `dip`
(akamehra's model, data, sif and container dirs are group-only), your own
`WANDB_API_KEY` exported, GitHub access to `ArEsKay3/RL` and
`ArEsKay3/Megatron-LM`, and /scratch quota: 369 GB per checkpoint rung, one rung
every 5 steps, plus ~1 GB per step of dumps.

## 1. Workspace

```bash
U=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users
WS=$U/$USER/workspaces/swe_dump && mkdir -p $WS && cd $WS
git clone -b rkirby/swe-v2-dump git@github.com:ArEsKay3/RL.git nemo_rl      # this branch; rkirby/swe-v2-minf 952eaf85b = plain MINF chain
git -C nemo_rl submodule update --init 3rdparty/Gym-workspace/Gym             # 354babf7e; the launcher bind-mounts it
git clone git@github.com:ArEsKay3/Megatron-LM.git Megatron-LM && git -C Megatron-LM checkout 880de0fce   # MINF only
cp nemo_rl/swe_dump_workspace/launch_swe_dump.sh .                            # launcher must sit next to nemo_rl/ and Megatron-LM/
cp -r nemo_rl/swe_dump_workspace/tools .                                      # HF export sbatch, dump analysis tools
mkdir -p analysis/logs && cp nemo_rl/swe_dump_workspace/sbatch/* analysis/logs/   # analysis sbatch files expect W/analysis/logs
```

Edit the launcher (and `tools/export_checkpoint_hf_v2.sbatch`): set `MINE=$U/$USER`
and `MY_HF_HOME` to a directory you can write. The first run converts the HF base
model to Megatron into `HF_HOME/nemo_rl/` (about 10 min); later runs reuse it.
Everything else (base model `akamehra/swe_e2e_corrected/base_model/step_18/hf`,
data, sif dir, container `rl-gym.63635108-zstd.sqsh`, sandbox) is read from
akamehra's tree.

## 2. Smoke first

```bash
SMOKE=1 ENGINE=vllm bash ./launch_swe_dump.sh     # 8 train + 8 gen nodes, QOS short, 2 steps, no checkpoints
SMOKE=1 ENGINE=minf bash ./launch_swe_dump.sh
```

Pass: Slurm `COMPLETED`, two `train step N/` lines in the SingleControllerActor log,
`[token-dump]` lines and non-empty `dumps/rollouts/*.jsonl` with `full_result`,
no `[rollout-dump] FAILED` or `[token-dump] FAILED`.

## 3. Full runs

Shape is fixed in the launcher: 32 training + 32 generation nodes (4 GPUs each),
training TP4 CP4 EP32, 32 prompts x 16 generations = GBS 512, in_order sampler
lag 1, `min_groups_for_streaming_train 8`, `max_buffered_rollouts 96`, saves every
5 steps. MINF: 32 engines of TP4 EP4, `NRL_MINF_SAMPLING_BACKEND=torch`,
`NRL_MINF_LOGPROBS_MODE=raw_logprobs`, `NVSHMEM_MAX_CTAS=2`, fork Megatron-LM
mounted. vLLM: TP4, `logprobs_mode=processed_logprobs`, container Megatron-LM.

Each invocation submits ONE 4-hour segment. The job name is a Slurm singleton, so
run the same command N times to queue a chain; every segment resumes from the
highest `checkpoints/step_*`. Always pass `+checkpointing.load_replay_buffer=false`
(replay-free resume: the prompt journal regenerates the exact in-order prompts).
Expect 7 to 9 steps per segment and multi-hour queue waits. `EXCLUDE_NODES=<list>`
is available if a node keeps failing.

```bash
# vLLM with dumps, from scratch, to step 60 (about 8 segments)
for i in $(seq 8); do ENGINE=vllm bash ./launch_swe_dump.sh +checkpointing.load_replay_buffer=false; sleep 60; done

# MINF with dumps, continuing from an existing chain's step_10 (about 7 segments)
SRC=$U/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf/checkpoints/step_10
EXP=nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-$(date +%Y%m%d)
mkdir -p $U/$USER/runs/$EXP/checkpoints && cp -a $SRC $U/$USER/runs/$EXP/checkpoints/step_10
rm -rf $U/$USER/runs/$EXP/checkpoints/step_10/hf                              # drop the HF export if the source had one
for i in $(seq 7); do SEED_CHECKPOINT=$SRC ENGINE=minf bash ./launch_swe_dump.sh +checkpointing.load_replay_buffer=false; sleep 60; done

# plain MINF chain from scratch (no dumps), from the swe_minf workspace
for i in $(seq 8); do bash ./launch_swe_minf.sh +checkpointing.load_replay_buffer=false; sleep 60; done
```

Different job names never wait on each other; same-name segments do.

## 4. Where things land

`$U/$USER/runs/<EXP>/`: `checkpoints/step_N/` (config.yaml, policy/weights,
pending_rollouts.pt, replay_buffer.pt, replacement_reserve.pt, training_info.json),
`runs/<YYYYMMDD-HHMM>/{slurm,logs/nemo_gym}`, `ray_logs/<job>-logs/ray-driver.log`,
the controller log `ray_logs/<job>-logs/ray/session_*/logs/worker-*.out` (the one
containing `SingleControllerActor`), `dumps/rollouts/target_step_NNNNN.jsonl`
(target_step N trains in `train step N+1`), `dumps/token_level/step_NNNNN_chunk_CCC.pt`,
and `gym_results/swebench_results_*/` (per-instance SWE artifacts).

## 5. Failure modes seen, all infrastructure

Host OOM on an inference node (Slurm kills the step; resubmit). nvshmem IB init
failure at the first refit (`IBRC QP modify INIT->RTR failed`; note the node,
resubmit). Setup hang: one engine rank never prints `prepare_for_generation END`
(count them across `ray_logs/<job>-logs/ray/*/session_*/logs/worker-*.out`; if
under 128 after 30 min, `scancel` so the chain moves on). Only a repeatable
traceback in nemo_rl or Megatron code justifies stopping a chain.

## 6. HF export of rungs

```bash
cd tools && env -u TMPDIR sbatch -J hf-export-<EXP> --array=5,10,15,20 export_checkpoint_hf_v2.sbatch
```

Writes `checkpoints/step_N/hf/` (14 shards, 62 GB, group-readable) in about 6 min
per step on one node; already-exported steps exit immediately. The job name must
keep the `hf-export-` prefix so it stays out of the training singleton.

Reference vLLM baseline for parity: `$U/akamehra/runs/nano35-swe-v2-stream128-inorder1-cmh-64n` (job 3541201).

## 7. Experiment arms (2026-09-18 to 2026-09-21, CMH)

All arms: same base model, same data, in_order sampler (the same 32 prompts at the same step in every
arm), 32 prompts x 16 generations, `grpo.seed` 42, dumps on, `+checkpointing.load_replay_buffer=false`
on every segment. EXP names are `nano35-swe-v2-stream128-inorder1-cmh-64n-<suffix>`.

| arm | suffix | engine, start point | enable_prefix_caching / overlap_param_gather | segments | launch script |
|---|---|---|---|---|---|
| main chain (MINF from scratch, original) | `minf` (swe_minf workspace, no dumps) | MINF, base model | true / true | 4 h | swe_minf/launch_swe_minf.sh |
| run A (vLLM from scratch) | `vllm_dump-20260918` | vLLM, base model | n/a / true | 4 h | `ENGINE=vllm bash ./launch_swe_dump.sh +checkpointing.load_replay_buffer=false` |
| run B (MINF from MINF step 10) | `minf_dump-from10-20260918` | MINF, seed = main chain step_10 | true / true | 4 h | `SEED_CHECKPOINT=... ENGINE=minf ...` (section 3) |
| chain C (MINF from MINF step 10, no prefix cache) | `minf_dump-nopg-noprefix-20260919` | MINF, seed = main chain step_10 | false / false | 4 h | launch/launch_minf_nopg_noprefix_20260919 (see log names) |
| chain D (vLLM from MINF step 10) | `vllm_dump-from10-nopg-20260919` | vLLM, seed = main chain step_10 | n/a / false | 4 h | as run B with ENGINE=vllm and overlap_param_gather=false |
| chain E (MINF from scratch, prefix cache on; cancelled after 2 steps) | `minf_dump-from0-20260920` | MINF, base model | true / true | 4 h | launch/launch_minf_from0_20260920.sh |
| chain F (MINF from vLLM step 10) | `minf_dump-fromvllm10-20260920` | MINF, seed = run A step_10 | true then false (see below) / true | 4 h | launch/launch_minf_fromvllm10_20260920.sh, launch/launch_minf_fromvllm10_cont_20260921.sh |
| chain G (MINF from scratch, no prefix cache) | `minf_dump-from0-nopg-noprefix-20260920` | MINF, base model | false / false | 4 h, then 8 h on batch_long | launch/launch_minf_from0_nopg_noprefix_20260920.sh |
| chain I / chain J (MINF from scratch, prefix cache on, replicas 1 and 2) | `minf_dump-from0-prefix-nopg-r1-20260921`, `...-r2-20260921` | MINF, base model | true / false | 8 h on batch_long | launch/launch_minf_from0_prefix_nopg_x2_20260921.sh |

`launch/launch_minf_from0_maxtok_nochunk_20260920.sh` (max_tokens 204800, no chunked prefill) was
prepared after a passing 16-node smoke but never run.

Caveat that bit chain F: the launcher's yaml (`swe_sc_cmh_minf.yaml`, inherited by the dump config)
is read when a segment STARTS, not when it is submitted. The MINF default in that yaml became
`enable_prefix_caching: false` on 2026-09-20 12:06 (this branch), so chain F steps 11-18 ran with
prefix caching on and steps 19-35 with it off. Pass the engine settings explicitly on the command
line for every arm; command-line overrides are baked into the sbatch script at submission and the
last occurrence of a key wins (the launcher's own MINF defaults come first, your trailing overrides
after them).

## 8. Reproducing chain I / chain J (MINF from scratch, prefix cache on, overlap_param_gather off)

Per segment (submit N times, one per minute; singleton job name, each resumes from the highest
`checkpoints/step_*`):

```bash
EXP=nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-$(date +%Y%m%d)
ENGINE=minf RUN_DATE=$(date +%Y%m%d) EXP_NAME=$EXP \
  SLURM_PARTITION=batch_long WALLTIME=8:00:00 SLURM_QOS=normal \
  bash ./launch_swe_dump.sh \
    +checkpointing.load_replay_buffer=false \
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false \
    policy.generation.mcore_generation_config.enable_prefix_caching=true
```

Use a second EXP_NAME (`...-r2-...`) for the replica. Launcher knobs: `SLURM_PARTITION` (default batch,
4 h max on CMH; batch_long allows 8 h), `WALLTIME` (only feeds sbatch --time; `scontrol update
JobId=<id> Partition=batch_long TimeLimit=08:00:00` on a pending segment is equivalent),
`SLURM_QOS`, `NUM_TRAIN_NODES`/`NUM_GEN_NODES` (32/32), `EXCLUDE_NODES`, `DRY_RUN=1` (prints the
rendered command; check the trailing overrides and `Walltime:`), `SEED_CHECKPOINT`, `SMOKE=1`.

On another cluster edit the top of `launch_swe_dump.sh`: `SWE_BASE` (base model
`swe_e2e_corrected/base_model/step_18/hf`, data jsonl, sif dir, containers
`rl-gym.63635108-zstd.sqsh` and `nemo-skills-sandbox-no-sync.sqsh`), `MINE` (runs/ and
persistent_cache/ root), `MY_HF_HOME`, `SLURM_ACCOUNT`, `CPUS_PER_WORKER` (140 on CMH), and
`SLURM_COMMENT` (CMH idle-GPU reaper exemption). The Megatron-LM fork mount
(`ArEsKay3/Megatron-LM` 880de0fce, branch rkirby/rlvr-nolap-repro) must contain
`_run_post_refit_hooks` in `megatron/core/resharding/refit.py`; the launcher checks.

Expected rate on CMH: 6-9 train steps per 4 h segment (first step 50-60 min after start), 12-16 per
8 h segment; 64 nodes; ~10 GB of dumps per step.

## 9. Analysis pipeline (tools/, sbatch/)

`sbatch/refresh_all.sbatch` (CPU partition) summarises new rollout and token dumps of every EXP
listed in its `EXPS` line into `analysis/{rollouts,tokens,think_stop,token_bias,think_close}` and
rewrites `analysis/think_close_by_step.md`; `python3 tools/close_vs_length.py` then renders
`analysis/close_vs_length.md|.svg` (per step: n, mean length, reward, truncation, `</think>` closes,
geo-mean P, 10th-pct P, low share P<0.905, think p95). `tools/first10_report.py collect|plot`
(sbatch/first10_collect.sbatch, sbatch/first10_plot.sbatch which renders inside the training
container because matplotlib is not on the login node) produces the matched-prompt first-ten-steps
study (`analysis/first10/first10_report.pdf`, `report.md`). `tools/close_prob_uncertainty.py`
gives rollout- and prompt-cluster bootstrap CIs for the `</think>` statistics;
`tools/first_token_mismatch.py` the per-turn first-token trainer/engine gap. Add a new EXP to
`refresh_all.sbatch` EXPS and to the run maps in `tools/close_vs_length.py`,
`tools/aggregate_think_close.py`, `tools/aggregate_think_audit.py`, `tools/aggregate_token_bias.py`.

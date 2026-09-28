# swe_mask_replay — masked-replay variant of the splice experiment (chain R family)

Prepared 2026-09-27 for the "Data Difference Deep Dive" session's experiment
(spec: /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/datadiff/EXPERIMENT_masked_replay.md).
Question: are chain G's rewarded rollouts with a >4,096-token think turn the poison that makes a
G-steps-1-10 replay (chain R) loop? Replay G steps 1-10 exactly like chain R, but give a listed set
of replayed sample_ids sample_mask 0 (rows stay in their groups, so siblings' leave-one-out
advantages are unchanged; only the listed members' loss contribution disappears), then go live on vLLM.
PREREGISTERED TARGET (rkirby, 2026-09-27 14:38): train step 25 = 10 replayed + 15 live, i.e. stop when
checkpoints/step_25 is complete (140 files, no tmp); read-out windows 11-15 / 16-20 / 21-25 exactly as chain R.

Overlay (the live swe_replay_splice checkout at nemo_rl 4f779abc is not edited): five files mounted over
/opt/nemo-rl/nemo_rl/... by launch_swe_mask_replay.sh, see patches.diff.
- nemo_rl/experience/foreign_rollout_source.py: ForeignRolloutSource(mask_file=...) loads a CSV
  (header, sample_id column; optional target_step column for expected counts), _pack_group sets
  sample_mask[row]=0 for listed members, lookup() prints per-group and per-step masked counts
  ("foreign_rollout: masked k/n replayed sample(s) in group ... step total m/expected").
- nemo_rl/algorithms/single_controller.py: forwards +foreign_rollout.mask_file to the source.
- nemo_rl/experience/rollout_dump.py, nemo_rl/algorithms/single_controller_utils/{config,setup}.py: the run
  manager's token_ids_dump.patch (workspaces/token_ids_dump.patch + _README.md; rkirby 2026-09-27): new key
  async_rl.dump.token_ids = off|digest|full; the arms run +async_rl.dump.token_ids=digest. ON THIS STACK that adds
  exactly one thing per message to dumps/rollouts: its own `token_ids` verbatim (+~0.18 GB/step, +1.8 %). The
  prompt_token_ids / generation_token_ids / compact_prompt_token_ids keys never reach the message dicts — they are
  popped in nemo_rl/environments/nemo_gym.py:861-862 right after the per-turn contiguity assert at :852-858
  ("Non-contiguous messages found!"), so no *_digest keys appear (manager's correction 2026-09-27 16:2x; the
  synthetic validation fixture carried those keys, production data does not). Off = byte-identical to today. Scope:
  replayed (foreign) groups never reach dumps/rollouts/*.jsonl -- generate_and_push_foreign skips write_group
  (nemo_rl/experience/rollout_manager.py ~1497-1512; chain R on disk has no target_step_00000..00009.jsonl, only
  00010+ at 512 rows), so per-message token_ids exist for the LIVE steps 11-25 only. Replay identity and mask
  integrity for steps 1-10 come from dumps/token_level/step_NNNNN_chunk_K.pt (sample_ids, input_ids,
  sample_mask_before/after, rewards, tags), written in _advantage_stage (single_controller.py:2182) for every
  committed group: chain X's input_ids must equal chain G's for the same sample_id, and sample_mask_before must be 0
  exactly on the listed sample_ids (set at pack time, before the seq-error AND, so before == after == 0 there).
Why the zero survives: grpo.compute_and_apply_seq_logprob_error_masking multiplies its threshold
mask by the incoming sample_mask (AND) before writing it back; GRPOAdvantageEstimator's leave-one-out
baseline uses torch.ones_like(rewards), so masking a member does not move its siblings' advantages.
Post-hoc check: dumps/token_level/step_NNNNN_chunk_*.pt sample_mask_before is 0 on the listed sample_ids.

Arms (ENGINE=vllm, same config/shape/seed as chain R nano35-swe-v2-splice-vllm-runGdata-to10-20260924,
64 nodes, 8 h segments in batch_long / hero-res / sla_res_nemotron_sw_post; to avoid colliding with the
existing chain letters A/B/C the arms are chains X, Y, Z — letters confirmed by the run manager 2026-09-27 14:3x, embedded in the job names as maskX/maskY/maskZ):
| arm | job name | source run | mask (masks/) |
|---|---|---|---|
| chain X (masked replay, test) (rewarded deep-think rollouts, 127) | nano35-swe-v2-splice-vllm-runGdata-to10-maskX-rewarded-deep-20260927 | chain G | poison_G_rewarded_deep_steps1-10.csv |
| chain Y (masked replay, control) (random rewarded, no deep think, 127) | nano35-swe-v2-splice-vllm-runGdata-to10-maskY-random-rewarded-20260927 | chain G | control_G_random_rewarded_steps1-10.csv |
| chain Z (masked replay, mirror; optional) (punished deep-think rollouts, 213) | nano35-swe-v2-splice-vllm-runMdata-to10-maskZ-punished-deep-20260927 | chain M | poison_M_punished_deep_steps1-10.csv |
Expected masked counts per train step 1..10 for chain X: 4,10,9,23,12,6,23,18,11,11.

Launch (from /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice, one command per segment; segments singleton-chain by name; chain R reached step 25 inside one 8 h segment; a second segment is queued as margin and is cancelled by the
step_25 stop):
  cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice && ENGINE=vllm RUN_DATE=20260927 WALLTIME=8:00:00 SLURM_QOS=hero-res SLURM_PARTITION=batch_long \
  SLURM_RESERVATION=sla_res_nemotron_sw_post EXP_NAME=<job name> bash /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay/launch_swe_mask_replay.sh \
    +foreign_rollout.source_run=<source run> +foreign_rollout.steps=1:10 \
    +foreign_rollout.mask_file=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay/masks/<csv> +checkpointing.load_replay_buffer=false \
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
Renders: logs/dry_run_render_chain{X,Y,Z}.out. Validation: test_mask_replay.sbatch -> logs/test_mask_replay_<job>.out
(test_overrides_mask.py = config path; test_mask_replay.py = mask semantics on chain G target_step 0, AND
semantics of the seq-error masking, estimator independence). Not launched until rkirby says so.
No auto-stop: at checkpoints/step_25 complete, cancel the running segment and the queue (manager's step_25 trigger).

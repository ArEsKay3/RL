swe_range_replay — step-range replay ablation (chains AA / AB / AC), BUILD ONLY until rkirby's word
====================================================================================================
Spec: swe_dump/analysis/datadiff/EXPERIMENTS.md section 6 (Data Difference, relaying rkirby 2026-09-27 ~21:00; ranges revised
by rkirby ~21:25 to 1:7 vs 8:10). Purpose: if chain X (masked replay, test) loops, locate which replayed steps of chain G carry
the effect. LR warmup is 10 iterations (3e-7 -> 3e-6), so the late steps carry most of the first ten steps' update.
Recipe: chain R's, i.e. the LIVE splice tree's plain launcher (workspaces/swe_replay_splice/launch_swe_splice.sh,
nemo_rl 4f779abc, ENGINE=vllm, container rl-gym.63635108-zstd.sqsh), no overlay, no mask, no digest:
  cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice && ENGINE=vllm RUN_DATE=<launch date> WALLTIME=8:00:00 SLURM_QOS=hero-res SLURM_PARTITION=batch_long \
    SLURM_RESERVATION=sla_res_nemotron_sw_post EXP_NAME=<name> bash launch_swe_splice.sh \
    +foreign_rollout.source_run=<source> +foreign_rollout.steps=<a:b> +checkpointing.load_replay_buffer=false \
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
  (two identical submissions per arm = two 8 h segments chained by singleton; add SLURM_DEPENDENCY=afterany:<ids> for ordering;
  if an arm is inserted ahead of chains Y/Z, HOLD all four Y/Z ids first: afterany fires on cancellation and would release Z)
Arms (labels AA/AB/AC confirmed by the run manager; date suffix = launch date, renders use 20260928):
  chain AA  nano35-swe-v2-splice-vllm-runGdata-8to10-<date>  source chain G (/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920)  steps 8:10  (live 1-7, 11-25)  PRIORITY 1
  chain AB  nano35-swe-v2-splice-vllm-runGdata-1to7-<date>   source chain G  steps 1:7   (live 8-25)
  chain AC  nano35-swe-v2-splice-vllm-runMdata-8to10-<date>  source chain M (/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923)  steps 8:10  (control, optional)
Target: checkpoints/step_25 per arm (manager's stop trigger cancels both segments by name).
Semantics verified in the splice code: steps "a:b" -> range(a, b+1) indexes only those train steps (source target_step a-1..b-1);
a (target_step, prompt_idx) miss returns None and dispatches live generation (single_controller.py ~695-716).
Caveat: in chain AA the policy at steps 8-10 was trained on its own live steps 1-7, so chain G's replayed rows are off-trajectory
there and the 2.0 seq-logprob-error threshold may zero more rows than chain R's two (steps 6 and 10); sample_mask_after in
dumps/token_level shows the count. Replayed groups write token_level chunks but no dumps/rollouts jsonl.
Superseded (first draft 6:10 / 1:5, never launched): logs/superseded_6to10_1to5/; their DRY_RUN renders left empty scaffolding
run dirs runs/nano35-swe-v2-splice-vllm-runGdata-6to10-20260928, ...-runGdata-1to5-20260928, ...-runMdata-6to10-20260928
(nothing deleted; harmless).
Files: logs/dry_run_render_chain{AA,AB,AC}.out, train_cmd_chain{AA,AB,AC}.txt, test_overrides_range.py, test_range_replay.py,
test_range_replay.sbatch (cpu partition, driver venv), logs/test_range_replay_<job>.out.

Validation record (cpu partition, driver venv):
  4057445 (21:16-21:41): ForeignRolloutSource range semantics PASS x3 (G 8:10 -> target steps 7-9, 96 keys, misses for 0/6 return None,
           hit packs 16 rows identical to source; G 1:7 -> target steps 0-6, 224 keys, miss for 7 returns None, hit identical; M 8:10 PASS).
           Config-path test printed correct values but reported FAIL because it hard-coded vllm_cfg.enable_prefix_caching=False; the key is
           ABSENT in chain R and chain X alike (vLLM default, ON on these GPUs); the `false` in the saved yaml is mcore_generation_config's.
  4057715 / 4058117: field-by-field comparison with chain R's saved config.yaml: the only differences are runtime-filled keys
           (policy.generation._mtp_weights_from_refit/_pad_token_id/model_name, vllm_cfg.load_format=dummy, megatron_cfg.train_iters=1156,
           grpo.max_num_steps 1000000 -> 1156) and data.train/validation being list-wrapped in the saved yaml; every setting is identical.
  4058559: same comparison with those runtime keys ignored still prints FAIL(values) for 8 keys that the data pipeline adds at run
           time (data.train/validation env_name=nemo_gym, processor=nemo_gym_data_processor, prompt_file=None, system_prompt_file=None);
           a pre-launch MasterConfig dump can never match a post-run saved yaml literally. Verdict on inspection: every user-settable
           value identical to chain R; range semantics PASS x3 in each job. No further validation runs.

UPDATE 2026-09-28 00:1x: rkirby (via the manager): "Let's extend X, Y, Z to 30 steps each. For now I do want those to run before any AA." So AA/AB/AC wait for chains X, Y and Z (each now to step_30) to finish; AA/AB/AC target stays step_25; renders dated 20260928 will need re-rendering with the actual launch date (likely 20260929).

LAUNCHED 2026-09-28 01:00 (rkirby via the manager: "Let's let AA and AB slip ahead of Y and Z."): chain AA = 4061467 (afterany:4055486:4055489) + 4061468 (singleton+afterany:4061467); chain AB = 4061469 (afterany:4061467:4061468) + 4061470 (singleton+afterany:4061469); chains Y/Z held by the manager (4055490-4055493 JobHeldUser) for release after AB; chain AC unlaunched. Gates verified with scontrol at submission. Per-run provenance: <run>/README_chain{AA,AB}_range_replay.txt.

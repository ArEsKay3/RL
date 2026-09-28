chain Z: masked replay MIRROR: vLLM replays chain M (vLLM from scratch, r1) steps 1-10 with the 213 punished deep-thinking rows given sample_mask 0, then 15 live steps
launched 2026-09-27 18:53 PDT by the Cross Train Experiment session on rkirby's word (via the run manager: "let X,Y,Z run sequentially")
jobs: seg 1 = 4055492 (afterany:4055490 AND afterany:4055491), seg 2 = 4055493 (singleton + afterany:4055492); 64 nodes, 8 h each, hero-res / batch_long / sla_res_nemotron_sw_post
target: checkpoints/step_25 (preregistered by rkirby, 10 replayed + 15 live steps); the run manager's *mask[XYZ]* step_25 trigger cancels both jobs by name
launcher: /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay/launch_swe_mask_replay.sh (md5 632ef64b298263e1ef71fd15663d2eab); stack = swe_replay_splice nemo_rl 4f779abc (rkirby/swe-v2-dump), ENGINE=vllm, container rl-gym.63635108-zstd.sqsh
overrides: +foreign_rollout.source_run=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923 +foreign_rollout.steps=1:10 +foreign_rollout.mask_file=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay/masks/poison_M_punished_deep_steps1-10.csv (md5 9241da0f132a657efd0368f538920f1f) +checkpointing.load_replay_buffer=false +async_rl.dump.token_ids=digest policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
overlay (5 files bind-mounted over /opt/nemo-rl, validated by cpu job 4053246, log /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay/logs/test_mask_replay_4053246.out):
    5763f77b92d967657462258b39d4e0d4  patches/nemo_rl/nemo_rl/algorithms/single_controller.py
    e3228b783fa9ce13cdbd07d367947dd3  patches/nemo_rl/nemo_rl/algorithms/single_controller_utils/config.py
    f30b91b933e321f9f4e37272352bcee9  patches/nemo_rl/nemo_rl/algorithms/single_controller_utils/setup.py
    88ad77fef08fa0283338c035469e46bd  patches/nemo_rl/nemo_rl/experience/foreign_rollout_source.py
    3d211a8defa1915cf4056f8b2cd84202  patches/nemo_rl/nemo_rl/experience/rollout_dump.py
expected masked rows per target_step 0..9 (train steps 1..10): 11,9,3,28,25,31,28,48,14,16 (213)
notes: replayed groups never reach dumps/rollouts/*.jsonl (first jsonl = target_step_00010); token_level/step_00001.. carry sample_mask_before = 0 exactly on the listed rows; ids are re-minted, join to the source by input_ids content (swe_dump/analysis/datadiff/verify_replay.py)
launch logs: /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay/logs/launch_chainZ_seg1_20260927_185320.out, /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay/logs/launch_chainZ_seg2_20260927_185320.out

UPDATE 2026-09-28 00:1x PDT: rkirby (via the run manager) extended chains X, Y and Z to checkpoints/step_30 ("extend X, Y, Z to 30 steps each"); the manager's stop trigger now fires on step_30. The step-range ablation arms AA/AB/AC run only after all three are done.

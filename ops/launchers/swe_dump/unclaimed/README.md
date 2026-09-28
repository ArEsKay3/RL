Uncommitted working-tree state of the two base checkouts on CMH at 2026-09-28 12:05 PDT, preserved verbatim; no session claimed them.

- replay_train.py: untracked file in swe_dump/nemo_rl/examples/nemo_gym/nemotron-3.5-nano/ (2026-09-23 08:06) - replays a source run's dumped rollouts through the real SC training pipeline without a generation engine; companion of launch_swe_replay.sh one level up.
- swe_dump_nemo_rl_worktree.diff: swe_sc_cmh_minf.yaml enable_prefix_caching false -> true (the launcher MINF_OVERRIDES forced true anyway; every MINF arm ran with prefix caching on unless its tag says otherwise).
- swe_minf_nemo_rl_worktree.diff: the opposite one-line edit in swe_minf's copy of the same yaml.

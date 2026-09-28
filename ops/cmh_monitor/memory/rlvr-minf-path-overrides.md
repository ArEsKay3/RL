---
name: rlvr-minf-path-overrides
description: "CMH resource paths for the RLVR dolphin-convergence launch: the HSG defaults in dolphin_convergence_common.sh do not exist here; a CLUSTER=HSG|CMH case (committed f4f89e7, pushed 2026-09-12) now supplies the CMH paths, container and CPUS_PER_WORKER automatically"
metadata:
  type: project
---

The reference-site (HSG) defaults in `RLVR/nemotron-3.5-nano/scripts/dolphin_convergence_common.sh` reference
`/lustre/fs1/...` and other users' scratch, none of which exist on CMH (`aws-cmh-slurm-1`). The CMH values, all
under `/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users` unless noted:

- `CONTAINER=rkirby/containers/nemo-rl_fd45cb8-67127697-gym.sqsh` (imported from `nvcr.io/nvidian/nemo-rl:fd45cb8-67127697-gym`; has nemo.lens, prebaked Gym venvs, libXrender). akamehra's `rl-gym.63635108.sqsh` lacks nemo.lens and no longer works with the V2 pin.
- `TRAIN_PATH=akamehra/training_data/curriculum_amplified_dolphin_v41_cheery_umbrette.train.jsonl/curriculum_amplified_dolphin_v41_cheery_umbrette.train.jsonl`
- `GENRM_MODEL=akamehra/models/hf_judge_models/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM/snapshots/116af7cb1a23ce9017b2c412945b0623252655d2`
- `GENRM_REASONING_PARSER=akamehra/evaluation/ultra_v3_reasoning_parser.py`
- `NL2BASH_JUDGE_MODEL=akamehra/models/Qwen3-235B-A22B-Instruct-2507-FP8`
- Safety judge: `akamehra/models/Nemotron-Content-Safety-Reasoning-4B`
- `SANDBOX_CONTAINER=akamehra/containers/nemo-skills-sandbox-no-sync.sqsh`
- Base model (rkirby's copy with the generation_config.json trailing-comma fix, NOT akamehra's original, whose
  trailing comma silently disables multi-EOS): `rkirby/base_models/nano_v35_sft-upsampled-iter6000-eosfix/hf`
- `PERSISTENT_CACHE=rkirby/persistent_cache`
- `HF_HOME=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/rkirby/hf_home`
- `RESULTS_ROOT=rkirby/runs` (the script asserts it lives under `/lustre/*`)
- `CPUS_PER_WORKER=140` ([[cmh-cpus-per-worker-140]])

**Why:** every one of the HSG defaults 404s here; each missing one cost a full 86-node job start (jobs 3629819,
3629838, 3630635 each died on a different missing path).

**How to apply:** all of the above are the `CMH` arm of the `case "$CLUSTER"` block in
`dolphin_convergence_common.sh` (pipeline branch `rlvr-convergence-minf`, commit f4f89e7, pushed to GitLab
2026-09-12). `CLUSTER` auto-detects from `scontrol show config` ClusterName (`aws-cmh-*` -> CMH, else HSG) and can
be forced by exporting it; individual `VAR=` overrides still win. On CMH no per-path env vars are needed. If the
branch ever loses that block, export the list above by hand. The README on the branch documents the same.

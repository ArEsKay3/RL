---
name: swe-dump-checkout-deletion-incident
description: 2026-09-19 20:37 PDT the live-mounted swe_dump/nemo_rl checkout (and most of its Gym submodule) was deleted by an unknown actor while dump segments 3847633/3847639 were running; restored 20:44-21:00 from the pushed branch and the swe_minf Gym checkout
metadata:
  type: project
---

What happened: at ~20:37 PDT on 2026-09-19 the contents of
`users/rkirby/workspaces/swe_dump/nemo_rl/` vanished except `.python-version`, `examples/` and parts of
`3rdparty/Gym-workspace/Gym/` (cache/ 1.6 TB survived; nemo_gym/, responses_api_*/ etc. were gone). `.git`
was removed too. Run B (3847639) failed every new rollout with FileNotFoundError on
`Gym/responses_api_agents/swe_agents/configs/oh_config.toml` (3926 errors) until the restore; run A had not
started new rollouts yet (0 errors). The mounted Megatron-LM worktree and swe_minf were untouched.

Cause: unknown. Not the checkpoint-verification agent (its transcript only has read-only commands on that
path), no rm/rsync/find process on the login node, the checkout was a standalone clone (not a linked
worktree), the surviving set mixes tracked and untracked files so it is not a `git worktree remove`.
Nothing in my own commands deletes there. Treat as an external actor (another session/job) and keep the
workspace read-only for others.

Restore (done): `git clone -b rkirby/swe-v2-dump git@github.com:ArEsKay3/RL.git nemo_rl.restore`, rsync
--ignore-existing into nemo_rl/, moved .git in (HEAD 33a9bf6a); Gym content rsynced from
`swe_minf/nemo_rl/3rdparty/Gym-workspace/Gym` (commit 354babf7e = pinned gitlink), submodule git metadata
copied from swe_minf/.git/modules, `.git` gitdir file written. Accidental copies of swe_minf's pre-staged
`swe_*_setup` dirs were removed again so the harness keeps using `Gym/cache/swe_agents/...`.
`nemo_rl.restore/` (clone without .git) can be deleted.

Impact: both running segments died with `ValueError: NeMo Gym returned a result with no generation data`
(3847633 FAILED 20:59:30 after train step 23; 3847639 FAILED 21:08:51 after train step 28). The next singleton
segments 3847664 (A) / 3847668 (B) went PENDING(Priority) and will resume from the latest checkpoints on the
restored tree (HEAD 33a9bf6a, git status clean; Gym 354babf7e clean + cache/).

**How to apply:** if the dump chains misbehave after 2026-09-19 21:00, first check that
`swe_dump/nemo_rl/nemo_rl` and `Gym/nemo_gym` still exist; the queued segments launch from this live tree
(USE_SNAPSHOT=0). Related: [[swe-dump-runs]], [[swe-length-growth-investigation]].

**Update 2026-09-19 23:35:** the 20:37 deletion ALSO removed files inside the OpenHands virtualenv in the Gym cache (`Gym/cache/swe_agents/swe_openhands_setup/nv-OpenHands-dd7d06ea/5f0180054732945df08ad2293903e6873f0492b6/OpenHands/.venv`, 750 vs 862 site-packages entries, numpy gone). Chain C first segment 3870192 FAILED at 10 min (22:53) with every rollout dying at `import pandas` in run_infer.py -> step 10 lost 4/32 prompt groups -> RuntimeError floor 0.9. The setup dir was rsynced (--ignore-existing, excluding evaluation/oh) from swe_minf; the other cached harness setups and miniforge3 were intact at shallow depth. 3870199 (chain C seg 2) was held during the restore and must be released after verification.

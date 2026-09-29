---
name: splice-family-handoff
description: "2026-09-28 HSG-move handoff for the splice/replay family: what was pushed where (rkirby/swe-v2-splice eb327474 = 4f779abc + splice_workspaces/, swe-915 branches), HANDOFF.md locations, and the uncommitted-but-identical Megatron guard fact"
metadata:
  type: project
  originSessionId: 824af663-9c1e-4cb4-9c9e-fd1feb14c5fa
---

rkirby (2026-09-28 11:53, via the manager): document every experiment and push all code to his forks for a possible move to HSG.

**Pushed 2026-09-28 12:4x (from a fresh clone at `workspaces/swe_handoff/RL`; no live tree touched):**
- `git@github.com:ArEsKay3/RL.git` **`rkirby/swe-v2-splice`** @ `eb327474` = the splice commit `4f779abc` (local branch `rkirby/swe-v2-dump` in `swe_replay_splice/nemo_rl`, never pushed onto the GitHub `rkirby/swe-v2-dump` 7f8a2b9d) + one commit adding `splice_workspaces/` (119 files: swe_prefix_keep, swe_lr0, swe_mask_replay incl. masks, swe_range_replay, launch_swe_splice.sh, token_ids_dump patch, run_readmes/, HANDOFF.md). Secrets-scanned before commit.
- `ArEsKay3/RL` `rkirby/swe-915-nomask` @ `3a7fde92c` and `ArEsKay3/Megatron-LM` `rkirby/fork-915` @ `a012970be` (swe_915 trees, owner = session 51d77432; pushed as snapshots).
- `swe_replay_splice/Megatron-LM` = 880de0fce + an UNCOMMITTED change in `dynamic_engine.py` that is byte-identical to GitHub `rkirby/mlm-880de0fce-dynengine-logprob-guard` @ 91cb08ea (diff 0 lines) → MINF arms reproducible from that branch.

**HANDOFF.md** (179 lines: repos, stack, workspaces, per-arm table P…AC with jobs/rungs/results/held ids, exact launch/resume commands, 10 traps, HSG changes, CMH-only locations) lives in the branch at `splice_workspaces/HANDOFF.md` and as copies in `workspaces/swe_mask_replay/`, `swe_range_replay/`, `swe_prefix_keep/`, `swe_lr0/`.

**Why:** rkirby wants to resume elsewhere; checkpoints/dumps stay on CMH, code+docs must be portable.
**How to apply:** future splice-family edits go to `rkirby/swe-v2-splice` (never force-push; the live trees stay bind-mounted); update HANDOFF.md when an arm's state changes; the manager compiles HANDOFF_INDEX ([[hsg-move-handoff]]). Related: [[swe-splice-experiments]], [[swe-mask-replay-experiment]], [[swe-prefix-keep-arm]].

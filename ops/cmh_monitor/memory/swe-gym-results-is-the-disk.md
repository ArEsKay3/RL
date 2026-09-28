---
name: swe-gym-results-is-the-disk
description: "gym_results (per-instance SWE harness output) is the dominant consumer on the nemotron Lustre quota - ~34 TiB inferred, invisible to du on the login node"
metadata:
  type: project
---

When the 100 TiB quota on `/scratch/fsw/portfolios/nemotron` (uid 6428) gets tight, **`runs/*/gym_results` is where the space is**, not checkpoints.

Accounting done 2026-09-27 against 93 TiB used: policy checkpoints ~32 TiB (counted, each `policy/` is exactly 368 GiB), HF exports ~3.9 TiB (each `hf/` exactly 62 GiB), rollout+token dumps ~5.75 TiB, `nemo-evaluator-rundirs` 3.08 TiB, workspaces+misc ~1.7 TiB, `ray_logs` negligible. That leaves **~34 TiB unaccounted, all of it `gym_results`** across 49 run directories.

**You cannot measure it from the login node.** Four separate `du`/`find` surveys were OOM-killed; a single `gym_results` directory could not even be file-counted in 40 s. Sampling works instead: `find -printf '%s %b\n' | head -20000` showed mean file size **4.06 MB** and 20,000 files alone = 75.65 GiB. Block overhead is not a factor (apparent vs allocated 1.01).

Per-step training cost measured the same day: dumps 9-12 GB/step, a rung 0.336 TiB, an HF export 0.06 TiB — but the empirical whole-system rate is **~0.14 TiB/step all-in** at 0.71 TiB/h across two concurrent 64-node arms. The excess over the modelled 0.09 is gym_results growing per step.

rkirby authorised deleting `gym_results` on all arms except the live ones on 2026-09-27 07:0x; script at /home/rkirby/.claude/jobs/e247f0cd/tmp/gymprune.sh (keep-list re-checked inside the loop, refuses any path that is not `<run>/gym_results`). Deleting 7 of 46 directories moved headroom 6.46 -> 8.91 TiB.

**How to apply:** it is re-derivable harness output, not weights — prefer it over checkpoints when freeing space. Delete per-arm and read `lfs quota` after each rather than trying to size it first. Related: [[feedback-never-prune-live-arms]], [[swe-p4-q4-run-to-70]].

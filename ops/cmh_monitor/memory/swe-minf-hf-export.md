---
name: swe-minf-hf-export
description: "Megatron->HF export of the SWE-E2E v2 + MINF chain rungs on CMH: sbatch users/rkirby/workspaces/swe_minf/tools/export_checkpoint_hf_v2.sbatch, job name hf-export-<EXP_NAME>, output checkpoints/step_N/hf (62 GB, group-readable); array 3830144 exported steps 5-35 in ~6 min each on 2026-09-17; rerun with the full step list as rungs 40-60 land"
metadata:
  type: project
---

User asked 2026-09-17 ~20:55 PDT to convert the chain's checkpoints to HF following their HSG handoff doc
(export_checkpoint_hf_v2.sbatch), adapted to CMH. Rungs are every 5 steps here (save_period 5), so the step list is
5,10,...,60; rolling resume checkpoints (step_37, step_39, ...) are never exported.

**Script:** `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_minf/tools/export_checkpoint_hf_v2.sbatch`
(logs in `tools/logs/<jobname>_<A>_<a>.{out,err}`). CMH adaptations: account nemotron_sw_post / partition batch /
`--qos=short` / `--gres=gpu:4` / 1h30 walltime; RUN_ROOT = `$MINE/runs/$EXP_NAME/checkpoints` (checkpoints live in a
`checkpoints/` subdir here); container = akamehra rl-gym.63635108-zstd.sqsh (same as training; has
/opt/nemo-rl/examples/converters/convert_megatron_to_hf.py and the
/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker venv, confirmed from the
training driver log); BASE_HF = akamehra swe_e2e_corrected/base_model/step_18/hf (tokenizer, chat template,
modeling_nemotron_h.py, ultra_v3_reasoning_parser.py copied from there); mounts mirror the training overlays
(workspace nemo_rl -> /opt/nemo-rl/nemo_rl, examples/converters, fork Megatron-LM 880de0fce over the Bridge's
nested Megatron-LM, runs/, akamehra swe_e2e_corrected/, HF_HOME on nemotron_sw_pre) so the checkpoint is loaded by
the code that wrote it; NRL_IGNORE_VERSION_MISMATCH=1 like training. Converter = nemo_rl
export_model_from_megatron (AutoBridge, gloo temp dist context, single process; node has 900 GB host RAM).

**Submit (idempotent, safe to pass the full list every time):**
```
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_minf/tools && \
env -u TMPDIR sbatch -J hf-export-nano35-swe-v2-stream128-inorder1-cmh-64n-minf --array=5,10,15,20,25,30,35,40,45,50,55,60 export_checkpoint_hf_v2.sbatch
```
Job name MUST keep the `hf-export-` prefix: the training chain is a singleton keyed on the bare EXP_NAME.
A task exits 0 at once if `step_N/hf/model.safetensors.index.json` exists, exits 1 on an incomplete checkpoint
(rung still saving) or a leftover `hf/` / `hf.exporting.*` (inspect, delete, rerun).

**Output:** `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf/checkpoints/step_N/hf/`
(sharded model-*.safetensors + index + config.json + tokenizer/template/modeling files), `chmod -R g+rX` before the
final rename (user asked for group read). Parent dirs are already drwxr-sr-x (group llmservice).

**Verify:** `.out` ends with `HF export complete: <path>`; or loop over step_* checking hf/model.safetensors.index.json.

**History:** array **3830144** (tasks 5,10,15,20,25,30,35) submitted 2026-09-17 21:00 PDT, all 7 COMPLETED exit 0 in 5m48s-6m15s each (all started at once on QOS short); verified 21:10: 14 shards + 25 files, 62 GB per step, 0 files without group read, config.json architectures NemotronHForCausalLM / model_type nemotron_h. Exported so far: step_5..step_35. Array **3830729** (tasks 5-40; only 40 does work) submitted 2026-09-17 21:50 PDT after step_40 landed; all 8 tasks COMPLETED (5-35 exited in ~20 s as already-exported no-ops, 40 in 6m14s); step_40/hf verified 22:09: 14 shards, 25 files, 62 GB, group-readable. Exported: step_5..step_40. Array **3839942** (task 45 only) submitted 2026-09-18 08:10 PDT, COMPLETED in 6m23s; step_45/hf verified 08:29 (14 shards, 25 files, 62 GB, group-readable). Exported: step_5..step_45. Array **3841235** (task 50 only) submitted 2026-09-18 09:30 PDT, COMPLETED in 5m53s; step_50/hf verified 09:49 (14 shards, 25 files, 62 GB, group-readable). Exported: step_5..step_50. Next: 55, 60 as they land (pass only the missing steps). Monitor ticks should
resubmit with the full list once step_40/45/50/55/60 land (the chain [[swe-minf-64n-run]] saves every 5 steps).
step_55 export 3851537 COMPLETED in 6m05s (2026-09-18 18:32-18:38), checkpoints/step_55/hf verified (14 shards, 25 files, 62 GB, group-readable). Exported: step_5..step_55. Remaining: 60 when it lands.

**Chain F export (user, 2026-09-21 06:3x "convert to hf all these checkpoints, proper read permissions"):** the swe_dump copy `workspaces/swe_dump/tools/export_checkpoint_hf_v2.sbatch` now differs from the swe_minf copy: logs in swe_dump/tools/logs, `EXPORT_WS` (workspace whose nemo_rl/Megatron-LM are mounted; pass the tree that wrote the checkpoint, swe_dump for the dump runs), `EXPORT_OUT_ROOT` (output parent; default = the step dir, so hf lands in step_N/hf), and `chmod -R g+rX,o+rX` (world read, per the user's checkpoint-permission rule). Submitted 06:36 PDT for EXP nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920 (chain F, MINF from vLLM step 10): array **3896149** steps 10,15,20,25 in place (`EXPORT_WS=.../swe_dump`), array **3896150** step 26 with `EXPORT_OUT_ROOT=runs/<EXP>/hf_exports/step_26` because step_26 is a rolling checkpoint that the trainer deletes when step_27 saves (in-place output would vanish). Rungs 30 and 35 will need the same command when they land. Verification script: /home/rkirby/.claude/jobs/e247f0cd/tmp/wait_hf_export.sh.
**Chain F export result (2026-09-21 06:43 PDT):** all 5 tasks COMPLETED exit 0 in 6m04s-6m27s (started together 06:36:23 on QOS short). Outputs verified: checkpoints/step_10/hf, step_15/hf, step_20/hf, step_25/hf and hf_exports/step_26/hf, each 14 shards, 25 files, 62 GB, 0 files without group or world read; parent dirs drwxr-sr-x. Exported for chain F (MINF from vLLM step 10): 10, 15, 20, 25, 26. Remaining: rungs 30 and 35 when they land (same command, array=30 or 35, in place).
**Chain F step_30 export (2026-09-21 07:56):** array 3897306 task 30 COMPLETED in 6m02s (submitted 07:49 right after the rung landed); checkpoints/step_30/hf verified 14 shards, 25 files, 62 GB, group+world readable. Exported for chain F: 10, 15, 20, 25, 26 (hf_exports), 30. Remaining: 35 when it lands (chain F stops at step_35).
**Chain F step_35 export (2026-09-21 09:26):** array 3898881 task 35 COMPLETED in 6m06s; checkpoints/step_35/hf verified 14 shards, 25 files, 62 GB, group+world readable. Chain F (MINF from vLLM step 10) export set is COMPLETE: checkpoints/step_{10,15,20,25,30,35}/hf + hf_exports/step_26/hf. Nothing further to export for chain F (chain stopped at step_35).
**Chain G exports (user, 2026-09-22 07:0x "convert all checkpoints and run evals for all checkpoints of this run, final checkpoint priority"):** EXP nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920 (chain G, MINF from scratch, no prefix cache; still training toward step_35 at the time). Array 3924077 steps 5,10,15,20,25,30 in place (`EXPORT_WS=.../swe_dump`) and 3924078 step 32 (rolling) -> `hf_exports/step_32/hf`, all COMPLETED 07:10-07:17 in ~6 min each, verified 14 shards / 25 files / 62 GB / group+world readable. Remaining: step_35 when the chain saves it (in place, array=35). Eval job dir: users/rkirby/evaluation/jobs/chainG-minf-from0-nopg-noprefix-swe (same tooling as chain F).
**Chain G final exports (2026-09-22 08:37):** 3925220 step_35 in place and 3925221 step_36 -> hf_exports/step_36/hf, both COMPLETED (~6 min) and verified (14 shards / 65,827,374,264 B / 6,513 tensors / group+world readable). Chain G exports complete: steps 5,10,15,20,25,30,35 in place; 32 and 36 under hf_exports/.

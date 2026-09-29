# SWE-Bench Verified eval runner on HSG (oci-hsg-cs-001)

Same evaluation as the CMH campaigns under `ops/evaluation/jobs/`: reviewed recipe `swe-bench-verified-nano-3.5`, NEL Next toolchain 569ff7e80669ff9e200b15666702c11994723403, eval image nemo-evaluator-next:0.5.0.1-harbor, 10 shards x 1 node x 4 GPUs, 500 problems x 5 repeats, vLLM TP2 DP2, OpenHands SDK 1.17.0, 200 turns, 10,800 s solve cap, temperature 1.0 / top-p 0.95. HSG numbers are comparable to the CMH ones.

Source config: `source_full_config_20260918_115143_3acfec8b.yaml` = ksanthanam's certified oci-hsg run of 2026-09-18 (config commit 81337a5b). Verified 2026-09-29 that `run_benchmark.py --recipe swe-bench-verified-nano-3.5 --cluster oci-hsg --resolve-only` on the shared checkout still resolves to the same commit and toolchain (Evergreen certificate) and that the resolved config differs only in schema defaults. Replaying the frozen file avoids the 3-minute compose, the login-node process cap and the SSH drops described in ksanthanam's SWEBENCH_RUNBOOK.md.

| item | value on HSG |
|---|---|
| checkpoints | /lustre/fsw/portfolios/llmservice/users/rkirby/runs/<run>/checkpoints/step_N/hf |
| job records | /lustre/fsw/portfolios/llmservice/users/rkirby/evaluation/jobs/<arm>/ (runs.json, results.md, step_N-config.diff, step_N-eval-attempted, submission logs) |
| eval rundirs | /lustre/fsw/portfolios/llmservice/users/rkirby/nemo-evaluator-rundirs/nano_v35/swebench-verified-fixed/<run_id>/ |
| venv | /lustre/fsw/portfolios/llmservice/users/rkirby/.efb-runner-cache/pinned-venvs/nel-next/569ff7e80669ff9e200b15666702c11994723403 |
| credentials | /lustre/fsw/portfolios/nemotron/projects/nemotron_n3_post/eval/.frontier_eval/.env (shared; never print) |
| Slurm | batch_long, 8 h, QOS normal, account nemotron_sw_post, reaper exemption 480 min in the sbatch comment |

Flow per rung: `verify_export.py` (structure + read-test of every shard, catches 0600 weights) -> `ARM=<arm> RUN=<run> ./submit_step.sh N` -> record run_id and job ids in runs.json -> poll shards -> `resume_dead_shards.py JOB_DIR` every tick until every shard has .shard_done with 250 verified rows (re-queue until done) -> `merge_step.sh RUN_DIR` if the auto-merge did not fire -> `pooled_check.py RUN_DIR` must say complete (500 problems, 2,500 rows) and match report.md -> results.md.

Never evaluate the rolling checkpoint; only closed rungs. A preempted shard can write a false .shard_done; the resume script treats fewer than 250 verified rows as incomplete. `afternotok` follow-ups stop on CANCELLED and after 3 infra retries, so dead shards need the resume script, not the chain.

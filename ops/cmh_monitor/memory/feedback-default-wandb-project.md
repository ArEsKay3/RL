---
name: feedback-default-wandb-project
description: "W&B projects for Nano 3.5 RLVR runs: the MR52 launcher defaults to nvidia/nano35-rlvr-main-tot (smoke: nano35-rlvr-main-tot-smoke) and the user accepted that for minf_v2_9_15; the older pipeline_A line logged to nano35-rlvr-convergence. Echo the resolved project before submitting."
metadata:
  type: feedback
---

On 2026-09-15 the user asked to send minf_v2_9_15 "to the default wandb". I used the MR52 launcher's default
(`WANDB_PROJ` -> nano35-rlvr-main-tot; run https://wandb.ai/nvidia/nano35-rlvr-main-tot/runs/mirzxvk5). The user
first expected nano35-rlvr-convergence (where minf-v2, job 3688117, lives) but then confirmed the launcher default
was the right call once they saw the launcher had changed.

**Why:** the project moved with the recipe rewrite; the user was not tracking that the MR52 branch defaults
differently from the pipeline_A line.

**How to apply:** for launches from pipeline_B (MR52) leave `WANDB_PROJ` at the launcher default unless told
otherwise; for pipeline_A-line launches use nano35-rlvr-convergence. In both cases print the resolved
`entity/project/run` from the dry-run summary before submitting so the user can catch a mismatch early.

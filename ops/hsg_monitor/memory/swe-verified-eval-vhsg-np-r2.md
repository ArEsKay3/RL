---
name: swe-verified-eval-vhsg-np-r2
description: "SWE-Bench Verified eval campaign on the three HSG no-prefix parity arms (V-HSG-np-r2 seeds 42/1234/4321), started 2026-09-30 22:2x CDT by the Eval Runner session; HF exports in place under checkpoints/step_N/hf; rungs 5/10/15 first, step_20 next"
metadata:
  type: project
---

Started 2026-09-30 ~22:20 CDT. The Eval Runner session says rkirby told it directly "Can you start evals for the 3 runs we're doing" (its account; rkirby has not yet repeated it to the run manager).

Arms: nano35-swe-v2-from0-parity-minf-noprefix-hsg-r2-20260930 (seed 42), -r2-seed1234-20260930, -r2-seed4321-20260930 under /lustre/fsw/portfolios/llmservice/users/rkirby/runs. Labels V-HSG-np-r2 / V-HSG2-np-r2 / V-HSG3-np-r2 (MINF from scratch, vLLM-parity build, no prefix cache, HSG), letters pending rkirby.

HF export: /home/rkirby/swe_dump/tools/export_checkpoint_hf_v2.sbatch (HSG-ported; batch/short, 1 node, nemotron_sw_post) with EXPORT_WS=/lustre/fsw/portfolios/llmservice/users/rkirby/workspaces/swe_vllm_parity so the Megatron tree that wrote the checkpoints does the conversion; arrays 7581196 / 7581197 / 7581199, --array=5,10,15; output checkpoints/step_N/hf. QOS short allows ~2 concurrent export tasks per user (QOSMaxJobsPerUserLimit).

Eval job dirs: /lustre/fsw/portfolios/llmservice/users/rkirby/evaluation/jobs/{vhsg-np-r2-seed42-swe,vhsg2-np-r2-seed1234-swe,vhsg3-np-r2-seed4321-swe}; rundirs under .../users/rkirby/nemo-evaluator-rundirs/nano_v35/swebench-verified-fixed/; each rung = 10 x 1-node jobs, batch_long 8 h, QOS normal.

**How to apply:** the run manager confirms each rung closed (140 files, 0 tmp, training_info.current_step = N) and messages the Eval Runner when step_20 (and later rungs) close; never evaluate or export the rolling checkpoint; exports add hf/ only, no pruning on live arms. tick.sh lists hf-export-* and nel-eval* jobs as "aux" lines. Related: [[feedback-vhsg-paused]], [[hsg-cluster-facts]], [[swe-minf-hf-export]], [[feedback-eval-requeue-until-done]].

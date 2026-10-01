---
name: chainw-hf-rungs-on-hsg
description: "2026-10-01: chain W (vLLM from scratch, NeMo RL main 09-24 stack, masking off) HF exports for rungs 5-35 were copied from CMH to HSG under runs/nano35-swe-main915-64n-vllm-20260930 (hf/ only, no policy/); Eval Runner says rkirby ordered their SWE-Bench Verified evals and paused the V-HSG-np-r2 evals"
metadata:
  type: project
---

Verified by the run manager 11:3x CDT 2026-10-01: /lustre/fsw/portfolios/llmservice/users/rkirby/runs/nano35-swe-main915-64n-vllm-20260930/checkpoints/step_{5,10,15,20,25,30,35}/hf each hold a complete HF export (model.safetensors.index.json present, 25 files per rung), no policy/ or training state; the checkpoints dir was written by akamehra at 11:06 CDT. These are copies of chain W's CMH rungs, so this run dir is eval-only on HSG, not a training arm.

The Eval Runner session says rkirby told it at 11:1x CDT: start evals on the seven chain W rungs and "pause all other eval if they are getting in the way". It cancelled all 112 V-HSG-np-r2 eval shards (cached verified attempts kept, resumable), and is submitting chain W evals 35 -> 5 (70 shards) under evaluation/jobs/chainW-main915-vllm-swe. It will still HF-export new r2 rungs when announced; their evals wait for rkirby to unpause. This is the Eval Runner's account of rkirby's order, not heard by the run manager.

**How to apply:** tick.sh maps nano35-swe-main915*vllm* to chain W; no training job should appear under that name on HSG unless rkirby launches one. Keep announcing closed r2 rungs to the Eval Runner. Related: [[swe-verified-eval-vhsg-np-r2]], [[swe-main915-workspace]], [[feedback-eval-requeue-until-done]].

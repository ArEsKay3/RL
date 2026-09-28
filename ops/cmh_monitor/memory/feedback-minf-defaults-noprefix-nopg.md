---
name: feedback-minf-defaults-noprefix-nopg
description: "User rule, REVISED 2026-09-24 - MINF default is now prefix caching ON (was off since 2026-09-20); overlap_param_gather stays OFF. The real default lives in the launcher MINF_OVERRIDES, not the yaml."
metadata:
  type: feedback
---

MINF defaults as of 2026-09-24 11:24: `enable_prefix_caching=true`, `overlap_param_gather=false`.

The prefix-caching half REVERSES the 2026-09-20 rule. The user authorised it after confirming chain J (MINF from scratch, prefix cache on, replica 2) ran prefix caching on: verified True in chain J's own resolved config at step_5, step_25 and step_50 — it never flipped — with overlap_param_gather False, seed 42, backend megatron. Chain I (replica 1) is True as well. Chain J reached its step_50 target cleanly, which is the evidence behind making it the default.

WHERE THE DEFAULT ACTUALLY LIVES — this is the part that is easy to get wrong. It is the hardcoded `MINF_OVERRIDES` array in the launcher, NOT the yaml:
- swe_dump/launch_swe_dump.sh lines 65-66
- swe_replay_splice/launch_swe_splice.sh lines 65-66
Both arrays are expanded BEFORE `"$@"` in the final `exec`, so a caller-supplied Hydra override on the command line still wins (that is how chain J got prefix caching true while the launcher said false). The vLLM branch sets `MINF_OVERRIDES=()`, so vLLM runs are untouched by any of this.

Changed 2026-09-24 in all four places so the launcher and the yaml agree: line 66 of both launchers false->true, and `enable_prefix_caching` at swe_sc_cmh_minf.yaml:32 false->true in both the swe_dump and swe_replay_splice checkouts (that yaml is byte-identical across them, now md5 a5219f478788). `overlap_param_gather=false` at line 65 of both launchers was deliberately left alone.

Config inheritance for reference: swe_sc_cmh_dump_minf.yaml -> swe_sc_cmh_minf.yaml -> swe_sc_cmh_stream4.yaml; swe_sc_cmh_dump_vllm.yaml -> swe_sc_cmh_stream4.yaml directly, so the vLLM path never reads swe_sc_cmh_minf.yaml. `overlap_param_gather: true` at swe_sc_cmh_common.yaml:254 is the raw yaml default, which is why the launcher has to force it false.

**Why:** the user had set prefix caching off as a 2026-09-20 default while it was under suspicion in the length-growth investigation; chain I and chain J cleared it.

**How to apply:** new MINF runs need no prefix-caching override at all now. Keep passing `overlap_param_gather=false` explicitly anyway — the user wants it stated, and any launcher that lacks the MINF_OVERRIDES block would otherwise inherit true. Related: [[swe-dump-runs-nopg]], [[feedback-arm-labels]], [[swe-length-growth-investigation]].

2026-09-26 **vLLM prefix caching DEFAULTS ON in this tree, and every vLLM arm so far ran with it ENABLED.** Verified by me, not taken on trust:
- `nemo_rl/models/generation/vllm/vllm_worker.py:84 _resolve_enable_prefix_caching` does `vllm_cfg.get("enable_prefix_caching", None)`; when absent it returns `torch.cuda.get_device_capability()[0] >= 8`, i.e. TRUE on these nodes.
- Exact-key yaml walk of the vLLM defaults chain (swe_sc_cmh_dump_vllm -> swe_sc_cmh_stream4 -> swe_sc_cmh_common -> ../../configs/grpo_math_1B) finds `policy.generation.vllm_cfg.enable_prefix_caching` ABSENT at every level.
**So run A, chain M, chain K, chain Q, chain R, chain R-prime and chain S all ran with vLLM prefix caching ON.** The `'enable_prefix_caching': False` that appears in those arms' MasterConfig dumps is the **mcore** key (policy.generation.mcore_generation_config), which is inert on a vLLM arm - reading it as the vLLM setting is the trap. chain Q-quadruple-prime (nano35-swe-v2-from0-noprefix-vllm-20260926) is the FIRST vLLM arm with it actually off, via +policy.generation.vllm_cfg.enable_prefix_caching=false.

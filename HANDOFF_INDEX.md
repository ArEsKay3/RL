# HANDOFF INDEX — CMH SWE-E2E v2 campaign (compiled by the run manager, 2026-09-28 12:03 PDT)

rkirby's instruction (2026-09-28 11:53 PDT): every session documents its workstream and pushes all code to his forks so work can resume on HSG.
Status per workstream (updated as sessions report; "claimed" = the session's report, "verified" = ls-remote by the manager):

| workstream | session | HANDOFF.md | repo / branch / commit | status |
|---|---|---|---|---|
| run manager: monitor, memory notes, eval records, base launchers | Manage Ongoing Runs | ops/cmh_monitor/HANDOFF.md (this branch) | ArEsKay3/RL rkirby/cmh-ops-monitor-20260928 | this commit |
| Megatron dynamic_engine guard (live under all swe-v2-dump MINF arms) | Manage Ongoing Runs | (commit message) | ArEsKay3/Megatron-LM rkirby/mlm-880de0fce-dynengine-logprob-guard @ 91cb08ea7 | verified pushed |
| chain V / V2 / V3 (vLLM numerical parity) | VLLM parity | workspaces/swe_vllm_parity/HANDOFF.md + nemo_rl/vllm_parity_workspace/HANDOFF.md | ArEsKay3/RL rkirby/swe-v2-vllm-parity @ f910fe13; ArEsKay3/Megatron-LM rkirby/vllm-parity-armV @ d37db1077 | verified pushed (ls-remote 12:0x) |
| chain U (NeMo RL main 09-24 stack) | Get Latest Main? | workspaces/swe_main915/HANDOFF.md = nemo_rl/swe_main915_workspace/HANDOFF.md in the branch | committed LOCALLY: RL rkirby/swe-main915-latest @ 6a6f38b8 (b97225bb6 + workspace/handoff commit); Megatron-LM mlm-main915-latest @ 475167fa4; Megatron-Bridge mlm-bridge-main915-latest @ 1f8873bb0 (= upstream pin); Gym rkirby/gym-main915 @ d54e6374e | **NOT PUSHED** - that session's permission policy blocked git push; push commands in its HANDOFF section 2 (targets ArEsKay3/RL, /Megatron-LM, /Megatron-Bridge, /Gym) |
| splice family P/Q/R/P'/Q'/P''/P'''/P''''/Q''''/P''''2/Q''''2/X/Y/Z/AA/AB/AC + overlays (prefix_keep, lr0, mask_replay, range_replay, token_ids_dump, per-arm run READMEs) | Cross Train Experiment | splice_workspaces/HANDOFF.md in the branch; copies at workspaces/swe_mask_replay, swe_range_replay, swe_prefix_keep, swe_lr0 | ArEsKay3/RL rkirby/swe-v2-splice @ eb327474 (= 4f779abc splice on 7f8a2b9d + splice_workspaces/, 119 files) | verified pushed (ls-remote 12:5x) |
| swe_915 stack (chains S/T) | Cross Train Experiment | (covered in splice HANDOFF) | ArEsKay3/RL rkirby/swe-915-nomask @ 3a7fde92c; ArEsKay3/Megatron-LM rkirby/fork-915 @ a012970be | verified pushed (ls-remote 12:5x) |
| data-difference analysis (datadiff tooling, specs, mask lists, LOG/SUMMARY/EXPERIMENTS, verify_replay.py) | Data Difference Deep Dive (2) | workspaces/swe_dump/analysis/datadiff/HANDOFF.md = analysis/swe_datadiff/HANDOFF.md on the branch | ArEsKay3/RL rkirby/swe-analysis-datadiff @ f7095cd (220 files; parts/ 8 GB excluded as regenerable) | verified pushed (ls-remote 12:0x) |
| log analysis: loop-share pipeline, reports, run-dir map, tools incl. dumpbrowse | Log Analysis | analysis/loopfeedback/HANDOFF.md (branch and on disk under swe_dump/analysis/loopfeedback/) | ArEsKay3/RL rkirby/swe-analysis-logs @ 889b0a80 (orphan branch: analysis/ + tools/ only) | verified pushed (ls-remote 12:0x) |
| dump browser (server, UI, start script, notebooks) | Data Browser | tools/dumpbrowse/HANDOFF.md (branch and on disk under swe_dump/tools/dumpbrowse/) | ArEsKay3/RL rkirby/swe-tools-dumpbrowse @ 36d45dd | verified pushed (ls-remote 12:0x) |
| SWE-Bench Verified eval runner | SWE Verified Eval Runner (not live at 11:54) | records copied to ops/evaluation/jobs by the manager | — | copied |

## Update 2026-09-29 08:15 PDT (run manager)

ls-remote at 08:10: RL rkirby/cmh-ops-monitor-20260928 (this push), rkirby/swe-v2-vllm-parity @ 576c3901 (moved on since f910fe13), rkirby/swe-v2-splice @ 494d17c6 (moved on since eb327474), rkirby/swe-tools-dumpbrowse @ 282c3f7f, rkirby/swe-analysis-datadiff @ f7095cd, rkirby/swe-analysis-logs @ 889b0a80, rkirby/swe-915-nomask @ 3a7fde92c, rkirby/swe-v2-dump @ 7f8a2b9d, rkirby/swe-v2-minf @ 952eaf85b, rkirby/engine-loop-test @ 561ccba2; Megatron-LM rkirby/mlm-880de0fce-dynengine-logprob-guard @ 91cb08ea7, rkirby/vllm-parity-armV @ d37db1077, rkirby/fork-915 @ a012970be, rkirby/rlvr-nolap-repro @ 880de0fce.

**Still NOT on GitHub: the chain U / chain W stack** (RL rkirby/swe-main915-latest, Megatron-LM mlm-main915-latest @ 475167fa4, Megatron-Bridge mlm-bridge-main915-latest @ 1f8873bb0, Gym rkirby/gym-main915 @ d54e6374e) — committed locally in workspaces/swe_main915; push commands in workspaces/swe_main915/HANDOFF.md section 2. rkirby must run them (that session is not allowed to push).

Queue state and the void never-cancel list: see ops/cmh_monitor/HANDOFF.md section 7.

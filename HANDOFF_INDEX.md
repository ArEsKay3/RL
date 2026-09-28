# HANDOFF INDEX — CMH SWE-E2E v2 campaign (compiled by the run manager, 2026-09-28 12:03 PDT)

rkirby's instruction (2026-09-28 11:53 PDT): every session documents its workstream and pushes all code to his forks so work can resume on HSG.
Status per workstream (updated as sessions report; "claimed" = the session's report, "verified" = ls-remote by the manager):

| workstream | session | HANDOFF.md | repo / branch / commit | status |
|---|---|---|---|---|
| run manager: monitor, memory notes, eval records, base launchers | Manage Ongoing Runs | ops/cmh_monitor/HANDOFF.md (this branch) | ArEsKay3/RL rkirby/cmh-ops-monitor-20260928 | this commit |
| Megatron dynamic_engine guard (live under all swe-v2-dump MINF arms) | Manage Ongoing Runs | (commit message) | ArEsKay3/Megatron-LM rkirby/mlm-880de0fce-dynengine-logprob-guard @ 91cb08ea7 | verified pushed |
| chain V / V2 / V3 (vLLM numerical parity) | VLLM parity | workspaces/swe_vllm_parity/HANDOFF.md + nemo_rl/vllm_parity_workspace/HANDOFF.md | ArEsKay3/RL rkirby/swe-v2-vllm-parity @ f910fe13; ArEsKay3/Megatron-LM rkirby/vllm-parity-armV @ d37db1077 | verified pushed (ls-remote 12:0x) |
| chain U (NeMo RL main 09-24 stack) | Get Latest Main? | pending | rkirby/swe-main915-latest, mlm-main915-latest, Bridge, Gym | requested 11:58 |
| splice family P/Q/R/P'/Q'/P'''/P''''/Q''''/X/Y/Z/AA/AB + overlays + swe_915 | Cross Train Experiment | pending | splice commit 4f779abc (new branch), swe-915 branches | requested 11:58 |
| data-difference analysis (datadiff tooling, specs, mask lists, LOG/SUMMARY/EXPERIMENTS, verify_replay.py) | Data Difference Deep Dive (2) | workspaces/swe_dump/analysis/datadiff/HANDOFF.md = analysis/swe_datadiff/HANDOFF.md on the branch | ArEsKay3/RL rkirby/swe-analysis-datadiff @ f7095cd (220 files; parts/ 8 GB excluded as regenerable) | verified pushed (ls-remote 12:0x) |
| log analysis: loop-share pipeline, reports, run-dir map, tools incl. dumpbrowse | Log Analysis | analysis/loopfeedback/HANDOFF.md (branch and on disk under swe_dump/analysis/loopfeedback/) | ArEsKay3/RL rkirby/swe-analysis-logs @ 889b0a80 (orphan branch: analysis/ + tools/ only) | verified pushed (ls-remote 12:0x) |
| dump browser (server, UI, start script, notebooks) | Data Browser | tools/dumpbrowse/HANDOFF.md (branch and on disk under swe_dump/tools/dumpbrowse/) | ArEsKay3/RL rkirby/swe-tools-dumpbrowse @ 36d45dd | verified pushed (ls-remote 12:0x) |
| SWE-Bench Verified eval runner | SWE Verified Eval Runner (not live at 11:54) | records copied to ops/evaluation/jobs by the manager | — | copied |

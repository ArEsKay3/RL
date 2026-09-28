---
name: swe-dumpbrowse-server
description: "Stdlib web browser for the SWE dump rollout jsonl files (runs/*/dumps/rollouts); where it lives, how to start it, port, cache, data-format gotchas"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6a167afc-de95-4dcd-b150-94f6a7197bfa
  modified: 2026-09-25T01:47:47.927Z
---

`swe_dump/tools/dumpbrowse/` (server.py + app.html + start.sh + README.md) built 2026-09-24 at rkirby's request ("browse the dumped json, click to see full turns"). Start with `start.sh [port]`; default port 8790 because 8765 on aws-cmh-slurm-1-vscode-01 is owned by someone else's process (do not kill it). Binds 127.0.0.1 only; user reaches it via `ssh -L 8790:localhost:8790` or VS Code port forwarding. pid in dumpbrowse.pid, log dumpbrowse.log. Cache (format v2, 2026-09-25) of line offsets + per-rollout summaries incl. loop stats (loop_turns, loop_tokens, loop_pct, min_zlib, runaway, max_turn_tokens): `swe_dump/analysis/dumpbrowse_cache/<run>__<file>.idx.json`; built by parsing every line in a 4-process spawn pool (~1 min per 10 GB; 56 s for the 23 GB chain G step-7 file); v1 caches upgrade automatically on the next visit; incremental for growing files.

**Data format (rollout jsonl):** one rollout per line, 4-40 MB each, 512 lines per target_step file (7-23 GB). Top-level scalars precede `messages` (fast head parse works); `full_result.response.output` is the Responses-API item list (reasoning / message / function_call with `prompt_str` (full chat prompt, grows to 260 KB per turn) and `generation_str` (raw generation incl. `</think>` + `<tool_call>`) / function_call_output); a runaway last turn has a reasoning item but no function_call. `instance_config.ng_global_config_dict_str` contains api_key lines: server redacts them.

**Restart-duplicate gotcha:** a 4 h segment TIMEOUT mid-step leaves both attempts in the append-only jsonl (chain G target_step 6/7 = 976/1008 lines, 61/63 groups); only the post-resume attempt is trained (token-level step N sample_ids == jsonl target_step N-1 of the later attempt). Training step = target_step + 1.

**Trainer-metric join (2026-09-28, requested via the Data Difference Deep Dive session):** server.py `Metrics` loads analysis/datadiff/browser/rollout_metrics_<ARM>.csv + long_turns_<ARM>.csv (BROWSER_HANDOFF.md documents columns; regenerate with analysis/datadiff/browser_export.py), joins on (run, target_step, sample_id) and turn index k; exposes mask_metric_X/Z, r_deep_contrib, in_X/Y/Z_mask_rank, group_* aggregates, per-turn deep_tokens/turn_contrib/close_p_tr; `trained` flag = has a metrics row (restart duplicates show `untrained`). Turn index alignment verified (n_tok and zlib identical on chain G step 7). Sorting puts nulls last in both directions.

**Why:** the token-level .pt dumps only carry ids; the rollout jsonl is the readable record of every turn, needed to look at loops by hand.

**How to apply:** for any "look at the rollouts" request point the user at the browser (or add an API endpoint) instead of ad-hoc scripts; keep it stdlib-only (no venv on the login node, see [[swe-length-growth-investigation]]).

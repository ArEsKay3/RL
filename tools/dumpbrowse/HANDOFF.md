# dumpbrowse — handoff (written 2026-09-28 for a possible move from CMH to HSG)

Web browser for the SWE-E2E v2 dump runs' rollout records. Python 3 standard library only
(no venv, no pip), one process, one port. Built 2026-09-24 to 09-28 by the "Data Browser"
Claude session for rkirby.

## Files (this directory = `tools/dumpbrowse/` in branch `rkirby/swe-tools-dumpbrowse` of git@github.com:ArEsKay3/RL.git)

| file | purpose |
|---|---|
| `server.py` | HTTP server + API: run/file listing, lazy jsonl indexing (parallel), rollout/turn views, loop detection, trainer-metric join, redaction |
| `app.html` | single-page front end (hash routes: `#/`, `#/run/<run>`, `#/run/<run>/file/<file>`, `.../rec/<i>`, `#/search/<q>`) |
| `start.sh [port]` | detached start (setsid nohup), pid in `dumpbrowse.pid`, log in `dumpbrowse.log`; default port 8790 |
| `arms.json` | run-directory name -> arm letter + description (self-contained copy of the CMH labelling) |
| `README.md` | user-facing feature list |
| `notebooks/browse_dumps*.ipynb` | earlier pandas notebooks over the same jsonl (outputs stripped) |

## Start on another cluster

    export DUMPBROWSE_ROOT=/path/to/runs            # directory holding <run>/dumps/rollouts/target_step_*.jsonl
    export DUMPBROWSE_CACHE=/path/to/writable/cache  # index cache, created if missing
    export DUMPBROWSE_METRICS=/path/to/datadiff/browser   # optional: rollout_metrics_<ARM>.csv + long_turns_<ARM>.csv
    export DUMPBROWSE_LABELS=/path/to/loopfeedback        # optional: jobs.txt + splice_arms.json (arms.json already covers CMH runs)
    ./start.sh 8790
    # then from a laptop:  ssh -L 8790:localhost:8790 <login node>   ->  http://localhost:8790

Equivalent flags: `python3 server.py --root ... --cache ... --metrics ... --host 127.0.0.1 --port 8790`.
Binds to localhost only; on a shared login node every user of that node can reach it through
their own tunnel. To expose it on all interfaces (`--host 0.0.0.0`) add authentication first.
The process dies with a node reboot (that happened on CMH 2026-09-28); an `@reboot` crontab
entry calling `start.sh` fixes that.

## On CMH (as of 2026-09-28)

* runs root `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs` (41 runs with rollout dumps, 5-25 GB per step file)
* code + cache under `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/` (`tools/dumpbrowse`, `analysis/dumpbrowse_cache`)
* trainer metrics from the Data Difference Deep Dive session: `analysis/datadiff/browser/` (regenerate with `analysis/datadiff/browser_export.py`; columns in `BROWSER_HANDOFF.md` there)
* running instance: login node aws-cmh-slurm-1-vscode-01, port 8790 (8765 there belongs to someone else)

## Data format (rollout jsonl, written by the `async_rl.dump` patch on branch rkirby/swe-v2-dump)

* One rollout per line (4-40 MB), 512 lines per training step, file `target_step_%05d.jsonl`
  where the number is 0-based: training step N (token-level dump `step_%05d`) == file `target_step_(N-1)`.
* Top level: `sample_id` (`<group uuid>_g<k>`), `group_id`, `completion_index`, `target_step`,
  `start/end_weight_version`, `committed_at`, `prompt_idx`, `prompt_metadata` (instance_id, repo,
  problem_statement), `reward`, `truncated`, `num_assistant_turns`, `generation_length`,
  `total_tokens`, `messages` (role + n_tokens per turn, no text), `rollout_metrics`,
  `prompt_input`, `full_result`.
* `full_result.response.output` is the Responses-API item list: per assistant turn a
  `reasoning` item (`summary[0].text` = the think block), a `message` item (usually "\n"), a
  `function_call` (name, arguments, plus `prompt_str` = the exact chat prompt sent to the
  engine, up to 260 KB, and `generation_str` = raw generation incl. `</think>` and
  `<tool_call>`), then `function_call_output`. A runaway last turn has a reasoning item but no
  function_call. `full_result` also has `resolved`, `patch_exists`, `model_patch`,
  `agent_error_kind` (max_iteration / context_window / stuck_in_loop / other), timings,
  `per_turn_metrics` (token_usages, response_latencies) and `instance_config`, whose
  `ng_global_config_dict_str` contains api_key lines (the server redacts them).
* Restart duplicates: a segment that times out mid-step leaves both attempts in the append-only
  file (e.g. chain G target_step 6/7 = 976/1008 lines); only the post-resume attempt was trained.
  With trainer metrics loaded the browser marks the earlier attempt `untrained`.

## Index cache

`<cache>/<run>__<file>.idx.json`, format v2: byte offsets and lengths of every line plus a
per-rollout summary (ids, reward, truncation, error kind, token counts, loop stats:
`loop_turns`, `loop_tokens`, `loop_pct`, `min_zlib`, `runaway`, `max_turn_tokens`). Built on
first visit by parsing every line in a 4-process pool (about 1 min per 10 GB), extended
incrementally when a file grows, rebuilt automatically if the format version is older.
The cache is derived data: safe to delete, not worth copying between clusters.

## Loop definition used by the browser

Repetitive turn = assistant turn of >= 1,000 tokens whose reasoning text compresses with
zlib level 6 to < 10 % of its size (same criterion as `analysis/loopfeedback`). Runaway =
rollout truncated and its last turn never reached a tool call.

## Trainer-metric join (optional)

`rollout_metrics_<ARM>.csv` / `long_turns_<ARM>.csv` are joined on (run, target_step,
sample_id) and the 0-based assistant-turn index. Exposed: adv, mask_metric_X / Z, r_deep_contrib,
mean_1mp_deep, n_deep_turns, in_X/Y/Z_mask_rank, group_* aggregates, per-turn deep_tokens,
turn_contrib, close_p_tr. Metrics are arm-local. Files are re-read within 30 s of changing.

## Related tooling not in this branch

`analysis/loopfeedback/` (loop-share tables, lf_worker.py, splice_*.py) and `tools/*.py`
(token-level dump analysis, pt_numpy.py) belong to the Log Analysis / Data Difference sessions'
handoffs.

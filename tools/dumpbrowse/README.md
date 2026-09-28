# dumpbrowse — browse the SWE dump rollout jsonl files

Stdlib-only HTTP server (Python 3) plus a single-page front end. Serves every
`<run>/dumps/rollouts/target_step_*.jsonl` under the runs root, one rollout per line.

    ./start.sh [port]           # default 8790 (8765 is taken on the login node), binds 127.0.0.1 only
    kill $(cat dumpbrowse.pid)  # stop
    python3 server.py --help    # --root, --cache, --host, --port

From a laptop: `ssh -L 8790:localhost:8790 <login node>` then open http://localhost:8790.
VS Code Remote forwards the port automatically when the server is running.

Views: runs (with arm letters from analysis/loopfeedback/jobs.txt and splice_arms.json) →
step files → rollout table (sort by any column, filter by reward / truncated / error kind /
free text) → one rollout: turns (reasoning, tool call, tool output, per-turn tokens and
latency, zlib ratio of the reasoning, raw generation_str and the prompt_str as sent to the
engine), problem statement and model patch, full field tree, the GRPO group of the same
prompt, raw JSON download. Keys j / k step through rollouts. The search box finds an
instance_id / sample_id / group_id / prompt_idx across all indexed files.

Indexing: the first visit to a file scans it once and parses every rollout in 4 worker
processes (about 1 min per 10 GB), caching line offsets plus a per-rollout summary in
analysis/dumpbrowse_cache/ (cache format v2). Files that grew since (a step still
streaming) show a re-index link; only the new tail is parsed.

Loop columns in the rollout table (also filterable with the "loops" selector): `loops` =
number of assistant turns whose reasoning is 1,000+ tokens with zlib ratio < 0.10 (the
repetitive-block criterion of the loop-share tables), `loop %` = their tokens over the
rollout's generated tokens, `min zlib` = lowest ratio over the rollout's long turns (red
below 0.10), `max turn tok` = largest assistant turn, and a `runaway` badge = rollout
truncated with its last turn never reaching a tool call.

Strings longer than 20,000 characters are elided from the record view and loaded on click.
Values under keys named like api_key / secret / password are redacted by the server.

Trainer metrics (optional): if analysis/datadiff/browser/rollout_metrics_<ARM>.csv and
long_turns_<ARM>.csv exist (written by analysis/datadiff/browser_export.py, see
BROWSER_HANDOFF.md there), the server joins them on (run, target_step, sample_id) and, for
turns, on the 0-based assistant-turn index. Files with a join get a "metrics" badge; the
rollout table gains sortable columns adv, mask X, mask Z (|adv| x sum of 1-p over think
tokens beyond position 4096, the ranking behind the chain X / chain Z mask lists),
R_deep contrib (signed), 1-p deep, deep turns, group net, cancel, a "mask list" column with
X #rank / Y #rank / Z #rank badges, and an `untrained` badge for jsonl rows with no trainer
row (restart duplicates). The rollout view highlights deep turns (violet border, jump
links) with per-turn contrib, 1-p deep and p(</think>); the group tab shows the 16
siblings' signed contributions plus the group net and cancel ratio. Metrics are arm-local.
The CSVs are re-read automatically when they change (checked every 30 s); /api/metrics
shows what is loaded. Override the directory with --metrics.

Mask-list view (`#/mask/<run>/X|Y|Z`, linked from the runs page, the rollout table note
and the rollout header): every rollout of a run that sits in the chain X / Y / Z list, across
all its steps in one sortable table (rank, step, instance, adv, mask metric, R_deep contrib,
deep turns and tokens, 1-p deep, loop flags, group net and cancel), with a link to the rollout
and an inline "deep turns" expander that shows the reasoning of the turn(s) carrying the
gradient. Steps whose file is not indexed yet are indexed on first visit (two files at a time).

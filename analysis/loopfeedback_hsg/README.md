# Loop-share analysis on HSG (vLLM-parity arms V-HSG / V-HSG2 / V-HSG3)

Port of `../loopfeedback` (see its HANDOFF.md for definitions) to the HSG cluster. Only paths changed; the labeling worker is the same code.

| item | value |
|---|---|
| run root | /lustre/fsw/portfolios/llmservice/users/rkirby/runs |
| tokenizer | pjin .../step_18/hf (same vocab and special ids 11/13 as the CMH base) |
| python | uv venv /home/rkirby/.venvs/loganalysis (numpy only) |
| arms | jobs_hsg.txt: VH, VH2, VH3 |

Run: `source hsg_env.sh; $LF_PY list_new_chunks.py > jobs_new.txt; xargs -P 4 -L 1 $LF_PY lf_worker.py < jobs_new.txt; $LF_PY hsg_loopshare.py; $LF_PY hsg_rollout_stats.py`.
Outputs: hsg_loopshare.txt/.csv (per step joint and strict share next to the CMH comparators from ../loopfeedback/joint_loopshare.csv), hsg_rollout_stats.json (per step reward, truncation, length quantiles from the rollout jsonl headers). parts/ and joint_gen_cache.json are regenerable and not committed.

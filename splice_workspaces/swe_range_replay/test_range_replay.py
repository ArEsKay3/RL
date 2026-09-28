"""Exercise ForeignRolloutSource with a train-step range that does not start at 1.

usage: test_range_replay.py <source_run> <start_step> <end_step>
Checks: index covers exactly target steps start-1..end-1 with 32 prompts each; a
lookup for a target step outside the range returns None (live fall-through); a
lookup inside the range packs a 16-row group with sample_mask all ones whose
input_ids match the source token-level rows.
"""
import glob
import json
import sys

import torch

from nemo_rl.experience.foreign_rollout_source import ForeignRolloutSource

src, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
steps = range(start, end + 1)
fs = ForeignRolloutSource(source_run=src, steps=steps, pad_token_id=0)
idx = fs._index
tsteps = sorted({k[0] for k in idx})
per = {t: sum(1 for k in idx if k[0] == t) for t in tsteps}
print(f"range {start}:{end} -> indexed target_steps {tsteps} prompts per step {per} total keys {len(idx)}")
ok_index = tsteps == [s - 1 for s in steps] and all(v == 32 for v in per.values())

def first_prompt_idx(target_step):
    path = f"{src}/dumps/rollouts/target_step_{target_step:05d}.jsonl"
    with open(path) as f:
        return json.loads(f.readline())["prompt_idx"]

outside = [t for t in (0, start - 2, end) if 0 <= t <= 9 and t not in tsteps]
miss_ok = True
for t in outside:
    r = fs.lookup(t, first_prompt_idx(t))
    print(f"lookup(target_step={t}, prompt_idx from its own jsonl) -> {'None (live fall-through)' if r is None else 'HIT (unexpected)'}")
    miss_ok &= r is None
r_ok = fs.lookup(999, 0) is None

t_in = start - 1
p_in = first_prompt_idx(t_in)
hit = fs.lookup(t_in, p_in)
hit_ok = hit is not None
if hit_ok:
    tb, tags = hit
    n = tb["input_ids"].shape[0]
    mask_ones = bool(torch.all(tb["sample_mask"] == 1.0))
    rows = fs._get_step_rows(t_in)
    gid = idx[(t_in, p_in)]
    members = [(sid, row) for sid, row in rows.items() if row["group_id"] == gid]
    ids_match = all(torch.equal(tb["input_ids"][i, : row["length"]], row["input_ids"]) for i, (_, row) in enumerate(members))
    chunk_files = sorted(glob.glob(f"{src}/dumps/token_level/step_{t_in + 1:05d}_chunk_*.pt"))
    print(f"lookup(target_step={t_in}, prompt_idx={p_in}) -> group {gid[:8]} rows={n} tags={len(tags)} sample_mask all ones={mask_ones} input_ids match source token_level={ids_match} (source files step_{t_in + 1:05d}: {len(chunk_files)} chunk(s))")
    hit_ok = n == 16 and len(tags) == 16 and mask_ones and ids_match
print("RESULT", "PASS" if (ok_index and miss_ok and r_ok and hit_ok) else "FAIL")

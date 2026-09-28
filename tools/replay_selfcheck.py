#!/usr/bin/env python3
"""Compare a replay run's step-1 advantages against the source run's real step-1
advantages, joined by sample_id. sample_ids are carried through unchanged by
replay_train.py, so they join directly. A match beyond float noise is strong
evidence the prompt_ids_for_adv reconstruction (grouping) in replay_train.py is
correct; a mismatch means it is wrong.
"""
import glob
import sys

import torch

replay_dir, source_dir = sys.argv[1], sys.argv[2]


def load_step(run_dir: str, step: int) -> dict[str, torch.Tensor]:
    pattern = f"{run_dir}/dumps/token_level/step_{step:05d}_chunk_*.pt"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    out: dict[str, torch.Tensor] = {}
    for p in paths:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        lengths = obj["input_lengths"].long().reshape(-1)
        adv_off = torch.cat([torch.zeros(1, dtype=torch.long), (lengths - 1).cumsum(0)])
        for i, sid in enumerate(obj["sample_ids"]):
            out[sid] = obj["advantages"][adv_off[i] : adv_off[i + 1]]
    return out


replay_adv = load_step(replay_dir, 1)
source_adv = load_step(source_dir, 1)

common = sorted(set(replay_adv) & set(source_adv))
print(f"replay rows: {len(replay_adv)}  source rows: {len(source_adv)}  common sample_ids: {len(common)}")
if not common:
    print("NO COMMON SAMPLE_IDS -- cannot compare. sample_id scheme diverged.")
    sys.exit(1)

max_abs_diff = 0.0
n_checked = 0
n_mismatched_shape = 0
for sid in common:
    a, b = replay_adv[sid], source_adv[sid]
    if a.shape != b.shape:
        n_mismatched_shape += 1
        continue
    d = (a - b).abs().max().item()
    max_abs_diff = max(max_abs_diff, d)
    n_checked += 1

print(f"checked: {n_checked}  shape-mismatched (skipped): {n_mismatched_shape}")
print(f"max |advantage diff| across all common sample_ids, step 1: {max_abs_diff:.6g}")
if max_abs_diff < 1e-3:
    print("PASS: advantages match beyond float noise -- prompt_ids_for_adv grouping is correct.")
else:
    print("FAIL (or needs inspection): advantages diverge beyond float noise.")

---
name: swe-rollout-jsonl-token-ids
description: "Optional per-message token-id fields in the SWE rollout jsonl dumps (async_rl.dump.token_ids off|digest|full), patch + README paths, cost, which parsers are affected"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 7cb50873-4185-4239-a055-62bd9f83c0ac
  modified: 2026-09-27T22:51:17.900Z
---

2026-09-27: the Manage Ongoing Runs session wrote a patch adding `async_rl.dump.token_ids` (default `"off"` = byte-identical output) to the rollout jsonl dumper.
- patch: /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/token_ids_dump.patch
- readme: /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/token_ids_dump_README.md
- `digest`: every message gains `token_ids`; assistant messages gain `generation_token_ids`, `prompt_token_ids_digest` {len, sha1, head, tail}, `compact_prompt_token_ids_digest`. `full`: verbatim `prompt_token_ids` / `compact_prompt_token_ids` instead of digests (+86 % size, O(turns^2)); digest costs +1.8 % on ~10 GB/step.
- Keys are only added, never renamed; absent-at-source keys stay absent; malformed keys are emitted as `null`.
- Each arm's session decides whether to enable it; assume `off` per arm until the manager says otherwise.

**How to apply:** loop-share tooling (lf_worker on token_level .pt, joint_loopshare) is unaffected. For per-turn token work, `digest` mode removes the reconstruction from packed .pt + n_tokens described in [[swe-dump-per-turn-structure]]; per-row prefix identity can be checked with the sha1 digests. Related: [[swe-length-growth-investigation]].

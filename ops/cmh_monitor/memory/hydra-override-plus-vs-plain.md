---
name: hydra-override-plus-vs-plain
description: "Hydra validates overrides against the YAML, not the Pydantic schema - a key added to a config class still needs + unless the yaml declares it; cost chain V a segment on 2026-09-27"
metadata: 
  node_type: memory
  type: reference
  originSessionId: a7f966fb-b99e-4db0-bb54-a178ecde5ae0
  modified: 2026-09-28T06:19:05.703Z
---

`parse_hydra_overrides` runs **before** Pydantic validation and resolves keys against the **YAML tree only**.

- Key present in the yaml -> plain form `a.b.c=value`. Using `+` raises *"Could not append to config: an item is already at a.b.c"*.
- Key absent from the yaml, even if it exists on the Pydantic config class -> **`+a.b.c=value` is required**. Plain form raises *"Could not override 'a.b.c'. To append to your config use +a.b.c=value"*.

**Incident 2026-09-27:** [[swe-token-ids-dump-patch]] added `token_ids` to the Pydantic `RolloutDumpConfig`. I concluded from that the key was "in the schema" and told the VLLM parity session to drop the `+`. It was right for `swe_main915`, whose yaml declares `token_ids: digest` at `swe_sc_cmh_dump_minf.yaml:15`, and wrong for `swe_vllm_parity`, whose yaml did not. chain V's continuation 4053258 died at config parse after 3:09 (`OverridesError`), and its two remaining continuations carried the same string. I held 4053259/4053260 before they could repeat it; the parity session cancelled them and resubmitted as **4059655** (resumed from step_17), ~35 min of arm downtime.

**Fix applied:** the parity session declared `token_ids: off` in `swe_vllm_parity/.../swe_sc_cmh_dump_minf.yaml:12` (commit 100a756f), matching swe_main915's convention, so the plain form is now correct for that tree. Verified the inheritance holds: `swe_sc_cmh_parity_minf.yaml` starts `defaults: swe_sc_cmh_dump_minf.yaml`.

**Why:** the failure is silent until a job starts and then costs ~3 minutes plus the queue slot, and a chained arm burns every queued segment the same way in minutes.

**Second failure, same incident:** the fix for the first one wrote `token_ids: off` in the yaml, which YAML 1.1 parses as the **boolean False**; Pydantic's `Literal["off","digest","full"]` rejected it (`input_value=False, input_type=bool`) and job 4059655 died at 2:57. Fixed by quoting (`token_ids: "off"`, commit 8257c406) **and** by hardening the schema — the shared patch now carries a `field_validator(mode="before")` coercing `False -> "off"`. `True` is still rejected.

**The root cause was the schema choice, not the yaml:** `Literal[...]` containing a bare `off`/`on`/`yes`/`no` is a landmine in any YAML-fed config. Prefer `none`/`disabled`. That one is mine.

**How to apply:** before advising an override form, `grep` the **target tree's** yaml for the key. Never carry an override conclusion from one workspace to another - the trees diverge. And **parse the config file with the actual parser** (`yaml.safe_load`) before trusting an edit to it; that needs no allocation and would have caught both of these. The agreed framing with the parity session: "advice did not transfer across trees without verification", not one person's error. Declaring the key in the yaml with a default removes the judgment call entirely and is the preferred fix. Related: [[swe-splice-experiments]] (the earlier `+`-append trap, same family).

**Second failure, same incident, different bug: unquoted `off` is a YAML 1.1 boolean.** The declare-it-in-the-yaml fix above was applied as `token_ids: off` (unquoted) and resubmitted as **4059655** — which FAILED at 2:57 with a *different* error, `pydantic_core.ValidationError: Input should be 'off', 'digest' or 'full' [input_value=False, input_type=bool]`. YAML's classic `on/off/yes/no` -> bool coercion turned the default into `False`, which `Literal["off","digest","full"]` correctly rejected. This makes the yaml default unusable standing alone: every segment submitted without an explicit override would die the same way. Fixed by quoting it, `token_ids: "off"` (commit 8257c406), verified with a plain `yaml.safe_load` before committing rather than reasoning from the schema again. Meanwhile the concurrently-queued **4059729**, which carried an explicit `async_rl.dump.token_ids=digest` override, was unaffected (the override replaces the yaml default before Pydantic ever sees it) and is the job that kept chain V running through both failures.

**Compounding lesson:** two failures in the same incident, each ~3 minutes, each invisible in `provenance.txt` (which only proves the override string was submitted, not that it parsed), each the direct byproduct of fixing the previous one. The generalizable rule: when a config default is a bare `on/off/yes/no/true/false`-shaped word going into a YAML file, quote it, or use a value that cannot collide with YAML's implicit typing (`none`/`disabled`/`enabled` instead of `off`/`on`). This is a defect in the token-ids patch's own schema choice, not something tree-specific.

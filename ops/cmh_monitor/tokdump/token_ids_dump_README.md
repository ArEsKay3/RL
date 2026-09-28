# Dumping prompt_token_ids / generation_token_ids in the rollout jsonl

rkirby, 2026-09-27: dump the round-trip token ids so they can be verified directly.
This is the bug class that killed chain U today — OpenHands' `nemo_gym_client.py`
carries a hardcoded 3-name allowlist and silently dropped `compact_prompt_token_ids`,
producing 1-turn episodes at reward 0.0000 while every other health signal read clean.

## Patch

`token_ids_dump.patch` (in this directory) — 3 files, applies with `patch -p1` from
the nemo_rl checkout root. Generated against `swe_dump/nemo_rl`; the three files are
byte-identical across the splice, prefix-keep and parity trees, so it should apply
cleanly everywhere. `swe_main915` diverges at the `single_controller.py` call site
only, which this patch does not touch.

- `nemo_rl/algorithms/single_controller_utils/config.py` — new field
  `async_rl.dump.token_ids: Literal["off","digest","full"] = "off"`
- `nemo_rl/experience/rollout_dump.py` — `_message_summary()` gains a mode argument;
  new `_as_id_list()` and `_id_digest()` helpers
- `nemo_rl/algorithms/single_controller_utils/setup.py` — pass the mode through and
  echo it in the "Rollout dump enabled" banner

`off` is the default and produces byte-identical output to today. Nothing changes for
a run that does not set the key.

## What each mode emits, per message

| mode | `token_ids` | `generation_token_ids` | `prompt_token_ids` | `compact_prompt_token_ids` |
|---|---|---|---|---|
| `off` | — | — | — | — |
| `digest` | verbatim | verbatim | `{len, sha1, head[8], tail[8]}` | `{len, sha1, head[8], tail[8]}` |
| `full` | verbatim | verbatim | verbatim | verbatim |

A key absent from the message is absent from the dump; a key present but unparseable
is emitted as `null`. That distinction is deliberate — "the field never arrived" and
"the field arrived malformed" are different bugs and today we could not tell them
apart from the dumps.

## Cost — measured, not estimated

From chain P⁗2 `target_step_00022.jsonl`, 48 rollouts, 40.8 assistant messages each:

| mode | per rollout | per step (512) | per hour (2 steps/h) |
|---|---|---|---|
| `off` | 0 | 0 | 0 |
| `digest` | +0.34 MB | **+0.18 GB** | +0.35 GB |
| `full` | +16.8 MB | **+8.59 GB** | +17.2 GB |

Existing dump volume is ~10 GB/step, so `full` is roughly **+86%** and `digest` is
**+1.8%**.

`full` is expensive because `prompt_token_ids` on turn N is the entire prefix up to
turn N, so dumping every turn is O(turns²) in tokens — 1.37 M ids per rollout against
8.3 k of actual generation, a 165x ratio. `compact_prompt_token_ids` doubles it again.

## Recommendation

Use `digest` on production arms and `full` only on smokes or a single diagnostic step.
`digest` is sufficient for every verification we actually want, because the cumulative
prefix is reconstructible from the per-message `token_ids` that `digest` dumps verbatim:

```python
# prefix identity: does turn N's prompt equal everything the harness sent before it?
run = []
for m in row["messages"]:
    if m["role"] == "assistant" and "prompt_token_ids_digest" in m:
        assert m["prompt_token_ids_digest"]["sha1"] == _id_digest(run)["sha1"], m
    run += m.get("token_ids", [])
```

That check — verified working in a unit test against the patched module — would have
caught chain U's failure on the first step instead of after 34 minutes on 64 nodes.

Two further checks `digest` supports:
- `generation_token_ids == token_ids` on every assistant message (engine's view vs the
  trainer's view of what was generated)
- `compact_prompt_token_ids_digest.len` non-zero on every assistant turn after the
  first — its absence is precisely the main@9-15 stitch failure

## YAML gotcha (added after it bit chain V)

`token_ids: off` written unquoted in a YAML file parses as the **boolean False**, not the
string `"off"`, and the `Literal["off","digest","full"]` then rejects it with
`input_value=False, input_type=bool`. This killed a chain V segment on 2026-09-27.

The patch now carries a `field_validator(mode="before")` that coerces `False -> "off"`, so
an unquoted `off` is accepted. `True` is still rejected. Either of these is safe:

```yaml
    token_ids: "off"     # quoted - unambiguous
    token_ids: off       # bool False, coerced by the validator
```

Verified against the real parser rather than by reasoning about the schema:

```
yaml 'token_ids: off'     -> {'token_ids': False}    -> model 'off'
yaml 'token_ids: "off"'   -> {'token_ids': 'off'}    -> model 'off'
yaml 'token_ids: digest'  -> {'token_ids': 'digest'} -> model 'digest'
```

## Not applied anywhere

I have not edited any tree. Every arm's checkout is bind-mounted live into running
jobs with `USE_SNAPSHOT=0`, so an edit lands in whatever container starts next.
Apply it to your own tree at a moment you choose.

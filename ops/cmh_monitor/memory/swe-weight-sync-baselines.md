---
name: swe-weight-sync-baselines
description: "Measured _sync_weights timings per node count and stack - startup sync is NOT monotonic in node count, so never compare a 16-node smoke to a 64-node arm"
metadata:
  type: reference
---

`grep -oE 'sync done in [0-9.]+s'` on `runs/<EXP>/runs/latest/logs/exp_001/wandb/wandb/run-*/files/output.log` (the SingleController's own output, not the driver log).

| run | stack | nodes | startup | steady state |
|---|---|---|---|---|
| nano35-swe-v2-minf_dump-smoke-16n | v2 MINF | 16 | 75.395 s | 1.905, 2.092 s |
| nano35-swe-v2-parity-minf-smoke-16n 4051233 | v2 + vLLM parity | 16 | 76.594 s | — |
| nano35-swe-v2-parity-minf-smoke-16n 4050516 | v2 + vLLM parity | 16 | 77.303 s | — |
| nano35-swe-main915-minf-smoke-16n 4051227 | main@9-15 | 16 | 0.546 s | 0.561 s |
| chain P⁗ / P⁗2 | v2 MINF nvshmem | 64 | 37.5 s | 4.9-6.8 s |

**Startup sync is NOT monotonic in node count:** 37.5 s at 64 nodes is *faster* than 75.4 s at 16. Do not reason "more nodes, slower sync" - it cost two wrong comparisons on 2026-09-27 (a "2.04x parity regression" and a "140x parity regression", both retracted; the real figure is +1.6%, i.e. nothing).

main@9-15's sub-second sync is a genuine outlier, 138x faster than the same operation at the same node count. It is probably a faster path on the newer MLM, not a skipped one: `start_weight_version`/`end_weight_version` in the rollout dumps advance one per step exactly like the working v2 arms.

**Why:** the only valid comparator is same stack, same node count, same harness. There was one (`nano35-swe-v2-minf_dump-smoke-16n`) and neither I nor the parity session looked for it first.

**How to apply:** before quoting a refit regression, find the matched control run dir under `runs/` and quote it. Related: [[minf-refit-engine-pause-verified]], [[swe-vllm-parity-arm-v]], [[main915-prefix-stitch-storm]].

**Use the 64-node band for 64-node arms.** The 1.8-5.2 s control band above is all 16-node smokes. For a 64-node arm the comparator is chain P⁗2: startup 37.488 s, steady state 5.07-6.80 s. chain V (4052187) should be flagged against that, not the smoke band.

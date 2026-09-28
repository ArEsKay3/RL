# Plan: chain L (MINF from chain K step_10) — a second vLLM-lineage continuation

Status: PLAN ONLY. Nothing submitted. Requires rkirby's word before any copy or sbatch.
Written 2026-09-23 08:2x PDT.

## Purpose

chain F (MINF from vLLM step 10) is the single arm behind the claim that MINF continuing
from vLLM-trained weights stays clean, while MINF-from-scratch lineages degrade. It is
n=1 and it forked from run A (vLLM from scratch), which has since been shown to degrade
on its own past step ~25. chain L repeats that fork on an independent vLLM lineage,
chain K (vLLM from scratch, seed 1234), so the conclusion no longer rests on one fork of
one parent.

## Design

| | chain F (MINF from vLLM step 10) | chain L (proposed) |
|---|---|---|
| parent | run A (vLLM from scratch), seed 42 | chain K (vLLM from scratch, seed 1234) |
| fork point | run A checkpoint step_10 | chain K checkpoint step_10 |
| engine | MINF | MINF |
| Megatron guard | segments straddle the 2026-09-22 07:47 patch | patched throughout |

Pairing chain L against chain J (MINF from scratch, prefix cache on, replica 2) isolates
one variable, the starting weights, provided the engine settings match chain J exactly.
Pairing it against chain F instead tests reproducibility of the original result. These
two goals want different settings, which is open decision 2 below.

## Seed checkpoint

chain K checkpoint step_10 is complete and usable as a seed:

    files=140  weights=134  bytes=394980279007

Note the byte total differs from run A step_10 (395124903238). The chain F launch script
hardcodes run A's number, so that assertion cannot be copied verbatim; chain L's script
must assert chain K's own count.

chain K rungs on disk: step_5, step_10, step_15, step_18, each 140 files and 369 GB.

## Procedure (mirrors chain F, which is the only recipe proven on this stack)

1. Copy U/<chainK exp>/checkpoints/step_10 to U/<chainL exp>/checkpoints/step_10.
   Use the parallel copy, not plain rsync; chain F's rsync died with rc=20 mid-copy on
   2026-09-20 and had to be redone.
2. Verify the copy by content, not only by file count and byte total. Checksum the 134
   weight shards against the source. chain F's only gate was a size assertion, which is
   what makes its audit non-trivial today.
3. Submit singleton segments with
   `ENGINE=minf EXP_NAME=<chainL exp> SEED_CHECKPOINT=<that path> bash ./launch_swe_dump.sh +checkpointing.load_replay_buffer=false`
   plus the engine flags chosen in decision 2, inside sla_res_nemotron_sw_post at
   QOS hero-res on batch_long with an 8 h wall.
4. Register the new arm in FIVE places before the first audit refresh:
   analysis/logs/refresh_all.sbatch EXPS; tools/close_vs_length.py RUNS and its `order`
   list; and the maps in tools/aggregate_think_close.py, aggregate_think_audit.py,
   aggregate_token_bias.py. Also tell the loop-share session so it lands in that table.

## Capacity and timing

Reservation sla_res_nemotron_sw_post now holds 144 healthy nodes and ends
2026-09-26 16:00, which is 79.6 h away. 128 nodes are allocated to chains J and K, and
16 are free, so a third 64-node segment cannot start until one of them finishes. Two
concurrent slots over the remaining window give about 159 node-slot hours.

Rough demand at observed pace (MINF ~35 min/step, vLLM ~45 min/step, plus ~25 min warm-up
per segment):

| arm | now | target | steps left | estimated hours |
|---|---|---|---|---|
| chain J (MINF from scratch, prefix on, replica 2) | 14 | 50 | 36 | 22 |
| chain K (vLLM from scratch, seed 1234) | 18 | 60 | 42 | 31 |
| chain L to step 35 | 10 | 35 | 25 | 15 |
| chain L to step 60 | 10 | 60 | 50 | 30 |

Total with chain L to step 60 is about 83 h of segment time against 159 available, so it
fits, but only in sequence: chain L starts when chain J or chain K finishes. chain J is
the earlier of the two, around 06:30 on 2026-09-24 at current pace.

Both chains J and K also need more segments than are currently queued. chain J has one
running plus one queued, which is 16 h against a 22 h need. chain K has one running plus
one queued, which is 13.5 h against a 31 h need.

## Open decisions for rkirby

1. Fork step. step_10 mirrors chain F exactly and is available now. step_15 or step_20
   would fork from weights closer to where run A began degrading, which tests a different
   question.
2. Engine settings. DECIDED by rkirby 2026-09-23: match chain J, i.e.
   enable_prefix_caching=true and overlap_param_gather=false, on the patched engine.

   This is two changes relative to chain F, and it also makes chain L internally
   consistent where chain F was not. chain F changed settings mid-run:

   | chain F segment | steps | prefix caching | overlap_param_gather |
   |---|---|---|---|
   | 3878147 | 11-18 | True | True |
   | 3878158 | 19-26 | False | True |
   | 3878199 | 27-35 | False | True |

   So chain F ran with overlap_param_gather ON for all 25 of its steps, which no
   from-scratch arm did, and its prefix caching flipped off after step 18. Any chain F
   versus chain G / chain I / chain J comparison currently confounds starting weights
   with both of those. chain L removes that confound.
3. Target step and therefore segment count.
4. Whether chain L displaces chain J or chain K, or waits for one to finish.

# chain F (MINF from vLLM step 10) continuation audit

Read-only forensic audit, 2026-09-23. No Slurm job was submitted, cancelled, held or released;
no file under the runs directories or under the mounted code/yaml trees was modified.

Runs root `U` = `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs`
Workspace `W` = `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump`

---

## VERDICT

**The continuation claim is CONFIRMED.** chain F (MINF from vLLM step 10) started from a
**bit-for-bit identical** copy of run A (vLLM from scratch) `step_10` (all 140 files, 395 GB,
full sha256) — checkpoint weights, optimizer
state, dataloader position and training bookkeeping all carry over exactly, the Adam step
counter reads 10 -> 15 -> 20 with no reset, and the prompt stream resumes at exactly the next
32 SWE instances. Every other arm that owns a `step_10` was explicitly excluded by content.

**But the "chain F stays clean while MINF-from-scratch lineages degrade" conclusion is confounded.**
chain F (MINF from vLLM step 10) ran with `enable_prefix_caching=true` **and**
`overlap_param_gather=true`, while chain G (MINF from scratch, no prefix cache) has both false
and chain I / chain J (MINF from scratch, prefix cache on, replicas 1 and 2) have
`overlap_param_gather=false`. chain F therefore differs from chain G in three ways at once
(lineage, prefix caching, overlap_param_gather) and from chain I / chain J in two
(lineage, overlap_param_gather). Separately, chain F ran entirely **before** the
2026-09-22 07:46:37 `dynamic_engine.py` log-prob guard, whereas chain I is guarded from step 16
onward and chain J is guarded throughout. The only near-clean contrast in the family is
**chain F vs the main chain (MINF from scratch, original)**, which differ in lineage alone.

---

## PART 1 - Did chain F (MINF from vLLM step 10) really continue from run A (vLLM from scratch) step_10?

### Summary

| # | check | verdict |
|---|---|---|
| 1 | which checkpoint chain F's first segment loaded | **CONFIRMED** |
| 2 | that path is run A's lineage, not a MINF lineage | **CONFIRMED** |
| 3 | byte-level identity of the seeded checkpoint | **CONFIRMED** (exhaustive, all 395 GB) |
| 4 | continuity of training bookkeeping | **CONFIRMED** |
| 5 | optimizer-state continuity | **CONFIRMED** |
| 6 | data ordering continuity | **CONFIRMED** (but not lineage-discriminating - see note) |

### Job IDs in submission order

From `W/analysis/logs/launch_minf_fromvllm10_20260920.log`, five singleton segments were
submitted 2026-09-20 09:32-09:37 at `Nice=2000`:
`3878147, 3878158, 3878199, 3878253, 3878294`. Only the first three ever ran.

```
$ sacct -j 3878147,3878158,3878199,3878253,3878294 -X -o JobID,State,Submit,Start,End,Elapsed,NNodes,Reason
JobID             State              Submit               Start                 End    Elapsed  NNodes     Reason
3878147         TIMEOUT 2026-09-20T09:32:48 2026-09-20T10:05:15 2026-09-20T14:05:33   04:00:18      64       None
3878158         TIMEOUT 2026-09-20T09:33:55 2026-09-21T01:34:09 2026-09-21T05:34:16   04:00:07      64 Dependency
3878199      CANCELLED+ 2026-09-20T09:35:07 2026-09-21T05:52:46 2026-09-21T09:19:33   03:26:47      64 Dependency
3878253      CANCELLED+ 2026-09-20T09:36:16                None 2026-09-21T09:19:33   00:00:00      64 Dependency
3878294      CANCELLED+ 2026-09-20T09:37:23                None 2026-09-21T09:19:33   00:00:00      64 Dependency
```

### Check 1 - which checkpoint path the first segment loaded: CONFIRMED

The mechanism is *seed-then-auto-resume*, not a cross-directory load.
`W/analysis/logs/launch_minf_fromvllm10_20260920.sh` copies the seed into chain F's **own**
experiment directory, asserts `files=140 weights=134 bytes=395124903238`, then submits with
`checkpointing.checkpoint_dir=<chainF>/checkpoints` and `+checkpointing.load_replay_buffer=false`.
The launch log records the assertion passing:

```
seed check 09:32:44: files=140 weights=134 bytes=395124903238
```

Segment 1 (`3878147`) driver log, `U/<chainF>/ray_logs/3878147-logs/ray-driver.log`:

```
101:📦 Restoring dataloader state from checkpoint: .../minf_dump-fromvllm10-20260920/checkpoints/step_10
679:(MegatronPolicyWorker pid=368510, ip=10.67.23.49)  loading distributed checkpoint from
    .../minf_dump-fromvllm10-20260920/checkpoints/step_10/policy/weights at iteration 0
```

Controller log (`.../ray/session_*/logs/worker-953606f6...-1189972.out`, the file containing
`SingleControllerActor`):

```
📦 Skipping replay buffer restore (checkpointing.load_replay_buffer=false); regenerating 64 untrained prompt(s).
📦 Restored 32 pooled spare prompt(s) from checkpoint: .../minf_dump-fromvllm10-20260920/checkpoints/step_10/replacement_reserve.pt
  regenerated 32 pending prompt(s) for target_step=10 (shortfall=0)
  regenerated 32 pending prompt(s) for target_step=11 (shortfall=0)
```

First train step in the controller log is `train step 11/1156`.

### Check 2 - is that checkpoint run A's, and not a MINF-lineage step_10: CONFIRMED

Six arms own a `step_10`. `training_info.json` `total_valid_tokens` is a clean lineage
fingerprint, and a sampled-content comparison excludes every non-run A candidate:

| step_10 owner | `total_valid_tokens` | vs chain F step_10 (sha256 content compare) |
|---|---|---|
| **run A (vLLM from scratch)** | **137353954.0** | **140/140 equal in full, 0 mismatch, 0 size mismatch** |
| run B (MINF from MINF step 10) | 140248871.0 | 131/136 mismatch + 4 size mismatch |
| chain C (MINF from MINF step 10, no prefix cache) | 140248871.0 | 131/136 mismatch + 4 size mismatch |
| chain D (vLLM from MINF step 10) | 140248871.0 | 131/136 mismatch + 4 size mismatch |
| main chain (MINF from scratch, original) | 140248871.0 | 131/136 mismatch + 4 size mismatch |
| chain G (MINF from scratch, no prefix cache) | 139165264.0 (step 10) | 131/136 mismatch + 4 size mismatch |
| chain I (MINF from scratch, prefix cache on, r1) | 139165264.0 | 131/136 mismatch + 4 size mismatch |

chain F's `step_10/training_info.json` is byte-identical to run A's:

```
{"consumed_samples": 320, "current_step": 10, "current_epoch": 0, "total_steps": 10,
 "total_valid_tokens": 137353954.0, "sampler_name": "in_order", "next_nemo_gym_task_index": 384}
```

chain F's `step_10/config.yaml` is also byte-identical to run A's (sha256 `eb4216d8d8d93a44...`,
18621 bytes) and is a **vLLM** config - it still says `backend: vllm` and carries no
`mcore_generation_config.offload_policy_before_refit` key, exactly as run A's does. chain F's own
configs (step_15 onward, 19242/19243 bytes) say `backend: megatron`. The seed is therefore
demonstrably a vLLM-lineage artefact.

### Check 3 - byte-level identity: CONFIRMED

Context (raised by the coordinator): the seed copy was **interrupted**. From
`W/analysis/logs/seed_copy_fromvllm10_20260920.log`:

```
Sun Sep 20 09:22:51 PDT 2026
rsync error: received SIGINT, SIGTERM, or SIGHUP (code 20) at rsync.c(716) [sender=3.2.7]
rsync error: received SIGUSR1 (code 19) at main.c(1635) [generator=3.2.7]
Sun Sep 20 09:24:56 PDT 2026
rsync rc=20
parallel copy start Sun Sep 20 09:24:57 PDT 2026
parallel copy end   Sun Sep 20 09:32:06 PDT 2026 rc=0
```

so the only integrity gate before launch was a file-count and total-byte assertion. That is not
sufficient, and the content checks below were done for that reason.

**Source integrity.** run A's `step_10` is not a moving target. All 140 non-`hf` files carry the
single mtime `2026-09-19 14:49`, i.e. they predate the 09-20 09:22 copy, and nothing under it has
been rewritten since:

```
$ find <runA>/checkpoints/step_10 -path .../hf -prune -o -type f -newermt "2026-09-20 09:22:00" -print
(no output)
$ find ... -printf '%TY-%Tm-%Td %TH:%TM\n' | sort | uniq -c
    140 2026-09-19 14:49
```
(run A kept training to `step_43`, but each rung is a separate directory; `step_10`'s only
post-copy addition is the `hf/` export subdirectory, created 2026-09-22 13:21 and excluded here.)

**Not a hardlink.** All files have link count 1 and distinct inodes on the two sides, so this is a
genuine second copy, not an alias:

```
chain F step_10/config.yaml : inode 414332905129620001 links 1
run A  step_10/config.yaml : inode 378303694719944265 links 1
```

**Manifest.** Name-for-name and size-for-size identical, computed independently of the launcher's
assertion:

```
chainF files: 140  bytes: 395124903238
runA   files: 140  bytes: 395124903238
diff <manifest A> <manifest F>  ->  IDENTICAL name-for-name and size-for-size
```

**Content - exhaustive, not sampled.** Every one of the 140 files was read end to end on both
sides and hashed with sha256. 395,124,903,238 bytes per side, 790 GB of reads, 1491 s:

```
manifest_equal True  n_a 140  n_f 140  only_in_a []  only_in_f []
n_compared 140  n_equal 140  mismatch []  size_mismatch []
modes: {'full': 140}          <- every file hashed in full, no sampling
total bytes hashed per side: 395124903238
```

Spot values from that run (full-file sha256, first 32 hex digits shown):

```
policy/weights/iter_0000000/__0_0.distcp    3226780182 B  35018aa58b8326ab620a8d0c6fd6b08e  (equal)
policy/weights/iter_0000000/__127_0.distcp  3226270397 B  b990bfba65f69feb9e981523c4188c36  (equal)
replay_buffer.pt                             154772307 B  68a9807423eeb7bd17dcaf4d3cce9725  (equal)
```

So the interrupted rsync left no damage: chain F's seed is **bit-for-bit identical** to run A's
`step_10`, across all 128 weight/optimizer shards and all 12 auxiliary files. (An earlier,
cheaper pass over 8 evenly spaced 2 MiB windows per shard agreed, and is what the negative
controls below use.)

**The method discriminates** - two negative controls run with the identical code path:

```
chain F step_10 vs chain F step_15 : compared 135, equal 5, mismatch 130, size_mismatch 5
chain F step_10 vs chain G step_10 : compared 136, equal 5, mismatch 131, size_mismatch 4
```

The first also rules out the NVRx stale-handle failure mode that `W/analysis/ckpt_verify/optim_bitcmp.py`
was written for: chain F's later checkpoints are genuinely new bytes, not re-dumps of the seed.

**Independent corroboration through a different code path.** Both `step_10` directories were later
exported to HF safetensors *independently* - chain F's on 2026-09-21 06:40, run A's on
2026-09-22 13:19, different inodes, no copying between them. The two exports agree:

```
compared 25, equal 24, mismatch ['model.safetensors.index.json'], size_mismatch 0
  (all 14 model-000NN-of-00014.safetensors equal; the index.json difference is key ordering only -
   parsed JSON is identical and both files are 638275 bytes)

$ sha256sum <runA>/checkpoints/step_10/hf/model-00001-of-00014.safetensors
dac6d3cc0551531bba923385f6a7505ec4a161d464d9129fa7a34c66ce7cb099
$ sha256sum <chainF>/checkpoints/step_10/hf/model-00001-of-00014.safetensors
dac6d3cc0551531bba923385f6a7505ec4a161d464d9129fa7a34c66ce7cb099
```

Two independent Megatron->HF conversions of the two directories producing the same 4.99 GB shard
byte-for-byte is strong evidence the underlying distributed checkpoints carry the same weights.

### Check 4 - continuity of training bookkeeping: CONFIRMED

Note on the expected increment: `consumed_samples` counts **prompts**, not sequences, so it
advances by `grpo.num_prompts_per_step = 32` per step, not by 512. (`train_global_batch_size = 512`
= 32 prompts x `num_generations_per_prompt` 16.) The brief's "+512 per step" does not match this
field's semantics; the observed +32/step is correct and consistent across all arms.

| checkpoint | `current_step` | `consumed_samples` | `total_valid_tokens` | `next_nemo_gym_task_index` |
|---|---|---|---|---|
| run A step_5 | 5 | 160 | 68265059 | 192 |
| **run A step_10** | **10** | **320** | **137353954** | **384** |
| **chain F step_10** (the seed) | **10** | **320** | **137353954** | **384** |
| chain F step_15 | 15 | 480 | 207666058 | 576 |
| chain F step_20 | 20 | 640 | 276168305 | 768 |
| chain F step_25 | 25 | 800 | 340076568 | 928 |
| chain F step_30 | 30 | 960 | 406101527 | 1120 |
| chain F step_35 | 35 | 1120 | 470443091 | 1280 |

`consumed_samples` is exactly `32 * current_step` at every rung with no gap or repeat across the
run A step_10 -> chain F step_11..35 boundary. For reference run A's own continuation gives
480 / 640 / 800 / 960 / 1120 at steps 15-35 - identical, as it must be.

One field does diverge after the boundary, legitimately: `next_nemo_gym_task_index` is 384 at
step 10 in both, then reaches 576 in chain F vs 544 in run A at step 15. That counter advances
with rollout *replacements*, which are engine- and failure-dependent; chain F pulled 192 gym tasks
over those five steps against run A's 160. This is a downstream behavioural difference, not a
bookkeeping discontinuity.

### Check 5 - optimizer-state continuity: CONFIRMED

`checkpointing.save_optimizer: true`, and the distributed-optimizer state lives inside the same
`.distcp` shards that check 3 verified. Reading the DCP index without torch (not installed on
`aws-cmh-slurm-1-vscode-01`; no Slurm job was submitted) shows 216 optimizer tensors per
checkpoint under keys of the form
`chained_{0,1}.optimizer.distributed.dp_group_idx_N.gbuf_idx_0.dtype_(torch.bfloat16, torch.bfloat16).bucket_idx_0.{exp_avg,exp_avg_sq,param}`.

The tensor payloads are `torch.save` zip archives inside the shards; they were decoded with
`zipfile` + numpy. Representative bucket `chained_0 / dp_group_idx_0 / gbuf_idx_0 / bucket_idx_0`,
`exp_avg_sq`, 17,616,324 fp32 elements from shard `__0_0.distcp`:

| checkpoint | frac nonzero | mean abs | L2 | max | sha256[:16] of first 256 values |
|---|---|---|---|---|---|
| run A step_5 | 1.0 | 1.439168e-14 | 1.014622e-08 | 8.501027e-09 | a4904fcfd4d44520 |
| **run A step_10** | 1.0 | 2.813201e-14 | 1.493487e-08 | 1.208335e-08 | **f031ea3acc85d64f** |
| **chain F step_10** | 1.0 | 2.813201e-14 | 1.493487e-08 | 1.208335e-08 | **f031ea3acc85d64f** |
| chain F step_15 | 1.0 | 3.966661e-14 | 1.981301e-08 | 1.594754e-08 | f96e3e26fb401609 |
| chain F step_20 | 1.0 | 5.383584e-14 | 2.496114e-08 | 1.820263e-08 | 77cd060b8a52b1ae |
| chain G step_10 | 1.0 | 2.598717e-14 | 8.606933e-09 | 4.417577e-09 | fe50c2407f3abe28 |

The Adam **first** moment `exp_avg` of the same bucket tells the same story:

| checkpoint | frac nonzero | mean abs | L2 | max | sha256[:16] of first 256 values |
|---|---|---|---|---|---|
| **run A step_10** | 1.0 | 6.917660e-08 | 1.544405e-03 | 2.476946e-04 | **4531b81c79bc2645** |
| **chain F step_10** | 1.0 | 6.917660e-08 | 1.544405e-03 | 2.476946e-04 | **4531b81c79bc2645** |
| chain F step_15 | 1.0 | 6.801866e-08 | 1.519575e-03 | 3.537501e-04 | 4bc740e75fa42075 |
| chain G step_10 | 1.0 | 6.910965e-08 | 1.510866e-03 | 2.939212e-04 | b2e8f7f2236d0cd9 |

chain F's seed reproduces run A's first moment to the last digit and to the same 256-value hash,
while chain G's independently-trained step_10 does not - the moments are close in magnitude (as
the weight-space study already found) but numerically distinct.

Adam second moments are fully populated (`frac_nonzero = 1.0`, no NaN/sentinel region in this
bucket), chain F's seed is numerically identical to run A's, and the second moment grows
monotonically 1.49e-8 -> 1.98e-8 -> 2.50e-8 across chain F steps 10 -> 15 -> 20, which is the
expected ramp for `beta2 = 0.999` still filling from its zero init. A fresh optimizer would read 0.

The Adam **step counter** was read out of the `...dp_group_idx_0.optimizer/shard_0_1` state blob:

```
runA_step_5    : param_groups[0].step = 5   param_groups[1].step = 5
runA_step_10   : param_groups[0].step = 10  param_groups[1].step = 10
chainF_step_10 : param_groups[0].step = 10  param_groups[1].step = 10
chainF_step_15 : param_groups[0].step = 15  param_groups[1].step = 15
chainF_step_20 : param_groups[0].step = 20  param_groups[1].step = 20
```

No reset to 0, continuous with run A.

**What could not be established.** The brief asked about chain F's `step_11`/`step_12` optimizer
state specifically. Those checkpoints no longer exist: `checkpointing.ft_save_period=1` with
`ft_keep_latest_k=1` saves every step but prunes all but the latest non-rung checkpoint (the
controller log shows `Removing checkpoint .../step_31 (step 31)` etc.). chain F's surviving
checkpoints are `step_10, 15, 20, 25, 30, 35`, so `step_15` is the earliest post-resume evidence.
A full bitwise optimizer comparison via `W/analysis/ckpt_verify/optim_bitcmp.py` was **not** run -
it needs torch, which is unavailable on this node, and running it needs a Slurm job, which is out
of scope for this audit. The numpy-based statistics above are a weaker but sufficient substitute
for the specific question asked.

### Check 6 - data ordering continuity: CONFIRMED (not lineage-discriminating)

Read from the cached per-rollout summaries in `W/analysis/rollouts/*.summary.jsonl` (derived from
the raw dumps by `W/tools/aggregate_dumps.py`) rather than re-reading ~10 GB per step. The cache
was spot-checked against the raw files and matches exactly:

```
RAW    <chainF>/dumps/rollouts/target_step_00010.jsonl first record:
       sample_id="46c6b0c7-f81c-4a5f-bc57-6aded7567678_g0" target_step=10 prompt_idx=360 instance_id="elastic__go-libaudit-67"
CACHED chainF summary 00010 first record: same four values
RAW    <runA>/dumps/rollouts/target_step_00009.jsonl first record:
       sample_id="9054f85f-a207-44b7-b914-832aa891f757_g0" target_step=9 prompt_idx=321 instance_id="bbc__psammead-587"
CACHED runA summary 00009 first record: same four values
```

Boundary (`shuffle: false`, `async_rl.sampler.name: in_order`):

```
runA   file 00009 (train step 10): 528 rows, 33 groups, 32 prompt_idx [320..351], 32 instance ids
chainF file 00010 (train step 11): 512 rows, 32 groups, 32 prompt_idx [352..383], 32 instance ids
runA   file 00010 (train step 11): 512 rows, 32 groups, 32 prompt_idx [352..383], 32 instance ids

runA_s10  vs chainF_s11 : prompt_idx overlap 0/32, instance_id overlap 0/32   (disjoint, as required)
runA_s11  vs chainF_s11 : prompt_idx overlap 32/32, instance_id overlap 32/32 (identical set)
```

chain F's step 11 is exactly the next 32 SWE instances after run A's step 10, and exactly the set
run A itself consumed at its own step 11.

Control - chain G (MINF from scratch, no prefix cache) at the same step numbers:

```
chainG file 00009 (train step 10): prompt_idx [320..351]  ->  overlap 32/32 with runA_s10
chainG file 00010 (train step 11): prompt_idx [352..383]  ->  overlap 32/32 with runA_s11
chainG_s10 vs chainG_s11 : overlap 0/32 (advances the same way)
```

**Important caveat.** Because the sampler is `in_order` with `shuffle: false` and the same data
path, *every* arm walks the identical prompt sequence. chain G from scratch shows the same
[320..351] -> [352..383] at steps 10 -> 11. This check therefore confirms that chain F's
dataloader position was restored correctly and did not repeat or skip work, but it does **not** by
itself identify run A as the parent. The lineage identification rests on checks 2, 3 and 5.

(run A's step 10 has 33 groups / 528 rows against 32 distinct `prompt_idx` - one replacement
rollout group. chain F's step 11 has a clean 32/512.)

---

## PART 2 - Settings comparison across arms

Source of truth: `checkpoints/step_N/config.yaml` inside each arm's own checkpoint directory
(this is the config the run actually executed), cross-checked against the `MasterConfig(...)`
dict printed in each arm's `ray-driver.log`. Step 15 was used wherever available; chain E (MINF
from scratch, prefix cache on, cancelled) only reached step 2 and chain J (MINF from scratch,
prefix cache on, replica 2) step 10.

### Settings that DIFFER across arms

Columns: A = run A (vLLM from scratch), B = run B (MINF from MINF step 10), C = chain C (MINF from
MINF step 10, no prefix cache), D = chain D (vLLM from MINF step 10), E = chain E (MINF from
scratch, prefix cache on, cancelled), F = chain F (MINF from vLLM step 10), G = chain G (MINF from
scratch, no prefix cache), I = chain I (MINF from scratch, prefix cache on, replica 1),
J = chain J (MINF from scratch, prefix cache on, replica 2), K = chain K (vLLM from scratch,
seed 1234), main = main chain (MINF from scratch, original).

| setting | A | B | C | D | E | F | G | I | J | K | main |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `engine backend` | vllm | megatron | megatron | vllm | megatron | megatron | megatron | megatron | megatron | vllm | megatron |
| `mcore.enable_prefix_caching` | False | True | False | False | True | True | False | True | True | False | True |
| `mcore.refit_backend` | nccl | nvshmem | nvshmem | nccl | nvshmem | nvshmem | nvshmem | nvshmem | nvshmem | nccl | nvshmem |
| `mcore.prefix_caching_coordinator_policy` | <absent> | longest_prefix | longest_prefix | <absent> | longest_prefix | longest_prefix | longest_prefix | longest_prefix | longest_prefix | <absent> | longest_prefix |
| `mcore.offload_policy_before_refit (dead key)` | <absent> | False | False | <absent> | False | False | False | False | False | <absent> | False |
| `mcore.mamba_inference_ssm_states_dtype` | <absent> | float32 | float32 | <absent> | float32 | float32 | float32 | float32 | float32 | <absent> | float32 |
| `ddp.overlap_param_gather` | True | True | False | False | True | True | False | False | False | True | True |
| `grpo.seed` | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 1234 | 42 |

Notes on the rows that are artefacts rather than real interventions:
`mcore.refit_backend`, `mcore.prefix_caching_coordinator_policy`,
`mcore.offload_policy_before_refit` and `mcore.mamba_inference_ssm_states_dtype` differ only
between the vLLM yaml and the MINF yaml; they are unused on the vLLM arms.

### Settings IDENTICAL across all eleven arms

Generation: `max_new_tokens 196608`, `temperature 1.0`, `top_p 1.0`, `top_k None`,
`refit_transport nccl_reshard`, `mcore.max_tokens 16384`, `mcore.max_model_len 196608`,
`mcore.enable_chunked_prefill true`, `mcore.block_size_tokens 256`,
`mcore.kv_cache_management_mode persist`, `mcore.cuda_graph_impl local`,
`vllm.max_model_len 196608`, `vllm.gpu_memory_utilization 0.85`, `vllm.enforce_eager false`,
`vllm_kwargs.max_num_batched_tokens 8480`.

DDP / offload: `overlap_grad_reduce false`, `grad_reduce_in_fp32 false`,
`megatron_cfg.optimizer.optimizer_cpu_offload false`, `policy.offload_optimizer_for_logprob false`.

GRPO / batch: `num_prompts_per_step 32`, `num_generations_per_prompt 16`,
`train_global_batch_size 512`, `seq_logprob_error_threshold 2.0`, `overlong_filtering false`,
`normalize_rewards true`, `use_leave_one_out_baseline true`, `advantage_clip -100/+100`,
`invalid_tool_call_advantage -5.0`, `malformed_thinking_advantage -5.0`.

Optimizer: `adam`, `lr 3e-06`, `min_lr 3e-06`, `weight_decay 0.0`, `adam_beta1 0.9`,
`adam_beta2 0.999`, `adam_eps 1e-08`, `clip_grad 1.0`, `policy.max_grad_norm 1.0`.
Scheduler: `lr_decay_style constant`, `lr_warmup_iters 10`, `lr_warmup_init 3e-07`,
`override_opt_param_scheduler true`.

Loss: `reference_policy_kl_penalty 0.0`, `use_kl_in_reward false`, `ratio_clip_min 0.2`,
`ratio_clip_max 0.28`, `ratio_clip_c None`, `token_level_loss true`,
`sequence_level_importance_ratios false`, `use_importance_sampling_correction true`,
`truncated_importance_sampling_type tis`, ratio `5.0` / min `0.2`,
`use_on_policy_kl_approximation true`, `force_on_policy_ratio true`, `use_cispo false`.

Data / async: `shuffle false`, `max_input_seq_length None`, `max_total_sequence_length 196608`,
`sampler in_order`, `max_lookahead_versions 1`, `min_groups_for_streaming_train 8`,
`max_inflight_prompts 64`, `max_buffered_rollouts 96`,
`recompute_kv_cache_after_weight_updates false`.

Checkpointing: `load_replay_buffer false`, `save_period 5`, `save_optimizer true`.
Parallelism: `TP 4`, `CP 4`, `EP 32`, `freeze_moe_router true`, `moe_router_enable_expert_bias true`.
Base model: `.../swe_e2e_corrected/base_model/step_18/hf`. Cluster: 64 nodes, 32 generation nodes.

### `policy.offload_optimizer_for_refit`

**This key does not exist in any arm's config.** The only related key present is
`policy.generation.mcore_generation_config.offload_policy_before_refit: false`, which is the dead
key already identified in the length-growth investigation, and it is present only in the MINF
yaml. So the "MINF offloads the optimizer at every refit" behaviour is unconditional in this code
version and is not configurable per-arm - it cannot be used to separate chain F from other MINF
arms, and it separates every MINF arm from every vLLM arm.

### Answer: what differs between chain F and run A

chain F (MINF from vLLM step 10) vs run A (vLLM from scratch):

| difference | chain F | run A | real intervention? |
|---|---|---|---|
| `policy.generation.backend` | `megatron` | `vllm` | **yes - this is the intended intervention** |
| `mcore.enable_prefix_caching` | true | false (unused on vLLM) | no - vLLM does not read it |
| `mcore.refit_backend` / coordinator policy / mamba dtype | MINF values | absent/unused | no |
| `ddp.overlap_param_gather` | true | true | **no difference** |
| `grpo.seed` | 42 | 42 | **no difference** |
| optimizer / scheduler / loss / data / batch | identical | identical | **no difference** |
| mounted Megatron-LM tree | fork `880de0fce` bind-mounted over the container tree | container tree (no mount for `ENGINE=vllm`) | **yes - see below** |

One difference beyond the engine switch is easy to miss: `W/launch_swe_dump.sh` bind-mounts
`W/Megatron-LM` (fork HEAD `880de0fce`) over
`/opt/nemo-rl/3rdparty/.../Megatron-LM` **only when `ENGINE=minf`**
(`launch_swe_dump.sh:133-135`). vLLM arms run the container's Megatron-LM. Because Megatron-LM is
also the *training* backend, chain F and run A do not merely differ in the generation engine -
they differ in the Megatron-LM used for the training step as well. That is inherent to every
MINF-vs-vLLM comparison in this family, not specific to chain F.

The `nemo_rl` commit is identical (`33a9bf6a4`, branch `rkirby/swe-v2-dump`) for chain F and
run A, and the `33a9bf6a4 -> 7f8a2b9dc` delta used by the later arms touches nothing outside the
`swe_dump_workspace/` tools directory except `examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_minf.yaml`
(the main chain's config, which none of the dump arms use):

```
$ git -C W/nemo_rl diff --stat 33a9bf6a4..7f8a2b9dc -- ':!swe_dump_workspace'
 examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_minf.yaml | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

So the NeMo-RL training/generation code is the same across every dump arm.

### Answer: what differs between chain F and the MINF-from-scratch arms

| | chain F (MINF from vLLM step 10) | chain G (MINF from scratch, no prefix cache) | chain I (MINF from scratch, prefix cache on, r1) | chain J (MINF from scratch, prefix cache on, r2) | main chain (MINF from scratch, original) |
|---|---|---|---|---|---|
| lineage | run A step_10 (vLLM) | scratch | scratch | scratch | scratch |
| `mcore.enable_prefix_caching` | **true** | **false** | true | true | true |
| `ddp.overlap_param_gather` | **true** | **false** | **false** | **false** | true |
| `grpo.seed` | 42 | 42 | 42 | 42 | 42 |
| `nemo_rl` commit | 33a9bf6a4 | 33a9bf6a4 | 33a9bf6a4 then 7f8a2b9dc | 7f8a2b9dc | 952eaf85b (different branch, `swe_sc_cmh_minf.yaml`) |
| `dynamic_engine.py` guard | **unguarded throughout** | unguarded throughout | **guarded from step 16** | **guarded throughout** | unguarded (predates it) |
| everything else in the key list | identical | identical | identical | identical | identical |

### Flagged confounds

1. **`overlap_param_gather` is true in chain F and false in chain G, chain I and chain J.**
   This is the single biggest problem for the "chain F stays clean" conclusion. The user rule of
   2026-09-20 was that every new MINF experiment sets `enable_prefix_caching=false` and
   `overlap_param_gather=false`; chain F was launched 2026-09-20 09:32 with **both true** and so
   predates/violates that rule. chain F is therefore not a valid control for the
   MINF-from-scratch arms on this axis.
2. **`enable_prefix_caching` is true in chain F and false in chain G.** chain F vs chain G differs
   in three ways simultaneously (lineage, prefix caching, overlap_param_gather); no single-factor
   attribution is possible from that pair.
3. **The `dynamic_engine.py` guard splits the family.** `W/Megatron-LM/megatron/core/inference/engines/dynamic_engine.py`
   was edited in place at `2026-09-22 07:46:37` (mtime; `git status` shows it still modified):
   ```
   -                        request.generated_log_probs.append(request_log_probs[-1])
   +                        if request.generated_tokens and len(request_log_probs) > 0:
   +                            request.generated_log_probs.append(request_log_probs[-1])
   ```
   This is on the MINF prefill path and changes the *per-token log-prob sequence*, which feeds the
   importance-sampling ratios in the loss - it is a behavioural change, not only a crash guard. The
   tree is bind-mounted live, so a segment is guarded iff its process started after that instant.
   - chain F (MINF from vLLM step 10): segments started 09-20 10:05, 09-21 01:34, 09-21 05:52 -
     **all steps 11-35 unguarded**. No chain F segment straddles the guard.
   - chain G (MINF from scratch, no prefix cache): last segment started 09-22 06:22, before the
     edit - **all steps 1-36 unguarded**. Comparable to chain F on this axis.
   - chain I (MINF from scratch, prefix cache on, r1): `3901530` steps 1-13 and `3901560`
     steps 14-15 unguarded; `3924762` (started 09-22 07:52) steps 16-28 and `3924764` steps 29-36
     **guarded**.
   - chain J (MINF from scratch, prefix cache on, r2): `3901613` (started 09-22 22:05) steps 1-12
     and `3943025` steps 13-14 **guarded throughout**.
   Any chain F vs chain I / chain J contrast past chain I's step 15 therefore also carries a code
   difference.
4. **chain F and run A do not use the same Megatron-LM** (mounted fork vs container tree), as
   described above. Inherent to MINF-vs-vLLM, but worth stating explicitly.
5. **The replay buffer is discarded at every segment boundary in every arm**
   (`checkpointing.load_replay_buffer: false` is in all eleven configs, and run A's own
   segments log `📦 Skipping replay buffer restore ... regenerating 64 untrained prompt(s)`).
   So chain F's `+checkpointing.load_replay_buffer=false` override was a no-op and does not make
   chain F special. It does mean chain F's step 11 began with a *freshly regenerated* async
   rollout pipeline, whereas run A trained step 10 -> 11 inside a single segment (`3847610`
   covered steps 9-18) with its in-flight rollouts intact. The boundary effect is the same kind
   run A itself experiences at its own steps 9, 19, 24 and 36, but it lands in a different place
   for chain F.
6. **The cleanest available contrast is chain F vs the main chain (MINF from scratch, original)** -
   same engine, same prefix caching, same `overlap_param_gather`, same seed, both unguarded -
   differing in lineage alone. It differs in `nemo_rl` commit and branch
   (`952eaf85b` / `rkirby/swe-v2-minf` vs `33a9bf6a4` / `rkirby/swe-v2-dump`) and config file, so
   it is not perfect either, but it is the nearest single-factor comparison in the family.
7. Not a confound, but worth recording: **chain K (vLLM from scratch, seed 1234)** is the only arm
   with a different `grpo.seed`; every other arm uses 42.

---

## PART 3 - chain F (MINF from vLLM step 10) segment history

### Segments

| # | job | state | start | end | elapsed | train steps produced | notes |
|---|---|---|---|---|---|---|---|
| 1 | 3878147 | TIMEOUT | 2026-09-20 10:05:15 | 2026-09-20 14:05:33 | 04:00:18 | **11-18** | resumed from the seeded `step_10`; hit the 4 h walltime |
| 2 | 3878158 | TIMEOUT | 2026-09-21 01:34:09 | 2026-09-21 05:34:16 | 04:00:07 | **19-26** | resumed from `step_18`; hit the 4 h walltime; 11 h 29 m queue wait after segment 1 |
| 3 | 3878199 | CANCELLED | 2026-09-21 05:52:46 | 2026-09-21 09:19:33 | 03:26:47 | **27-35** | resumed from `step_26`; cancelled at 09:19:33, 33 min short of walltime |
| 4 | 3878253 | CANCELLED | never started | 2026-09-21 09:19:33 | 00:00:00 | - | cancelled while pending |
| 5 | 3878294 | CANCELLED | never started | 2026-09-21 09:19:33 | 00:00:00 | - | cancelled while pending |

Segments 3, 4 and 5 were all cancelled at the identical timestamp `2026-09-21 09:19:33`, i.e. a
single `scancel` of the remaining chain rather than a failure. No segment was requeued: `sacct`
shows only `.batch`/`.extern`/`.0-.2` steps with no `REQUEUED` state, and there is no
`NODE_FAIL` or `FAILED` anywhere in chain F.

Six further run directories exist under `U/<chainF>/runs/` dated `20260921-1001` through
`20260921-1007`, created at submit time by a later launcher invocation. They contain only empty
`logs/` and `slurm/` directories plus a `provenance.txt`; no matching jobs appear in `sacct` and
no `ray_logs` were produced. Those submissions never ran.

### Steps, checkpoints and dumps

- Steps 11 through 35 were produced **contiguously**: 11-18, 19-26, 27-35. **No step was re-run
  on resume and there is no gap.** Each segment resumed from the previous segment's last
  fault-tolerance checkpoint (`step_10` -> `step_18` -> `step_26`), which is why no work was
  repeated.
- **No step was generated but left un-checkpointed.** `ft_save_period=1` saves every step; the
  controller logs show `Saving checkpoint for step N...` for every N in 11-35.
- Rollout dumps `target_step_00010.jsonl` .. `target_step_00034.jsonl` (zero-based -> train steps
  11..35) are all present, as are token-level dumps for every step 11-35. There is no
  `target_step_00035`: segment 3's tail shows step 36's rollouts at `Collecting rollouts 44% 7/16`
  when the job was cancelled, so step 36 was in flight and never completed, dumped or trained.
- **Surviving checkpoint rungs are `step_10, 15, 20, 25, 30, 35`.** The intermediate per-step
  checkpoints were pruned by `ft_keep_latest_k=1` (`Removing checkpoint .../step_31 (step 31)`,
  `.../step_32 (step 32)` etc. in segment 3's driver log). This is the reason checks on
  `step_11`/`step_12` are impossible - it is designed behaviour, not data loss.
- `checkpoints/latest_checkpoint_status.json` reads `{"last_checkpoint_step": 35}`, consistent.

### A logging artefact worth knowing about

The `ray-driver.log` of segment 3 ends at `train step 33` and contains **no** `step_34`/`step_35`
string at all, which at first looks like two missing steps. The controller's own worker log -
`ray_logs/3878199-logs/ray/session_*/logs/worker-a8fbd349...-1896645.out`, the file containing
`SingleControllerActor` - shows `train step 27..35` and `Saving checkpoint for step 27..35`.
Ray's log forwarding simply lost the last ~3 minutes of controller output when the job was killed.
`step_35/` on disk (written 09:16:54-09:17:01, before the 09:19:33 cancel) confirms the controller
log. **For step accounting in this family, trust the controller worker log, not `ray-driver.log`.**

### MINF engine crash signatures

Counted over every file under each segment's `ray_logs` tree (driver log, 64 ray-worker logs and
all Ray session worker logs):

| segment | `Coordinator: removed engine` | `post_process_requests` | `Traceback (most recent call last)` |
|---|---|---|---|
| 3878147 | 0 | 0 | 0 |
| 3878158 | 0 | 0 | 0 |
| 3878199 | 0 | 0 | 0 |

**Caveat: this is not discriminating evidence.** The same counts over chain G (MINF from scratch,
no prefix cache) are also 0/0/0 across all five of its segment log trees. The signature does not
occur anywhere in this experiment family, so its absence in chain F says nothing about chain F
specifically.

---

## Anything that looks wrong or could not be verified

1. **The seed copy had no content verification at the time.** The gate was
   `files=140 weights=134 bytes=395124903238` after an rsync that died with
   `code 20` and a `parallel copy` of unrecorded mechanism. The copy turned out to be correct -
   but the log does not even record what the "parallel copy" command was, so the audit had to
   establish correctness after the fact. Future seeds should record a checksum manifest.
2. **chain F violates the 2026-09-20 MINF defaults rule.** It ran with
   `enable_prefix_caching=true` and `overlap_param_gather=true`. Any published claim comparing
   chain F to chain G / chain I / chain J should state these differences alongside the lineage
   difference, or the comparison should be re-run with matched flags.
3. **The `dynamic_engine.py` guard was applied to a live bind-mounted tree while arms were
   running,** with no record in any run's provenance. `provenance.txt` records the `nemo_rl`
   commit but not the mounted Megatron-LM working-tree state, so nothing on disk tells you which
   segments were guarded except the file mtime and the segment start times. chain I (MINF from
   scratch, prefix cache on, r1) is internally inconsistent because of it: steps 1-15 unguarded,
   steps 16-36 guarded.
4. **`policy.offload_optimizer_for_refit` does not exist** in this code version, so the key named
   in the audit brief could not be compared. Only the dead
   `mcore_generation_config.offload_policy_before_refit` is present.
5. **`consumed_samples` advances by 32, not 512,** per step. Any analysis assuming 512 has the
   wrong sample accounting by a factor of 16.
6. **Not established:** a *tensor-level* bitwise walk of the optimizer state in the style of
   `optim_bitcmp.py` (needs torch / a Slurm job), and any check on chain F's `step_11`/`step_12`
   optimizer state (those checkpoints were pruned by `ft_keep_latest_k=1`). Note the seed-identity
   question is already settled at a stronger level - the full-file sha256 pass covers the
   optimizer shards byte for byte - so what is missing is only the per-tensor breakdown of how
   chain F's optimizer evolved after step 10, for which the moment statistics and the Adam step
   counter stand in.
7. **Six phantom run directories** under `U/<chainF>/runs/20260921-100{1,2,3,5,6,7}` from
   submissions that never ran. Harmless, but they make the run directory look like chain F had
   nine segments when it had three.

---

*Tooling used (all read-only): `sacct`, `stat`, `find`, `diff`, `sha256sum`, and throwaway python
in `/tmp/cfaudit/` (`ckpt_cmp.py` manifest+hash compare, `adam2.py` numpy/zipfile DCP reader,
`optstate.py` Adam step-counter reader, `order_check.py` prompt-set compare, `cfg_extract.py`
config table). Existing workspace artefacts read but not modified:
`W/tools/close_vs_length.py`, `W/analysis/rollouts/*.summary.jsonl`,
`W/analysis/ckpt_verify/optim_bitcmp.py`, `W/analysis/ckpt_weightspace/headline.md`,
`W/analysis/logs/launch_minf_fromvllm10_20260920.{sh,log}`,
`W/analysis/logs/seed_copy_fromvllm10_20260920.log`.*

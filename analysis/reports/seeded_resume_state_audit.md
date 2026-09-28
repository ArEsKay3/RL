# Seeded-resume state audit

Read-only audit, 2026-09-23. Question: when a run is started from another run's checkpoint
(a "seeded" continuation) rather than resuming its own, does any model or trainer state
silently fail to load or get re-initialised from defaults, and could a refit to the
inference engine mask such a gap?

## Verdict

**No silent state gap is specific to a seeded start.** A seeded run and a self-resume take
the *identical* code path: the loader is keyed only on the filesystem path
`<run>/checkpoints/step_*`, never on run name, experiment name or any identifier of the
run that wrote the checkpoint, and the one content-keyed guard in the path (dataset name)
matches. Every piece of state the stack restores at all is restored the same way for both.
**Refit cannot mask a training-side gap**: refit is a one-way copy of the trainer's
parameters *and persistent buffers* (including MoE `expert_bias`) into the engine, so a
wrong trainer-side buffer is propagated to the engine, not hidden by it; and the engine is
overwritten before the first rollout in either case. The real residual risk is not a
seeded-vs-self difference at all but a **single silent fallback**: if a seed copy is
missing `policy/weights/latest_train_state.pt` *and*
`policy/weights/latest_checkpointed_iteration.txt`, Megatron-Bridge quietly falls back to
the **base model** with `finetune=True` (no weights from the parent, no optimizer, no LR
position) while NeMo-RL's own step counter, dataloader cursor and prompt journal still
resume at step 10. The `launch_swe_dump.sh` seed guard does not check either tracker file.
The staged chain L (MINF from chain K step_10) copy is byte-identical to chain K
(vLLM from scratch, seed 1234) step_10 and does contain both trackers, so it is safe.

## Paths and source caveat

* `W` = `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump`
* `NRL` = `W/nemo_rl` (the mounted training checkout, `7f8a2b9d`, branch `rkirby/swe-v2-dump`)
* `MLM` = `W/Megatron-LM` (the fork mounted for the megatron generation backend)
* `MB` = `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mlmmain/Megatron-Bridge`
  at `ea5dded83`.

**Caveat (source fidelity).** `W/nemo_rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
is an empty submodule directory; the bridge that the runs actually execute lives inside the
container image and is not on this filesystem. All `MB/...` citations below are from the
`swe_mlmmain` checkout, cross-checked against an independent second copy (a Ray runtime-env
upload under `/tmp/ray/session_latest/runtime_resources/working_dir_files/_ray_pkg_e6f758c0b20dd632/`);
the two agree on every semantic point, only line numbers shift. Treat bridge *semantics* as
verified and bridge *line numbers* as indicative.

## State table

"Seeded" and "self-resume" are the same column because the code path is identical; the table
is split only where the answer is conditional.

| State | Restored? | Where |
|---|---|---|
| Model parameters | **yes** | `MB/src/megatron/bridge/training/checkpointing.py:3341` `_load_model_state_dict(model[0], state_dict["model"], load_strict)`, fed by the dist-ckpt load at `:3781` |
| Persistent module buffers (incl. MoE `expert_bias`, `qb_beta`) | **yes** | same load; the buffers are in `sharded_state_dict` because `MegatronModule._save_to_state_dict` emits params + persistent buffers. Confirmed on disk: 24 `*.mlp.router.expert_bias` keys in the chain L seed `.metadata` |
| Non-persistent buffers (`local_tokens_per_expert`, `global_tokens_per_expert`, `ga_steps`, `qb_beta_accum/count`, experts `_fc1_weight/_fc2_weight`) | **no — re-zeroed every process start, by design** | `MLM/megatron/core/transformer/moe/router.py:182-190` (`persistent=False`), `:203-252`; `MLM/megatron/core/transformer/moe/experts.py:1148-1149`. They are per-step accumulators consumed and reset inside the step, so there is no cross-step carry-over |
| TE `_extra_state` (6282 keys in this checkpoint's `.metadata`) | **yes** | part of the model state dict; a strict-load mismatch on `_extra_state` is explicitly tolerated at `MB/.../checkpointing.py:2842-2868` |
| Optimizer state (Adam `exp_avg`, `exp_avg_sq`, fp32 master params) | **conditional — yes here** | gate `MB/.../checkpointing.py:3353`; requires `optimizer_path is not None` (`NRL/nemo_rl/models/megatron/setup.py:1276 load_optim=optimizer_path is not None`) and `finetune == False`. `optimizer_path` is non-`None` iff the checkpoint's `common_state` contains an `optimizer` key (`NRL/nemo_rl/utils/checkpoint.py:222-232`). Verified present in the chain L seed |
| Optimizer step counter | **yes** (same gate) | read out of the chain L seed's `common_state`: `param_groups[0].step = 10`, `lr = 3e-06` |
| LR scheduler position (`num_steps`) | **yes** (same gate as optimizer) | `MB/.../checkpointing.py:3385/3387` → `MLM/megatron/core/optimizer_param_scheduler.py:390 self.step(increment=num_steps)`. Not suppressed by `override_opt_param_scheduler`, which only guards the hyper-parameters (`:318-339`) |
| LR scheduler hyper-parameters | **overridden from YAML, not the checkpoint** | `override_opt_param_scheduler: true` (`NRL/examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_common.yaml:249`) → `_check_and_set` returns the class value (`MLM/megatron/core/optimizer_param_scheduler.py:328-330`) |
| Gradient scaler | **n/a** | `policy.precision: "bfloat16"` (`swe_sc_cmh_common.yaml:151`, `megatron_cfg.model.bf16: true` at `:223`); Megatron only builds a grad scaler for fp16. If one existed it would ride in `state_dict["optimizer"]` |
| RNG states (python / numpy / torch / cuda / TP tracker) | **no — hard-disabled** | `NRL/nemo_rl/models/megatron/setup.py:1280 load_rng=False` (policy) and `:2024` (reference model), so the bridge gate at `MB/.../checkpointing.py:3412` is always false. The checkpoint *does* contain 128 `rng_state/shard_*` objects; they are never read. Workers instead re-seed deterministically from `rng.seed = 1234` (the Megatron default, `MLM/megatron/training/config/common_config.py:10`) |
| Megatron `iteration` / `train_state` | **restored, then ignored** | `MB/.../checkpointing.py:3275`; NeMo-RL discards the return value of `load_checkpoint` (`NRL/nemo_rl/models/megatron/setup.py:1881`). On disk the value is always zero: the chain L seed's `iter_0000000/train_state.pt` decodes to `step=0, consumed_train_samples=0, do_train=False`, and `latest_checkpointed_iteration.txt` is `0` |
| RL step counter / epoch / consumed samples / valid tokens / gym task index | **yes** | `training_info.json` → `_get_grpo_save_state` (`NRL/nemo_rl/algorithms/grpo.py:341-353`) → `NRL/nemo_rl/algorithms/single_controller.py:202-203, 281-283`. Chain L seed: `current_step=10, consumed_samples=320, next_nemo_gym_task_index=352` |
| Sampler dispatch index | **yes** | `NRL/nemo_rl/algorithms/single_controller.py:211 self._sampler.set_dispatch_index(actor_args.save_state.current_step)` |
| Dataloader position | **yes, with a content-keyed guard** | `NRL/nemo_rl/algorithms/single_controller_utils/setup.py:662-664` → `NRL/nemo_rl/data/utils.py:59-103`. Chain L seed state is `{'samples_yielded': 416, '_sampler_iter_yielded': 13}` |
| Replay buffer | **no — deliberately skipped** | `checkpointing.load_replay_buffer=false` → `NRL/nemo_rl/algorithms/single_controller.py:386-394` |
| Pending rollouts (prompt journal) | **yes, and mandatory** | `NRL/nemo_rl/algorithms/single_controller.py:431-548`; absent file raises (`:447-453`) |
| Replacement reserve | **yes** | `NRL/nemo_rl/algorithms/single_controller.py:550-584` |
| Rerun state machine | **no** | gate at `MB/.../checkpointing.py` requires `"rerun_state_machine" in state_dict`; the chain L seed's `common_state` has only `checkpoint_version, iteration, optimizer, opt_param_scheduler, content_metadata` |

## Q1 — the resume code path

Driver side, `NRL/nemo_rl/algorithms/single_controller_utils/setup.py:623-629`:

```
checkpointer = CheckpointManager(master_config.checkpointing)
last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
loaded_state = checkpointer.load_training_info(last_checkpoint_path)
save_state = _get_grpo_save_state(loaded_state)
weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)
```

* `get_latest_checkpoint_path` (`NRL/nemo_rl/utils/checkpoint.py:555-572`) is a pure glob of
  `checkpoint_dir/step_*` with an `int(name.split("_")[1])` sort. **No run identity anywhere.**
  A copied `step_10` is indistinguishable from a self-written one.
* `get_resume_paths` (`:193-242`) returns `(<ckpt>/policy/weights, <ckpt>/policy/optimizer)`.
  For Megatron, `optimizer_path` is only a flag; the detection reads the checkpoint's
  `common_state` and checks for an `optimizer` key (`:222-232`). If it is absent the code
  emits `warnings.warn(... "Optimizer will be freshly initialized.")` (`:234-239`) and returns
  `optimizer_path = None` — which also silently disables the LR-scheduler restore (see Q6).
* Worker side, `NRL/nemo_rl/models/policy/workers/megatron_policy_worker.py:413` (`__init__`)
  → `:499 validate_and_set_config(...)` → `NRL/nemo_rl/models/megatron/setup.py:1272-1282`
  builds the `CheckpointConfig` with `load=weights_path`,
  `load_optim=optimizer_path is not None`, `pretrained_checkpoint=<HF conversion cache>`,
  `load_rng=False`.
* `:542 setup_model_and_optimizer(...)` → `NRL/nemo_rl/models/megatron/setup.py:1837 get_model(...)`
  (one construction) and `:1881 load_checkpoint(state, model, optimizer, scheduler, ...)`
  (one load). `MegatronPolicyWorker.load_checkpoint` itself is a stub that raises
  (`megatron_policy_worker.py:3465-3475`) — nothing loads a checkpoint after `__init__`.

## Q2 — base-HF-then-overwrite, and load strictness

**The premise does not hold: the live model is never populated from the base HF weights on a
resume.** `handle_model_import` (`NRL/nemo_rl/models/megatron/setup.py:1934`) is a *disk-side*
HF→Megatron conversion into a shared cache under `HF_HOME`; it returns immediately when the
cache exists. The in-memory model is built once by `get_model` (`:1837`) with
`model_cfg.perform_initialization = True` (set at `:1322`), i.e. **fresh random init**, and is
then filled by exactly one `load_checkpoint` (`:1881`). Selection between the resume checkpoint
and the pretrained cache happens inside the bridge:

```
setup.py:1698-1705  resume_checkpoint_exists   = checkpoint.load is not None and checkpoint_exists(checkpoint.load)
                    pretrained_checkpoint_exists = ... checkpoint_exists(checkpoint.pretrained_checkpoint)
setup.py:1872-1875  should_load_checkpoint = resume_checkpoint_exists or (pretrained_checkpoint_exists and ...)
MB/.../checkpointing.py:2562-2575  if pretrained_dir is not None and not has_resume_checkpoint:
                                       load_dir = pretrained_dir ; cfg.checkpoint.finetune = True
```

Consequence: anything missing from the checkpoint would retain **random init**, not base-model
values. But it cannot go missing silently:

* **dist-checkpoint layer.** `dist_ckpt_strictness` is never set by NeMo-RL (the YAML
  forwarding whitelist is only 5 keys, `NRL/nemo_rl/models/megatron/setup.py:1288-1294`), so
  the Megatron default `"assume_ok_unexpected"` applies
  (`MLM/megatron/training/config/training_config.py:584-593`). With `ASSUME_OK_UNEXPECTED`,
  `requires_explicit_ckpt_mismatch_check` is `False`
  (`MLM/megatron/core/dist_checkpointing/validation.py:86-89`), so **no** key diff is computed
  and nothing is stripped from the request (`:163-184`). A key the model asks for that is
  absent from the checkpoint therefore reaches the planner and **hard-raises**:

  ```
  MLM/megatron/core/dist_checkpointing/strategies/torch.py:504-507
      if sh_ten.key not in metadata.state_dict_metadata:
          raise KeyError(f"{sh_ten.key} from model not in state dict: ...")
  ```

  Keys present in the checkpoint but not requested by the model are ignored
  (`validation.py:266-275`: `missing_keys = set()` under the `*_UNEXPECTED` group).
* **`torch.nn.Module.load_state_dict` layer.** NeMo-RL passes no `strict`, so the bridge
  default `strict=True` flows through to `_load_model_state_dict`
  (`MB/.../checkpointing.py:2842-2868`), which catches the exception, retries with
  `strict=False`, and only **prints** a rank-0 warning for non-`_extra_state` mismatches. This
  layer is therefore soft. It is not load-bearing here because the dist layer already raised
  for any genuinely absent tensor.

Net: the only things "not overwritten" are the non-persistent buffers listed in the state
table, and they are zero-initialised accumulators with no cross-step meaning — the same on
a self-resume and on a fresh start.

## Q3 — what `+checkpointing.load_replay_buffer=false` actually skips

It skips **`replay_buffer.pt` only**.

* `NRL/nemo_rl/algorithms/single_controller.py:386-394`: when false, `_maybe_restore_replay_buffer`
  calls `_load_pending_rollouts_for_regeneration()` and returns before touching
  `replay_buffer.pt`.
* `train_dataloader.pt` — still loaded, unconditionally on `last_checkpoint_path is not None`
  (`setup.py:662-664`). Unaffected by the flag.
* `pending_rollouts.pt` — not only still loaded but **required**: absence raises
  `RuntimeError` (`single_controller.py:447-453`), a `partition_id` mismatch raises (`:469-474`),
  and `state["complete"] is not True` raises (`:475-480`). This is the launcher's third seed
  precondition for good reason.
* `replacement_reserve.pt` — still loaded (`:550-584`), silently skipped if absent (which is the
  normal case for `on_dropped_prompt != "replace"`).

Quantified: **a seeded run loses nothing relative to a self-resume**, because both pass the
flag. What either loses relative to `load_replay_buffer=true` is the partially-generated
rollout cohort, which is then regenerated prompt-for-prompt from the journal
(`single_controller.py:802-837 _regenerate_checkpoint_prompts`). Only rollouts whose
`target_step >= self._trainer_version` are re-queued (`:523-524`), so nothing already trained
on is re-run.

## Q4 — anything keyed on run name, experiment name or absolute path

Nothing is keyed on run or experiment name. The complete list of identity checks in the load
path, and how each behaves for chain L (MINF from chain K step_10):

| Check | Keyed on | Behaviour on mismatch | Chain L |
|---|---|---|---|
| checkpoint discovery | path only (`checkpoint_dir/step_*` glob) | n/a | matches |
| `load_dataloader_state` | `data.train.dataset_name` read from the checkpoint's **own** `config.yaml` vs the live config | **silently** skips the restore, dataloader starts at index 0, printed warning only (`NRL/nemo_rl/data/utils.py:91-101`) | same dataset → restores |
| replay-buffer restore | `sampler_name` in `training_info.json` vs `async_rl.sampler.name` | skips restore with a warning (`single_controller.py:403-412`) | not reached (`load_replay_buffer=false`) |
| pending-rollout restore | `partition_id` (constant `"rollout_data"`, `setup.py:569`) | **raises** | seed file contains `'rollout_data'` |
| replay-buffer load | `expected_partition_id`, `expected_group_size` | truncates/ignores | not reached |
| bridge resume-vs-pretrained | **existence of a tracker file** under `<ckpt>/policy/weights` | **silently falls back to the base model** (see below) | both trackers present |
| bridge TP/PP guard | `run_config.yaml` inside `iter_0000000` vs live `model.tensor_model_parallel_size`/`pipeline_model_parallel_size` | RNG and rerun state ignored (both already off); **raises** for a non-reshardable distributed optimizer | identical shapes |
| optimizer / RNG availability | `run_config["checkpoint"]["save_optim"] / ["save_rng"]` from the checkpoint's own `run_config.yaml` (`MB/.../checkpointing.py:3047, 3080`) | falls through to a fresh optimizer with a printed note (`:3107`) | both `true` in the seed |

Contents of the three in-checkpoint metadata files, read directly:

* `iter_0000000/metadata.json` = `{"sharded_backend": "torch_dist", "sharded_backend_version": 1}`.
  Carries no run identity.
* `iter_0000000/run_config.yaml` is the **writer's** config snapshot. It contains stale absolute
  paths from chain K (`checkpoint.save` points at the shared base-model conversion cache,
  `checkpoint.load: null`, `checkpoint.finetune: true` — chain K started from scratch). None of
  those are applied to the resuming run; only `model.*` TP/PP and `checkpoint.save_optim` /
  `save_rng` are read.
* `iter_0000000/train_state.pt` decodes to all zeros (`step`, `consumed_train_samples`,
  `skipped_train_samples`, `consumed_valid_samples`, `floating_point_operations_so_far` = 0;
  `do_train/do_valid/do_test` = False). NeMo-RL always writes iteration 0 and drives the step
  counter from `training_info.json`, so this file is inert.
* `checkpoints/latest_checkpoint_status.json` is write-only monitoring output
  (`NRL/nemo_rl/algorithms/grpo.py:2207-2222`); nothing reads it at resume. Its absence from a
  seeded run directory is harmless.

**The one silent fallback.** `checkpoint_exists(<ckpt>/policy/weights)`
(`MB/src/megatron/bridge/training/utils/checkpoint_utils.py:260-291`) returns True only if the
directory is itself an iteration dir (it is not — it *contains* `iter_0000000/`), or
`latest_train_state.pt` exists (`:280-283`, `TRACKER_PREFIX="latest"`, `TRAIN_STATE_FILE="train_state.pt"`,
`:32-33`), or `latest_checkpointed_iteration.txt` exists (`:286`). If a seed copy omitted both,
`resume_checkpoint_exists` would be `False`, the bridge would print
*"Checkpoint file not found in load directory ... Attempting to finetune with checkpoint in
&lt;pretrained_dir&gt;"* and set `finetune = True` (`MB/.../checkpointing.py:2567-2575`). The run
would then start from the **base model**, with no optimizer and no LR position, while NeMo-RL's
SingleController still resumed step 10, the dataloader at sample 416, the prompt journal and the
reserve — and W&B would show a healthy-looking run continuing at step 11.
`launch_swe_dump.sh:120` checks only `policy/weights/iter_0000000`, `config.yaml` and
`pending_rollouts.pt`; it does **not** check either tracker file.

## Q5 — what refit transfers, and whether it can mask a gap (the crux)

Refit transfers **parameters plus persistent buffers**, and never `_extra_state`.

* MINF (`backend: megatron`) — the transfer set comes from
  `MLM/megatron/core/resharding/utils.py:221-231`:

  ```
  def named_refit_tensors(module):
      yield from module.named_parameters(recurse=True)
      for full_name, _sub, _buf_name, buf in named_persistent_buffers(module):
          yield full_name, buf
  ```

  `named_persistent_buffers` (`utils.py:202-219`) walks `_buffers` and skips
  `_non_persistent_buffers_set`. The planner is driven by the **destination** roster
  (`MLM/megatron/core/resharding/planner.py:328-330`) and **raises** if a destination
  parameter or persistent buffer has no source (`:343-347`). `expert_bias` is explicitly
  in scope (`planner.py:363-365`) with a dtype-harmonisation step at
  `MLM/megatron/core/resharding/refit.py:412-459`. `_extra_state` appears nowhere under
  `megatron/core/resharding/`.
* vLLM (`load_format: dummy`, `_mtp_weights_from_refit: true`, `refit_transport: nccl_reshard`)
  — the source iterates `itertools.chain(model.named_parameters(), persistent_buffers(model))`
  with `if "_extra_state" in local_name: continue` inside the bridge's conversion-task builder.
  `expert_bias` is mapped to the HF name `...mixer.gate.e_score_correction_bias` (NemotronH
  bridge) and rides the misc packed-broadcast leg
  (`NRL/nemo_rl/weight_sync/nccl_reshard_utils.py:207-209`). The destination load is **not**
  strict: `NRL/nemo_rl/models/generation/vllm/vllm_backend.py:199` calls
  `model_runner.model.load_weights(weights=policy_weights)`, and the only completeness check
  compares against the *source's* manifest, so a vLLM parameter nobody wrote keeps its dummy
  random init.

**Direct answer.** Refit is a one-way mirror from trainer to engine. If a training-side buffer
were wrong or stale, refit would **copy the wrong value into the engine**, so the engine would
agree with the trainer rather than conceal the difference. Refit is therefore *irrelevant* to
the failure mode in the question: it cannot restore a value the trainer does not have, and it
cannot hide a value the trainer has wrong. The only thing refit genuinely masks is an
**engine-side** initialisation gap — which is exactly why `load_format: dummy` is safe for
chain K and run A (vLLM from scratch / vLLM arms) — and that masking is identical for seeded
and from-scratch arms.

One correction to the stated premise: the MINF inference model *does* read the checkpoint from
disk at construction. `_build_generation` passes `weights_path=weights_path` with
`skip_weight_load=False` (`NRL/nemo_rl/algorithms/single_controller_utils/setup.py:752, 262-269`
→ `NRL/nemo_rl/models/generation/megatron/megatron_generation.py:155-165`). It is belt-and-braces:
`SingleControllerActor.run` does `await self._sync_weights()` before starting any pump
(`NRL/nemo_rl/algorithms/single_controller.py:305-306`), so the engine is refit from the trainer
before the first rollout either way.

## Q6 — LR schedule and optimizer step counter

**Both resume at the parent's position, and the schedule is *not* moot.**

The recipe is not a flat constant LR. `swe_sc_cmh_common.yaml:241-249` sets
`lr: 3.0e-6`, `min_lr: 3.0e-6`, `lr_decay_style: "constant"`, but also
**`lr_warmup_iters: 10`** and **`lr_warmup_init: 3e-7`**. Megatron applies warmup before the
decay style is consulted:

```
MLM/megatron/core/optimizer_param_scheduler.py:230-233
    if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
        return self.init_lr + ((max_lr - self.init_lr) * float(self.num_steps) / float(self.lr_warmup_steps))
```

So LR ramps 3e-7 → 3e-6 over the first 10 optimizer steps. A scheduler reset would run the
whole warmup again at a 10x-low starting LR — a real, silent, weight-preserving change.

It does not happen. The chain L seed's `common_state` (extracted from
`iter_0000000/__0_0.distcp` via the `.metadata` storage index) reads:

```
iteration = 0
opt_param_scheduler: max_lr 3e-06, min_lr 3e-06, lr_warmup_steps 5120, num_steps 5120,
                     lr_decay_style 'constant', lr_decay_steps 591872, wd 0.0 constant
optimizer: {0: {param_state_sharding_type: 'dp_reshardable'}, 1: {...}}
```

`num_steps = 5120 = 10 steps x global_batch_size 512`, i.e. warmup exactly complete, and
`lr_warmup_steps` is likewise 5120 in sample units. `OptimizerParamScheduler.load_state_dict`
restores it with `self.step(increment=num_steps)` at
`MLM/megatron/core/optimizer_param_scheduler.py:390`, which is **outside** the
`override_opt_param_scheduler` guard (`:318-339`) — that flag only forces the *hyper-parameters*
to the YAML values. So the position is restored even with `override_opt_param_scheduler: true`.

The Adam step counter is restored too. The optimizer common object
`chained_0.optimizer.distributed.dp_group_idx_0.optimizer/shard_0_1` decodes to
`param_groups[0] = {max_lr 3e-06, min_lr 3e-06, lr 3e-06, betas (0.9, 0.999), eps 1e-08,
weight_decay 0.0, step 10}`. Bias correction therefore continues from step 10 rather than
restarting.

**The conditional to watch.** Both restores sit behind the *same* gate
(`MB/.../checkpointing.py:3353`): `not release and not finetune and load_optim and not
ignore_optimizer_state`. If `optimizer_path` came back `None` — which happens whenever the
`common_state` optimizer key cannot be read — the run would lose Adam moments *and* silently
restart the warmup at 3e-7, with nothing louder than a `warnings.warn` from
`NRL/nemo_rl/utils/checkpoint.py:234-239`.

## On-disk verification of the staged chain L seed

Seed: `.../runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromk10-prefix-nopg-20260923/checkpoints/step_10`,
copied from chain K (vLLM from scratch, seed 1234) `step_10`.

* `find -printf "%p %s"` output is **identical** between source and copy (file list + sizes).
* md5 identical for `training_info.json`, `config.yaml`, `train_dataloader.pt`,
  `pending_rollouts.pt`, `replacement_reserve.pt`, `replay_buffer.pt`,
  `policy/weights/latest_train_state.pt`, `policy/weights/latest_checkpointed_iteration.txt`,
  `iter_0000000/.metadata`, `iter_0000000/run_config.yaml`, `iter_0000000/train_state.pt`,
  `iter_0000000/metadata.json`, and a sampled 3 GB shard `__24_0.distcp`.
* `.metadata` holds 7214 keys, of which 6282 are `_extra_state` (note: the prior audit's
  figure of 12564 is exactly 2x this; counted by `state_dict_metadata` keys here),
  24 `*.mlp.router.expert_bias`, 0 `local_tokens_per_expert`, 128 `rng_state/shard_*`,
  1 `common_state/shard_0_1`, and `chained_0.optimizer.distributed.dp_group_idx_*` exp_avg /
  exp_avg_sq / param / optimizer / per_bucket_numel entries (36 each).
* `decoder.layers.1.mlp.router.expert_bias` decodes to 128 fp32 values, min 41.2863,
  max 41.3903, mean 41.3606, all non-zero. The buffer is initialised to **zeros**
  (`MLM/.../router.py:191-198`), so this is genuine accumulated state — and it *is* live:
  `freeze_moe_router: true` only clears `router.weight.requires_grad`
  (`NRL/nemo_rl/models/megatron/setup.py:1736-1738`); it never sets `frozen_expert_bias`, so
  the ±`moe_router_bias_update_rate` 1e-3 update at
  `MLM/megatron/core/distributed/finalize_model_grads.py:351-371` runs every step.
  A failure to load this buffer would zero the routing bias — large and silent. It cannot
  happen silently: the key is present, it is in the model's request, and an absent key raises
  (`strategies/torch.py:504-507`).
* `train_dataloader.pt` = `{'samples_yielded': 416, '_sampler_iter_yielded': 13,
  '_index_sampler_state': None}`. `data.shuffle: false` (`swe_sc_cmh_common.yaml:374`), so
  index 416 is a deterministic position independent of any seed.
* `pending_rollouts.pt` header carries `partition_id = 'rollout_data'`, matching the SC default.

## Things that look wrong, or that I could not verify

1. **Seed guard is incomplete (actionable).** `W/launch_swe_dump.sh:120` checks
   `policy/weights/iter_0000000`, `config.yaml` and `pending_rollouts.pt`. It should also
   require `policy/weights/latest_train_state.pt` **or**
   `policy/weights/latest_checkpointed_iteration.txt` (the two files
   `checkpoint_exists` accepts) and `training_info.json`. Without the tracker the run silently
   trains from the base model; `training_info.json` at least fails loudly
   (`NRL/nemo_rl/utils/checkpoint.py:589`).
2. **`grpo.seed` mismatch between chain L (MINF from chain K step_10) and its parent.** Chain K
   (vLLM from scratch, seed 1234) recorded `grpo.seed: 1234`; the launcher default from
   `swe_sc_cmh_common.yaml:74` is 42. Traced consequences are narrow — `set_seed` only touches
   the SC driver process (`NRL/nemo_rl/algorithms/utils.py:277-282`), the Megatron workers use
   `rng.seed = 1234` (the Megatron default, unaffected by `grpo.seed`), and vLLM derives its
   seed from node/bundle index (`NRL/nemo_rl/models/generation/vllm/vllm_worker.py:160-169`).
   I did **not** enumerate every consumer of the driver-side RNG, so pass
   `grpo.seed=1234` on the chain L command line for cleanliness.
3. **Dataloader-restore mismatch is silent.** `NRL/nemo_rl/data/utils.py:91-101` prints a
   warning and restarts at index 0 on a dataset-name mismatch. Harmless for chain L (same
   dataset) but it is the one content-keyed check in the path that degrades silently rather
   than raising.
4. **`load_optim` silently degrades.** As in Q6, a `None` `optimizer_path` costs the Adam state
   *and* the LR position behind a `warnings.warn`. Worth a hard assert for seeded runs.
5. **Bridge line numbers unverified against the container.** See the source caveat. Semantics
   were confirmed against two independent bridge checkouts; exact lines may differ in the image.
6. **Could not determine statically:** whether the container's `megatron_bridge` differs
   behaviourally from `ea5dded83`. Cheapest runtime check: at the start of the next segment,
   grep the driver log for the bridge's own resume messages —
   `successfully loaded checkpoint from <...>/checkpoints/step_N/policy/weights` must appear
   (and *not* `Attempting to finetune with checkpoint in`), together with
   `> overriding learning rate value to 3e-06` from `_check_and_set`. A second, equally cheap
   confirmation: the first logged `lr` for step N+1 must be `3e-06`, not `3e-07` or an
   intermediate warmup value.

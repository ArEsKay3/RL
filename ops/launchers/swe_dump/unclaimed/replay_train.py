#!/usr/bin/env python3
"""Replay a source run's disk-dumped rollout data through the real SC training pipeline.

No generation engine is ever constructed. Only the source of rollout content
changes; every downstream stage runs exactly as it does in a live chain:

  * the real InOrderSampler (admit/select/evict against target_step)
  * the real prev_logprobs recompute (trainer's own logprob inference)
  * the real _advantage_stage (real GRPOAdvantageEstimator, grouped on
    prompt_ids_for_adv, writing the real token-level dump as a side effect)
  * the real Megatron train step
  * the real CheckpointManager save path

This is done by constructing the real SingleControllerActor + real TQPolicy +
real TQReplayBuffer with gen_handle=None (mirroring
tests/unit/single_controller/test_train_pump.py::test_train_pump_drives_mcore_training_step),
pre-populating the buffer directly from a source run's
dumps/token_level/step_NNNNN_chunk_*.pt files reshaped into the buffer's
native pre-advantage schema (input_ids, input_lengths, generation_logprobs,
token_mask, sample_mask, prompt_ids_for_adv, total_reward -- NOT advantages or
prev_logprobs, which the real pipeline computes fresh), then calling the real
_train_pump() once (it loops internally until grpo.max_num_steps, per
single_controller.py:1128).

Each pre-populated group's target_step is stamped directly (1..num_steps, in
source-step order) instead of via the sampler's own admit() gate, since there
is no live rollout pump to call admit() for us; see the module docstring in
nemo_rl/algorithms/async_utils/staleness_sampler.py (InOrderSampler.select
keys strictly on target_step == current_train_weight, independent of
weight_version) for why this is the correct substitute.

prompt_ids_for_adv (nemo_rl/experience/payload.py: record_to_train_batch) is
the padded prompt-only token ids, grouped by exact tensor equality in
calculate_baseline_and_std_per_prompt (nemo_rl/algorithms/utils.py). The
token-level dump does not store prompt/completion boundaries directly, so
this script derives each row's prompt length as the position of the first
token_mask==1 entry (nemo_rl/algorithms/grpo.py:
add_grpo_token_loss_masks_and_generation_logprobs masks every non-generated
message, including prompt messages, to 0; the first generated assistant
token is therefore the first 1). This is the single biggest unverified
assumption in this script -- see the self-check printed at the end.

ENTRYPOINT CONTRACT -- matches examples/run_grpo_single_controller.py exactly
(--config <path> plus trailing Hydra key=value/+key=value overrides) so this
runs through the unmodified real launcher (launch_swe_dump.sh /
nano35_launch.sh) via TRAIN_ENTRYPOINT, not a bespoke CLI. The launcher
already supplies policy.model_name, cluster.num_nodes,
policy.generation.colocated.resources.num_nodes, checkpointing.checkpoint_dir,
logger.log_dir, async_rl.dump.dir, and grpo.max_num_steps (from NRL_MAX_STEPS)
exactly as it does for a live chain -- this script does not re-derive any of
those. Replay-specific inputs arrive as new `replay.*` config keys (MasterConfig
is `extra="allow"`, same mechanism +checkpointing.load_replay_buffer=false
already uses):

    +replay.source_run=/path/to/source/run/dir
    +replay.steps=1:10                          # start:end, 1-indexed, inclusive
    +replay.weights_path=...                    # optional: resume from a real
    +replay.optimizer_path=...                  #   checkpoint instead of a fresh
                                                  #   HF import (policy.model_name
                                                  #   is ignored when this is set)

Node shape is NOT a script flag: it is read back out of cluster.num_nodes and
policy.generation.colocated.resources.num_nodes (both launcher-supplied),
mirroring _build_clusters's own non-colocated arithmetic
(single_controller_utils/setup.py:157-177) so this script's train-only Ray
cluster is sized identically to what a live chain's trainer actually gets,
even though the generation-node allocation (required by nano35_launch.sh's
own `NUM_GEN_NODES must be > 0` check, single_controller_utils path not
involved) sits idle here.

Usage (through the real launcher, not directly):
    EXP_NAME=... SMOKE=1 ENGINE=vllm TRAIN_ENTRYPOINT='--no-sync ./examples/nemo_gym/nemotron-3.5-nano/replay_train.py' \\
      bash launch_swe_dump.sh +replay.source_run=/scratch/.../runs/.../vllm_dump-20260918 +replay.steps=1:2

Add --dry-run (a real CLI flag, parsed before the Hydra passthrough, same as
--config) to load/reshape the dump data and print buffer/partition sizing
without touching Ray or a GPU cluster -- catches data-shape bugs on a
CPU/login node before spending a node allocation.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from types import SimpleNamespace

import torch
import yaml
from omegaconf import OmegaConf
from tensordict import TensorDict

# Same sys.path hygiene as run_grpo_single_controller.py: this file lives
# under examples/nemo_gym/nemotron-3.5-nano/, which has no __init__.py, so it
# must not shadow the real nemo_gym package as a namespace package.
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir in sys.path:
    sys.path.remove(current_dir)

# Union of DP_TRAIN_FIELDS (nemo_rl/data_plane/schema.py) + the rollout-stage
# extras (payload.py: pack_payload/record_to_train_batch) that _advantage_stage
# and TQPolicy's training fetch both read from. advantages/prev_logprobs are
# registered because the schema is shared, but this script never writes them --
# _advantage_stage and the prev-logprob recompute stage do, for real, at
# runtime.
REGISTERED_FIELDS = [
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "prev_logprobs",
    "reference_policy_logprobs",
    "advantages",
    "token_mask",
    "sample_mask",
    "total_reward",
    "prompt_ids_for_adv",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Mirrors run_grpo_single_controller.py's parse_args: --config plus passthrough
    Hydra overrides, with one extra real flag (--dry-run) parsed the same way."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load/reshape dump data and print sizing only; no Ray, no cluster, no training",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _load_step_rows(source_run: str, step: int) -> dict[str, dict]:
    """Load every token_level chunk for one source step; return sample_id -> row dict.

    Row tensors are CPU, unpadded (native dump trim), keyed exactly like
    write_token_level_chunk (nemo_rl/experience/rollout_dump.py) wrote them:
    input_ids/token_mask trimmed from offset 0, generation_logprobs from
    offset 1 (logprob_offset), rewards/masks per-row.
    """
    pattern = os.path.join(
        source_run, "dumps", "token_level", f"step_{step:05d}_chunk_*.pt"
    )
    chunk_paths = sorted(glob.glob(pattern))
    if not chunk_paths:
        raise FileNotFoundError(f"No token_level chunks for step {step} under {source_run}")

    rows: dict[str, dict] = {}
    for path in chunk_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        lengths = obj["input_lengths"].long().reshape(-1)
        n = int(lengths.shape[0])
        ids_off = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        lp_off = torch.cat([torch.zeros(1, dtype=torch.long), (lengths - 1).cumsum(0)])
        sample_ids = obj["sample_ids"]
        tags = obj.get("tags") or [{}] * n
        input_ids_flat = obj["input_ids"]
        token_mask_flat = obj["token_mask"]
        gen_lp_flat = obj["generation_logprobs"]
        rewards = obj["rewards"]

        for i in range(n):
            length = int(lengths[i])
            input_ids_i = input_ids_flat[ids_off[i] : ids_off[i + 1]].long()
            token_mask_i = token_mask_flat[ids_off[i] : ids_off[i + 1]].long()
            gen_lp_i = gen_lp_flat[lp_off[i] : lp_off[i + 1]].float()

            # token_mask_i[0] is the offset-0 dummy/never-loss-read column (see
            # write_token_level_chunk); real per-position mask starts at index 1
            # and add_grpo_token_loss_masks_and_generation_logprobs
            # (nemo_rl/algorithms/grpo.py) sets it to 0 for every non-generated
            # message (prompt AND any interleaved tool/user turns) and 1 only for
            # actually-sampled assistant tokens, so the first 1 is the first
            # generated token -- i.e. the end of the prompt.
            nz = (token_mask_i[1:] == 1).nonzero(as_tuple=True)[0]
            prompt_length = (int(nz[0]) + 1) if len(nz) else length

            sample_id = sample_ids[i]
            rows[sample_id] = {
                "group_id": sample_id.rsplit("_g", 1)[0],
                "input_ids": input_ids_i,
                "token_mask": token_mask_i,
                "generation_logprobs": gen_lp_i,
                "length": length,
                "prompt_length": prompt_length,
                "reward": float(rewards[i]),
                "weight_version": (tags[i] or {}).get("weight_version", 0),
            }
    return rows


def _group_rows(rows: dict[str, dict]) -> list[tuple[str, list[tuple[str, dict]]]]:
    """Bucket rows by group_id, preserving first-seen order (stable replay order)."""
    groups: dict[str, list[tuple[str, dict]]] = {}
    order: list[str] = []
    for sample_id in sorted(rows.keys()):
        row = rows[sample_id]
        gid = row["group_id"]
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append((sample_id, row))
    return [(gid, groups[gid]) for gid in order]


def _pack_group(members: list[tuple[str, dict]], pad_token_id: int) -> tuple[list[str], "TensorDict"]:
    """Build (sample_ids, TensorDict) for one group in the buffer's native pre-advantage
    schema (payload.py: record_to_train_batch's output shape), sourced from dump rows
    instead of a live PromptGroupRecord. advantages/prev_logprobs are deliberately absent:
    the real pipeline computes both fresh (prev_logprobs by the trainer's own recompute
    stage, advantages by the real advantage estimator in _advantage_stage).
    """
    n = len(members)
    seq_len = max(row["length"] for _, row in members)
    prompt_len = max(row["prompt_length"] for _, row in members)

    input_ids = torch.full((n, seq_len), pad_token_id, dtype=torch.long)
    token_mask = torch.zeros((n, seq_len), dtype=torch.long)
    gen_lp = torch.zeros((n, seq_len), dtype=torch.float32)
    input_lengths = torch.zeros((n,), dtype=torch.long)
    prompt_ids_for_adv = torch.full((n, prompt_len), pad_token_id, dtype=torch.long)
    total_reward = torch.zeros((n,), dtype=torch.float32)
    # float32, not bool/long: payload.py:154 writes it this way at commit time,
    # and grpo.py's seq-logprob-error masking (~line 2677) later overwrites
    # this same DataPlane field with an explicit .float() tensor -- the
    # DataPlane enforces one dtype per field across all writes, so the initial
    # write must already match or a live TransferQueueController job errors
    # with "dtype mismatch" the first time _advantage_stage runs.
    sample_mask = torch.ones((n,), dtype=torch.float32)
    # Placeholders only: prev_logprobs is overwritten by the trainer's live
    # recompute before _advantage_stage reads it; reference_policy_logprobs is
    # never read because every SWE dump config has reference_policy_kl_penalty
    # == 0 (loss_functions.py:391 gates the read on that). Zeros are inert either way.
    prev_logprobs = torch.zeros((n, seq_len), dtype=torch.float32)
    reference_policy_logprobs = torch.zeros((n, seq_len), dtype=torch.float32)

    sample_ids = [sid for sid, _ in members]
    for row_idx, (_sid, row) in enumerate(members):
        length = row["length"]
        input_ids[row_idx, :length] = row["input_ids"]
        token_mask[row_idx, :length] = row["token_mask"]
        gen_lp[row_idx, 1:length] = row["generation_logprobs"]
        input_lengths[row_idx] = length
        p = row["prompt_length"]
        prompt_ids_for_adv[row_idx, :p] = row["input_ids"][:p]
        total_reward[row_idx] = row["reward"]

    fields = TensorDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "generation_logprobs": gen_lp,
            "prev_logprobs": prev_logprobs,
            "reference_policy_logprobs": reference_policy_logprobs,
            "total_reward": total_reward,
            "prompt_ids_for_adv": prompt_ids_for_adv,
        },
        batch_size=(n,),
    )
    return sample_ids, fields


def _find_source_train_iters(source_run: str) -> int | None:
    """Best-effort: read the SOURCE run's own real megatron_cfg.train_iters (or
    grpo.max_num_steps) out of whichever checkpoints/step_*/config.yaml it saved,
    so the replay's LR schedule sits at the same position in the same schedule the
    source run's own steps actually used -- NOT the replay's own short step count
    (those are unrelated: a live chain's train_iters is fixed for its whole horizon,
    e.g. 1156, regardless of how many steps a given 4h segment manages to run).
    Returns None if no source config can be found/parsed; caller must not silently
    guess in that case.
    """
    candidates = sorted(glob.glob(os.path.join(source_run, "checkpoints", "step_*", "config.yaml")))
    for path in reversed(candidates):  # prefer the latest available rung
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            return int(cfg["policy"]["megatron_cfg"]["train_iters"])
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return None


def main() -> None:
    args, overrides = parse_args()

    if not args.config:
        raise SystemExit("--config is required (same contract as run_grpo_single_controller.py)")

    # Imports deferred past --config/--dry-run parsing so `--help` and config-only
    # errors don't require nemo_rl's full (GPU-adjacent) import graph.
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    cfg = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        cfg = parse_hydra_overrides(cfg, overrides)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig

    master_config = MasterConfig(**config_dict)

    replay_cfg = getattr(master_config, "replay", None)
    if not replay_cfg or "source_run" not in replay_cfg:
        raise SystemExit(
            "Missing +replay.source_run=<path> (and +replay.steps=start:end) -- "
            "these are the replay-specific inputs; see this script's module docstring."
        )
    source_run = replay_cfg["source_run"]
    start_step, end_step = (int(x) for x in replay_cfg.get("steps", "1:10").split(":"))
    if start_step > end_step:
        raise ValueError(f"replay.steps start must be <= end, got {replay_cfg.get('steps')!r}")
    num_steps = end_step - start_step + 1
    weights_path = replay_cfg.get("weights_path")
    optimizer_path = replay_cfg.get("optimizer_path")

    print(f"Loading source dump rows for steps {start_step}..{end_step} from {source_run}")
    step_rows = {
        source_step: _load_step_rows(source_run, source_step)
        for source_step in range(start_step, end_step + 1)
    }
    total_rows = sum(len(rows) for rows in step_rows.values())
    total_groups = sum(len(_group_rows(rows)) for rows in step_rows.values())
    print(f"Loaded {total_rows} rows across {total_groups} groups, {num_steps} steps")

    if args.dry_run:
        for source_step, rows in step_rows.items():
            groups = _group_rows(rows)
            sizes = sorted({len(members) for _, members in groups})
            print(f"  step {source_step}: {len(rows)} rows, {len(groups)} groups, group sizes {sizes}")
        print("Dry run only -- no Ray, no cluster, no training. Re-run without --dry-run to execute.")
        return

    # grpo.max_num_steps drives _train_pump's own termination
    # (`while self._train_steps < grpo_cfg.max_num_steps`, single_controller.py:1128).
    # Force it to match exactly what we pre-populate -- NRL_MAX_STEPS (if the
    # launcher set it) is a default, not a guarantee it matches --steps, and a
    # mismatch in the "keep looping past the pre-populated data" direction would
    # hang forever waiting for rollouts that will never arrive (no rollout pump).
    master_config.grpo.max_num_steps = num_steps

    # Mirrors _maybe_inject_megatron_train_iters, but sourced from the SOURCE
    # run's own real train_iters, not this replay's step count -- see
    # _find_source_train_iters's docstring.
    if master_config.policy.get("megatron_cfg", {}).get("enabled", False):
        source_train_iters = _find_source_train_iters(source_run)
        if source_train_iters is not None:
            master_config.policy["megatron_cfg"]["train_iters"] = source_train_iters
            print(f"train_iters set to source run's own value: {source_train_iters}")
        else:
            print(
                "[WARN] Could not read the source run's own train_iters from any "
                "checkpoints/step_*/config.yaml; leaving the launcher-supplied "
                "value in place. Verify the LR schedule position is still valid "
                "for the step range being replayed before trusting results.",
                file=sys.stderr,
            )

    import ray
    from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
    from nemo_rl.algorithms.grpo import _create_advantage_estimator, _initial_grpo_save_state
    from nemo_rl.algorithms.loss import ClippedPGLossFn
    from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
    from nemo_rl.algorithms.single_controller import SingleControllerActor
    from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerActorArgs
    from nemo_rl.algorithms.utils import get_tokenizer, set_seed
    from nemo_rl.data_plane import KVBatchMeta
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray

    set_seed(master_config.grpo.seed)
    init_ray()
    tokenizer = get_tokenizer(master_config.policy["tokenizer"])
    pad_token_id = int(tokenizer.pad_token_id or 0)

    # Train-only cluster, sized identically to a live chain's own trainer cluster:
    # mirrors _build_clusters's non-colocated branch
    # (single_controller_utils/setup.py:157-177) exactly. The launcher-allocated
    # generation-node pool (required by its own `NUM_GEN_NODES must be > 0` check)
    # is real Slurm/Ray allocation but sits completely idle -- this script never
    # builds a generation cluster or reads policy.generation.colocated.resources
    # for anything beyond this arithmetic.
    total_nodes = master_config.cluster["num_nodes"]
    gpus_per_node = master_config.cluster["gpus_per_node"]
    gen_resources = master_config.policy["generation"]["colocated"]["resources"]
    gen_nodes = gen_resources.get("num_nodes") or 1
    if total_nodes == 1:
        train_gpus_per_node = gpus_per_node - gen_resources["gpus_per_node"]
        train_nodes = 1
    else:
        train_gpus_per_node = gpus_per_node
        train_nodes = total_nodes - gen_nodes
    assert train_nodes > 0, f"train_nodes must be > 0: {total_nodes} - {gen_nodes} = {train_nodes}"
    print(f"Train cluster: {train_nodes} nodes x {train_gpus_per_node} gpus/node "
          f"(of {total_nodes} total allocated; {gen_nodes} generation nodes idle)")

    train_cluster = RayVirtualCluster(
        name="replay_train_cluster",
        bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
        use_gpus=True,
        num_gpus_per_node=train_gpus_per_node,
        max_colocated_worker_groups=1,
    )

    from nemo_rl.models.policy.tq_policy import TQPolicy

    partition_id = "train"
    trainer = TQPolicy(
        cluster=train_cluster,
        config=master_config.policy,
        tokenizer=tokenizer,
        dp_cfg=master_config.data_plane,
        tq_partition_id=partition_id,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=master_config.loss_fn.reference_policy_kl_penalty > 0,
    )

    try:
        dp_client = trainer.dp_client
        dp_client.register_partition(
            partition_id=partition_id,
            fields=REGISTERED_FIELDS,
            num_samples=total_rows,
            consumer_tasks=["prev_lp", "ref_lp", "train"],
        )

        tq_buffer = TQReplayBuffer(
            dp_client,
            partition_id=partition_id,
            pad_value_dict={"input_ids": pad_token_id, "token_ids": pad_token_id},
        )

        # target_step is stamped directly here (0..num_steps-1, in source-step
        # order) instead of via the sampler's admit() gate: admit() exists to
        # pace a LIVE rollout pump against the trainer's current position, and
        # there is no rollout pump running. InOrderSampler.select matches
        # target_step == current_train_weight (_trainer_version); both
        # _trainer_version (single_controller.py:281) and _dispatch_index
        # (staleness_sampler.py:133, pre-incremented before the first stamp,
        # so its first-ever value is 0) start at 0 -- confirmed by reading, not
        # assumed: an earlier 1-indexed version of this stamping deadlocked
        # _train_pump forever (target_step=1 is never selectable while
        # current_train_weight=0, and nothing to select means nothing trains,
        # means current_train_weight never advances -- _rollout_exhausted is
        # never set without a rollout pump, so the 5ms retry in
        # single_controller.py:1187 just spins). Replay step i (0-indexed
        # within this run, independent of the source run's own step numbers)
        # must carry target_step == i for _train_pump to ever select it.
        for replay_idx, source_step in enumerate(range(start_step, end_step + 1)):
            groups = _group_rows(step_rows[source_step])
            for group_id, members in groups:
                sample_ids, fields = _pack_group(members, pad_token_id)
                tags = [{"weight_version": replay_idx} for _ in sample_ids]
                dp_client.put_samples(
                    sample_ids=sample_ids,
                    partition_id=partition_id,
                    fields=fields,
                    tags=tags,
                )
                meta = KVBatchMeta(
                    partition_id=partition_id,
                    task_name="train",
                    sample_ids=sample_ids,
                    fields=list(fields.keys()),
                    sequence_lengths=[fields["input_ids"].shape[1]] * len(sample_ids),
                    tags=[dict(t) for t in tags],
                )
                # reserve() keeps all 8 of TQReplayBuffer's parallel lists in
                # sync (meta_list, start_weight_list, end_weight_list,
                # target_step_list, ready_list, _group_ids, prompt_list,
                # journal_id_list -- replay_buffer.py:738-749). An earlier
                # version of this script appended to only 6 of them directly;
                # the desync doesn't surface at population time, only later as
                # IndexError inside TQReplayBuffer.remove() the first time a
                # group is actually selected (replay_buffer.py:924-931 deletes
                # from all 8 in lockstep by index). prompt=None/journal_id=None
                # are fine: both only matter for regenerating a dropped prompt
                # from a live rollout pump, which never runs here.
                tq_buffer.reserve(
                    weight_version=replay_idx,
                    target_step=replay_idx,
                    group_id=group_id,
                )
                idx = tq_buffer._group_ids.index(group_id)
                tq_buffer.meta_list[idx] = meta
                tq_buffer.end_weight_list[idx] = replay_idx
                tq_buffer.ready_list[idx] = True
        print(f"Pre-populated {total_groups} groups across {num_steps} target steps")

        loss_fn = ClippedPGLossFn(master_config.loss_fn)
        advantage_estimator = _create_advantage_estimator(master_config)

        # No live generation engine: nothing to refit, nothing to hand a
        # weight version to. Mirrors test_train_pump.py's stubs exactly.
        weight_synchronizer = SimpleNamespace(sync_weights=lambda *, kv_scales=None: None)
        rollout_manager = SimpleNamespace(
            _tq_buffer=None,
            set_next_nemo_gym_task_index=lambda _value: None,
            get_next_nemo_gym_task_index=lambda: 0,
            set_weight_version=lambda _v: None,
        )

        actor_args = SingleControllerActorArgs(
            gen_handle=None,
            trainer_handle=trainer,
            env_handles={},
            train_cluster=train_cluster,
            inference_cluster=None,  # type: ignore[arg-type]
            dp_client=dp_client,
            dataloader=None,  # type: ignore[arg-type]  # _rollout_pump never starts
            weight_synchronizer=weight_synchronizer,  # type: ignore[arg-type]
            advantage_estimator=advantage_estimator,
            loss_fn=loss_fn,
            rollout_manager=rollout_manager,  # type: ignore[arg-type]
            tq_buffer=tq_buffer,
            partition_id=partition_id,
            save_state=_initial_grpo_save_state(),
            last_checkpoint_path=None,
        )

        ctrl = SingleControllerActor.remote(
            master_config=master_config,
            actor_args=actor_args,
            setup_timing_metrics=SetupTimingMetrics(),
        )
        print("Driving _train_pump() (loops internally through grpo.max_num_steps)...")
        ray.get(ctrl._train_pump.remote())
        state = ray.get(ctrl.ping.remote())
        print(f"Replay complete: {state}")
        print(f"Checkpoints under: {master_config.checkpointing['checkpoint_dir']}")
        print(f"Token-level dumps under: {master_config.async_rl.dump.dir}/token_level")
        print(
            "Recommended validation: for the FIRST replayed step, compare "
            "advantages per sample_id between this run's own step dump and the "
            "source run's dump for the same source step -- sample_ids are "
            "carried through unchanged, so they join directly. A mismatch beyond "
            "float noise means the prompt_ids_for_adv reconstruction (grouping) "
            "is wrong; a match is strong evidence it is right."
        )
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()

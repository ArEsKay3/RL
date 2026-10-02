# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Regression coverage for the BF16 backport of NeMo RL PR #3739."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from nemo_rl.weight_sync.nccl_reshard_utils import check_nccl_reshard_refit_support


def _config():
    return OmegaConf.create(
        {
            "policy": {
                "precision": "bfloat16",
                "megatron_cfg": {
                    "enabled": True,
                    "tensor_model_parallel_size": 4,
                    "expert_model_parallel_size": 32,
                    "expert_tensor_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "sequence_parallel": True,
                    "fp8_cfg": {"enabled": False, "fp8_param": False},
                },
                "generation": {
                    "backend": "megatron",
                    "refit_transport": "nccl_reshard",
                    "colocated": {"enabled": False},
                    # Stale vLLM options inherited by the Megatron recipe are irrelevant.
                    "vllm_cfg": {"precision": "fp8"},
                    "mcore_generation_config": {
                        "tensor_model_parallel_size": 4,
                        "expert_model_parallel_size": 1,
                        "expert_tensor_parallel_size": 4,
                        "pipeline_model_parallel_size": 1,
                        "transformer_impl": "inference_optimized",
                        "refit_backend": None,
                    },
                },
            }
        }
    )


def test_parity_expert_tensor_layout_is_preserved():
    from nemo_rl.models.generation.megatron.megatron_generation import (
        MegatronGeneration,
    )
    from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
        NcclReshardWeightSynchronizer,
    )

    config = _config()
    check_nccl_reshard_refit_support(config)
    generation = object.__new__(MegatronGeneration)
    generation.cfg = config.policy.generation
    sync = NcclReshardWeightSynchronizer(
        SimpleNamespace(cfg=config.policy), generation, None, None
    )
    assert sync._train_parallelism() == {
        "tp_size": 4,
        "ep_size": 32,
        "etp_size": 1,
        "pp_size": 1,
    }
    assert sync._gen_parallelism() == {
        "tp_size": 4,
        "ep_size": 1,
        "etp_size": 4,
        "pp_size": 1,
    }


def test_recipe_selects_nccl_reshard_through_factory():
    from pathlib import Path

    from nemo_rl.models.generation.megatron.megatron_generation import (
        MegatronGeneration,
    )
    from nemo_rl.utils.config import load_config_with_inheritance
    from nemo_rl.weight_sync.factory import create_weight_synchronizer
    from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
        NcclReshardWeightSynchronizer,
    )

    root = Path(__file__).resolve().parents[3]
    config = load_config_with_inheritance(
        root
        / "examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_parity_minf_nccl_reshard.yaml"
    )
    check_nccl_reshard_refit_support(config)
    generation = object.__new__(MegatronGeneration)
    generation.cfg = config.policy.generation
    sync = create_weight_synchronizer(
        SimpleNamespace(cfg=config.policy),
        generation,
        "megatron",
        False,
        train_cluster=object(),
        inference_cluster=object(),
    )
    assert isinstance(sync._transport, NcclReshardWeightSynchronizer)
    assert sync._transport._gen_parallelism()["etp_size"] == 4


@pytest.mark.parametrize(
    "unsupported", ["colocated", "pp", "fp8_storage", "fp8_generation"]
)
def test_backport_rejects_unimplemented_modes(unsupported):
    config = _config()
    if unsupported == "colocated":
        config.policy.generation.colocated.enabled = True
    elif unsupported == "pp":
        config.policy.generation.mcore_generation_config.pipeline_model_parallel_size = 2
    elif unsupported == "fp8_storage":
        config.policy.megatron_cfg.fp8_cfg.fp8_param = True
    else:
        config.policy.generation.mcore_generation_config.fp8_cfg = {"enabled": True}
    with pytest.raises(ValueError):
        check_nccl_reshard_refit_support(config)


def test_nccl_reshard_lifecycle_and_failed_transfer_stays_suspended(monkeypatch):
    from nemo_rl.weight_sync import megatron_weight_synchronizer as transport_module
    from nemo_rl.weight_sync.megatron_weight_synchronizer import (
        MegatronWeightSynchronizer,
    )

    calls = []
    transport = MagicMock()
    transport.sync_weights.side_effect = lambda **kwargs: calls.append("transfer")
    monkeypatch.setattr(
        transport_module, "NcclReshardWeightSynchronizer", lambda *args: transport
    )
    policy = MagicMock()
    policy.offload_before_refit.side_effect = lambda: calls.append("offload")
    gen = MagicMock()
    gen.cfg = _config().policy.generation
    gen.suspend_for_refit.side_effect = lambda: calls.append("suspend")
    gen.prepare_for_generation.side_effect = lambda tags: calls.append(tags[0])
    gen.resume_after_refit.side_effect = lambda: calls.append("resume")
    sync = MegatronWeightSynchronizer(
        policy, gen, colocated=False, train_cluster=object(), inference_cluster=object()
    )
    sync.init_communicator()
    transport.init_communicator.assert_called_once()
    sync.sync_weights()
    assert calls == ["suspend", "offload", "weights", "transfer", "kv_cache", "resume"]
    policy.swap_weights_via_reshard.assert_not_called()
    assert not sync.is_stale

    calls.clear()
    sync._stale = True
    transport.sync_weights.side_effect = RuntimeError("transfer failed")
    with pytest.raises(RuntimeError, match="transfer failed"):
        sync.sync_weights()
    assert calls == ["suspend", "offload", "weights"]
    assert sync.is_stale


def _make_refit_task(
    *,
    param_name: str,
    destination: object,
    dependencies: tuple[str, ...],
    local_specs: tuple[tuple[str, str], ...] = (),
    mapping: object | None = None,
):
    from nemo_rl.models.generation.megatron.megatron_worker import (
        _MegatronRefitTask,
    )

    specs = ()
    if local_specs:
        from megatron.bridge.models.conversion.param_mapping import LocalHFParamSpec

        specs = tuple(
            LocalHFParamSpec(
                name,
                None if component == "full" else -2,
                1 if component == "up" else 0,
                1 if component == "full" else 2,
            )
            for name, component in local_specs
        )

    def combine_local_hf_weights(weights):
        if len(specs) == 1:
            return weights[specs[0].name]
        return torch.cat(
            [
                weights[spec.name]
                for spec in sorted(specs, key=lambda item: item.split_index)
            ],
            dim=-2,
        )

    conversion_task = SimpleNamespace(
        param_name=param_name,
        global_param_name=param_name,
        vp_stage=0,
        hf_param_names=dependencies,
        mapping=mapping if mapping is not None else MagicMock(),
        local_hf_param_specs=lambda: specs,
        combine_local_hf_weights=combine_local_hf_weights,
    )
    return _MegatronRefitTask(
        conversion_task=conversion_task,
        destination=destination,
        target_id=id(destination),
    )


def test_megatron_m2n_unstacks_grouped_experts_into_local_weights() -> None:
    from nemo_rl.models.generation.megatron.megatron_worker import (
        MegatronGenerationRefitMixin,
    )

    tasks = []
    destinations = []
    for expert in range(2):
        gate_name = f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
        up_name = f"model.layers.0.mlp.experts.{expert}.up_proj.weight"
        destination = torch.nn.Parameter(
            torch.zeros((4, 2), dtype=torch.bfloat16), requires_grad=False
        )
        destinations.append(destination)
        tasks.append(
            _make_refit_task(
                param_name=(
                    f"decoder.layers.0.mlp.experts.local_experts.{expert}.linear_fc1.weight"
                ),
                destination=destination,
                dependencies=(gate_name, up_name),
                local_specs=((gate_name, "gate"), (up_name, "up")),
            )
        )

    grouped_gate = "model.layers.0.mlp.experts.gate_proj.weight"
    grouped_up = "model.layers.0.mlp.experts.up_proj.weight"
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {"name": grouped_gate, "grouped_expert_proj": "gate_proj"},
                {"name": grouped_up, "grouped_expert_proj": "up_proj"},
            ]
        },
    }
    worker = object.__new__(MegatronGenerationRefitMixin)
    param_map = worker._build_destination_hf_to_local_param_map(refit_info, tasks)

    orphaned_storage = [destination.data for destination in destinations]
    for destination in destinations:
        destination.data = torch.full_like(destination, -1)

    gate_spec = param_map.get(grouped_gate)
    gate_ctx = gate_spec.pre(None)
    gate_ctx.buf[0].fill_(1)
    gate_ctx.buf[1].fill_(2)
    gate_spec.post(gate_ctx)
    up_spec = param_map.get(grouped_up)
    up_ctx = up_spec.pre(None)
    up_ctx.buf[0].fill_(3)
    up_ctx.buf[1].fill_(4)
    up_spec.post(up_ctx)

    assert torch.equal(destinations[0][:2], torch.ones_like(destinations[0][:2]))
    assert torch.equal(destinations[0][2:], torch.full_like(destinations[0][2:], 3))
    assert torch.equal(destinations[1][:2], torch.full_like(destinations[1][:2], 2))
    assert torch.equal(destinations[1][2:], torch.full_like(destinations[1][2:], 4))
    assert all(torch.count_nonzero(storage).item() == 0 for storage in orphaned_storage)


def test_megatron_m2n_resolves_direct_destination_after_parameter_rebind() -> None:
    from nemo_rl.models.generation.megatron.megatron_worker import (
        MegatronGenerationRefitMixin,
    )

    gate_name = "model.layers.0.mlp.gate_proj.weight"
    up_name = "model.layers.0.mlp.up_proj.weight"
    destination = torch.nn.Parameter(
        torch.zeros((4, 2), dtype=torch.bfloat16), requires_grad=False
    )
    task = _make_refit_task(
        param_name="decoder.layers.0.mlp.linear_fc1.weight",
        destination=destination,
        dependencies=(gate_name, up_name),
        local_specs=((gate_name, "gate"), (up_name, "up")),
    )
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {"model.layers.0": [{"name": gate_name}]},
    }
    worker = object.__new__(MegatronGenerationRefitMixin)
    param_map = worker._build_destination_hf_to_local_param_map(refit_info, [task])
    gate_spec = param_map.get(gate_name)
    assert gate_spec is not None
    assert gate_spec.pre is not None

    orphaned_storage = destination.data
    destination.data = torch.full_like(destination, -1)
    gate_ctx = gate_spec.pre(gate_spec.base)
    gate_ctx.buf.fill_(7)

    assert torch.equal(destination[:2], torch.full_like(destination[:2], 7))
    assert torch.equal(destination[2:], torch.full_like(destination[2:], -1))
    assert torch.count_nonzero(orphaned_storage).item() == 0


@pytest.mark.parametrize(
    ("param_info", "expected_message"),
    [
        (
            {"name": "model.layers.0.mlp.gate_proj.weight"},
            "No local Megatron destination maps",
        ),
        (
            {
                "name": "model.layers.0.mlp.experts.gate_proj.weight",
                "grouped_expert_proj": "gate_proj",
            },
            "No local Megatron experts map",
        ),
    ],
)
def test_megatron_m2n_rejects_weights_with_no_local_destination(
    param_info: dict[str, str], expected_message: str
) -> None:
    """A train/gen geometry mismatch must fail while the plan is built.

    Every shipped bulk weight has to land somewhere local. If it doesn't, the
    receive plan is silently short a destination and the M-to-N collective would
    post a mismatched set of transfers at the first refit.
    """
    from nemo_rl.models.generation.megatron.megatron_worker import (
        MegatronGenerationRefitMixin,
    )

    worker = object.__new__(MegatronGenerationRefitMixin)
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {"model.layers.0": [param_info]},
    }

    with pytest.raises(ValueError, match=expected_message):
        worker._build_destination_hf_to_local_param_map(refit_info, [])

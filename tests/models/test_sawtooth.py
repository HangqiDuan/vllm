# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cheap model tests catch wrong weight keys and physical-rank stage exits."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.distributed.sawtooth import SawtoothConfig, StagePlan
from vllm.model_executor.models.sawtooth import (
    SawtoothModelMixin,
    forward_sawtooth_decoder,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
)


def test_disjoint_ownership_keeps_global_checkpoint_keys(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=2),
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.get_offloader",
        lambda: SimpleNamespace(wrap_modules=list),
    )
    plan = StagePlan.build(SawtoothConfig(True, 2), 8, 2)
    names = []

    def create_layer(prefix):
        names.append(prefix)
        return nn.Linear(1, 1, bias=False)

    _, _, layers = make_layers(8, create_layer, "model.layers", sawtooth_plan=plan)
    model = nn.Module()
    model.layers = layers
    assert names == [
        "model.layers.0",
        "model.layers.1",
        "model.layers.4",
        "model.layers.5",
    ]
    assert list(dict(model.named_parameters())) == [
        "layers.0.weight",
        "layers.1.weight",
        "layers.4.weight",
        "layers.5.weight",
    ]
    assert isinstance(layers[2], PPMissingLayer)
    assert is_pp_missing_parameter("layers.2.weight", model)
    assert not is_pp_missing_parameter("layers.4.weight", model)


def test_wraparound_does_not_repeat_embedding_or_apply_norm_early():
    plan = StagePlan.build(SawtoothConfig(True, 2), 8, 2)
    calls = []

    class Layer:
        def __init__(self, index):
            self.index = index

        def __call__(self, positions, hidden, residual):
            calls.append(self.index)
            return hidden + self.index + 1, hidden

    def embed(ids):
        calls.append("embed")
        return ids.float().unsqueeze(-1)

    def norm(hidden, residual):
        calls.append("norm")
        return hidden * 2, residual

    model = SimpleNamespace(
        layers=[Layer(i) for i in range(8)], embed_input_ids=embed, norm=norm
    )
    carry = None
    for stage in plan.stages:
        carry = forward_sawtooth_decoder(
            model, stage, torch.tensor([2]), torch.tensor([0]), carry, None
        )
    assert calls == ["embed", *range(8), "norm"]
    torch.testing.assert_close(carry, torch.tensor([[76.0]]))


def test_later_group_moe_uses_global_names_and_rank_local_ordinals():
    model = SawtoothModelMixin()
    model.sawtooth_plan = StagePlan.build(SawtoothConfig(True, 2), 8, 2)
    names = [f"model.layers.{i}.mlp.experts" for i in (0, 1, 4, 5)]
    model._sawtooth_vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_all_moe_layers=names)
    )
    assert model.get_sawtooth_moe_mapping(2) == (
        (4, 2, names[2]),
        (5, 3, names[3]),
    )
    assert model.get_sawtooth_moe_layers(2) == tuple(names[2:])

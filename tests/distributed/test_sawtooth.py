# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan tests guard layer order and ownership before distributed execution."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from vllm.distributed.sawtooth import (
    SawtoothConfig,
    StagePlan,
    get_sawtooth_plan,
)


@pytest.mark.parametrize("pipeline_size", [2, 3, 4])
@pytest.mark.parametrize("virtual_size", [2, 3, 4])
def test_interleaving_preserves_global_layer_order(pipeline_size, virtual_size):
    layers = pipeline_size * virtual_size * 3 + 1
    plan = StagePlan.build(SawtoothConfig(True, virtual_size), layers, pipeline_size)
    assert [i for stage in plan.stages for i in stage.layer_ids] == list(range(layers))
    for rank in range(pipeline_size):
        stages = plan.local_stages(rank)
        assert [s.group_id for s in stages] == list(
            range(rank, plan.num_groups, pipeline_size)
        )
        assert all(
            plan.stage_for_round(rank, r) == stage for r, stage in enumerate(stages)
        )
    with pytest.raises(FrozenInstanceError):
        plan.stages[0].start_layer = 1


@pytest.mark.parametrize(
    "boundaries",
    [(0, 4, 2, 6, 8), (0, 2, 2, 6, 8), (1, 2, 4, 6, 8), (0, 2, 4, 6, 7)],
)
def test_reject_incomplete_or_reordered_layer_coverage(boundaries):
    with pytest.raises(ValueError, match="layer_boundaries"):
        StagePlan.build(SawtoothConfig(True, 2, boundaries), 8, 2)


def test_plan_identity_changes_with_boundaries_not_live_batch_window():
    a = StagePlan.build(SawtoothConfig(True, 2, max_live_batches=2), 10, 2)
    b = StagePlan.build(SawtoothConfig(True, 2, max_live_batches=4), 10, 2)
    c = StagePlan.build(SawtoothConfig(True, 2, (0, 1, 3, 7, 10)), 10, 2)
    assert a.plan_id == b.plan_id
    assert a.plan_id != c.plan_id


@pytest.mark.parametrize("enabled", [True, False])
def test_early_admission_is_policy_not_stage_identity(enabled):
    config = SimpleNamespace(
        additional_config={
            "sawtooth_pipeline": {
                "enabled": True,
                "virtual_size": 2,
                "enable_early_admission": enabled,
            }
        }
    )
    policy = SawtoothConfig.from_vllm_config(config)
    assert policy.enable_early_admission is enabled
    a = StagePlan.build(policy, 8, 2)
    b = StagePlan.build(SawtoothConfig(True, 2), 8, 2)
    assert a.plan_id == b.plan_id
    assert a.stages == b.stages


def test_early_admission_defaults_on_and_unknown_options_fail():
    assert SawtoothConfig().enable_early_admission is True
    config = SimpleNamespace(additional_config={"sawtooth_pipeline": {}})
    assert SawtoothConfig.from_vllm_config(config).enable_early_admission is True
    config.additional_config["sawtooth_pipeline"]["enable_early_admissions"] = False
    with pytest.raises(ValueError, match="Unknown sawtooth_pipeline"):
        SawtoothConfig.from_vllm_config(config)


@pytest.mark.parametrize("value", [0, 1, None, "true", "false", [], {}])
def test_early_admission_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="enable_early_admission must be a boolean"):
        SawtoothConfig(enable_early_admission=value)
    config = SimpleNamespace(
        additional_config={
            "sawtooth_pipeline": {
                "enable_early_admission": value,
            }
        }
    )
    with pytest.raises(ValueError, match="enable_early_admission must be a boolean"):
        SawtoothConfig.from_vllm_config(config)


def test_one_virtual_stage_matches_ordinary_uneven_pp_partition():
    plan = StagePlan.build(SawtoothConfig(True, 1), 10, 4)
    assert [(s.start_layer, s.end_layer) for s in plan.stages] == [
        (0, 2),
        (2, 5),
        (5, 8),
        (8, 10),
    ]


def test_disabled_keeps_ordinary_pp_and_enabled_rejects_speculation():
    config = SimpleNamespace(additional_config={})
    assert get_sawtooth_plan(config, 8) is None
    config.additional_config = {
        "sawtooth_pipeline": {"enabled": True, "virtual_size": 2}
    }
    config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    config.speculative_config = object()
    with pytest.raises(ValueError, match="speculative"):
        get_sawtooth_plan(config, 8)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from vllm.v1.core import kv_cache_utils as utils


@pytest.mark.parametrize("override", [None, 10])
def test_aligned_pool_passes_config_to_platform_helper(monkeypatch, override):
    config = NS(
        num_prefill_lookahead_tokens=0,
        cache_config=NS(num_gpu_blocks_override=override),
        model_config=NS(original_max_model_len=16, max_model_len=16),
        parallel_config=NS(tensor_parallel_size=1, decode_context_parallel_size=1),
    )
    groups = [NS(layer_names=["layer"])]
    monkeypatch.setattr(
        utils.KVCacheSpecRegistry, "check_kv_cache_spec_registry", lambda specs: None
    )
    monkeypatch.setattr(utils, "get_kv_cache_groups", lambda cfg, specs: groups)
    monkeypatch.setattr(
        utils, "_project_kv_cache_groups_to_worker", lambda all_groups, specs: groups
    )
    calls = []

    def pool_bytes(cfg, worker_groups):
        assert cfg is config
        assert worker_groups is groups
        calls.append(cfg)
        return 10

    monkeypatch.setattr(utils, "_pool_bytes_per_block", pool_bytes)
    check = Mock()
    monkeypatch.setattr(utils, "_check_enough_kv_cache_memory", check)
    allocated = []

    def allocate(cfg, worker_groups, memory):
        allocated.append(memory)
        return NS(num_blocks=memory // 10, kv_cache_groups=worker_groups)

    monkeypatch.setattr(utils, "get_kv_cache_config_from_groups", allocate)
    monkeypatch.setattr(utils, "kv_cache_groups_tp_replicas", lambda *args: 1)
    configs = utils.get_kv_cache_configs(config, [{"layer": object()}], [103])
    assert calls
    assert allocated == [80]
    assert check.call_args.args[0] == 70  # Null block is reserved for admission.
    assert configs[0].num_blocks == 8

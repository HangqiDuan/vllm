# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable layer ownership and execution contract for interleaved PP.

This module deliberately imports neither torch nor distributed process state so
the scheduler and workers can independently construct and compare the same plan.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from itertools import accumulate
from typing import Any, Protocol


@dataclass(frozen=True)
class SawtoothConfig:
    enabled: bool = False
    virtual_size: int = 1
    layer_boundaries: tuple[int, ...] | None = None
    max_live_batches: int | None = None
    enable_early_admission: bool = True
    enable_prefill_chunk_overlap: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if type(self.enable_early_admission) is not bool:
            raise ValueError("enable_early_admission must be a boolean")
        if type(self.enable_prefill_chunk_overlap) is not bool:
            raise ValueError("enable_prefill_chunk_overlap must be a boolean")
        if type(self.virtual_size) is not int or self.virtual_size < 1:
            raise ValueError("virtual_size must be positive")
        if self.max_live_batches is not None and (
            type(self.max_live_batches) is not int or self.max_live_batches < 1
        ):
            raise ValueError("max_live_batches must be positive")
        if self.layer_boundaries is not None:
            if not isinstance(self.layer_boundaries, (tuple, list)) or any(
                type(index) is not int for index in self.layer_boundaries
            ):
                raise ValueError("layer_boundaries must be an integer list")
            object.__setattr__(self, "layer_boundaries", tuple(self.layer_boundaries))

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "SawtoothConfig":
        additional = vllm_config.additional_config
        if not isinstance(additional, dict):
            return cls()
        raw = additional.get("sawtooth_pipeline", {})
        if not isinstance(raw, dict):
            raise ValueError("sawtooth_pipeline must be an object")
        unknown = set(raw) - {
            "enabled",
            "virtual_size",
            "layer_boundaries",
            "max_live_batches",
            "enable_early_admission",
            "enable_prefill_chunk_overlap",
        }
        if unknown:
            raise ValueError(f"Unknown sawtooth_pipeline options: {sorted(unknown)}")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("sawtooth_pipeline.enabled must be a boolean")
        virtual_size = raw.get("virtual_size", 1)
        if type(virtual_size) is not int or virtual_size < 1:
            raise ValueError("sawtooth_pipeline.virtual_size must be positive")
        window = raw.get("max_live_batches")
        if window is not None and (type(window) is not int or window < 1):
            raise ValueError("sawtooth_pipeline.max_live_batches must be positive")
        boundaries = raw.get("layer_boundaries")
        if boundaries is not None:
            if not isinstance(boundaries, (list, tuple)) or any(
                type(index) is not int for index in boundaries
            ):
                raise ValueError("layer_boundaries must be an integer list")
            boundaries = tuple(boundaries)
        return cls(
            enabled,
            virtual_size,
            boundaries,
            window,
            raw.get("enable_early_admission", True),
            raw.get("enable_prefill_chunk_overlap", False),
        )


@dataclass(frozen=True)
class StageSpec:
    group_id: int
    owner_rank: int
    round_id: int
    start_layer: int
    end_layer: int
    is_first: bool
    is_last: bool

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.start_layer, self.end_layer))


@dataclass(frozen=True)
class StagePlan:
    config: SawtoothConfig
    num_layers: int
    pipeline_size: int
    stages: tuple[StageSpec, ...]
    plan_id: str

    @classmethod
    def build(
        cls, config: SawtoothConfig, num_layers: int, pipeline_size: int
    ) -> "StagePlan":
        if type(pipeline_size) is not int or pipeline_size < 1:
            raise ValueError("pipeline_size must be positive")
        groups = pipeline_size * config.virtual_size
        if type(num_layers) is not int or num_layers < groups:
            raise ValueError("Every sawtooth stage must own at least one layer")
        boundaries = config.layer_boundaries
        if boundaries is None:
            # Match ordinary PP for V=1, keeping the final norm/head group
            # out of the remainder distribution.
            partitions = [num_layers // groups] * groups
            for offset in range(2, num_layers % groups + 2):
                partitions[-offset] += 1
            boundaries = tuple(accumulate(partitions, initial=0))
        if (
            len(boundaries) != groups + 1
            or boundaries[0] != 0
            or boundaries[-1] != num_layers
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError(
                "layer_boundaries must contain G+1 strictly increasing layer "
                "offsets covering [0, num_layers] in execution order"
            )
        stages = tuple(
            StageSpec(
                g,
                g % pipeline_size,
                g // pipeline_size,
                boundaries[g],
                boundaries[g + 1],
                g == 0,
                g == groups - 1,
            )
            for g in range(groups)
        )
        identity = json.dumps(
            [1, pipeline_size, config.virtual_size, boundaries], separators=(",", ":")
        )
        return cls(
            config,
            num_layers,
            pipeline_size,
            stages,
            sha256(identity.encode()).hexdigest()[:16],
        )

    @property
    def virtual_size(self) -> int:
        return self.config.virtual_size

    @property
    def num_groups(self) -> int:
        return len(self.stages)

    @property
    def plan_version(self) -> str:
        return self.plan_id

    def local_stages(self, pp_rank: int) -> tuple[StageSpec, ...]:
        if not 0 <= pp_rank < self.pipeline_size:
            raise ValueError(f"Invalid PP rank {pp_rank}")
        return tuple(stage for stage in self.stages if stage.owner_rank == pp_rank)

    def local_layer_ids(self, pp_rank: int) -> tuple[int, ...]:
        return tuple(i for stage in self.local_stages(pp_rank) for i in stage.layer_ids)

    def stage_for_round(self, pp_rank: int, round_id: int) -> StageSpec:
        if not 0 <= round_id < self.virtual_size:
            raise ValueError(f"Invalid sawtooth round {round_id}")
        return self.local_stages(pp_rank)[round_id]


def is_sawtooth_enabled(vllm_config: Any) -> bool:
    return SawtoothConfig.from_vllm_config(vllm_config).enabled


def validate_sawtooth_capabilities(
    vllm_config: Any, platform_config: Any = None
) -> None:
    """Reject unimplemented combinations without changing feature-off defaults.

    This validates engine configuration, not request-scoped features such as
    structured output. A default StructuredOutputsConfig does not enable one.
    Platform config is optional so the upstream planner stays platform-neutral.
    """
    if not is_sawtooth_enabled(vllm_config):
        return

    unsupported = []
    if not getattr(vllm_config, "use_v2_model_runner", True):
        unsupported.append("Model Runner V1 (set VLLM_USE_V2_MODEL_RUNNER=1)")
    if getattr(vllm_config, "speculative_config", None) is not None:
        unsupported.append("speculative decoding")
    parallel = vllm_config.parallel_config
    for field, label in (
        ("data_parallel_size", "data parallelism"),
        ("prefill_context_parallel_size", "PCP"),
        ("decode_context_parallel_size", "DCP"),
    ):
        if getattr(parallel, field, 1) > 1:
            unsupported.append(label)
    if getattr(parallel, "enable_dbo", False):
        unsupported.append("DBO")
    for field, label in (
        ("lora_config", "LoRA"),
        ("kv_transfer_config", "KV connectors"),
        ("ec_transfer_config", "encoder connectors"),
    ):
        if getattr(vllm_config, field, None) is not None:
            unsupported.append(label)
    cache = getattr(vllm_config, "cache_config", None)
    if getattr(cache, "kv_offloading_size", None) is not None:
        unsupported.append("KV offloading")
    offload = getattr(vllm_config, "offload_config", None)
    if (
        getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0) > 0
        or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0) > 0
    ):
        unsupported.append("weight offloading")

    model = getattr(vllm_config, "model_config", None)
    if model is not None:
        if getattr(model, "runner_type", "generate") != "generate":
            unsupported.append("pooling/non-generation runners")
        architectures = getattr(getattr(model, "hf_config", None), "architectures", ())
        supported = {
            "Qwen2ForCausalLM",
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "DeepseekV4ForCausalLM",
        }
        if not architectures or not supported.intersection(architectures):
            unsupported.append(f"model adapter for {architectures!r}")
        if getattr(model, "model_impl", "auto") not in ("auto", "vllm"):
            unsupported.append("non-vLLM model implementations")

    if platform_config is not None:
        if getattr(getattr(platform_config, "kvpp_config", None), "size", 1) > 1:
            unsupported.append("KVPP")
        if getattr(
            getattr(platform_config, "sparse_kv_offload_config", None), "enabled", False
        ):
            unsupported.append("sparse KV offloading")
        if getattr(platform_config, "enable_dsa_cp", False):
            unsupported.append("DSA context parallelism")
    if unsupported:
        raise ValueError("Sawtooth does not support " + ", ".join(unsupported))


def get_sawtooth_plan(
    vllm_config: Any, num_layers: int, pp_rank: int | None = None
) -> StagePlan | None:
    config = SawtoothConfig.from_vllm_config(vllm_config)
    if not config.enabled:
        return None
    pipeline_size = vllm_config.parallel_config.pipeline_parallel_size
    if pipeline_size <= 1 or config.virtual_size < 2:
        raise ValueError("Sawtooth requires PP > 1 and virtual_size >= 2")
    validate_sawtooth_capabilities(vllm_config)
    plan = StagePlan.build(config, num_layers, pipeline_size)
    if pp_rank is not None:
        plan.local_stages(pp_rank)
    return plan


class SupportsSawtoothPipeline(Protocol):
    """Model adapter contract; every nonterminal stage returns named carry.

    A stage callable's layer set and entry/exit behavior never change. Schema
    shapes come from CPU scheduling metadata; dtype is the backend dtype object.
    The caller saves every returned tensor into a transport lease before replay
    may reuse stage outputs. Only the final stage returns final hidden states.
    """

    sawtooth_plan: StagePlan | None

    def get_sawtooth_stage(self, group_id: int) -> Callable[..., Any]: ...

    def get_sawtooth_carry_schema(
        self, group_id: int, num_tokens: int
    ) -> dict[str, tuple[tuple[int, ...], Any]]: ...

    def get_sawtooth_moe_mapping(
        self, group_id: int
    ) -> tuple[tuple[int, int, str], ...]: ...

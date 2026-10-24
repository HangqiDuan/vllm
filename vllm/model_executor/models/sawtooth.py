# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed stage callables; original models retain checkpoint parameter names."""

import re
from copy import copy

import torch
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.distributed.sawtooth import StageSpec
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.sequence import IntermediateTensors


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class _CompiledSawtoothStage(nn.Module):
    """One immutable group with its own compiler and graph identity."""

    def __init__(self, *, vllm_config: VllmConfig, owner: nn.Module, stage: StageSpec):
        super().__init__()
        self.decoder = owner.model
        self.stage = stage
        self.config = owner.config
        self.vllm_config = vllm_config

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.decoder.forward_sawtooth_stage(
            self.stage, input_ids, positions, intermediate_tensors, inputs_embeds
        )


class SawtoothStageModule(nn.Module):
    """Set group-local MoE call order outside the compiled stage boundary."""

    def __init__(self, *, vllm_config: VllmConfig, owner: nn.Module, stage: StageSpec):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.stage = stage
        self.config = owner.config
        self.moe_layer_names = list(owner.get_sawtooth_moe_layers(stage.group_id))
        self.moe_mapping = owner.get_sawtooth_moe_mapping(stage.group_id)
        # The ordinary decorator uses one model tag and AOT config hash for
        # every decoder. Both must identify this fixed stage, even at equal T.
        self.vllm_config = copy(vllm_config)
        self.vllm_config.compilation_config = copy(vllm_config.compilation_config)
        self.vllm_config.compilation_config.local_cache_dir = ""
        identity = f"sawtooth_{owner.sawtooth_plan.plan_id}_g{stage.group_id}"
        self.vllm_config.additional_config = {
            **vllm_config.additional_config,
            "sawtooth_stage_identity": identity,
        }
        with set_current_vllm_config(self.vllm_config), set_model_tag(identity):
            self.compiled_stage = _CompiledSawtoothStage(
                vllm_config=self.vllm_config, owner=owner, stage=stage
            )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_owner"), name)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        # Preserve the platform's call-scoped eager warmup (e.g. Ascend MC2)
        # without disabling normal stage compilation or graph capture.
        call_stage = (
            self.compiled_stage.forward
            if getattr(self._owner.model, "do_not_compile", False)
            else self.compiled_stage
        )
        with set_current_vllm_config(self.vllm_config):
            if not is_forward_context_available():
                return call_stage(
                    input_ids, positions, intermediate_tensors, inputs_embeds
                )
            context = get_forward_context()
            previous = context.all_moe_layers, context.moe_layer_index
            context.all_moe_layers = self.moe_layer_names
            context.moe_layer_index = 0
            try:
                return call_stage(
                    input_ids, positions, intermediate_tensors, inputs_embeds
                )
            finally:
                context.all_moe_layers, context.moe_layer_index = previous

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        if self.stage.is_first:
            return IntermediateTensors({})
        schema = self._owner.get_sawtooth_carry_schema(
            self.stage.group_id - 1, batch_size
        )
        return IntermediateTensors(
            {
                name: torch.zeros(shape, dtype=tensor_dtype, device=device)
                for name, (shape, tensor_dtype) in schema.items()
            }
        )


class SawtoothModelMixin:
    def init_sawtooth_stages(self, vllm_config: VllmConfig) -> None:
        self.sawtooth_plan = self.model.sawtooth_plan
        self._sawtooth_vllm_config = vllm_config
        # A plain dict deliberately avoids registering weights a second time.
        self._sawtooth_stages: dict[int, SawtoothStageModule] = {}

    def get_sawtooth_stage(self, group_id: int) -> SawtoothStageModule:
        if self.sawtooth_plan is None:
            raise ValueError("Sawtooth is not enabled")
        if not 0 <= group_id < self.sawtooth_plan.num_groups:
            raise ValueError(f"Invalid sawtooth group {group_id}")
        stage = self.sawtooth_plan.stages[group_id]
        if stage.owner_rank != get_pp_group().rank_in_group:
            raise ValueError(f"Sawtooth group {group_id} belongs to another rank")
        if group_id not in self._sawtooth_stages:
            self._sawtooth_stages[group_id] = SawtoothStageModule(
                vllm_config=self._sawtooth_vllm_config, owner=self, stage=stage
            )
        return self._sawtooth_stages[group_id]

    def get_sawtooth_moe_layers(self, group_id: int) -> tuple[str, ...]:
        return tuple(name for _, _, name in self.get_sawtooth_moe_mapping(group_id))

    def get_sawtooth_moe_mapping(
        self, group_id: int
    ) -> tuple[tuple[int, int, str], ...]:
        """Return (global layer, rank-local MoE ordinal, registered prefix)."""
        stage = self.sawtooth_plan.stages[group_id]
        names = self._sawtooth_vllm_config.compilation_config.static_all_moe_layers
        return tuple(
            (int(match[1]), ordinal, name)
            for ordinal, name in enumerate(names)
            if (match := re.search(r"(?:^|\.)layers\.(\d+)\.", name))
            and stage.start_layer <= int(match[1]) < stage.end_layer
        )

    def get_sawtooth_carry_schema(
        self, group_id: int, num_tokens: int
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        return self.model.get_sawtooth_carry_schema(group_id, num_tokens)


def forward_sawtooth_decoder(
    model, stage, input_ids, positions, intermediate_tensors, inputs_embeds
):
    """Dense/Qwen MoE residual-stream execution in global layer order."""
    if stage.is_first:
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else model.embed_input_ids(input_ids)
        )
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]
    for index in range(stage.start_layer, stage.end_layer):
        hidden, residual = model.layers[index](positions, hidden, residual)
    if stage.is_last:
        hidden, _ = model.norm(hidden, residual)
        return hidden
    return IntermediateTensors({"hidden_states": hidden, "residual": residual})


def decoder_carry_schema(model, group_id, num_tokens):
    stage = model.sawtooth_plan.stages[group_id]
    if stage.is_last:
        return {}
    shape = (num_tokens, model.config.hidden_size)
    return {key: (shape, model.sawtooth_dtype) for key in ("hidden_states", "residual")}

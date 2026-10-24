# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine.input_processor import InputProcessor


def test_unvalidated_grammar_request_fails_before_engine_admission():
    processor = InputProcessor.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(
        additional_config={"sawtooth_pipeline": {"enabled": True, "virtual_size": 2}}
    )
    params = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    with pytest.raises(VLLMValidationError, match="Structured output requests"):
        processor._validate_params(params, ("generate",))


@pytest.mark.parametrize("enabled", [False, True])
def test_default_sampling_keeps_the_existing_task_validation(enabled):
    processor = InputProcessor.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(
        additional_config={"sawtooth_pipeline": {"enabled": enabled, "virtual_size": 2}}
    )
    with pytest.raises(VLLMValidationError, match="does not support generation"):
        processor._validate_params(SamplingParams(), ())

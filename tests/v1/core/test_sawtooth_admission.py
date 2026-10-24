# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request cohorts fill logical slots without shrinking prefill token chunks.

These use the real scheduler and KV allocator. The cap is an admission policy,
so no model, worker, or simulated pipeline is needed to catch its failure modes.
"""

import json
from types import SimpleNamespace

import pytest

from vllm.lora.request import LoRARequest
from vllm.v1.request import RequestStatus

from .test_sawtooth_scheduler import result
from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def admission_scheduler(tmp_path):
    config = {
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "vocab_size": 50304,
        "hidden_size": 64,
        "word_embed_proj_dim": 64,
        "ffn_dim": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "max_position_embeddings": 32768,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def build(*, async_scheduling=True, enabled=True, **kwargs):
        sched = create_scheduler(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_num_batched_tokens=10240,
            max_model_len=16384,
            pipeline_parallel_size=2,
            use_v2_model_runner=True,
            async_scheduling=async_scheduling,
            **kwargs,
        )
        sched.sawtooth_enabled = enabled
        sched.sawtooth_available_batches = 1
        return sched

    return build


def _admit_wave(sched, slots):
    outputs = []
    for free_slots in range(slots, 0, -1):
        sched.sawtooth_available_batches = free_slots
        outputs.append(sched.schedule())
    return outputs


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("waiting_count", [0, 64])
def test_decode_cohorts_fill_slots_even_with_a_full_running_set(
    admission_scheduler, async_scheduling, waiting_count
):
    sched = admission_scheduler(max_num_seqs=64, async_scheduling=async_scheduling)
    requests = create_requests(64, num_tokens=8)
    for request in requests:
        sched.add_request(request)
    prefill = sched.schedule()
    sched.update_from_output(prefill, result(prefill))
    for request in create_requests(
        waiting_count,
        num_tokens=8,
        req_ids=[f"waiting-{i}" for i in range(waiting_count)],
    ):
        sched.add_request(request)

    outputs = _admit_wave(sched, 4)
    cohorts = [set(output.num_scheduled_tokens) for output in outputs]
    assert [len(cohort) for cohort in cohorts] == [16, 16, 16, 16]
    assert len(set.union(*cohorts)) == 64
    assert all(set(output.num_scheduled_tokens.values()) == {1} for output in outputs)
    assert len(sched.waiting) == waiting_count


def test_prefill_keeps_10240_token_chunk_while_short_requests_fill_other_slots(
    admission_scheduler,
):
    sched = admission_scheduler(max_num_seqs=4)
    lengths = [12288, 3, 5, 7]
    for index, length in enumerate(lengths):
        (request,) = create_requests(1, num_tokens=length, req_ids=[str(index)])
        sched.add_request(request)
    outputs = _admit_wave(sched, 4)
    assert [output.total_num_scheduled_tokens for output in outputs] == [
        10240,
        3,
        5,
        7,
    ]
    assert all(len(output.num_scheduled_tokens) == 1 for output in outputs)
    assert sched.requests["0"].is_prefill_chunk


def test_running_and_waiting_share_one_request_cap(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8)
    for request in create_requests(2, num_tokens=8):
        sched.add_request(request)
    prefill = sched.schedule()
    sched.update_from_output(prefill, result(prefill))
    for request in create_requests(
        6, num_tokens=16, req_ids=[f"new-{i}" for i in range(6)]
    ):
        sched.add_request(request)
    first, second = _admit_wave(sched, 2)
    assert len(first.num_scheduled_tokens) == len(second.num_scheduled_tokens) == 4
    assert first.num_scheduled_tokens["0"] == first.num_scheduled_tokens["1"] == 1
    assert len(first.scheduled_new_reqs) == 2
    assert set(first.num_scheduled_tokens).isdisjoint(second.num_scheduled_tokens)


def test_blocked_waiters_do_not_inflate_available_work(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8)
    requests = create_requests(8, num_tokens=8)
    for index, request in enumerate(requests):
        if index >= 4:
            request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        sched.add_request(request)
    outputs = _admit_wave(sched, 4)
    assert [len(output.num_scheduled_tokens) for output in outputs] == [1] * 4
    assert set.union(*(set(o.num_scheduled_tokens) for o in outputs)) == {
        request.request_id for request in requests[:4]
    }


def test_ready_skipped_waiters_count_toward_cohort_size(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8)
    for request in create_requests(8, num_tokens=8):
        sched.add_request(request)
    for _ in range(4):
        sched.skipped_waiting.add_request(sched.waiting.pop_request())
    outputs = _admit_wave(sched, 4)
    assert [len(output.num_scheduled_tokens) for output in outputs] == [2] * 4


def test_lora_skip_does_not_consume_successful_admission_capacity(
    admission_scheduler,
):
    sched = admission_scheduler(max_num_seqs=8)
    sched.lora_config = SimpleNamespace(max_loras=1)
    requests = create_requests(5, num_tokens=8)
    for index, request in enumerate(requests):
        adapter = 2 if index == 1 else 1
        request.lora_request = LoRARequest(str(adapter), adapter, "/unused")
        sched.add_request(request)
    sched.sawtooth_available_batches = 3
    output = sched.schedule()
    assert set(output.num_scheduled_tokens) == {"0", "2"}
    assert requests[1] in sched.skipped_waiting


def test_small_cohorts_do_not_preempt_another_live_batch(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=4, num_blocks=5)
    requests = create_requests(4, num_tokens=32, max_tokens=1)
    for request in requests:
        sched.add_request(request)
    sched.sawtooth_available_batches = 4
    first = sched.schedule()
    sched.sawtooth_available_batches = 3
    second = sched.schedule()
    sched.sawtooth_available_batches = 2
    blocked = sched.schedule()
    assert len(first.num_scheduled_tokens) == len(second.num_scheduled_tokens) == 1
    assert not blocked.num_scheduled_tokens
    assert all(request.num_preemptions == 0 for request in requests)
    assert requests[0].num_in_flight_tokens == requests[1].num_in_flight_tokens == 32
    sched.update_from_output(first, result(first))
    resumed = sched.schedule()
    assert len(resumed.num_scheduled_tokens) == 1
    assert requests[1].num_in_flight_tokens == 32


def test_feature_off_preserves_original_admission(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8, enabled=False)
    for request in create_requests(8, num_tokens=8):
        sched.add_request(request)
    sched.sawtooth_available_batches = 4
    assert len(sched.schedule().num_scheduled_tokens) == 8

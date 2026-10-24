# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical-step KV ownership; no model construction or device execution."""

import json
import sys
from contextlib import suppress
from dataclasses import replace
from functools import partial

import pytest
import torch

from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def local_model_config(tmp_path, monkeypatch):
    """The shared helper needs architecture metadata, never HF network access."""
    config = {
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "vocab_size": 50304,
        "hidden_size": 64,
        "word_embed_proj_dim": 64,
        "ffn_dim": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "max_position_embeddings": 8192,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        sys.modules[__name__],
        "create_scheduler",
        partial(create_scheduler, model=str(tmp_path), skip_tokenizer_init=True),
    )


def scheduler(**kwargs):
    instance = create_scheduler(
        async_scheduling=kwargs.pop("async_scheduling", True),
        pipeline_parallel_size=2,
        use_v2_model_runner=True,
        enable_prefix_caching=kwargs.pop("enable_prefix_caching", True),
        **kwargs,
    )
    # Exercise the scheduler in isolation from platform/model capability
    # validation. The existing fixture builds config, never model weights.
    instance.sawtooth_enabled = True
    return instance


def result(output, sample=True):
    ids = list(output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={request_id: i for i, request_id in enumerate(ids)},
        sampled_token_ids=[[7] if sample else [] for _ in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_same_request_cannot_advance_while_any_virtual_round_is_live():
    sched = scheduler()
    (request,) = create_requests(1, num_tokens=32, max_tokens=4)
    sched.add_request(request)
    first = sched.schedule()
    assert first.num_scheduled_tokens == {request.request_id: 32}
    assert request.num_in_flight_tokens == 32
    assert request.num_output_placeholders == 1
    assert first.sawtooth_token_ends == {request.request_id: 32}

    for _ in range(3):
        blocked = sched.schedule()
        assert not blocked.num_scheduled_tokens
        assert request.num_in_flight_tokens == 32
        assert request.num_output_placeholders == 1
    sched.update_from_output(first, result(first))
    assert request.num_in_flight_tokens == 0
    assert request.num_output_placeholders == 0
    second = sched.schedule()
    assert second.num_scheduled_tokens == {request.request_id: 1}


def test_prefix_is_private_until_terminal_then_published_once(monkeypatch):
    sched = scheduler()
    first, probe = create_requests(2, num_tokens=32, same_prompt=True)
    sched.add_request(first)
    output = sched.schedule()
    assert sched.kv_cache_manager.get_computed_blocks(probe)[1] == 0
    calls = []
    original = sched.kv_cache_manager.cache_blocks

    def commit(request, count):
        blocks = sched.kv_cache_manager.get_blocks(request.request_id).blocks
        assert all(block.ref_cnt > 0 for group in blocks for block in group)
        calls.append((request.request_id, count))
        return original(request, count)

    monkeypatch.setattr(sched.kv_cache_manager, "cache_blocks", commit)
    sched.update_from_output(output, result(output))
    assert calls == [(first.request_id, 32)]
    assert sched.kv_cache_manager.get_computed_blocks(probe)[1] > 0


def test_partial_prefill_commits_only_its_completed_input_boundary():
    sched = scheduler(max_num_batched_tokens=16, max_model_len=128)
    first, probe = create_requests(2, num_tokens=40, same_prompt=True)
    sched.add_request(first)
    output = sched.schedule()
    assert output.total_num_scheduled_tokens == 16
    assert first.is_prefill_chunk
    assert first.num_output_placeholders == 0
    assert sched.kv_cache_manager.get_computed_blocks(probe)[1] == 0
    assert not sched.schedule().num_scheduled_tokens

    sched.update_from_output(output, result(output, sample=False))
    assert first.num_output_tokens == 0
    assert first.num_in_flight_tokens == 0
    assert sched.kv_cache_manager.get_computed_blocks(probe)[1] == 16
    continuation = sched.schedule()
    assert continuation.total_num_scheduled_tokens == 16
    assert continuation.sawtooth_token_ends[first.request_id] == 32


def test_abort_keeps_kv_and_request_identity_until_terminal():
    sched = scheduler()
    (request,) = create_requests(1, num_tokens=32)
    sched.add_request(request)
    output = sched.schedule()
    blocks = [
        block
        for group in sched.kv_cache_manager.get_blocks(request.request_id).blocks
        for block in group
    ]
    sched.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert sched.requests[request.request_id] is request
    assert all(block.ref_cnt > 0 for block in blocks)
    assert request.request_id not in sched.finished_req_ids
    assert not sched.running
    assert not sched.schedule().num_scheduled_tokens

    sched.update_from_output(output, result(output))
    assert request.num_output_tokens == 0
    assert request.request_id not in sched.requests
    assert request.request_id in sched.finished_req_ids
    assert all(block.ref_cnt == 0 for block in blocks)


def test_memory_pressure_never_preempts_a_live_batch():
    sched = scheduler(num_blocks=5)
    first, second, waiting = create_requests(3, num_tokens=32, max_tokens=1)
    sched.add_request(first)
    first_output = sched.schedule()
    sched.add_request(second)
    second_output = sched.schedule()
    assert second_output.num_scheduled_tokens
    sched.add_request(waiting)
    blocked = sched.schedule()
    assert not blocked.num_scheduled_tokens
    assert first.num_preemptions == second.num_preemptions == 0
    assert first.num_in_flight_tokens == second.num_in_flight_tokens == 32

    sched.update_from_output(first_output, result(first_output))
    admitted = sched.schedule()
    assert waiting.request_id in admitted.num_scheduled_tokens
    assert second.num_in_flight_tokens == 32


def test_prefix_reset_refuses_live_leases_without_removing_running_requests():
    sched = scheduler()
    (request,) = create_requests(1, num_tokens=32)
    sched.add_request(request)
    output = sched.schedule()
    before = list(sched.running)
    # Either explicit rejection policy is acceptable; mutating a live running
    # queue before rejecting the reset is not.
    with suppress(RuntimeError):
        assert sched.reset_prefix_cache(reset_running_requests=True) is False
    assert sched.running == before
    assert sched.requests[request.request_id] is request
    sched.update_from_output(output, result(output))
    assert sched.schedule().num_scheduled_tokens


def enable_successor(sched, output):
    sched.sawtooth_mark_tail_enqueued(output)
    sched.sawtooth_allow_early_admission = True


@pytest.fixture
def refill_boundary():
    """Build real async state through committed outputs, never forged counters.

    This fixture describes the pre-split behavior. Future dispatch-kind tests
    can reuse it without changing request/KV/placeholder construction.
    """

    def build(num_running=31, committed_outputs=62, max_tokens=64):
        sched = scheduler(
            max_num_seqs=64,
            max_num_batched_tokens=10240,
            max_model_len=8192,
            enable_prefix_caching=False,
        )
        sched.sawtooth_available_batches = 1
        requests = create_requests(
            num_running, num_tokens=8, max_tokens=max_tokens, ignore_eos=True
        )
        for request in requests:
            sched.add_request(request)
        for _ in range(committed_outputs):
            committed = sched.schedule()
            assert len(committed.num_scheduled_tokens) == num_running
            # Match Engine: final round and sample RPC are enqueued before
            # terminal output commits, even when early admission is disabled.
            sched.sawtooth_mark_tail_enqueued(committed)
            sched.update_from_output(committed, result(committed))
        predecessor = sched.schedule()
        enable_successor(sched, predecessor)
        (fresh,) = create_requests(
            1, num_tokens=128, req_ids=["fresh"], ignore_eos=True
        )
        sched.add_request(fresh)
        return sched, requests, predecessor, fresh

    return build


def test_refill_boundary_current_auto_mixes_final_decode_with_new_prefill(
    refill_boundary,
):
    sched, requests, predecessor, fresh = refill_boundary()
    assert all(request.num_output_tokens == 62 for request in requests)
    assert all(request.num_output_placeholders == 1 for request in requests)
    mixed = sched.schedule()
    assert mixed.total_num_scheduled_tokens == 159
    assert mixed.sawtooth_has_decode  # any marker, not a pure-D assertion
    assert mixed.num_scheduled_tokens[fresh.request_id] == 128
    for request in requests:
        assert mixed.num_scheduled_tokens[request.request_id] == 1
        step = mixed.sawtooth_request_steps[request.request_id]
        assert (
            step.predecessor
            == predecessor.sawtooth_request_steps[request.request_id].key
        )
        assert (step.token_start, step.token_end) == (70, 71)
        assert request.num_output_placeholders == 2
    assert sched.max_num_scheduled_tokens == 10240


@pytest.mark.parametrize("capacity_bound", [False, True])
def test_refill_boundary_dp_throttle_is_not_strict_decode_selection(
    refill_boundary, capacity_bound
):
    sched, requests, _, fresh = refill_boundary(num_running=2, committed_outputs=1)
    sched.prefill_capacity_bound = capacity_bound
    output = sched.schedule(throttle_prefills=True)
    assert all(output.num_scheduled_tokens[r.request_id] == 1 for r in requests)
    assert (fresh.request_id in output.num_scheduled_tokens) is capacity_bound


def test_refill_boundary_pending_final_sample_is_not_decode_eligible(
    refill_boundary,
):
    sched, requests, _, fresh = refill_boundary(num_running=2, committed_outputs=63)
    output = sched.schedule()
    assert output.num_scheduled_tokens == {fresh.request_id: 128}
    assert all(request.num_output_placeholders == 1 for request in requests)


@pytest.mark.parametrize("prefill_length", [1, 17])
def test_refill_boundary_throttle_admits_one_token_new_prefill(prefill_length):
    sched = scheduler()
    sched.sawtooth_available_batches = 1
    source, fresh = create_requests(
        2,
        num_tokens=prefill_length,
        same_prompt=True,
        max_tokens=1,
        req_ids=["prefix-source", "fresh"],
    )
    if prefill_length > 1:
        sched.add_request(source)
        cached = sched.schedule()
        sched.sawtooth_mark_tail_enqueued(cached)
        sched.update_from_output(cached, result(cached))
        assert sched.kv_cache_manager.get_computed_blocks(fresh)[1] == 16
    (old,) = create_requests(1, num_tokens=8, req_ids=["decode"])
    sched.add_request(old)
    prefill = sched.schedule()
    sched.sawtooth_mark_tail_enqueued(prefill)
    sched.update_from_output(prefill, result(prefill))
    predecessor = sched.schedule()
    enable_successor(sched, predecessor)
    sched.add_request(fresh)
    sched.prefill_capacity_bound = False
    output = sched.schedule(throttle_prefills=True)
    assert output.num_scheduled_tokens == {old.request_id: 1, fresh.request_id: 1}
    (new_data,) = output.scheduled_new_reqs
    assert new_data.req_id == fresh.request_id
    # Worker V2 initializes prefill_len from these token IDs. A query length of
    # one is insufficient for FULL decode: this new row still has_prefill.
    assert len(new_data.prefill_token_ids) == prefill_length
    assert new_data.num_computed_tokens == prefill_length - 1


@pytest.mark.parametrize("capacity_bound", [False, True])
def test_strict_decode_skips_waiting_path_and_keeps_full_budget(
    refill_boundary, monkeypatch, capacity_bound
):
    sched, requests, _, fresh = refill_boundary(num_running=2, committed_outputs=1)
    sched.prefill_capacity_bound = capacity_bound
    sched.sawtooth_decode_only = True

    def forbidden(*args, **kwargs):
        pytest.fail("decode-only accessed waiting promotion/prefix lookup")

    monkeypatch.setattr(sched, "_try_promote_blocked_waiting_request", forbidden)
    monkeypatch.setattr(sched, "_get_local_prefix_cache_hit", forbidden)
    fresh.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    before = sched.current_step
    assert sched.sawtooth_has_decode_candidate()
    assert sched.sawtooth_has_prefill_candidate()  # blocked P may try once later
    assert sched.current_step == before
    output = sched.schedule(throttle_prefills=True)
    assert output.num_scheduled_tokens == {r.request_id: 1 for r in requests}
    assert output.sawtooth_has_decode
    assert sched.current_step == before + 1
    assert sched.max_num_scheduled_tokens == 10240
    assert fresh.num_computed_tokens == 0
    assert fresh in sched.waiting


def test_candidate_hint_is_read_only_and_shares_final_token_guard(
    refill_boundary, monkeypatch
):
    sched, requests, _, _ = refill_boundary(num_running=2, committed_outputs=63)

    def forbidden(*args, **kwargs):
        pytest.fail("candidate hint allocated or promoted a request")

    monkeypatch.setattr(sched.kv_cache_manager, "allocate_slots", forbidden)
    monkeypatch.setattr(sched, "_try_promote_blocked_waiting_request", forbidden)
    before = (sched.current_step, [r.num_in_flight_tokens for r in requests])
    assert not sched.sawtooth_has_decode_candidate()
    assert sched.sawtooth_has_prefill_candidate()
    assert before == (sched.current_step, [r.num_in_flight_tokens for r in requests])
    sched.sawtooth_decode_only = True
    assert not sched.schedule().num_scheduled_tokens


def test_decode_hint_uses_next_step_cadence_and_actual_model_limit(refill_boundary):
    sched, (request,), _, _ = refill_boundary(num_running=1, committed_outputs=1)
    request.next_decode_eligible_step = sched.current_step + 2
    assert not sched.sawtooth_has_decode_candidate()
    request.next_decode_eligible_step -= 1
    assert sched.sawtooth_has_decode_candidate()
    sched.max_model_len = request.num_computed_tokens + 1
    assert not sched.sawtooth_has_decode_candidate()
    sched.sawtooth_decode_only = True
    assert not sched.schedule().num_scheduled_tokens


def test_decode_hint_can_fail_allocation_without_rescheduling(
    refill_boundary, monkeypatch
):
    sched, (request,), _, fresh = refill_boundary(num_running=1, committed_outputs=1)
    calls = []

    def no_space(req, *args, **kwargs):
        calls.append(req.request_id)
        return None

    monkeypatch.setattr(sched.kv_cache_manager, "allocate_slots", no_space)
    assert sched.sawtooth_has_decode_candidate()
    assert not calls
    before = sched.current_step
    in_flight = request.num_in_flight_tokens
    sched.sawtooth_decode_only = True
    output = sched.schedule()
    assert not output.num_scheduled_tokens
    assert not output.sawtooth_has_decode
    assert calls == [request.request_id]
    assert sched.current_step == before + 1
    assert request.num_in_flight_tokens == in_flight
    assert fresh.num_computed_tokens == 0


def test_strict_decode_cap_counts_only_decode_candidates(refill_boundary):
    sched, requests, _, _ = refill_boundary(num_running=12, committed_outputs=1)
    for request in create_requests(
        20, num_tokens=128, req_ids=[f"p{i}" for i in range(20)]
    ):
        sched.add_request(request)
    sched.sawtooth_available_batches = 3
    sched.sawtooth_decode_only = True
    assert sched._sawtooth_request_cap(False) == 4
    output = sched.schedule()
    assert len(output.num_scheduled_tokens) == 4
    assert set(output.num_scheduled_tokens) <= {r.request_id for r in requests}
    assert output.total_num_scheduled_tokens == 4


def test_prefill_opportunity_excludes_decode_then_auto_restores(refill_boundary):
    sched, requests, _, fresh = refill_boundary(num_running=2, committed_outputs=1)
    sched.sawtooth_prefill_only = True
    assert sched._sawtooth_request_cap(False) == 1
    prefill = sched.schedule(throttle_prefills=True)
    assert prefill.num_scheduled_tokens == {fresh.request_id: 128}
    assert not prefill.sawtooth_has_decode
    assert sched._sawtooth_prefill_lens[fresh.request_id] == 128
    sched.sawtooth_prefill_only = False
    output = sched.schedule()
    assert output.num_scheduled_tokens == {r.request_id: 1 for r in requests}
    assert output.sawtooth_has_decode
    assert not prefill.sawtooth_has_decode  # immutable admission classification
    assert replace(output).sawtooth_has_decode


def test_strict_modes_reject_invalid_combinations_without_scheduling():
    sched = scheduler()
    before = sched.current_step
    sched.sawtooth_decode_only = sched.sawtooth_prefill_only = True
    with pytest.raises(ValueError, match="exclusive"):
        sched.schedule()
    sched.sawtooth_prefill_only = False
    sched.sawtooth_enabled = False
    with pytest.raises(ValueError, match="requires sawtooth"):
        sched.schedule()
    assert sched.current_step == before


@pytest.mark.parametrize("prefill_length", [1, 17])
def test_new_single_query_prefill_is_not_a_decode_candidate(prefill_length):
    sched = scheduler()
    source, fresh = create_requests(
        2,
        num_tokens=prefill_length,
        same_prompt=True,
        max_tokens=1,
        req_ids=["prefix-source", "fresh"],
    )
    if prefill_length > 1:
        sched.add_request(source)
        output = sched.schedule()
        sched.sawtooth_mark_tail_enqueued(output)
        sched.update_from_output(output, result(output))
    sched.add_request(fresh)
    assert not sched.sawtooth_has_decode_candidate()
    sched.sawtooth_decode_only = True
    assert not sched.schedule().num_scheduled_tokens
    sched.sawtooth_decode_only = False
    sched.sawtooth_prefill_only = True
    prefill = sched.schedule()
    assert prefill.num_scheduled_tokens == {fresh.request_id: 1}
    assert not prefill.sawtooth_has_decode
    assert sched._sawtooth_prefill_lens[fresh.request_id] == prefill_length


def test_chunked_and_replayed_prefill_use_worker_boundary():
    sched = scheduler(enable_prefix_caching=False)
    (request,) = create_requests(1, num_tokens=8, max_tokens=16)
    sched.add_request(request)
    for _ in range(3):
        output = sched.schedule()
        sched.sawtooth_mark_tail_enqueued(output)
        sched.update_from_output(output, result(output))
    assert request.num_tokens == 11
    assert sched._sawtooth_prefill_lens[request.request_id] == 8
    assert sched.reset_prefix_cache(reset_running_requests=True)
    sched.max_num_scheduled_tokens = 1
    # A resumed V2 allocation must replay eleven tokens, not just prompt8.
    for _ in range(10):
        output = sched.schedule()
        assert output.total_num_scheduled_tokens == 1
        assert not output.sawtooth_has_decode
        sched.sawtooth_mark_tail_enqueued(output)
        sched.update_from_output(output, result(output, sample=False))
    assert request.num_computed_tokens == 10
    assert sched._sawtooth_prefill_lens[request.request_id] == 11
    assert request.num_computed_tokens > request.num_prompt_tokens
    assert not sched.sawtooth_has_decode_candidate()
    assert sched.sawtooth_has_prefill_candidate()
    sched.sawtooth_decode_only = True
    assert not sched.schedule().num_scheduled_tokens
    sched.sawtooth_decode_only = False
    sched.sawtooth_prefill_only = True
    final_prefill = sched.schedule()
    assert final_prefill.total_num_scheduled_tokens == 1
    assert not final_prefill.sawtooth_has_decode
    sched.sawtooth_prefill_only = False
    enable_successor(sched, final_prefill)
    assert sched.sawtooth_has_decode_candidate()


def test_prefill_boundary_survives_abort_until_last_lease_settles():
    sched = scheduler()
    (request,) = create_requests(1, num_tokens=8)
    sched.add_request(request)
    first = sched.schedule()
    enable_successor(sched, first)
    second = sched.schedule()
    sched.sawtooth_mark_tail_enqueued(second)
    sched.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    sched.update_from_output(first, result(first))
    assert sched._sawtooth_prefill_lens[request.request_id] == 8
    sched.update_from_output(second, result(second))
    assert request.request_id not in sched._sawtooth_prefill_lens


def test_prefill_opportunity_preserves_full_single_prompt_budget(tmp_path):
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 16384
    config_path.write_text(json.dumps(config))
    sched = scheduler(
        max_num_batched_tokens=10240,
        max_model_len=16384,
        enable_prefix_caching=False,
    )
    (request,) = create_requests(1, num_tokens=12288, max_tokens=4)
    sched.add_request(request)
    sched.sawtooth_prefill_only = True
    output = sched.schedule(throttle_prefills=True)
    assert output.num_scheduled_tokens == {request.request_id: 10240}
    assert not output.sawtooth_has_decode
    assert sched._sawtooth_prefill_lens[request.request_id] == 12288


def test_decode_successor_has_a_distinct_lease_before_predecessor_output():
    sched = scheduler()
    (request,) = create_requests(1, num_tokens=32, max_tokens=4)
    sched.add_request(request)
    first = sched.schedule()
    enable_successor(sched, first)
    second = sched.schedule()
    assert second.num_scheduled_tokens == {request.request_id: 1}
    assert request.num_in_flight_tokens == 33
    assert request.num_output_placeholders == 2
    a = first.sawtooth_request_steps[request.request_id]
    b = second.sawtooth_request_steps[request.request_id]
    assert b.predecessor == a.key
    assert (a.token_start, a.token_end, b.token_start, b.token_end) == (0, 32, 32, 33)
    enable_successor(sched, second)
    assert not sched.schedule().num_scheduled_tokens  # bounded two leases
    sched.update_from_output(first, result(first))
    assert request.num_in_flight_tokens == 1
    assert request.num_output_placeholders == 1
    assert (
        sched.sawtooth_request_ledger.requests[request.request_id].completed_end == 32
    )
    sched.update_from_output(second, result(second))
    assert request.num_in_flight_tokens == request.num_output_placeholders == 0
    assert request.num_output_tokens == 2
    assert sched.running.count(request) == 1


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_known_prefill_successor_does_not_require_sample_or_publish_later_prefix(
    async_scheduling, monkeypatch
):
    sched = scheduler(
        async_scheduling=async_scheduling,
        max_num_batched_tokens=16,
        max_model_len=128,
    )
    request, probe = create_requests(2, num_tokens=48, same_prompt=True)
    sched.add_request(request)
    first = sched.schedule()
    enable_successor(sched, first)
    second = sched.schedule()
    assert not first.sawtooth_request_steps[request.request_id].produces_sample
    assert not second.sawtooth_request_steps[request.request_id].produces_sample
    assert request.num_in_flight_tokens == 32
    assert request.num_output_placeholders == 0
    assert sched.kv_cache_manager.get_computed_blocks(probe)[1] == 0
    sched.update_from_output(first, result(first, sample=False))
    assert sched.kv_cache_manager.get_computed_blocks(probe)[1] == 16
    assert request.num_in_flight_tokens == 16
    sched.update_from_output(second, result(second, sample=False))
    assert sched.kv_cache_manager.get_computed_blocks(probe)[1] == 32
    assert not request.num_output_tokens


def test_sync_decode_has_no_unknown_token_placeholder_to_schedule():
    sched = scheduler(async_scheduling=False)
    (request,) = create_requests(1, num_tokens=8, max_tokens=3)
    sched.add_request(request)
    first = sched.schedule()
    enable_successor(sched, first)
    assert not request.num_output_placeholders
    assert not sched.schedule().num_scheduled_tokens
    sched.update_from_output(first, result(first))
    assert sched.schedule().num_scheduled_tokens == {request.request_id: 1}


@pytest.mark.parametrize("abort", [False, True])
def test_stop_or_abort_retains_both_leases_and_discards_successor(abort):
    sched = scheduler()
    (request,) = create_requests(1, num_tokens=32, max_tokens=8)
    request.sampling_params.stop_token_ids = [7]
    request.sampling_params.all_stop_token_ids.add(7)
    sched.add_request(request)
    first = sched.schedule()
    enable_successor(sched, first)
    second = sched.schedule()
    blocks = [
        b
        for group in sched.kv_cache_manager.get_blocks(request.request_id).blocks
        for b in group
    ]
    if abort:
        sched.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    sched.update_from_output(first, result(first))
    assert request.is_finished()
    assert request.num_in_flight_tokens == 1
    assert sched.requests[request.request_id] is request
    assert request.request_id not in sched.finished_req_ids
    assert all(b.ref_cnt > 0 for b in blocks)
    tokens = request.num_output_tokens
    assert not sched.schedule().num_scheduled_tokens
    sched.update_from_output(second, result(second))
    assert request.num_output_tokens == tokens
    assert request.request_id in sched.finished_req_ids
    assert request.request_id not in sched.requests
    assert request.request_id not in sched.sawtooth_request_ledger.requests


def test_allocation_recycles_only_the_oldest_completed_boundary(monkeypatch):
    sched = scheduler(max_num_batched_tokens=16, max_model_len=128)
    (request,) = create_requests(1, num_tokens=64)
    sched.add_request(request)
    first = sched.schedule()
    removed_at = []
    original = sched.kv_cache_manager.coordinator.remove_skipped_blocks

    def record(rid, completed, **kwargs):
        removed_at.append(completed)
        return original(rid, completed, **kwargs)

    monkeypatch.setattr(
        sched.kv_cache_manager.coordinator, "remove_skipped_blocks", record
    )
    enable_successor(sched, first)
    second = sched.schedule()
    assert removed_at[-1] == 0  # optimistic start=16, predecessor still live
    enable_successor(sched, second)
    sched.update_from_output(first, result(first, sample=False))
    third = sched.schedule()
    assert third.num_scheduled_tokens == {request.request_id: 16}
    assert removed_at[-1] == 16  # second and third remain owned


def test_sliding_window_does_not_drop_blocks_used_by_either_live_step():
    sched = scheduler(
        max_num_batched_tokens=16,
        max_model_len=128,
        num_blocks=12,
        enable_prefix_caching=False,
        kv_cache_spec=SlidingWindowSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
            sliding_window=16,
        ),
    )
    (request,) = create_requests(1, num_tokens=96)
    sched.add_request(request)
    first = sched.schedule()
    enable_successor(sched, first)
    second = sched.schedule()
    enable_successor(sched, second)
    table = lambda: sched.kv_cache_manager.get_blocks(request.request_id).blocks[0]
    assert not table()[0].is_null  # both intervals still read the original window
    sched.update_from_output(first, result(first, sample=False))
    third = sched.schedule()
    enable_successor(sched, third)
    assert not table()[0].is_null  # second step still requires this block
    sched.update_from_output(second, result(second, sample=False))
    fourth = sched.schedule()
    assert fourth.num_scheduled_tokens == {request.request_id: 16}
    assert table()[0].is_null  # now below the oldest live step's window
    assert request.num_in_flight_tokens == 32
    assert request.num_preemptions == 0

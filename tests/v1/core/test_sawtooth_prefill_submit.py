# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completed prefill pairs can submit continuations before old host retirement."""
import pytest
from concurrent.futures import Future

from .test_sawtooth_admission import (
    _drain_tail_engine,
    _tail_engine,
    admission_scheduler,
)
from .utils import create_requests

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize('capacity', [3, 4])
def test_next_full_prefill_pair_submits_tails_without_retiring_old_pair(
    admission_scheduler, capacity,
):
    engine, sched, executor, request = _tail_engine(
        admission_scheduler, 2048, capacity=capacity,
    )
    (partner,) = create_requests(1, num_tokens=2048, max_tokens=1, req_ids=['partner'])
    sched.add_request(partner)
    for _ in range(20):
        engine.step()
        heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
        if len(heads) == 4:
            break
    assert len(heads) == 4
    old = [b for b in engine.active if b.batch_id < 2]
    assert old and all(not b.terminal_future.done() for b in old)
    free = sched.kv_cache_manager.block_pool.get_num_free_blocks()
    engine.step()
    last = executor.commands[-1].sawtooth_execution
    assert (last.batch_id, last.round_id) == (2, 1)
    assert all(not b.terminal_future.done() for b in old)
    assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == free
    engine.step()
    last = executor.commands[-1].sawtooth_execution
    assert (last.batch_id, last.round_id) == (3, 1)
    assert all(not b.terminal_future.done() for b in old)
    _drain_tail_engine(engine, sched)
    assert request.num_output_tokens == partner.num_output_tokens == 1
    assert request.num_in_flight_tokens == partner.num_in_flight_tokens == 0
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [1024] * 4
    commands = [(o.sawtooth_execution.batch_id, o.sawtooth_execution.round_id)
                for o in executor.commands]
    assert commands == [(b, r) for start in (0, 2)
                        for r in (0, 1) for b in (start, start + 1)]


@pytest.mark.parametrize('capacity', [3, 4])
@pytest.mark.parametrize('max_tokens', [1, 2])
def test_repeated_pair_advance_recycles_slots_and_drains_decode(
    admission_scheduler, capacity, max_tokens,
):
    engine, sched, executor, first = _tail_engine(
        admission_scheduler, 2048, capacity=capacity, max_tokens=max_tokens,
    )
    others = create_requests(2, num_tokens=2048, max_tokens=max_tokens,
                             req_ids=['second', 'third'])
    for request in others:
        sched.add_request(request)
    peak = _drain_tail_engine(engine, sched)
    assert peak <= capacity
    assert not engine.active and not engine.rounds
    assert len(engine.slots) == capacity
    assert not sched.sawtooth_request_ledger.requests
    for request in (first, *others):
        assert request.num_output_tokens == max_tokens
        assert request.num_in_flight_tokens == 0
        assert request.num_preemptions == 0
    commands = [o.sawtooth_execution for o in executor.commands]
    heads = [c for c in commands if c.is_first_round]
    assert len({(c.slot_id, c.slot_generation) for c in heads}) == len(heads)
    for head in heads:
        assert [c.round_id for c in commands if c.batch_id == head.batch_id] == [0, 1]
    assert max(c.slot_generation for c in heads) > 1


def four_heads(admission_scheduler):
    engine, sched, executor, first = _tail_engine(admission_scheduler, 2048, capacity=4)
    (second,) = create_requests(1, num_tokens=2048, max_tokens=1, req_ids=['second'])
    sched.add_request(second)
    for _ in range(20):
        engine.step()
        if sum(o.sawtooth_execution.is_first_round for o in executor.commands) == 4:
            break
    assert len(engine.active) == 4
    return engine, sched, executor, first, second


def test_pending_metadata_control_preserves_retirement_boundary(admission_scheduler):
    engine, sched, executor, _, _ = four_heads(admission_scheduler)
    old = engine.active[0]
    # A control RPC has no logical slot. An unread response must retain the
    # existing retirement/drain path rather than advance the prefill phase.
    control = (None, Future())
    engine.controls.append(control)
    count = len(executor.commands)
    engine.step()
    assert len(executor.commands) == count
    assert old.terminal_future.done()
    assert engine.phase_limit == 1
    assert engine.controls.pop() is control
    _drain_tail_engine(engine, sched)


@pytest.mark.parametrize('failed_batch', [0, 2])
def test_submission_failure_after_pair_advance_does_not_publish_failed_kv(
    admission_scheduler, failed_batch,
):
    engine, sched, executor, first, second = four_heads(admission_scheduler)
    batch = next(b for b in engine.active if b.batch_id == failed_batch)
    def fail():
        raise RuntimeError('injected submission failure')
    batch.command_futures[0][1].response = fail
    engine.step()
    engine.step()
    assert engine.phase_limit == 3
    assert len(engine.active) == 4
    with pytest.raises(RuntimeError, match='injected submission failure'):
        _drain_tail_engine(engine, sched)
    failed_request = first if failed_batch == 0 else second
    assert failed_request.num_output_tokens == 0
    assert failed_request.request_id in sched.requests
    assert failed_request.request_id in sched.sawtooth_request_ledger.requests
    assert failed_request.num_in_flight_tokens == 2048

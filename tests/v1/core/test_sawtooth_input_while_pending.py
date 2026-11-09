# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from .test_sawtooth_admission import _drain_tail_engine, _tail_engine, admission_scheduler
from .utils import create_requests

pytestmark = pytest.mark.cpu_test


def pending_pair(admission_scheduler, capacity=3):
    engine, sched, executor, request = _tail_engine(admission_scheduler, 2048, capacity=capacity)
    for _ in range(10):
        engine.step()
        if len(executor.commands) == 4:
            break
    assert len(executor.commands) == 4 and len(engine.active) == 2
    original = engine.active[0].terminal_future
    pending = Future()
    pending.can_poll = True
    pending.poll = Mock(return_value=False)
    pending.result = Mock(side_effect=AssertionError('blocking retirement'))
    engine.active[0].terminal_future = pending
    return engine, sched, executor, request, original, pending


def test_arriving_prefill_uses_free_slot_while_old_terminal_is_pending(admission_scheduler):
    engine, sched, executor, first, original, pending = pending_pair(admission_scheduler)
    free = sched.kv_cache_manager.block_pool.get_num_free_blocks()
    oldest = engine.active[0]
    assert engine.step() == (None, False)
    assert engine.active[0] is oldest and len(engine.slots) == 1
    assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == free
    pending.result.assert_not_called()
    pending.poll.assert_called_once()
    (second,) = create_requests(1, num_tokens=2048, max_tokens=1, req_ids=['new-arrival'])
    sched.add_request(second)
    engine.step()
    last = executor.commands[-1].sawtooth_execution
    assert last.batch_id == 2 and last.is_first_round
    assert engine.active[0] is oldest and not engine.slots
    pending.result.assert_not_called()
    oldest.terminal_future = original
    _drain_tail_engine(engine, sched)
    assert first.num_output_tokens == second.num_output_tokens == 1
    assert not sched.sawtooth_request_ledger.requests


@pytest.mark.parametrize('guard', ['control', 'decode', 'full', 'legacy'])
def test_pending_yield_keeps_existing_blocking_boundaries(admission_scheduler, guard):
    engine, sched, executor, _, _, pending = pending_pair(
        admission_scheduler, capacity=2 if guard == 'full' else 3)
    if guard == 'control':
        engine.controls.append((None, Future()))
    elif guard == 'decode':
        engine.active[0].scheduler_output.sawtooth_has_decode = True
    elif guard == 'legacy':
        pending.can_poll = False
    with pytest.raises(AssertionError, match='blocking retirement'):
        engine.step()
    pending.poll.assert_not_called()

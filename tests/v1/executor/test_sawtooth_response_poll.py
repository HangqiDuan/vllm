# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-reader readiness preserves bytes and FIFO until complete responses exist."""
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor.multiproc_executor import FutureWrapper


class Buffer:
    def __init__(self, written=0, read=0, overflow=0):
        self.meta = bytearray([written, read])
        self.data = bytearray([overflow])

    @contextmanager
    def get_metadata(self, index):
        assert index == 0
        yield self.meta

    @contextmanager
    def get_data(self, index):
        assert index == 0
        yield self.data


def local_queue(written=0, read=0, overflow=0, socket_ready=False):
    q = object.__new__(MessageQueue)
    q._is_local_reader = True
    q._is_remote_reader = False
    q.shutting_down = False
    q.current_idx = 0
    q.local_reader_rank = 0
    q.buffer = Buffer(written, read, overflow)
    q.local_socket = SimpleNamespace(poll=Mock(return_value=socket_ready))
    return q


@pytest.mark.parametrize('written,read,overflow,socket_ready,expected', [
    (0, 0, 0, False, False), (1, 1, 0, False, False),
    (1, 0, 0, False, True), (1, 0, 1, False, False),
    (1, 0, 1, True, True),
])
def test_local_readiness_never_consumes_header_or_advances_cursor(
    written, read, overflow, socket_ready, expected,
):
    q = local_queue(written, read, overflow, socket_ready)
    before = bytes(q.buffer.meta), bytes(q.buffer.data)
    assert q.is_ready() is expected
    assert q.is_ready() is expected
    assert (bytes(q.buffer.meta), bytes(q.buffer.data)) == before
    assert q.current_idx == 0
    if written and not read and overflow:
        q.local_socket.poll.assert_called_with(timeout=0)
    else:
        q.local_socket.poll.assert_not_called()


def test_remote_readiness_and_shutdown():
    q = object.__new__(MessageQueue)
    q._is_local_reader = False
    q._is_remote_reader = True
    q.shutting_down = False
    q.remote_socket = SimpleNamespace(poll=Mock(side_effect=[0, 1]))
    assert q.is_ready() is False
    assert q.is_ready() is True
    q.shutting_down = True
    with pytest.raises(RuntimeError, match='cancelled'):
        q.is_ready()


def test_poll_drains_ready_fifo_only_through_target():
    pending = deque()
    ready = [False, False, True]
    calls = []
    def poll(i):
        if not ready[i]:
            return False, None
        calls.append(i)
        return True, i
    futures = [FutureWrapper(pending, lambda i=i: calls.append(i) or i,
                             poll_response=lambda i=i: poll(i))
               for i in range(3)]
    assert not futures[1].poll()
    assert not calls and len(pending) == 3
    ready[0] = True
    assert not futures[1].poll()
    assert calls == [0] and len(pending) == 2
    ready[1] = True
    assert futures[1].poll()
    assert calls == [0, 1] and len(pending) == 1
    assert futures[1].result() == 1
    assert futures[1].poll()
    assert not futures[2].done()
    assert futures[2].result() == 2
    assert calls == [0, 1, 2] and not pending


def test_poll_preserves_failed_future_and_legacy_blocking_result():
    pending = deque()
    error = RuntimeError('worker failure')
    def fail():
        raise error
    first = FutureWrapper(pending, fail, poll_response=fail)
    second = FutureWrapper(pending, lambda: 2)
    assert first.poll()
    with pytest.raises(RuntimeError) as caught:
        first.result()
    assert caught.value is error
    assert not second.poll()
    assert second.result() == 2


def polling_executor():
    from .test_sawtooth_all_rank_rpc import executor_fixture
    executor, reads = executor_fixture()
    for mq in executor.response_mqs:
        mq.is_ready = lambda mq=mq: bool(mq.items)
    return executor, reads


def test_partial_ranks_are_saved_and_not_read_twice():
    from .test_sawtooth_all_rank_rpc import output, acknowledgements
    from vllm.v1.executor.multiproc_executor import WorkerProc
    executor, reads = polling_executor()
    scheduled = output()
    future = executor.execute_model(scheduled, non_block=True)
    acks = acknowledgements(scheduled.sawtooth_execution)
    executor.response_mqs[0].enqueue((WorkerProc.ResponseStatus.SUCCESS, acks[0]))
    assert not future.poll()
    assert reads == [0]
    assert not future.poll()
    assert reads == [0]
    for rank in range(1, 8):
        executor.response_mqs[rank].enqueue((WorkerProc.ResponseStatus.SUCCESS, acks[rank]))
    assert future.poll()
    assert future.result() == acks[4]
    assert reads == list(range(8))


def test_partial_poll_then_blocking_result_and_next_rpc():
    from .test_sawtooth_all_rank_rpc import output, acknowledgements
    from vllm.v1.executor.multiproc_executor import WorkerProc
    executor, reads = polling_executor()
    scheduled = output()
    first = executor.execute_model(scheduled, non_block=True)
    second = executor.collective_rpc('check_health', non_block=True)
    acks = acknowledgements(scheduled.sawtooth_execution)
    executor.response_mqs[0].enqueue((WorkerProc.ResponseStatus.SUCCESS, acks[0]))
    assert not first.poll()
    for rank in range(8):
        if rank:
            executor.response_mqs[rank].enqueue((WorkerProc.ResponseStatus.SUCCESS, acks[rank]))
        executor.response_mqs[rank].enqueue((WorkerProc.ResponseStatus.SUCCESS, rank))
    assert first.result() == acks[4]
    assert reads == list(range(8))
    assert second.result() == list(range(8))
    assert reads == list(range(8)) * 2


@pytest.mark.parametrize('method', ['execute_model', 'check_health'])
def test_rank_failure_does_not_wait_for_absent_peer_or_read_next_rpc(method):
    from .test_sawtooth_all_rank_rpc import output
    from vllm.v1.executor.multiproc_executor import WorkerProc
    executor, reads = polling_executor()
    first = executor.collective_rpc(method, args=(output(),) if method == 'execute_model' else (), non_block=True)
    terminal = executor.sample_tokens(None, non_block=True)
    executor.response_mqs[0].enqueue((WorkerProc.ResponseStatus.FAILURE, 'rank failed'))
    with pytest.raises(RuntimeError, match='rank failed'):
        terminal.poll()
    assert first.done()
    assert reads == [0]
    with pytest.raises(RuntimeError, match='rank failed'):
        terminal.result()
    assert reads == [0]


def test_target_deadline_expires_behind_old_no_deadline_control():
    executor, reads = polling_executor()
    old = executor.collective_rpc('check_health', non_block=True)
    terminal = executor.collective_rpc('sample_tokens', args=(None,), timeout=0,
                                       unique_reply_rank=4, non_block=True)
    with pytest.raises(TimeoutError):
        terminal.poll()
    assert not reads
    with pytest.raises(TimeoutError):
        old.result()
    assert executor.is_failed and not reads


def test_real_shm_readiness_inline_overflow_and_ring_wrap():
    import time

    writer = MessageQueue(n_reader=1, n_local_reader=1,
                          max_chunk_bytes=128, max_chunks=2)
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    try:
        writer.wait_until_ready()
        reader.wait_until_ready()
        assert not reader.is_ready()
        for i in range(6):
            payload = {'i': i, 'data': 'x' * (2048 if i % 2 else 1)}
            index = reader.current_idx
            writer.enqueue(payload)
            deadline = time.monotonic() + 5
            while not reader.is_ready():
                assert time.monotonic() < deadline
                time.sleep(0.001)
            assert reader.is_ready() and reader.current_idx == index
            with reader.buffer.get_metadata(index) as metadata:
                assert metadata[0] == 1 and metadata[1] == 0
            assert reader.dequeue(timeout=1) == payload
            assert reader.current_idx == (index + 1) % 2
            assert not reader.is_ready()
    finally:
        writer.shutdown()
        reader.shutdown()
        for socket in (
            writer.local_socket,
            writer._spin_condition.local_notify_socket,
            reader.local_socket,
            reader._spin_condition.local_notify_socket,
            reader._spin_condition.read_cancel_socket,
            reader._spin_condition.write_cancel_socket,
        ):
            socket.close(linger=0)

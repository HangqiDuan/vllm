# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real executor/FutureWrapper/WorkerProc with in-memory response transports.

No executor or worker constructors are called: these tests need no accelerator.
Imports are deliberately real; missing runtime dependencies are not skipped.
"""

import queue
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine.sawtooth import LogicalBatch, SawtoothEngine
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor,
    WorkerProc,
    _use_async_output,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput
from vllm.v1.sawtooth import SawtoothExecution, SawtoothRoundSubmitted


class Queue:
    def __init__(self, rank, reads):
        self.rank = rank
        self.reads = reads
        self.items = deque()

    def enqueue(self, value):
        self.items.append(value)

    def dequeue(self, **kwargs):
        self.reads.append(self.rank)
        if not self.items:
            raise AssertionError(f"Unexpected response read on rank {self.rank}")
        value = self.items.popleft()
        if isinstance(value, Exception):
            raise value
        return value


def executor_fixture(enabled=True):
    executor = object.__new__(MultiprocExecutor)
    executor.vllm_config = SimpleNamespace(
        additional_config={"sawtooth_pipeline": {"enabled": enabled}}
    )
    executor._sawtooth_enabled = enabled
    executor._sawtooth_failure = None
    executor.is_failed = False
    executor.output_rank = 4
    executor.world_size = 8
    executor.futures_queue = deque()
    executor.kv_output_aggregator = None
    executor.ec_output_aggregator = None
    reads = []
    executor.rpc_broadcast_mq = Queue(-1, [])
    executor.response_mqs = [Queue(rank, reads) for rank in range(8)]
    return executor, reads


def output(round_id=0, command_seq=0):
    return SimpleNamespace(
        sawtooth_execution=SawtoothExecution(
            0, round_id, 2, command_seq, 0, 1, "test-plan"
        )
    )


def responses(executor, values, failure_rank=None):
    status = WorkerProc.ResponseStatus
    for rank, value in enumerate(values):
        executor.response_mqs[rank].enqueue(
            (status.FAILURE if rank == failure_rank else status.SUCCESS, value)
        )


def acknowledgements(command):
    return [SawtoothRoundSubmitted(command) for _ in range(8)]


def test_nonblocking_all_rank_ack_and_final_none():
    executor, reads = executor_fixture()
    scheduled = output()
    first = executor.execute_model(scheduled, non_block=True)
    last = executor.execute_model(output(1, 1), non_block=True)
    assert not reads
    assert [request[3] for request in executor.rpc_broadcast_mq.items] == [None, None]
    responses(executor, acknowledgements(scheduled.sawtooth_execution))
    responses(executor, [None] * 8)
    assert last.result() is None
    assert first.result() == SawtoothRoundSubmitted(scheduled.sawtooth_execution)
    assert reads == list(range(8)) * 2


def test_sample_selects_physical_output_rank_four_not_seven():
    executor, reads = executor_fixture()
    values = [EMPTY_MODEL_RUNNER_OUTPUT] * 4 + [object() for _ in range(4)]
    future = executor.sample_tokens(None, non_block=True)
    assert not reads
    responses(executor, values)
    assert future.result() is values[4]
    assert reads == list(range(8))


@pytest.mark.parametrize("failed_rank", [0, 5])
def test_failure_poisons_later_metadata_and_sample_without_mq_reads(failed_rank):
    executor, reads = executor_fixture()
    scheduled = output()
    first = executor.execute_model(scheduled, non_block=True)
    control = executor.execute_model(
        SimpleNamespace(sawtooth_execution=None), non_block=True
    )
    utility = executor.collective_rpc("check_health", non_block=True)
    terminal = executor.sample_tokens(None, non_block=True)
    values = acknowledgements(scheduled.sawtooth_execution)
    values[failed_rank] = "non-reply failure"
    responses(executor, values, failure_rank=failed_rank)
    # There are intentionally no responses for the later RPCs. They must never
    # consume the unread tail of the failed command as a later response.
    with pytest.raises(RuntimeError, match="non-reply failure") as caught:
        terminal.result()
    assert reads == list(range(failed_rank + 1))
    for future in (first, control, utility, terminal):
        assert future.done()
        assert future.exception() is caught.value
    queued = len(executor.rpc_broadcast_mq.items)
    with pytest.raises(RuntimeError) as repeated:
        executor.execute_model(scheduled, non_block=True)
    assert repeated.value is caught.value
    assert len(executor.rpc_broadcast_mq.items) == queued


@pytest.mark.parametrize("final", [False, True])
def test_all_rank_command_or_final_shape_mismatch_is_fatal(final):
    executor, reads = executor_fixture()
    scheduled = output(int(final), int(final))
    first = executor.execute_model(scheduled, non_block=True)
    terminal = executor.sample_tokens(None, non_block=True)
    values = [None] * 8 if final else acknowledgements(scheduled.sawtooth_execution)
    values[5] = SawtoothRoundSubmitted(output(0, 99).sawtooth_execution)
    responses(executor, values)
    with pytest.raises(RuntimeError):
        terminal.result()
    assert first.exception() is terminal.exception()
    assert reads == list(range(8))


def test_timeout_poisons_future_before_another_rpc_read():
    executor, reads = executor_fixture()
    first = executor.execute_model(output(), non_block=True)
    terminal = executor.sample_tokens(None, non_block=True)
    executor.response_mqs[0].enqueue(TimeoutError("injected dequeue timeout"))
    with pytest.raises(TimeoutError):
        terminal.result()
    assert first.exception() is terminal.exception()
    assert reads == [0]


def test_nonreply_async_output_failure_reaches_executor():
    executor, reads = executor_fixture()
    future = executor.sample_tokens(None, non_block=True)

    class FailedCopy(AsyncModelRunnerOutput):
        def get_output(self):
            raise RuntimeError("rank5 async copy failed")

    for rank in range(8):
        worker = object.__new__(WorkerProc)
        worker.rank = rank
        worker.use_async_scheduling = True
        worker.async_output_queue = queue.Queue()
        worker.worker_response_mq = executor.response_mqs[rank]
        result = FailedCopy() if rank == 5 else EMPTY_MODEL_RUNNER_OUTPUT
        worker.worker = SimpleNamespace(sample_tokens=lambda result=result: result)
        worker._execute_worker_rpc(("sample_tokens", (), {}, None))
        assert not worker.worker_response_mq.items
        worker.enqueue_output(worker.async_output_queue.get_nowait())
    with pytest.raises(RuntimeError, match="rank5 async copy failed"):
        future.result()
    assert reads == list(range(6))


def test_metadata_control_keeps_empty_output_semantics():
    executor, reads = executor_fixture()
    future = executor.execute_model(
        SimpleNamespace(sawtooth_execution=None), non_block=True
    )
    responses(executor, [EMPTY_MODEL_RUNNER_OUTPUT] * 8)
    assert future.result() is EMPTY_MODEL_RUNNER_OUTPUT
    assert reads == list(range(8))


def test_feature_off_keeps_unique_rank_wire_and_reads():
    executor, reads = executor_fixture(enabled=False)
    first = executor.execute_model(output(), non_block=True)
    terminal = executor.sample_tokens(None, non_block=True)
    assert [request[3] for request in executor.rpc_broadcast_mq.items] == [4, 4]
    status = WorkerProc.ResponseStatus.SUCCESS
    executor.response_mqs[4].enqueue((status, "execute"))
    executor.response_mqs[4].enqueue((status, "sample"))
    assert terminal.result() == "sample"
    assert first.result() == "execute"
    assert reads == [4, 4]


def test_feature_off_connector_aggregation_is_unchanged():
    executor, reads = executor_fixture(enabled=False)
    aggregator = SimpleNamespace(
        aggregate=Mock(side_effect=lambda values, output_rank: values[output_rank])
    )
    executor.kv_output_aggregator = aggregator
    values = [object() for _ in range(8)]
    future = executor.sample_tokens(None, non_block=True)
    responses(executor, values)
    assert executor.rpc_broadcast_mq.items[0][3] is None
    assert future.result() is values[4]
    aggregator.aggregate.assert_called_once_with(values, output_rank=4)
    assert reads == list(range(8))


@pytest.mark.parametrize("name", ["kv_output_aggregator", "ec_output_aggregator"])
def test_sawtooth_rejects_connector_before_enqueue(name):
    executor, _ = executor_fixture()
    setattr(executor, name, object())
    with pytest.raises((ValueError, RuntimeError), match="connector"):
        executor.sample_tokens(None, non_block=True)
    assert not executor.rpc_broadcast_mq.items


def test_engine_does_not_commit_or_release_slot_on_all_rank_failure():
    executor, _ = executor_fixture()
    scheduled = output()
    execution_future = executor.execute_model(scheduled, non_block=True)
    terminal = executor.sample_tokens(None, non_block=True)
    responses(executor, ["rank0 fails"] + [None] * 7, failure_rank=0)
    batch = LogicalBatch(0, 0, 1, scheduled)
    batch.exec_future = execution_future
    batch.terminal_future = terminal
    batch.command_futures = [(scheduled.sawtooth_execution, execution_future)]
    engine = object.__new__(SawtoothEngine)
    engine.trace = SimpleNamespace(enabled=False, span=lambda *args: nullcontext())
    engine.active = deque([batch])
    engine.controls = deque()
    engine.slots = deque()
    engine.core = SimpleNamespace(
        scheduler=SimpleNamespace(update_from_output=Mock()),
        _process_aborts_queue=Mock(),
    )
    with pytest.raises(RuntimeError, match="rank0 fails"):
        engine._retire()
    engine.core.scheduler.update_from_output.assert_not_called()
    engine.core._process_aborts_queue.assert_not_called()
    assert not engine.slots


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("sawtooth", [False, True])
def test_output_queue_decision_preserves_scheduler_config(async_scheduling, sawtooth):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        additional_config={"sawtooth_pipeline": {"enabled": sawtooth}},
    )
    assert _use_async_output(config) is (async_scheduling or sawtooth)
    assert config.scheduler_config.async_scheduling is async_scheduling


def test_sync_sawtooth_response_queue_keeps_original_objects_fifo_and_failure():
    worker = object.__new__(WorkerProc)
    worker.rank = 5
    worker.use_async_scheduling = False
    worker.use_async_output = True
    worker.async_output_queue = queue.Queue()
    queued_responses = []
    worker.worker_response_mq = SimpleNamespace(enqueue=queued_responses.append)
    original = object()

    def fail():
        raise RuntimeError("sync sawtooth worker failure")

    worker.worker = SimpleNamespace(first=lambda: original, fail=fail)
    worker._execute_worker_rpc(("first", (), {}, None))
    worker._execute_worker_rpc(("fail", (), {}, None))
    assert not queued_responses
    first = worker.async_output_queue.get_nowait()
    failure = worker.async_output_queue.get_nowait()
    assert first is original
    assert isinstance(failure, RuntimeError)
    worker.enqueue_output(first)
    worker.enqueue_output(failure)
    status = WorkerProc.ResponseStatus
    assert queued_responses == [
        (status.SUCCESS, original),
        (status.FAILURE, "sync sawtooth worker failure"),
    ]
    assert worker.use_async_scheduling is False


def test_sync_feature_off_still_enqueues_response_directly():
    worker = object.__new__(WorkerProc)
    worker.rank = 5
    worker.use_async_scheduling = False
    worker.use_async_output = False
    queued_responses = []
    worker.worker_response_mq = SimpleNamespace(enqueue=queued_responses.append)
    original = object()
    worker.handle_output(original)
    assert queued_responses == [(WorkerProc.ResponseStatus.SUCCESS, original)]
    assert not hasattr(worker, "async_output_queue")

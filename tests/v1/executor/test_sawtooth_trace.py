# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import real WorkerProc; bypass only its device-initializing constructor.

Missing executor dependencies must fail collection rather than skip or replace
the module. The separate Windows task harness is not the repository test path.
"""

import json
import pickle
import queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import vllm.v1.executor.multiproc_executor as executor
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.sawtooth import SawtoothExecution
from vllm.v1.sawtooth_trace import SawtoothTrace, TracedOutput


def worker_fixture(monkeypatch, trace):
    monkeypatch.setattr(executor, "get_sawtooth_trace", lambda: trace)
    worker = object.__new__(executor.WorkerProc)
    worker.rank = 7
    worker.use_async_scheduling = True
    worker.async_output_queue = queue.Queue()
    responses = []
    worker.worker_response_mq = SimpleNamespace(enqueue=responses.append)
    return worker, responses


@pytest.mark.parametrize("failure", [False, True])
def test_off_keeps_original_queue_object_and_response(monkeypatch, failure):
    trace = SawtoothTrace()
    worker, responses = worker_fixture(monkeypatch, trace)
    value = RuntimeError("original worker failure") if failure else object()
    worker.handle_output(value)
    assert worker.async_output_queue.get_nowait() is value
    worker.enqueue_output(value)
    status = executor.WorkerProc.ResponseStatus
    expected = (status.FAILURE, str(value)) if failure else (status.SUCCESS, value)
    assert responses == [expected]
    assert trace.events == 0


def test_off_rpc_does_not_construct_context(monkeypatch):
    worker, _ = worker_fixture(monkeypatch, SawtoothTrace())
    value = object()
    worker.worker = SimpleNamespace(execute_model=lambda output: value)

    def forbidden(output):
        raise AssertionError("Disabled tracing inspected RPC metadata")

    monkeypatch.setattr(executor, "command_context", forbidden)
    worker._execute_worker_rpc(("execute_model", (object(),), {}, 7))
    assert worker.async_output_queue.get_nowait() is value


def test_async_context_matches_rpc_even_for_repeated_none_outputs(
    tmp_path, monkeypatch
):
    trace = SawtoothTrace(str(tmp_path))
    worker, responses = worker_fixture(monkeypatch, trace)
    result = {"req_ids": ["r"], "sampled_token_ids": [[11]]}

    class Sample(AsyncModelRunnerOutput):
        def get_output(self):
            return result

    worker.worker = SimpleNamespace(
        execute_model=lambda output: None, sample_tokens=lambda: Sample()
    )
    for index in range(2):
        command = SawtoothExecution(0, index, 2, index, 0, 1, "plan")
        output = SimpleNamespace(
            sawtooth_execution=command,
            sawtooth_request_steps={},
            num_scheduled_tokens={"r": 1},
            total_num_scheduled_tokens=1,
        )
        worker._execute_worker_rpc(("execute_model", (output,), {}, 7))
    worker._execute_worker_rpc(("sample_tokens", (), {}, 7))

    def consume():
        for _ in range(3):
            item = worker.async_output_queue.get_nowait()
            assert isinstance(item, TracedOutput)
            worker.enqueue_output(item)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(consume).result()
    expected = [
        (executor.WorkerProc.ResponseStatus.SUCCESS, value)
        for value in (None, None, result)
    ]
    assert responses == expected
    # Only the unchanged status/output pairs reach the response transport;
    # no envelope or diagnostic CPU context survives serialization.
    assert pickle.loads(pickle.dumps(responses)) == expected
    trace.close()
    (path,) = tmp_path.glob("*.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    queued = [r for r in rows if r["event"] == "worker.handle_output.enter"]
    sent = [r for r in rows if r["event"] == "worker.response_enqueue.exit"]
    assert [r["command_seq"] for r in queued] == [0, 1, 1]
    assert [r["rpc_method"] for r in queued] == [
        "execute_model",
        "execute_model",
        "sample_tokens",
    ]
    assert [r["rpc_id"] for r in queued] == [r["rpc_id"] for r in sent]
    assert all(a["thread_id"] != b["thread_id"] for a, b in zip(queued, sent))
    copied = [r for r in rows if r["event"] == "worker.get_output.exit"]
    assert len(copied) == 1 and copied[0]["status"] == "ok"
    starts = [r for r in rows if r["event"] == "worker.rpc.enter"]
    ends = [r for r in rows if r["event"] == "worker.rpc.exit"]
    assert len(starts) == len(ends) == 3
    for start, end, queued_output in zip(starts, ends, queued):
        assert start["span_id"] == end["span_id"]
        assert start["rpc_id"] == end["rpc_id"] == queued_output["rpc_id"]
        assert start["monotonic_ns"] <= end["monotonic_ns"]
        assert end["monotonic_ns"] <= queued_output["monotonic_ns"]
        assert end["status"] == "ok"


def test_rpc_failure_trace_preserves_failure_response(tmp_path, monkeypatch):
    trace = SawtoothTrace(str(tmp_path))
    worker, responses = worker_fixture(monkeypatch, trace)

    def fail(output):
        raise RuntimeError("original execute failure")

    worker.worker = SimpleNamespace(execute_model=fail)
    output = SimpleNamespace(
        sawtooth_execution=SawtoothExecution(0, 0, 2, 0, 0, 1, "plan"),
        sawtooth_request_steps={},
        num_scheduled_tokens={"r": 1},
        total_num_scheduled_tokens=1,
    )
    worker._execute_worker_rpc(("execute_model", (output,), {}, 7))
    worker.enqueue_output(worker.async_output_queue.get_nowait())
    assert responses == [
        (executor.WorkerProc.ResponseStatus.FAILURE, "original execute failure")
    ]
    trace.close()
    (path,) = tmp_path.glob("*.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    (start,) = [r for r in rows if r["event"] == "worker.rpc.enter"]
    (end,) = [r for r in rows if r["event"] == "worker.rpc.exit"]
    assert start["span_id"] == end["span_id"]
    assert start["rpc_id"] == end["rpc_id"]
    assert end["status"] == "error" and end["error_type"] == "RuntimeError"
    assert start["monotonic_ns"] <= end["monotonic_ns"]


@pytest.mark.parametrize("enabled", [False, True])
def test_get_output_failure_keeps_original_failure_response(
    tmp_path, monkeypatch, enabled
):
    trace = SawtoothTrace(str(tmp_path) if enabled else None)
    worker, responses = worker_fixture(monkeypatch, trace)

    class Sample(AsyncModelRunnerOutput):
        def get_output(self):
            raise RuntimeError("copy failed")

    context = {"rpc_method": "sample_tokens", "rpc_id": 1} if enabled else None
    worker.handle_output(Sample(), context)
    worker.enqueue_output(worker.async_output_queue.get_nowait())
    assert responses == [(executor.WorkerProc.ResponseStatus.FAILURE, "copy failed")]
    trace.close()
    if not enabled:
        assert not list(tmp_path.iterdir())


def test_response_enqueue_exception_is_not_swallowed(tmp_path, monkeypatch):
    worker, _ = worker_fixture(monkeypatch, SawtoothTrace(str(tmp_path)))
    original = OSError("response queue failed")

    def fail(value):
        raise original

    worker.worker_response_mq.enqueue = fail
    with pytest.raises(OSError) as caught:
        worker.enqueue_output(TracedOutput(None, {"rpc_id": 1}))
    assert caught.value is original

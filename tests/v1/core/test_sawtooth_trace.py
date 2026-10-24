# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import vllm.v1.sawtooth_trace as tracing
from vllm.v1.sawtooth import SawtoothExecution, SawtoothRequestStep


def read_trace(path):
    files = list(path.glob("*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_disabled_trace_does_no_serialization_or_io(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_SAWTOOTH_TRACE_DIR", raising=False)
    monkeypatch.setattr(tracing, "_trace", None)
    trace = tracing.get_sawtooth_trace()

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled diagnostics touched serializer or filesystem")

    monkeypatch.setattr(tracing.json, "dumps", forbidden)
    monkeypatch.setattr(tracing.os, "makedirs", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    trace.emit("unused", {"not_serializable": object()})
    with trace.span("unused", {"not_serializable": object()}):
        pass
    trace.close()
    assert not trace.enabled and trace.events == 0
    assert not list(tmp_path.iterdir())


def test_enabled_trace_has_paired_clocks_and_atomic_thread_lines(tmp_path):
    trace = tracing.SawtoothTrace(str(tmp_path), 1000)
    barrier = threading.Barrier(4)

    def run(index):
        barrier.wait(timeout=5)
        for iteration in range(25):
            with trace.span("work", {"worker": index, "iteration": iteration}):
                pass

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(4)))
    trace.close()
    rows = read_trace(tmp_path)
    assert len(rows) == 201
    pairs = {}
    for row in rows[:-1]:
        assert row["pid"] == trace.pid
        assert row["monotonic_ns"] > 0 and row["wall_ns"] > 0
        assert row["thread_id"] > 0
        pairs.setdefault(row["span_id"], []).append(row)
    assert len(pairs) == 100
    assert len({row["thread_id"] for row in rows[:-1]}) == 4
    for enter, exit_ in pairs.values():
        assert (enter["event"], exit_["event"]) == ("work.enter", "work.exit")
        assert enter["thread_id"] == exit_["thread_id"]
        assert exit_["monotonic_ns"] >= enter["monotonic_ns"]
        assert exit_["status"] == "ok"
    assert rows[-1]["dropped_events"] == 0


def test_cap_bounds_file_and_reports_dropped_events(tmp_path):
    trace = tracing.SawtoothTrace(str(tmp_path), 3)
    for index in range(20):
        trace.emit("event", {"index": index})
    trace.close()
    rows = read_trace(tmp_path)
    assert len(rows) == 5  # cap + one marker + one exit summary
    assert rows[-2]["event"] == "trace.cap_reached"
    assert rows[-1]["written_events"] == 3
    assert rows[-1]["dropped_events"] == 17


def test_trace_failure_cannot_replace_application_exception(tmp_path, monkeypatch):
    trace = tracing.SawtoothTrace(str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("diagnostic disk full")

    monkeypatch.setattr(trace, "_write", fail)
    original = RuntimeError("original RPC failure")
    with pytest.raises(RuntimeError) as caught, trace.span("execute"):
        raise original
    assert caught.value is original
    assert not trace.enabled and trace.errors == 1


def test_environment_is_read_once(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "_trace", None)
    monkeypatch.setenv("VLLM_SAWTOOTH_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SAWTOOTH_TRACE_MAX_EVENTS", "12")
    first = tracing.get_sawtooth_trace()
    monkeypatch.setenv("VLLM_SAWTOOTH_TRACE_MAX_EVENTS", "1")
    assert tracing.get_sawtooth_trace() is first
    assert first.max_events == 12
    first.close()


def test_separate_processes_write_separate_files(tmp_path):
    parent = tracing.SawtoothTrace(str(tmp_path))
    parent.emit("parent")
    parent.close()
    env = dict(
        os.environ,
        VLLM_SAWTOOTH_TRACE_DIR=str(tmp_path),
        VLLM_SAWTOOTH_TRACE_MAX_EVENTS="10",
    )
    code = (
        "import runpy,sys; m=runpy.run_path(sys.argv[1]); "
        "t=m['get_sawtooth_trace'](); t.emit('child'); t.close()"
    )
    subprocess.run(
        [sys.executable, "-c", code, tracing.__file__], env=env, check=True, timeout=10
    )
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 2
    rows = [json.loads(path.read_text().splitlines()[0]) for path in files]
    assert len({row["pid"] for row in rows}) == 2
    assert {row["event"] for row in rows} == {"parent", "child"}


@pytest.mark.parametrize("limit", ["0", "-1", "invalid"])
def test_invalid_limit_disables_diagnostic_without_affecting_service(
    tmp_path, monkeypatch, limit
):
    monkeypatch.setattr(tracing, "_trace", None)
    monkeypatch.setenv("VLLM_SAWTOOTH_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SAWTOOTH_TRACE_MAX_EVENTS", limit)
    assert not tracing.get_sawtooth_trace().enabled
    assert not list(tmp_path.iterdir())


def test_command_metadata_contains_only_cpu_version_identities():
    step = SawtoothRequestStep("r", 4, 5, 6, 7, True)
    output = SimpleNamespace(
        sawtooth_execution=SawtoothExecution(2, 1, 2, 8, 0, 3, "plan"),
        sawtooth_request_steps={"r": step},
        num_scheduled_tokens={"r": 1},
        total_num_scheduled_tokens=1,
        sawtooth_has_decode=True,
        scheduled_new_reqs=object(),  # Never inspect/serialize prompt payload.
    )
    context = tracing.command_context(output)
    assert context["command_seq"] == 8
    assert context["has_decode"] is True
    assert context["req_ids"] == ["r"]
    assert context["request_steps"] == [
        {
            "req_id": "r",
            "generation": 4,
            "step_id": 5,
            "token_start": 6,
            "token_end": 7,
            "produces_sample": True,
        }
    ]
    json.dumps(context)

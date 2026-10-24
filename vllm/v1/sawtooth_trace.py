# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, bounded CPU diagnostics; never use these runs as performance results."""

import atexit
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TracedOutput:
    """Process-local queue envelope, removed before any response serialization."""

    output: object
    context: dict


class SawtoothTrace:
    def __init__(self, directory=None, max_events=200_000):
        self.enabled = bool(directory)
        self.directory = directory
        self.max_events = max_events
        self.pid = os.getpid()
        self.started_ns = time.time_ns()
        self.events = 0
        self.dropped = 0
        self.errors = 0
        self._ids = 0
        self._file = None
        self._lock = threading.Lock()

    def next_id(self):
        with self._lock:
            self._ids += 1
            return self._ids

    def _write(self, event, context):
        if self._file is None:
            os.makedirs(self.directory, exist_ok=True)
            path = os.path.join(
                self.directory, f"sawtooth-{self.pid}-{self.started_ns}.jsonl"
            )
            # Keep the process-local handle open until close() emits the summary.
            self._file = open(  # noqa: SIM115
                path, "a", encoding="utf-8", buffering=1
            )
        record = dict(context or {})
        record.update(
            monotonic_ns=time.monotonic_ns(),
            wall_ns=time.time_ns(),
            pid=self.pid,
            thread_id=threading.get_ident(),
            event=event,
        )
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def emit(self, event, context=None):
        if not self.enabled:
            return
        try:
            with self._lock:
                if self.events >= self.max_events:
                    self.dropped += 1
                    if self.dropped == 1:
                        self._write(
                            "trace.cap_reached", {"max_events": self.max_events}
                        )
                    return
                self._write(event, context)
                self.events += 1
        except Exception:
            # Diagnostics must neither fail an RPC nor replace its exception.
            self.errors += 1
            self.enabled = False

    @contextmanager
    def span(self, name, context=None):
        if not self.enabled:
            yield
            return
        fields = dict(context or {}, span_id=self.next_id())
        self.emit(name + ".enter", fields)
        try:
            yield
        except BaseException as error:
            self.emit(
                name + ".exit",
                dict(fields, status="error", error_type=type(error).__name__),
            )
            raise
        else:
            self.emit(name + ".exit", dict(fields, status="ok"))

    def close(self):
        if self.pid != os.getpid():
            return  # An inherited atexit hook must not touch the parent's log.
        try:
            with self._lock:
                if self._file is not None:
                    self._write(
                        "trace.summary",
                        {
                            "written_events": self.events,
                            "dropped_events": self.dropped,
                            "write_errors": self.errors,
                        },
                    )
                    self._file.close()
                    self._file = None
                self.enabled = False
        except Exception:
            self.enabled = False


_trace = None
_init_lock = threading.Lock()


def _after_fork():
    global _trace, _init_lock
    _trace = None
    _init_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def get_sawtooth_trace():
    """Read the two environment options once per process, including after fork."""
    global _trace
    if _trace is None:
        with _init_lock:
            if _trace is None:
                directory = os.environ.get("VLLM_SAWTOOTH_TRACE_DIR")
                try:
                    limit = int(
                        os.environ.get("VLLM_SAWTOOTH_TRACE_MAX_EVENTS", "200000")
                    )
                    if limit < 1:
                        raise ValueError
                except ValueError:
                    directory, limit = None, 200_000
                _trace = SawtoothTrace(directory, limit)
                if _trace.enabled:
                    atexit.register(_trace.close)
    return _trace


def command_context(output):
    """Whitelist CPU identities; never serialize a scheduler payload or tensor."""
    try:
        command = getattr(output, "sawtooth_execution", None)
        if command is None:
            return {"control_only": True}
        steps = getattr(output, "sawtooth_request_steps", None) or {}
        return {
            "batch_id": command.batch_id,
            "slot_id": command.slot_id,
            "slot_generation": command.slot_generation,
            "round_id": command.round_id,
            "command_seq": command.command_seq,
            "num_rounds": command.num_rounds,
            "plan_id": command.plan_id,
            "num_tokens": output.total_num_scheduled_tokens,
            "has_decode": getattr(output, "sawtooth_has_decode", False),
            "num_reqs": len(getattr(output, "num_scheduled_tokens", steps)),
            "req_ids": list(getattr(output, "num_scheduled_tokens", steps)),
            "request_steps": [
                {
                    "req_id": rid,
                    "generation": step.request_generation,
                    "step_id": step.step_id,
                    "token_start": step.token_start,
                    "token_end": step.token_end,
                    "produces_sample": step.produces_sample,
                }
                for rid, step in steps.items()
            ],
        }
    except Exception as error:
        return {"metadata_error": type(error).__name__}

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-plane tests with the executor's lazy FIFO response semantics."""

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm.v1.engine.sawtooth import SawtoothEngine
from vllm.v1.sawtooth import SawtoothRequestLedger, SawtoothRoundSubmitted

pytestmark = pytest.mark.cpu_test


@dataclass
class Output:
    request_id: str
    total_num_scheduled_tokens: int = 1
    sawtooth_execution: object = None
    sawtooth_request_steps: object = None
    sawtooth_has_decode: bool = False


class LazyFuture(Future):
    """Like Multiproc FutureWrapper, done() itself never reads responses."""

    def __init__(self, queue, response):
        super().__init__()
        self.queue = queue
        self.response = response
        queue.append(self)

    def result(self, timeout=None):
        while not self.done():
            prior = self.queue.popleft()
            try:
                prior.set_result(prior.response())
            except Exception as error:
                prior.set_exception(error)
        return super().result(timeout)


class Scheduler:
    def __init__(self, count):
        self.waiting = deque(f"request-{index}" for index in range(count))
        self.skipped_waiting = []
        self.running = []
        self.requests = {}
        self.schedule_calls = 0
        self.available_batches = []
        self.updates = []

    def has_requests(self):
        return bool(self.waiting or self.running)

    def sawtooth_can_schedule(self, request):
        return not request.num_in_flight_tokens

    def sawtooth_has_prefill_candidate(self, *, require_eligible=False):
        return bool(self.waiting)

    def sawtooth_mark_tail_enqueued(self, output):
        pass

    def schedule(self, throttle):
        self.schedule_calls += 1
        self.available_batches.append(self.sawtooth_available_batches)
        request_id = self.waiting.popleft()
        request = SimpleNamespace(
            request_id=request_id,
            num_in_flight_tokens=1,
            is_finished=lambda: False,
        )
        self.running.append(request)
        self.requests[request_id] = request
        return Output(request_id)

    def get_grammar_bitmask(self, output):
        return None

    def update_from_output(self, scheduled, result):
        assert scheduled.request_id == result.request_id
        assert scheduled.request_id not in self.updates
        request = self.requests.pop(scheduled.request_id)
        request.num_in_flight_tokens = 0
        if request in self.running:
            self.running.remove(request)
        self.updates.append(scheduled.request_id)
        return {0: scheduled.request_id}


class Executor:
    def __init__(self):
        self.responses = deque()
        self.commands = []
        self.samples = []
        self.fail_first = False
        self.invalid_ack = False
        self.before_response = lambda: None

    def execute_model(self, output, non_block):
        assert non_block
        self.commands.append(output)
        execution = output.sawtooth_execution

        def response():
            self.before_response()
            if self.fail_first and execution.command_seq == 0:
                raise RuntimeError("stage submission failed")
            if execution.is_last_round:
                return None
            if self.invalid_ack:
                return None
            return SawtoothRoundSubmitted(execution)

        return LazyFuture(self.responses, response)

    def sample_tokens(self, grammar, non_block):
        assert non_block
        output = self.commands[-1]
        assert output.sawtooth_execution.is_last_round
        self.samples.append(output.request_id)
        return LazyFuture(
            self.responses,
            lambda: SimpleNamespace(request_id=output.request_id),
        )


def make_engine(count=4, capacity=2, rounds=3, enable_early_admission=None):
    config = SimpleNamespace(
        additional_config={
            "sawtooth_pipeline": {"enabled": True, "virtual_size": rounds}
        },
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=12),
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
            runner_type="generate",
        ),
        speculative_config=None,
        use_v2_model_runner=True,
        max_concurrent_batches=capacity,
    )
    scheduler = Scheduler(count)
    if enable_early_admission is not None:
        config.additional_config["sawtooth_pipeline"]["enable_early_admission"] = (
            enable_early_admission
        )
    executor = Executor()
    core = SimpleNamespace(
        vllm_config=config,
        is_pooling_model=False,
        scheduler=scheduler,
        model_executor=executor,
        _should_throttle_prefills=lambda: False,
        _process_aborts_queue=lambda: None,
    )
    return SawtoothEngine(core), scheduler, executor


@pytest.mark.parametrize("rounds", [2, 3, 4])
@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize("enable_early_admission", [None, False])
def test_lazy_fifo_futures_drain_without_rescheduling_continuations(
    rounds,
    capacity,
    enable_early_admission,
):
    engine, scheduler, executor = make_engine(
        7, capacity, rounds, enable_early_admission
    )
    for _ in range(100):
        if not scheduler.has_requests() and not engine:
            break
        engine.step()
        assert len(engine.active) <= capacity
    assert not engine
    assert scheduler.schedule_calls == 7
    assert len(scheduler.updates) == len(executor.samples) == 7
    assert not executor.responses
    groups = {}
    generations = {}
    for output in executor.commands:
        command = output.sawtooth_execution
        groups.setdefault(command.batch_id, []).append(command.round_id)
        if command.is_first_round:
            previous = generations.get(command.slot_id, 0)
            assert command.slot_generation > previous
            generations[command.slot_id] = command.slot_generation
    assert list(groups.values()) == [list(range(rounds))] * 7
    first_rounds = [
        output.sawtooth_execution.round_id for output in executor.commands[:capacity]
    ]
    assert first_rounds == [0] * capacity


def test_aborting_every_request_still_drains_admitted_rounds():
    engine, scheduler, executor = make_engine(1, 2, 4)
    engine.step()
    scheduler.running.clear()
    assert not scheduler.has_requests()
    assert engine
    for _ in range(5):
        if not engine:
            break
        engine.step()
    assert not engine
    assert len(executor.commands) == 4
    assert scheduler.updates == ["request-0"]
    assert not scheduler.requests


@pytest.mark.parametrize("failure", ["fail_first", "invalid_ack"])
def test_terminal_does_not_commit_when_an_earlier_round_failed(failure):
    engine, scheduler, executor = make_engine(1, 1, 3)
    setattr(executor, failure, True)
    for _ in range(3):
        engine.step()
    with pytest.raises(RuntimeError):
        engine.step()
    assert scheduler.updates == []
    assert not engine.slots


def test_empty_step_returns_scheduler_errors_to_the_client():
    """Grammar compilation failures can finish on a zero-token step."""
    engine, scheduler, executor = make_engine(0)
    scheduler.waiting.append("compilation-pending")
    scheduler.schedule = lambda throttle: Output("empty", 0)
    executor.execute_model = lambda output, non_block: LazyFuture(
        executor.responses, lambda: SimpleNamespace(request_id="empty")
    )
    expected = {0: SimpleNamespace(error="grammar compilation failed")}
    scheduler.update_from_output = lambda scheduled, result: expected
    outputs, executed = engine.step()
    assert outputs == expected
    assert not executed


def test_empty_lifecycle_failure_is_not_silently_lost():
    """An empty first RPC can fail even if a later RPC returns normally."""
    engine, scheduler, executor = make_engine(0)
    scheduler.waiting.append("compilation-pending")
    scheduler.schedule = lambda throttle: Output("empty", 0)
    calls = 0

    def execute(output, non_block):
        nonlocal calls
        calls += 1

        def response():
            if calls_at_submit == 1:
                raise RuntimeError("empty lifecycle failed")
            return SimpleNamespace(request_id="empty")

        calls_at_submit = calls
        return LazyFuture(executor.responses, response)

    executor.execute_model = execute
    scheduler.update_from_output = lambda scheduled, result: {}
    with pytest.raises(RuntimeError, match="empty lifecycle failed"):
        engine.step()


def test_control_failure_before_terminal_prevents_kv_and_output_commit():
    engine, scheduler, executor = make_engine(1, 2, 2)
    engine.step()
    scheduler.waiting.append("blocked")
    scheduler.schedule = lambda throttle: Output("empty", 0)
    execute = executor.execute_model

    def with_failed_control(output, non_block):
        if output.sawtooth_execution is not None:
            return execute(output, non_block)

        def response():
            raise RuntimeError("earlier control failed")

        return LazyFuture(executor.responses, response)

    executor.execute_model = with_failed_control
    with pytest.raises(RuntimeError, match="earlier control failed"):
        for _ in range(4):
            engine.step()
    assert scheduler.updates == []


@pytest.mark.parametrize("rounds", [2, 3, 4])
@pytest.mark.parametrize("count", [1, 2, 7])
@pytest.mark.parametrize("enable_early_admission", [None, False])
def test_cohort_refills_before_new_tails_without_early_host_wait(
    rounds,
    count,
    enable_early_admission,
):
    """PP2's three slots refill in FIFO order before another cohort's tails."""
    capacity = 3
    engine, scheduler, executor = make_engine(
        count, capacity, rounds, enable_early_admission
    )
    trace = []

    def check_no_early_wait():
        # Called on actual FIFO response reads, not just Future.done(). Every
        # member of the old cohort must already have its terminal submitted.
        limit = engine.phase_limit
        assert limit is not None
        assert all(
            batch.terminal_future is not None
            for batch in engine.active
            if batch.batch_id <= limit
        )

    executor.before_response = check_no_early_wait
    for _ in range(100):
        if not scheduler.has_requests() and not engine:
            break
        commands_before = len(executor.commands)
        updates_before = len(scheduler.updates)
        engine.step()
        trace.extend(
            (
                "dispatch",
                out.sawtooth_execution.batch_id,
                out.sawtooth_execution.round_id,
            )
            for out in executor.commands[commands_before:]
        )
        trace.extend(
            ("retire", int(request_id.split("-")[1]))
            for request_id in scheduler.updates[updates_before:]
        )
        assert len(engine.active) + len(engine.slots) == capacity

    expected = [("dispatch", batch, 0) for batch in range(min(capacity, count))]
    for start in range(0, count, capacity):
        cohort = range(start, min(start + capacity, count))
        expected.extend(
            ("dispatch", batch, round_id)
            for round_id in range(1, rounds)
            for batch in cohort
        )
        for batch in cohort:
            expected.append(("retire", batch))
            if batch + capacity < count:
                expected.append(("dispatch", batch + capacity, 0))
    assert trace == expected
    assert not executor.responses
    assert scheduler.available_batches == [
        capacity - batch if batch < capacity else 1 for batch in range(count)
    ]


def test_arrival_during_multiround_cohort_cannot_block_older_continuations():
    """A deferred new tail can sit ahead of old rounds in the round queue."""
    engine, scheduler, executor = make_engine(1, 3, 4)
    engine.step()  # A0
    engine.step()  # A1 freezes cohort A.
    scheduler.waiting.append("request-1")
    engine.step()  # B0, tail deferred even though a free slot remains.
    for _ in range(20):
        if not scheduler.has_requests() and not engine:
            break
        engine.step()
    assert [
        (output.sawtooth_execution.batch_id, output.sawtooth_execution.round_id)
        for output in executor.commands
    ] == [(0, 0), (0, 1), (1, 0), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    assert scheduler.updates == ["request-0", "request-1"]


def test_other_batch_fifo_failure_prevents_terminal_commit():
    """A terminal's FIFO read must not silently consume another batch's error."""
    engine, scheduler, executor = make_engine(2, 3, 2)
    execute = executor.execute_model

    def fail_second_batch(output, non_block):
        command = output.sawtooth_execution
        if command.batch_id == 1 and command.round_id == 0:

            def failed_response():
                raise RuntimeError("second batch submission failed")

            return LazyFuture(executor.responses, failed_response)
        return execute(output, non_block)

    executor.execute_model = fail_second_batch
    for _ in range(4):
        engine.step()
    with pytest.raises(RuntimeError, match="second batch submission failed"):
        engine.step()
    assert not scheduler.updates


def test_empty_controls_do_not_force_wait_before_cohort_rounds_are_submitted():
    engine, scheduler, executor = make_engine(1, 3, 4)
    engine.step()
    scheduler.waiting.append("blocked")
    scheduler.schedule = lambda throttle: Output("empty", 0)
    execute = executor.execute_model
    update = scheduler.update_from_output
    controls = []

    def with_control(output, non_block):
        if output.sawtooth_execution is not None:
            return execute(output, non_block)

        def response():
            assert len(executor.commands) == 4
            return SimpleNamespace(request_id="empty")

        future = LazyFuture(executor.responses, response)
        controls.append(future)
        return future

    def with_control_update(output, result):
        if output.request_id == "empty":
            scheduler.waiting.clear()
            return {}
        return update(output, result)

    executor.execute_model = with_control
    scheduler.update_from_output = with_control_update
    for _ in range(20):
        if not scheduler.has_requests() and not engine:
            break
        engine.step()
    assert scheduler.updates == ["request-0"]
    assert controls and all(future.done() for future in controls)
    assert not engine


class ChainedScheduler(Scheduler):
    """One request, two independent input steps and terminal-owned slots."""

    def __init__(self):
        super().__init__(1)
        self.ledger = SawtoothRequestLedger()
        self.admitted = 0

    def sawtooth_can_schedule(self, request):
        return self.admitted < 2 and (
            not request.num_in_flight_tokens
            or (
                self.sawtooth_allow_early_admission
                and self.ledger.can_advance(request.request_id)
            )
        )

    def schedule(self, throttle):
        if not self.running:
            rid = self.waiting.popleft()
            request = SimpleNamespace(
                request_id=rid, num_in_flight_tokens=0, is_finished=lambda: False
            )
            self.running.append(request)
            self.requests[rid] = request
        request = self.running[0]
        step = self.ledger.admit(request.request_id, self.admitted, 1, True)
        request.num_in_flight_tokens += 1
        self.admitted += 1
        return Output(
            request.request_id, sawtooth_request_steps={request.request_id: step}
        )

    def sawtooth_mark_tail_enqueued(self, output):
        for step in output.sawtooth_request_steps.values():
            self.ledger.mark_tail_enqueued(step)

    def update_from_output(self, output, result):
        request = self.requests[output.request_id]
        request.num_in_flight_tokens -= 1
        self.ledger.settle(
            output.sawtooth_request_steps[output.request_id],
            request.num_in_flight_tokens,
        )
        self.updates.append(output.request_id)
        if len(self.updates) == 2:
            self.running.clear()
            self.requests.clear()
        return {0: output.request_id}


def test_same_request_prefill_heads_precede_both_tails_without_terminal_wait():
    class PrefillScheduler(ChainedScheduler):
        def __init__(self):
            super().__init__()
            self.sawtooth_request_ledger = self.ledger

        def sawtooth_can_schedule(self, request):
            return self.admitted < 2 and (
                not request.num_in_flight_tokens
                or self.ledger.can_overlap_prefill(request.request_id)
            )

        def sawtooth_mark_head_enqueued(self, output):
            for step in output.sawtooth_request_steps.values():
                self.ledger.mark_head_enqueued(step)

        def schedule(self, throttle):
            if not self.running:
                rid = self.waiting.popleft()
                request = SimpleNamespace(
                    request_id=rid, num_in_flight_tokens=0,
                    is_finished=lambda: False,
                )
                self.running.append(request)
                self.requests[rid] = request
            request = self.running[0]
            step = self.ledger.admit(
                request.request_id, self.admitted, 1, self.admitted == 1
            )
            request.num_in_flight_tokens += 1
            self.admitted += 1
            return Output(
                request.request_id,
                sawtooth_request_steps={request.request_id: step},
            )

    initial, _, executor = make_engine(1, 2, 2)
    core = initial.core
    core.vllm_config.additional_config["sawtooth_pipeline"][
        "enable_prefill_chunk_overlap"
    ] = True
    core.scheduler = scheduler = PrefillScheduler()
    engine = SawtoothEngine(core)
    reads = []
    executor.before_response = lambda: reads.append(True)
    for _ in range(4):
        engine.step()
    assert [
        (out.sawtooth_execution.batch_id, out.sawtooth_execution.round_id)
        for out in executor.commands
    ] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert not reads
    engine.step()
    engine.step()
    assert len(scheduler.updates) == 2 and not engine


@pytest.mark.parametrize("rounds", [2, 3, 4])
@pytest.mark.parametrize("enable_early_admission", [None, True])
def test_same_request_head_uses_free_slot_before_predecessor_terminal(
    rounds,
    enable_early_admission,
):
    engine, _, executor = make_engine(1, 2, rounds, enable_early_admission)
    scheduler = ChainedScheduler()
    engine.core.scheduler = scheduler
    reads = []
    executor.before_response = lambda: reads.append(True)
    for _ in range(rounds + 1):
        engine.step()
    commands = [o.sawtooth_execution for o in executor.commands]
    assert [(c.batch_id, c.round_id) for c in commands] == [
        (0, r) for r in range(rounds)
    ] + [(1, 0)]
    assert not reads and not scheduler.updates
    assert len(engine.active) == 2 and not engine.slots
    assert commands[0].slot_id != commands[-1].slot_id
    assert all(c.slot_generation == 1 for c in commands)
    for _ in range(20):
        if not engine and not scheduler.has_requests():
            break
        engine.step()
    assert len(scheduler.updates) == 2
    assert not executor.responses and not engine


@pytest.mark.parametrize("rounds", [2, 3, 4])
def test_disabled_early_admission_waits_for_same_request_terminal(rounds):
    engine, _, executor = make_engine(1, 2, rounds, False)
    scheduler = ChainedScheduler()
    engine.core.scheduler = scheduler
    for _ in range(rounds):
        engine.step()
    assert len(executor.commands) == rounds
    assert not scheduler.updates and scheduler.admitted == 1
    assert len(engine.slots) == 1  # A spare slot must not bypass the fallback.
    engine.step()
    assert scheduler.updates == ["request-0"]
    assert scheduler.admitted == 1 and len(executor.commands) == rounds
    assert scheduler.sawtooth_allow_early_admission is False
    engine.step()
    assert scheduler.admitted == 2
    assert executor.commands[-1].sawtooth_execution.batch_id == 1
    assert executor.commands[-1].sawtooth_execution.round_id == 0
    for _ in range(20):
        if not engine and not scheduler.has_requests():
            break
        engine.step()
    assert len(scheduler.updates) == 2
    assert not executor.responses and not engine


def test_failed_sample_enqueue_does_not_certify_a_predecessor():
    engine, _, executor = make_engine(1, 2, 2)
    scheduler = ChainedScheduler()
    engine.core.scheduler = scheduler
    engine.step()

    def fail(*args, **kwargs):
        raise RuntimeError("sample enqueue failed")

    executor.sample_tokens = fail
    with pytest.raises(RuntimeError, match="sample enqueue failed"):
        engine.step()
    assert not scheduler.ledger.can_advance("request-0")
    assert scheduler.admitted == 1


def test_trace_pairs_dispatch_sample_and_terminal_without_changing_order(
    tmp_path, monkeypatch
):
    import json

    import vllm.v1.engine.sawtooth as engine_module
    from vllm.v1.sawtooth_trace import SawtoothTrace

    trace = SawtoothTrace(str(tmp_path))
    monkeypatch.setattr(engine_module, "get_sawtooth_trace", lambda: trace)
    engine, scheduler, executor = make_engine(2, 2, 2)
    for _ in range(20):
        if not scheduler.has_requests() and not engine:
            break
        engine.step()
    trace.close()
    (path,) = tmp_path.glob("*.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    dispatch = [r for r in rows if r["event"] == "engine.dispatch.enter"]
    assert [r["command_seq"] for r in dispatch] == [0, 1, 2, 3]
    assert [(r["batch_id"], r["round_id"]) for r in dispatch] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]
    for event, count in (
        ("dispatch", 4),
        ("sample_enqueue", 2),
        ("retire_wait", 2),
        ("retire_commit", 2),
    ):
        starts = [r for r in rows if r["event"] == f"engine.{event}.enter"]
        ends = [r for r in rows if r["event"] == f"engine.{event}.exit"]
        assert len(starts) == len(ends) == count
        assert [r["span_id"] for r in starts] == [r["span_id"] for r in ends]
        assert all(r["status"] == "ok" for r in ends)
    assert scheduler.updates == ["request-0", "request-1"]
    assert not executor.responses


def test_engine_trace_off_does_not_construct_request_metadata(monkeypatch):
    import vllm.v1.engine.sawtooth as engine_module
    from vllm.v1.sawtooth_trace import SawtoothTrace

    monkeypatch.setattr(engine_module, "get_sawtooth_trace", lambda: SawtoothTrace())

    def forbidden(output):
        raise AssertionError("Disabled tracing constructed metadata")

    monkeypatch.setattr(engine_module, "command_context", forbidden)
    engine, scheduler, _ = make_engine(1, 2, 2)
    for _ in range(10):
        if not scheduler.has_requests() and not engine:
            break
        engine.step()
    assert scheduler.updates == ["request-0"]


class AdmissionScheduler(Scheduler):
    """Running-first heads with independent D and cold waiting P candidates."""

    def __init__(self, decode_count=0):
        super().__init__(0)
        self.ready_decodes = deque()
        for index in range(decode_count):
            rid = f"decode-{index}"
            self.ready_decodes.append(rid)
            request = SimpleNamespace(
                request_id=rid, num_in_flight_tokens=0, is_finished=lambda: False
            )
            self.running.append(request)
            self.requests[rid] = request
        self.modes = []

    def sawtooth_has_decode_candidate(self):
        return bool(self.ready_decodes)

    def sawtooth_has_prefill_candidate(self, *, require_eligible=False):
        return bool(self.waiting)

    def schedule(self, throttle):
        self.schedule_calls += 1
        self.available_batches.append(self.sawtooth_available_batches)
        decode_only = self.sawtooth_decode_only
        prefill_only = self.sawtooth_prefill_only
        assert not (decode_only and prefill_only)
        self.modes.append((decode_only, prefill_only))
        if not prefill_only and self.ready_decodes:
            rid = self.ready_decodes.popleft()
            request = self.requests[rid]
            request.num_in_flight_tokens = 1
            return Output(rid, sawtooth_has_decode=True)
        assert not decode_only
        rid = self.waiting.popleft()
        request = SimpleNamespace(
            request_id=rid, num_in_flight_tokens=1, is_finished=lambda: False
        )
        self.running.append(request)
        self.requests[rid] = request
        return Output(rid)


def command_order(executor):
    return [
        (out.request_id, out.sawtooth_execution.round_id) for out in executor.commands
    ]


@pytest.mark.parametrize("rounds", [2, 3, 4])
def test_fresh_prefill_follows_admitted_decode_tail_without_terminal_wait(rounds):
    engine, _, executor = make_engine(0, 3, rounds)
    scheduler = AdmissionScheduler(1)
    engine.core.scheduler = scheduler
    reads = []
    executor.before_response = lambda: reads.append(True)
    engine.step()  # D0, no cohort frozen yet.
    scheduler.waiting.append("fresh-prefill")
    for _ in range(rounds):
        engine.step()
    assert command_order(executor) == [("decode-0", r) for r in range(rounds)] + [
        ("fresh-prefill", 0)
    ]
    assert executor.samples == ["decode-0"]
    assert not reads and not scheduler.updates
    assert scheduler.modes == [(False, False), (False, True)]
    assert len({b.slot_id for b in engine.active}) == 2


@pytest.mark.parametrize("rounds", [2, 3, 4])
@pytest.mark.parametrize("capacity", [2, 3, 5])
def test_decode_cohort_keeps_initial_and_refill_window_order(rounds, capacity):
    count = 2 * capacity + 1
    engine, _, executor = make_engine(0, capacity, rounds)
    scheduler = AdmissionScheduler(count)
    engine.core.scheduler = scheduler
    events = []
    for _ in range(200):
        if not scheduler.has_requests() and not engine:
            break
        before_commands = len(executor.commands)
        before_updates = len(scheduler.updates)
        engine.step()
        events.extend(
            ("dispatch", out.request_id, out.sawtooth_execution.round_id)
            for out in executor.commands[before_commands:]
        )
        events.extend(("retire", rid) for rid in scheduler.updates[before_updates:])
    expected = [("dispatch", f"decode-{b}", 0) for b in range(capacity)]
    for start in range(0, count, capacity):
        cohort = range(start, min(start + capacity, count))
        expected.extend(
            ("dispatch", f"decode-{b}", r) for r in range(1, rounds) for b in cohort
        )
        for b in cohort:
            expected.append(("retire", f"decode-{b}"))
            if b + capacity < count:
                expected.append(("dispatch", f"decode-{b + capacity}", 0))
    assert events == expected
    assert not executor.responses and not engine


@pytest.mark.parametrize("rounds", [2, 3, 4])
def test_waiting_prefill_gets_a_slot_after_bounded_decode_cohort(rounds):
    engine, _, executor = make_engine(0, 3, rounds)
    scheduler = AdmissionScheduler(30)
    engine.core.scheduler = scheduler
    engine.step()
    scheduler.waiting.append("waiting-prefill")
    # Ready D heads still fill the initial window despite a waiting P.
    engine.step()
    engine.step()
    assert command_order(executor) == [(f"decode-{b}", 0) for b in range(3)]
    for _ in range(3 * (rounds - 1)):
        engine.step()
    assert len(executor.commands) == 3 * rounds
    assert not scheduler.updates
    engine.step()  # No free slot: retire precisely the oldest terminal.
    assert scheduler.updates == ["decode-0"]
    engine.step()  # P-only opportunity must beat the other 27 ready decodes.
    assert command_order(executor)[-1] == ("waiting-prefill", 0)
    assert scheduler.modes[-1] == (False, True)
    assert len(executor.commands) == 3 * rounds + 1


@pytest.mark.parametrize("rounds", [2, 3, 4])
def test_deferred_decode_waits_for_old_phase_but_not_for_its_own_terminal(rounds):
    engine, _, executor = make_engine(1, 3, rounds)
    # Freeze and finish the first P cohort's submissions without retiring it.
    for _ in range(rounds):
        engine.step()
    original = engine.core.scheduler
    scheduler = AdmissionScheduler(1)
    scheduler.running.extend(original.running)
    scheduler.requests.update(original.requests)
    engine.core.scheduler = scheduler
    engine.step()  # New D0 is outside the old frozen phase.
    scheduler.waiting.append("next-prefill")
    engine.step()  # Old P terminal must retire before the new D tail is legal.
    assert scheduler.updates == ["request-0"]
    for _ in range(rounds):
        engine.step()
    assert command_order(executor) == [("request-0", r) for r in range(rounds)] + [
        ("decode-0", r) for r in range(rounds)
    ] + [("next-prefill", 0)]
    assert scheduler.updates == ["request-0"]


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_failed_admission_attempt_does_not_retry_or_stop_continuation_progress(mode):
    engine, _, executor = make_engine(0, 3, 2)
    scheduler = AdmissionScheduler(2)
    engine.core.scheduler = scheduler
    engine.step()
    scheduler.waiting.append("waiting-prefill")
    schedule = scheduler.schedule
    attempts = []

    def allocation_blocked(throttle):
        selected = (
            scheduler.sawtooth_decode_only
            if mode == "decode"
            else scheduler.sawtooth_prefill_only
        )
        if selected:
            attempts.append(mode)
            return Output("empty", 0)
        return schedule(throttle)

    execute = executor.execute_model

    def with_control(output, non_block):
        if output.sawtooth_execution is None:
            return LazyFuture(
                executor.responses, lambda: SimpleNamespace(request_id="empty")
            )
        return execute(output, non_block)

    scheduler.schedule = allocation_blocked
    executor.execute_model = with_control
    if mode == "prefill":
        # Suppress the second D until the protected cohort has submitted its
        # tail; otherwise independent D must fill the spare slots first.
        scheduler.ready_decodes.clear()
        engine.step()
    engine.step()
    assert attempts == [mode]
    assert command_order(executor) == [("decode-0", 0), ("decode-0", 1)]
    assert len(engine.controls) == 1
    if mode == "prefill":
        assert not engine.prefill_turn
        assert scheduler.updates == ["decode-0"]
    else:
        assert not scheduler.updates


def test_decode_drain_survives_old_prefill_phase_retirement():
    """A drain started on an old P tail must protect a deferred D cohort too."""
    engine, original, executor = make_engine(1, 3, 4)
    engine.step()  # P0
    engine.step()  # P1 freezes the old cohort.
    scheduler = AdmissionScheduler(1)
    scheduler.running.extend(original.running)
    scheduler.requests.update(original.requests)
    engine.core.scheduler = scheduler
    engine.step()  # D0 is outside the old frozen cohort.
    engine.step()  # P2 starts the protected drain because D still has debt.
    assert engine.decode_draining
    assert engine.phase_limit == 0
    assert command_order(executor)[-1] == ("request-0", 2)

    # Both an independent D and a P arrive with a free slot. Neither may
    # extend the set being drained, including across phase_limit clearing.
    rid = "later-decode"
    request = SimpleNamespace(
        request_id=rid, num_in_flight_tokens=0, is_finished=lambda: False
    )
    scheduler.ready_decodes.append(rid)
    scheduler.running.append(request)
    scheduler.requests[rid] = request
    scheduler.waiting.append("later-prefill")
    engine.step()  # P3, not a new head.
    assert engine.decode_draining
    engine.step()  # Retire old P, making D's cohort legal.
    assert scheduler.updates == ["request-0"]
    assert engine.phase_limit is None
    assert engine.decode_draining
    assert len(engine.slots) == 2
    for _ in range(3):
        engine.step()  # D1, D2, D3; the spare slots stay unfilled.
    assert command_order(executor) == [
        ("request-0", 0),
        ("request-0", 1),
        ("decode-0", 0),
        ("request-0", 2),
        ("request-0", 3),
        ("decode-0", 1),
        ("decode-0", 2),
        ("decode-0", 3),
    ]
    engine.step()  # The P-only opportunity precedes the ready independent D.
    assert command_order(executor)[-1] == ("later-prefill", 0)
    assert scheduler.modes[-1] == (False, True)
    assert scheduler.updates == ["request-0"]  # No wait for D's own terminal.
    assert list(scheduler.ready_decodes) == [rid]
    for _ in range(100):
        if not engine and not scheduler.has_requests():
            break
        engine.step()
    assert scheduler.updates == [
        "request-0",
        "decode-0",
        "later-prefill",
        "later-decode",
    ]
    assert not engine and not executor.responses


def test_failed_strict_decode_control_prevents_terminal_commit():
    """A positive D hint is not an allocation promise or a successful RPC."""
    engine, _, executor = make_engine(0, 3, 2)
    scheduler = AdmissionScheduler(2)
    engine.core.scheduler = scheduler
    engine.step()
    schedule = scheduler.schedule
    attempts = []

    def allocation_blocked(throttle):
        if scheduler.sawtooth_decode_only:
            attempts.append(True)
            return Output("empty", 0)
        return schedule(throttle)

    execute = executor.execute_model
    control_futures = []

    def with_failed_control(output, non_block):
        if output.sawtooth_execution is not None:
            return execute(output, non_block)

        def failed_response():
            raise RuntimeError("strict decode control failed")

        future = LazyFuture(executor.responses, failed_response)
        control_futures.append(future)
        return future

    scheduler.schedule = allocation_blocked
    executor.execute_model = with_failed_control
    engine.step()  # One empty control, then D1; no host response read.
    assert attempts == [True]
    assert command_order(executor) == [("decode-0", 0), ("decode-0", 1)]
    assert not control_futures[0].done()
    assert not scheduler.updates
    with pytest.raises(RuntimeError, match="strict decode control failed"):
        for _ in range(10):
            engine.step()
    assert control_futures[0].done()
    assert not scheduler.updates  # No output/KV commit after the failed RPC.


def test_prefill_only_empty_opportunity_without_active_batch_is_consumed_once():
    """After W=1 retirement, an empty P control must drain without rescheduling."""
    engine, _, executor = make_engine(0, 1, 2)
    scheduler = AdmissionScheduler(1)
    engine.core.scheduler = scheduler
    engine.step()
    scheduler.waiting.append("waiting-prefill")
    engine.step()  # D1, no free slot for the P-only turn yet.
    engine.step()  # Retire D; preserve the pending P-only opportunity.
    assert not engine.active and engine.prefill_turn
    assert scheduler.updates == ["decode-0"]
    attempts = []
    updates = []

    def blocked_prefill(throttle):
        assert scheduler.sawtooth_prefill_only
        assert not scheduler.sawtooth_decode_only
        attempts.append(True)
        return Output("empty", 0)

    def execute_control(output, non_block):
        assert output.sawtooth_execution is None and non_block
        return LazyFuture(
            executor.responses, lambda: SimpleNamespace(request_id="empty")
        )

    def update_control(output, result):
        assert output.request_id == result.request_id == "empty"
        updates.append(True)
        return {0: "metadata-progress"}

    scheduler.schedule = blocked_prefill
    scheduler.update_from_output = update_control
    executor.execute_model = execute_control
    assert engine.step() == ({0: "metadata-progress"}, False)
    assert attempts == updates == [True]
    assert not engine.prefill_turn
    assert not engine and not executor.responses
    assert list(scheduler.waiting) == ["waiting-prefill"]


@pytest.mark.parametrize("rounds", [2, 3, 4])
def test_single_slot_decode_guard_releases_prefill_after_slot_retirement(rounds):
    engine, _, executor = make_engine(0, 1, rounds)
    scheduler = AdmissionScheduler(1)
    engine.core.scheduler = scheduler
    engine.step()
    scheduler.waiting.append("waiting-prefill")
    for _ in range(rounds - 1):
        engine.step()
    assert command_order(executor) == [("decode-0", r) for r in range(rounds)]
    assert not scheduler.updates and not engine.slots
    engine.step()
    assert scheduler.updates == ["decode-0"]
    assert engine.prefill_turn and len(engine.slots) == 1
    engine.step()
    assert command_order(executor)[-1] == ("waiting-prefill", 0)
    assert scheduler.modes[-1] == (False, True)
    for _ in range(20):
        if not engine and not scheduler.has_requests():
            break
        engine.step()
    assert scheduler.updates == ["decode-0", "waiting-prefill"]
    assert not engine and not executor.responses


@pytest.mark.parametrize("overlap", [False, True])
def test_dependent_decode_policy_tracks_prefill_candidates(overlap):
    initial, scheduler, executor = make_engine(0, 2, 2)
    initial.core.vllm_config.additional_config["sawtooth_pipeline"][
        "enable_prefill_chunk_overlap"
    ] = overlap
    scheduler.sawtooth_request_ledger = SawtoothRequestLedger()
    engine = SawtoothEngine(initial.core)
    # A policy hint alone must not fabricate a batch or block engine progress.
    for available in (True, False, True):
        scheduler.sawtooth_has_prefill_candidate = lambda **kwargs: available
        assert engine.step() == ({}, False)
        assert scheduler.sawtooth_defer_dependent_decode == (overlap and available)
    assert not executor.commands

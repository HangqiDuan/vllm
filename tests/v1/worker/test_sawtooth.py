# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.sawtooth import SawtoothRequestStep
from vllm.v1.worker.gpu.sawtooth import KeyedPPHandler, copy_stage_inputs


def test_raw_stage_rows_restore_into_a_larger_graph_bucket():
    buffers = IntermediateTensors(
        {
            "hidden": torch.full((8, 2), 99.0),
            "tokens": torch.full((8,), 99, dtype=torch.int32),
        }
    )
    incoming = IntermediateTensors(
        {
            "hidden": torch.arange(6.0).view(3, 2),
            "tokens": torch.tensor([2, 4, 6], dtype=torch.int32),
        }
    )
    restored = copy_stage_inputs(buffers, incoming, 4)
    for name, source in incoming.tensors.items():
        torch.testing.assert_close(restored[name][:3], source)
        assert not torch.count_nonzero(restored[name][3])
        assert torch.all(buffers[name][4:] == 99)


def test_payload_cannot_exceed_its_graph_bucket():
    with pytest.raises(ValueError, match="does not match"):
        copy_stage_inputs(
            IntermediateTensors({"hidden": torch.zeros(8, 2)}),
            IntermediateTensors({"hidden": torch.zeros(5, 2)}),
            4,
        )


def test_sample_consumption_uses_request_identity_not_virtual_round_count():
    class Native:
        def __init__(self):
            self.queue = deque([None, None])

        def receive(self, batch):
            self.queue[-1] = SimpleNamespace(
                request_id=batch.req_ids[0],
                event=SimpleNamespace(query=lambda: False),
            )
            return True

        def get_prev_sampled_outputs(self, draft_tokens):
            return {"producer": self.queue.popleft().request_id}

    native = Native()
    handler = KeyedPPHandler(native)
    consumed = []
    runner = SimpleNamespace(
        req_states=SimpleNamespace(draft_tokens=None),
        postprocess_sampled=lambda **output: consumed.append(output["producer"]),
    )
    handler.begin_batch(0, ["A"], runner)
    handler.receive(SimpleNamespace(req_ids=["A"]))
    handler.begin_batch(1, ["B"], runner)
    handler.receive(SimpleNamespace(req_ids=["B"]))
    for _ in range(8):
        assert handler.get_prev_sampled_outputs() is None
    assert not consumed
    handler.begin_batch(2, ["A"], runner)
    assert consumed == ["A"]
    assert tuple(handler.pending) == (1,)
    assert len(native.queue) == 2


def ready_fixture(*, last=False):
    waits, updates = [], []
    native = SimpleNamespace(
        queue=deque(),
        is_last_rank=last,
        main_stream=SimpleNamespace(
            record_event=lambda: SimpleNamespace(query=lambda: False),
            wait_event=lambda event: waits.append(event),
        ),
    )

    def receive(batch):
        native.queue[-1] = SimpleNamespace(
            req_ids=batch.req_ids,
            need_sampled_mask=[
                handler.sample_steps[r].produces_sample for r in batch.req_ids
            ],
            event=SimpleNamespace(query=lambda: False),
        )
        return True

    native.receive = receive

    def previous(_):
        slot = native.queue.popleft()
        return {
            "producer": tuple(
                rid
                for rid, sampled in zip(slot.req_ids, slot.need_sampled_mask)
                if sampled
            )
        }

    native.get_prev_sampled_outputs = previous
    handler = KeyedPPHandler(native)
    runner = SimpleNamespace(
        req_states=SimpleNamespace(draft_tokens=None),
        postprocess_sampled=lambda **out: updates.append(out["producer"]),
    )
    return handler, runner, waits, updates


def request_step(rid, step_id, sampled=True, generation=0):
    return SawtoothRequestStep(rid, generation, step_id, step_id, step_id + 1, sampled)


def publish(handler, runner, batch_id, steps):
    handler.begin_batch(batch_id, steps, runner, steps)
    handler.sample_steps = steps
    if not handler.native.is_last_rank and any(
        s.produces_sample for s in steps.values()
    ):
        handler.receive(SimpleNamespace(req_ids=list(steps)))
    handler.complete_sample()


def test_exact_dependency_survives_partial_producer_batch_consumption():
    handler, runner, waits, updates = ready_fixture()
    publish(handler, runner, 0, {r: request_step(r, 0) for r in ("A", "B")})
    publish(handler, runner, 1, {"A": request_step("A", 1)})
    assert updates == [("A", "B")]
    # A has a newer pending producer, but B still consumes its exact old key,
    # already applied when A needed that batch. It must not apply it twice.
    publish(handler, runner, 2, {"B": request_step("B", 1)})
    assert updates == [("A", "B")]
    handler.begin_batch(3, ["A"], runner, {"A": request_step("A", 2)})
    assert updates == [("A", "B"), ("A",)]
    assert len(waits) == 3


@pytest.mark.parametrize("last", [False, True])
def test_unsampled_prefill_uses_local_state_event_without_receive(last):
    handler, runner, waits, updates = ready_fixture(last=last)
    publish(handler, runner, 0, {"A": request_step("A", 0, sampled=False)})
    event = handler.producers["A"].event
    assert not handler.pending
    handler.begin_batch(1, ["A"], runner, {"A": request_step("A", 1, sampled=False)})
    assert waits == [event]
    assert not updates and not handler.pending


def test_real_native_unsampled_receive_does_not_accumulate_empty_pending():
    import numpy as np

    from vllm.v1.worker.gpu.pp_utils import PPHandler

    handler, runner, _, updates = ready_fixture()
    native = handler.native
    original_queue = deque([None, None])
    native.queue = original_queue
    # Run the real native early-return path without constructing a device
    # handler. No stream or distributed group is needed for unsampled rows.
    native.receive = PPHandler.receive.__get__(native)
    for index in range(20):
        steps = {"A": request_step("A", index, sampled=False)}
        handler.begin_batch(index, steps, runner, steps)
        handler.sample_steps = steps
        batch = SimpleNamespace(
            req_ids=["A"],
            num_computed_tokens_np=np.array([index]),
            num_scheduled_tokens=np.array([1]),
            prefill_len_np=np.array([64]),
        )
        assert handler.receive(batch) is False
        handler.complete_sample()
        assert native.queue is original_queue
        assert list(native.queue) == [None, None]
        assert not handler.pending
    assert not updates


@pytest.mark.parametrize("wrong", ["missing", "step", "generation"])
def test_successor_cannot_guess_sample_dependencies_from_request_id(wrong):
    handler, runner, _, _ = ready_fixture()
    if wrong != "missing":
        publish(handler, runner, 0, {"A": request_step("A", 0)})
    step = request_step(
        "A", 2 if wrong == "step" else 1, generation=int(wrong == "generation")
    )
    with pytest.raises(RuntimeError, match="exact sawtooth"):
        handler.begin_batch(1, ["A"], runner, {"A": step})


def test_cleanup_consumes_pending_before_generation_reuse():
    handler, runner, _, updates = ready_fixture()
    publish(handler, runner, 0, {"A": request_step("A", 0)})
    handler.begin_batch(-1, ["A"], runner, {})
    handler.forget_requests({"A"})
    assert updates == [("A",)]
    assert not handler.producers and not handler.current_steps
    publish(handler, runner, 1, {"A": request_step("A", 0, generation=1)})
    assert handler.producers["A"].step.request_generation == 1
    assert handler.retired and handler.retired_events


def test_last_rank_dependency_does_not_postprocess_samples_twice():
    handler, runner, waits, updates = ready_fixture(last=True)
    publish(handler, runner, 0, {"A": request_step("A", 0)})
    publish(handler, runner, 1, {"A": request_step("A", 1)})
    assert len(waits) == 1 and not updates and not handler.pending


@pytest.mark.parametrize("next_b_sampled", [False, True])
def test_no_sample_row_can_advance_while_a_peer_row_still_needs_tokens(next_b_sampled):
    handler, runner, _, updates = ready_fixture()
    publish(
        handler,
        runner,
        0,
        {
            "A": request_step("A", 0),
            "B": request_step("B", 0, sampled=False),
        },
    )
    assert handler.pending[0][0] == ("A",)
    publish(handler, runner, 1, {"B": request_step("B", 1, sampled=next_b_sampled)})
    assert not updates
    # The real native postprocess masks B in batch0. Its newer producer must
    # neither be rejected nor marked applied by consuming A's old token batch.
    handler.begin_batch(2, ["A"], runner, {"A": request_step("A", 1)})
    assert updates == [("A",)]
    assert handler.producers["B"].step.step_id == 1
    assert handler.producers["B"].applied is (not next_b_sampled)
    assert 0 not in handler.pending

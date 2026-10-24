# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical-step sampled-token dependencies for an interleaved pipeline."""

from collections import deque
from dataclasses import dataclass

from vllm.v1.sawtooth import SawtoothRequestStep


def copy_stage_inputs(buffers, incoming, padded_tokens):
    """Restore raw transport rows and clear the graph-only padding tail."""
    tensors = {}
    for name, buffer in buffers.tensors.items():
        source = incoming.tensors[name]
        target = buffer[:padded_tokens]
        actual = source.shape[0]
        if actual > padded_tokens or source.shape[1:] != target.shape[1:]:
            raise ValueError("Pipeline payload does not match the graph input")
        target[:actual].copy_(source, non_blocking=True)
        if actual < padded_tokens:
            target[actual:].zero_()
        tensors[name] = target
    return tensors


@dataclass
class _ProducerReady:
    step: SawtoothRequestStep
    batch_id: int
    event: object
    applied: bool


class KeyedPPHandler:
    """Reuse native broadcasts, matching receives to their logical producers."""

    def __init__(self, native):
        self.native = native
        self.batch_id = None
        self.pending = {}
        self.retired = []
        self.producers: dict[str, _ProducerReady] = {}
        self.current_steps: dict[str, SawtoothRequestStep] = {}
        self.sample_steps: dict[str, SawtoothRequestStep] | None = None
        self.retired_events: dict[int, object] = {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _consume(self, key, runner):
        req_ids, pending = self.pending[key]
        for rid in req_ids:
            producer = self.producers.get(rid)
            if producer is not None and producer.batch_id != key:
                raise RuntimeError("Older samples would overwrite a newer request step")
        original_queue = self.native.queue
        try:
            self.native.queue = deque([pending])
            outputs = self.native.get_prev_sampled_outputs(
                runner.req_states.draft_tokens
            )
        finally:
            self.native.queue = original_queue
        if outputs is not None:
            runner.postprocess_sampled(**outputs)
        for rid in req_ids:
            producer = self.producers.get(rid)
            if producer is not None:
                producer.applied = True
        self.retired.append(pending)
        del self.pending[key]

    def begin_batch(self, batch_id, request_ids, runner, steps=None):
        self.batch_id = batch_id
        wanted = set(request_ids)
        waited_events = set()
        if steps is not None:
            for rid, step in steps.items():
                producer = self.producers.get(rid)
                if step.predecessor is None:
                    if producer is not None or rid in self.current_steps:
                        raise RuntimeError("Request generation reused without cleanup")
                else:
                    if producer is None or producer.step.key != step.predecessor:
                        raise RuntimeError(
                            "Missing exact sawtooth sample/state predecessor"
                        )
                    if not producer.applied:
                        self._consume(producer.batch_id, runner)
                    # This includes no-sample prompt chunks: their local count
                    # update is sufficient, without a last-rank broadcast.
                    if id(producer.event) not in waited_events:
                        self.native.main_stream.wait_event(producer.event)
                        waited_events.add(id(producer.event))
                self.current_steps[rid] = step
            # Remaining IDs are lifecycle cleanup, not wildcard dependencies.
            wanted.difference_update(steps)
        for key, (req_ids, _) in tuple(self.pending.items()):
            if wanted.intersection(req_ids):
                self._consume(key, runner)
        # query() is nonblocking. Retain side-stream storage until its receive
        # event completes, including entries discarded by request generation.
        self.retired = [slot for slot in self.retired if not slot.event.query()]
        self.retired_events = {
            key: event
            for key, event in self.retired_events.items()
            if not event.query()
        }

    def complete_sample(self):
        """Called after sample RPC enqueues all persistent state updates."""
        steps = self.sample_steps
        if steps is None:
            raise RuntimeError("Sawtooth sample has no final execution identity")
        event = self.native.main_stream.record_event()
        for rid, step in steps.items():
            if self.current_steps.get(rid) != step:
                raise RuntimeError("Sampling a superseded sawtooth input version")
            previous = self.producers.get(rid)
            if previous is not None:
                if previous.step.key != step.predecessor or not previous.applied:
                    raise RuntimeError("Unconsumed or duplicate sawtooth producer")
                self.retired_events[id(previous.event)] = previous.event
            # On the last rank postprocess_sampled already ran. On other ranks
            # only rows producing samples need deferred token writeback.
            applied = self.native.is_last_rank or not step.produces_sample
            if not applied and (
                self.batch_id not in self.pending
                or rid not in self.pending[self.batch_id][0]
            ):
                raise RuntimeError("Sampled step has no pending token receive")
            self.producers[rid] = _ProducerReady(step, self.batch_id, event, applied)
        self.sample_steps = None

    def forget_requests(self, request_ids):
        for rid in request_ids:
            producer = self.producers.pop(rid, None)
            if producer is not None:
                if not producer.applied:
                    raise RuntimeError(
                        "Freeing a request before sample dependency cleanup"
                    )
                self.retired_events[id(producer.event)] = producer.event
            self.current_steps.pop(rid, None)

    def get_prev_sampled_outputs(self, *args, **kwargs):
        # Virtual execution commands never advance the native PP-size FIFO.
        return None

    def receive(self, input_batch):
        if self.batch_id is None:
            raise RuntimeError("Sampling has no logical pipeline producer")
        if self.batch_id in self.pending:
            raise RuntimeError("Logical pipeline step received samples twice")
        original_queue = self.native.queue
        try:
            self.native.queue = deque([None])
            result = self.native.receive(input_batch)
            pending = self.native.queue[-1]
        finally:
            self.native.queue = original_queue
        if pending is not None:
            mask = getattr(pending, "need_sampled_mask", None)
            if mask is None and self.sample_steps is not None:
                mask = [
                    self.sample_steps[rid].produces_sample
                    for rid in input_batch.req_ids
                ]
            if (
                mask is not None
                and self.sample_steps is not None
                and any(
                    bool(sampled) != self.sample_steps[rid].produces_sample
                    for rid, sampled in zip(input_batch.req_ids, mask)
                )
            ):
                raise RuntimeError("Sample dependency differs from its input version")
            # Native postprocess masks no-sample rows. Those requests can have
            # already advanced using their local event, independently of this
            # batch's sampled rows; never mark or validate them as token writes.
            req_ids = tuple(
                rid
                for index, rid in enumerate(input_batch.req_ids)
                if mask is None or mask[index]
            )
            self.pending[self.batch_id] = (req_ids, pending)
        return result

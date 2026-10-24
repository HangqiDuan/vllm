# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded interleaved PP dispatch without re-scheduling virtual rounds."""

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

from vllm.distributed.sawtooth import get_sawtooth_plan
from vllm.v1.sawtooth import SawtoothExecution, SawtoothRoundSubmitted
from vllm.v1.sawtooth_trace import command_context, get_sawtooth_trace


@dataclass
class LogicalBatch:
    batch_id: int
    slot_id: int
    generation: int
    scheduler_output: Any
    next_round: int = 1
    exec_future: Any = None
    terminal_future: Any = None
    command_futures: list = field(default_factory=list)


class SawtoothEngine:
    def __init__(self, core):
        self.core = core
        self.trace = get_sawtooth_trace()
        config = core.vllm_config
        self.plan = get_sawtooth_plan(
            config, config.model_config.hf_text_config.num_hidden_layers
        )
        assert self.plan is not None
        if not config.use_v2_model_runner:
            raise ValueError("Sawtooth requires Model Runner V2")
        if core.is_pooling_model:
            raise ValueError("Sawtooth requires a generation model")
        self.capacity = config.max_concurrent_batches
        self.slots = deque(range(self.capacity))
        self.generations = [0] * self.capacity
        self.active: deque[LogicalBatch] = deque()
        self.rounds: deque[LogicalBatch] = deque()
        self.controls = deque()
        self.next_batch_id = 0
        self.command_seq = 0
        self.phase_limit: int | None = None
        self.decode_draining = False
        self.prefill_turn = False

    def __bool__(self):
        return bool(self.active or self.controls)

    def _dispatch(self, batch, round_id):
        command = SawtoothExecution(
            batch.batch_id,
            round_id,
            self.plan.virtual_size,
            self.command_seq,
            batch.slot_id,
            batch.generation,
            self.plan.plan_id,
        )
        self.command_seq += 1
        output = replace(batch.scheduler_output, sawtooth_execution=command)
        context = command_context(output) if self.trace.enabled else None
        if context is not None:
            context.update(
                phase_limit=self.phase_limit,
                free_slots=len(self.slots),
                active_batches=len(self.active),
                decode_draining=self.decode_draining,
                prefill_turn=self.prefill_turn,
            )
        with self.trace.span("engine.dispatch", context):
            batch.exec_future = self.core.model_executor.execute_model(
                output, non_block=True
            )
        batch.command_futures.append((command, batch.exec_future))
        if command.is_last_round:
            grammar = self.core.scheduler.get_grammar_bitmask(batch.scheduler_output)
            with self.trace.span("engine.sample_enqueue", context):
                batch.terminal_future = self.core.model_executor.sample_tokens(
                    grammar, non_block=True
                )
            # Every worker receives final execute + sample before a successor's
            # round zero on the same FIFO. This is not a device completion.
            self.core.scheduler.sawtooth_mark_tail_enqueued(batch.scheduler_output)

    def _can_admit(self):
        scheduler = self.core.scheduler
        if not self.slots or not scheduler.has_requests():
            return False
        if scheduler.sawtooth_decode_only:
            return scheduler.sawtooth_has_decode_candidate()
        if scheduler.sawtooth_prefill_only:
            return scheduler.sawtooth_has_prefill_candidate()
        return bool(scheduler.waiting or scheduler.skipped_waiting) or any(
            scheduler.sawtooth_can_schedule(req) and not req.is_finished()
            for req in scheduler.running
        )

    def _retire(self):
        batch = self.active.popleft()
        context = None
        if self.trace.enabled:
            context = command_context(
                replace(
                    batch.scheduler_output,
                    sawtooth_execution=batch.command_futures[-1][0],
                )
            )
            context.update(phase_limit=self.phase_limit, free_slots=len(self.slots))
        with self.trace.span("engine.retire_wait", context):
            output = batch.terminal_future.result()
        for submitted in (batch, *self.active):
            for command, future in submitted.command_futures:
                if not future.done():
                    continue
                acknowledgement = future.result()
                if not command.is_last_round and (
                    not isinstance(acknowledgement, SawtoothRoundSubmitted)
                    or acknowledgement.execution != command
                ):
                    raise RuntimeError("Mismatched virtual pipeline submission")
        # Reading a later Multiproc future marks older failed futures done,
        # but does not raise their exceptions. Validate completed submissions
        # (including other batches) and controls before publishing KV or tokens.
        for _, future in self.controls:
            if not future.done():
                break
            future.result()
        if output is None:
            batch.exec_future.result()
            raise RuntimeError("Final sawtooth sampling returned no result")
        with self.trace.span("engine.retire_commit", context):
            self.core._process_aborts_queue()
            result = self.core.scheduler.update_from_output(
                batch.scheduler_output, output
            )
            self.slots.append(batch.slot_id)
            if batch.batch_id == self.phase_limit:
                self.phase_limit = None
        return result, True

    def _retire_control(self):
        output, future = self.controls.popleft()
        result = future.result()
        if result is None:
            raise RuntimeError("Metadata-only pipeline command returned no output")
        self.core._process_aborts_queue()
        return self.core.scheduler.update_from_output(output, result), False

    def _schedule(self):
        self.core.scheduler.sawtooth_available_batches = len(self.slots)
        return self.core.scheduler.schedule(self.core._should_throttle_prefills())

    def step(self):
        # An early dependent head must not insert a sample wait ahead of older
        # ready continuations.
        self.core.scheduler.sawtooth_allow_early_admission = (
            self.plan.config.enable_early_admission
            and (
                not self.rounds
                or (
                    self.phase_limit is not None
                    and all(batch.batch_id > self.phase_limit for batch in self.rounds)
                )
            )
        )
        if self.controls and self.controls[0][1].done():
            return self._retire_control()

        scheduler = self.core.scheduler
        decode_debt = any(
            batch.scheduler_output.sawtooth_has_decode for batch in self.rounds
        )
        if self.decode_draining and not decode_debt:
            self.decode_draining = False
            self.prefill_turn = True
        scheduler.sawtooth_decode_only = False
        scheduler.sawtooth_prefill_only = False
        if not self.decode_draining:
            if decode_debt:
                # Fill spare slots with independent decodes before freezing
                # their cohort. A fresh P head must not precede an admitted
                # decode's still-unsubmitted continuation on the worker FIFO.
                scheduler.sawtooth_decode_only = True
            elif self.prefill_turn:
                if scheduler.sawtooth_has_prefill_candidate():
                    # Running-first auto selection plus the request cap can
                    # otherwise give every slot to D indefinitely. One P-only
                    # attempt supplies a real opportunity at this safe boundary.
                    scheduler.sawtooth_prefill_only = True
                else:
                    self.prefill_turn = False

        if not self.decode_draining and self._can_admit():
            output = self._schedule()
            if scheduler.sawtooth_prefill_only:
                # An empty result can carry native allocation/grammar/cleanup
                # progress. Do not retry schedule or spin until P is accepted.
                self.prefill_turn = False
            if output.total_num_scheduled_tokens:
                slot = self.slots.popleft()
                self.generations[slot] += 1
                batch = LogicalBatch(
                    self.next_batch_id, slot, self.generations[slot], output
                )
                self.next_batch_id += 1
                self.active.append(batch)
                self._dispatch(batch, 0)
                self.rounds.append(batch)
                return None, True
            # Empty scheduling can still carry cleanup/preemption notifications.
            # Its result is not a logical GPU completion and must not retire KV.
            future = self.core.model_executor.execute_model(output, non_block=True)
            self.controls.append((output, future))

        if self.phase_limit is None and self.rounds:
            # Freeze the cohort when its first continuation is submitted. A
            # refill may submit round zero, but its tail must wait until every
            # old terminal has retired and had an opportunity to refill.
            self.phase_limit = self.next_batch_id - 1
        eligible = None
        if self.phase_limit is not None:
            eligible = next(
                (
                    index
                    for index, batch in enumerate(self.rounds)
                    if batch.batch_id <= self.phase_limit
                ),
                None,
            )
        if eligible is not None:
            if decode_debt:
                # No new head may extend this bounded drain. Release it as soon
                # as the existing D tails and sample RPCs have been submitted;
                # never wait for their device or host completion to admit P.
                self.decode_draining = True
            batch = self.rounds[eligible]
            del self.rounds[eligible]
            self._dispatch(batch, batch.next_round)
            batch.next_round += 1
            if batch.next_round < self.plan.virtual_size:
                self.rounds.append(batch)
            return None, True

        if self.active:
            # done() on a Multiproc FutureWrapper does not poll responses.
            # Explicitly drain only after all rounds of this cohort have been
            # submitted; otherwise a host wait can stall the physical pipe.
            return self._retire()
        if self.controls:
            return self._retire_control()

        if self.core.scheduler.has_requests():
            output = self._schedule()
            if output.total_num_scheduled_tokens:
                raise RuntimeError("Schedulable work bypassed sawtooth admission")
            future = self.core.model_executor.execute_model(output, non_block=True)
            result = future.result()
            if result is not None:
                return self.core.scheduler.update_from_output(output, result), False
        return {}, False

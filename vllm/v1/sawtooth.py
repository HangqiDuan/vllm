# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control-plane identities for a logical interleaved pipeline step."""

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SawtoothExecution:
    batch_id: int
    round_id: int
    num_rounds: int
    command_seq: int
    slot_id: int
    slot_generation: int
    plan_id: str

    def __post_init__(self) -> None:
        if self.batch_id < 0 or self.command_seq < 0:
            raise ValueError("Pipeline command identities must be nonnegative")
        if self.num_rounds < 2 or not 0 <= self.round_id < self.num_rounds:
            raise ValueError("Invalid sawtooth virtual round")
        if self.slot_id < 0 or self.slot_generation < 0:
            raise ValueError("Invalid pipeline buffer lease")

    @property
    def is_first_round(self) -> bool:
        return self.round_id == 0

    @property
    def is_last_round(self) -> bool:
        return self.round_id + 1 == self.num_rounds


@dataclass(frozen=True)
class SawtoothRoundSubmitted:
    execution: SawtoothExecution


@dataclass(frozen=True)
class SawtoothRequestStep:
    """Immutable input interval and producer identity, independent of slots."""

    request_id: str
    request_generation: int
    step_id: int
    token_start: int
    token_end: int
    produces_sample: bool

    @property
    def key(self) -> tuple[str, int, int]:
        return self.request_id, self.request_generation, self.step_id

    @property
    def predecessor(self) -> tuple[str, int, int] | None:
        if self.step_id == 0:
            return None
        return self.request_id, self.request_generation, self.step_id - 1


@dataclass
class _RequestSteps:
    generation: int
    scheduled_end: int
    completed_end: int
    next_step: int = 0
    tail_enqueued: int = -1
    live: deque[SawtoothRequestStep] = field(default_factory=deque)


class SawtoothRequestLedger:
    """FIFO token leases. Submission permits compute, never reclamation."""

    def __init__(self):
        self.requests: dict[str, _RequestSteps] = {}
        self.next_generation = 0

    def can_advance(self, request_id: str) -> bool:
        state = self.requests.get(request_id)
        return bool(
            state is not None
            and len(state.live) == 1
            and state.tail_enqueued == state.live[-1].step_id
        )

    def admit(self, request_id, start, count, produces_sample):
        state = self.requests.get(request_id)
        if state is None:
            state = _RequestSteps(self.next_generation, start, start)
            self.next_generation += 1
            self.requests[request_id] = state
        if count <= 0 or state.scheduled_end != start or len(state.live) >= 2:
            raise RuntimeError("Invalid sawtooth request token lease")
        if state.live and not self.can_advance(request_id):
            raise RuntimeError("Sawtooth predecessor has unsent terminal commands")
        step = SawtoothRequestStep(
            request_id,
            state.generation,
            state.next_step,
            start,
            start + count,
            produces_sample,
        )
        state.next_step += 1
        state.scheduled_end = step.token_end
        state.live.append(step)
        return step

    def mark_tail_enqueued(self, step: SawtoothRequestStep) -> None:
        state = self.requests[step.request_id]
        if step not in state.live or step.step_id != state.tail_enqueued + 1:
            raise RuntimeError("Out-of-order sawtooth terminal submission")
        state.tail_enqueued = step.step_id

    def settle(self, step: SawtoothRequestStep, in_flight_after: int) -> None:
        state = self.requests[step.request_id]
        if (
            not state.live
            or state.live[0] != step
            or state.completed_end != step.token_start
            or state.scheduled_end - in_flight_after != step.token_end
        ):
            raise RuntimeError("Non-contiguous sawtooth request completion")
        state.live.popleft()
        state.completed_end = step.token_end

    def forget(self, request_id: str) -> None:
        state = self.requests.get(request_id)
        if state is not None:
            if state.live:
                raise RuntimeError("Cannot release a live sawtooth request")
            del self.requests[request_id]

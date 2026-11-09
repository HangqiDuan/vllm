# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request cohorts fill logical slots without shrinking prefill token chunks.

These use the real scheduler and KV allocator. The cap is an admission policy,
so no model, worker, or simulated pipeline is needed to catch its failure modes.
"""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm.lora.request import LoRARequest
from vllm.v1.request import RequestStatus

from .test_sawtooth_scheduler import result
from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def admission_scheduler(tmp_path):
    config = {
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "vocab_size": 50304,
        "hidden_size": 64,
        "word_embed_proj_dim": 64,
        "ffn_dim": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "max_position_embeddings": 32768,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def build(*, async_scheduling=True, enabled=True, **kwargs):
        sched = create_scheduler(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_num_batched_tokens=kwargs.pop("max_num_batched_tokens", 10240),
            max_model_len=kwargs.pop("max_model_len", 16384),
            pipeline_parallel_size=2,
            use_v2_model_runner=True,
            async_scheduling=async_scheduling,
            **kwargs,
        )
        sched.sawtooth_enabled = enabled
        sched.sawtooth_available_batches = 1
        return sched

    return build


def _tail_engine(admission_scheduler, length, *, capacity=3, max_tokens=1, **kwargs):
    """Real scheduler/KV ownership driven by the engine's lazy FIFO executor."""
    from tests.v1.engine.test_sawtooth_engine import Executor, LazyFuture, make_engine
    from vllm.v1.engine.sawtooth import SawtoothEngine
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

    sched = admission_scheduler(max_num_batched_tokens=1024, **kwargs)
    initial, _, _ = make_engine(count=0, capacity=capacity, rounds=2)
    core = initial.core
    core.scheduler = sched
    core.vllm_config.additional_config["sawtooth_pipeline"][
        "enable_prefill_chunk_overlap"
    ] = True

    class PrefillExecutor(Executor):
        def execute_model(self, output, non_block):
            if output.sawtooth_execution is None:
                return LazyFuture(self.responses, lambda: EMPTY_MODEL_RUNNER_OUTPUT)
            if output.sawtooth_execution.is_first_round:
                for rid, step in output.sawtooth_request_steps.items():
                    blocks = sched.kv_cache_manager.get_blocks(rid).get_block_ids()[0]
                    assert len(blocks) * 16 >= step.token_end
            # Multiprocess RPC serializes admission data before request mutation.
            return super().execute_model(deepcopy(output), non_block)

        def sample_tokens(self, grammar, non_block):
            assert non_block
            output = self.commands[-1]
            assert output.sawtooth_execution.is_last_round
            self.samples.append(output)
            ids = list(output.num_scheduled_tokens)
            return LazyFuture(
                self.responses,
                lambda: ModelRunnerOutput(
                    req_ids=ids,
                    req_id_to_index={rid: i for i, rid in enumerate(ids)},
                    sampled_token_ids=[
                        [7]
                        if output.sawtooth_request_steps[rid].produces_sample
                        else []
                        for rid in ids
                    ],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )

    core.model_executor = executor = PrefillExecutor()
    engine = SawtoothEngine(core)
    (request,) = create_requests(1, num_tokens=length, max_tokens=max_tokens)
    sched.add_request(request)
    return engine, sched, executor, request


def _drain_tail_engine(engine, sched):
    peak_live = 0
    for _ in range(100):
        if not engine and not sched.has_requests():
            return peak_live
        engine.step()
        peak_live = max(peak_live, len(engine.active))
    pytest.fail("Tail pipeline failed to drain")


@pytest.mark.parametrize("capacity", [2, 3])
@pytest.mark.parametrize(
    "length,counts",
    [
        (7168, [1024] * 6 + [512, 512]),
        (3072, [1024] * 2 + [512, 512]),
        (4096, [1024] * 4),
        (6144, [1024] * 6),
        (6656, [1024] * 6 + [256, 256]),
        (2548, [1024, 1024, 250, 250]),
        (2549, [1024, 1024, 251, 250]),
        (2051, [1024, 1024, 2, 1]),
        (2049, [1024, 1024, 1]),
        (1536, [1024, 512]),
        (2, [1, 1]),
        (1, [1]),
    ],
)
def test_unpaired_tail_is_two_contiguous_full_model_steps(
    admission_scheduler, capacity, length, counts
):
    engine, sched, executor, request = _tail_engine(
        admission_scheduler, length, capacity=capacity
    )
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == counts
    steps = [o.sawtooth_request_steps[request.request_id] for o in heads]
    end = 0
    for step, count in zip(steps, counts):
        assert step.token_start == end
        end += count
        assert step.token_end == end
    for output, step in zip(heads, steps):
        new = output.scheduled_new_reqs
        if new:
            assert len(new) == 1 and new[0].req_id == request.request_id
            assert new[0].num_computed_tokens == step.token_start
            assert len(new[0].prefill_token_ids) == length
        else:
            cached = output.scheduled_cached_reqs
            assert cached.req_ids == [request.request_id]
            assert cached.num_computed_tokens == [step.token_start]
    assert end == length
    assert [s.produces_sample for s in steps] == [False] * (len(steps) - 1) + [True]
    assert len({s.request_generation for s in steps}) == 1
    assert request.num_in_flight_tokens == 0
    assert request.num_output_tokens == 1
    assert request.request_id not in sched.requests
    assert request.request_id not in sched.sawtooth_request_ledger.requests
    commands = [o.sawtooth_execution for o in executor.commands]
    for head in heads:
        bid = head.sawtooth_execution.batch_id
        assert [c.round_id for c in commands if c.batch_id == bid] == [0, 1]
    if len(counts) >= 2 and counts[-2] + counts[-1] <= 1024:
        ca, cb = [o.sawtooth_execution.batch_id for o in heads[-2:]]
        order = [(c.batch_id, c.round_id) for c in commands]
        assert order.index((cb, 0)) < order.index((ca, 1))


def test_tail_with_an_independent_waiting_partner_keeps_its_chunk(
    admission_scheduler,
):
    engine, sched, executor, request = _tail_engine(admission_scheduler, 3072)
    # A and B are admitted and their tails enqueued, but have not retired yet.
    for _ in range(4):
        engine.step()
    (partner,) = create_requests(1, num_tokens=512, max_tokens=1, req_ids=["partner"])
    sched.add_request(partner)
    _drain_tail_engine(engine, sched)
    counts = [
        o.num_scheduled_tokens[request.request_id]
        for o in executor.commands
        if o.sawtooth_execution.is_first_round
        and request.request_id in o.num_scheduled_tokens
    ]
    assert counts == [1024, 1024, 1024]


@pytest.mark.parametrize("admit_second", [False, True])
def test_split_tail_abort_retains_kv_until_all_admitted_halves_retire(
    admission_scheduler, admit_second
):
    engine, sched, executor, request = _tail_engine(admission_scheduler, 1023)
    engine.step()
    assert executor.commands[0].total_num_scheduled_tokens == 512
    if admit_second:
        engine.step()
        assert executor.commands[1].total_num_scheduled_tokens == 511
    blocks = [
        b for group in sched.kv_cache_manager.get_blocks(request.request_id).blocks
        for b in group
    ]
    sched.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert request.request_id in sched.requests
    assert all(b.ref_cnt > 0 for b in blocks)
    _drain_tail_engine(engine, sched)
    assert request.num_in_flight_tokens == 0
    assert request.num_output_tokens == 0
    assert request.request_id not in sched.requests
    assert request.request_id not in sched.sawtooth_request_ledger.requests
    assert all(b.ref_cnt == 0 for b in blocks)


def test_split_halves_share_partial_kv_block_without_reusing_token_slots(
    admission_scheduler,
):
    engine, sched, executor, request = _tail_engine(admission_scheduler, 513)
    engine.step()
    first_blocks = sched.kv_cache_manager.get_blocks(
        request.request_id
    ).get_block_ids()[0]
    engine.step()
    second_blocks = sched.kv_cache_manager.get_blocks(
        request.request_id
    ).get_block_ids()[0]
    assert second_blocks[:len(first_blocks)] == first_blocks
    a, b = [o.sawtooth_request_steps[request.request_id] for o in executor.commands]
    assert (a.token_start, a.token_end) == (0, 257)
    assert (b.token_start, b.token_end) == (257, 513)
    slots = [second_blocks[p // 16] * 16 + p % 16 for p in range(513)]
    assert len(set(slots)) == 513
    assert first_blocks[-1] == second_blocks[b.token_start // 16]
    assert request.num_in_flight_tokens == 513 and request.num_output_tokens == 0
    # Both heads and tails precede retirement; settling Ca cannot publish a token.
    engine.step()
    engine.step()
    engine.step()
    assert request.num_in_flight_tokens == 256
    assert request.num_output_tokens == 0
    assert request.request_id in sched.requests
    _drain_tail_engine(engine, sched)
    assert request.num_output_tokens == 1


def test_split_tail_does_not_change_following_decode_steps(admission_scheduler):
    engine, sched, executor, request = _tail_engine(
        admission_scheduler, 1023, max_tokens=3
    )
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [512, 511, 1, 1]
    assert [o.sawtooth_has_decode for o in heads] == [False, False, True, True]
    assert request.num_output_tokens == 3


def test_delayed_second_half_is_not_split_again_after_first_retires(
    admission_scheduler, monkeypatch
):
    engine, sched, executor, request = _tail_engine(admission_scheduler, 1023)
    allocate = sched.kv_cache_manager.allocate_slots

    def delay_second(req, *args, **kwargs):
        if req.num_computed_tokens == 512 and req.num_in_flight_tokens:
            return None
        return allocate(req, *args, **kwargs)

    monkeypatch.setattr(sched.kv_cache_manager, "allocate_slots", delay_second)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [512, 511]
    assert request.num_preemptions == 0


def test_failed_first_allocation_does_not_commit_a_split_marker(
    admission_scheduler, monkeypatch
):
    engine, sched, executor, request = _tail_engine(admission_scheduler, 1023)
    allocate = sched.kv_cache_manager.allocate_slots
    monkeypatch.setattr(sched.kv_cache_manager, "allocate_slots", lambda *a, **k: None)
    engine.step()
    assert not executor.commands
    assert request.request_id not in sched.sawtooth_request_ledger.requests
    monkeypatch.setattr(sched.kv_cache_manager, "allocate_slots", allocate)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [512, 511]


@pytest.mark.parametrize("capacity,overlap", [(1, True), (3, False)])
def test_tail_falls_back_without_two_overlapping_slots(
    admission_scheduler, capacity, overlap
):
    from vllm.v1.engine.sawtooth import SawtoothEngine

    engine, sched, executor, _ = _tail_engine(
        admission_scheduler, 3072, capacity=capacity
    )
    engine.core.vllm_config.additional_config["sawtooth_pipeline"][
        "enable_prefill_chunk_overlap"
    ] = overlap
    engine = SawtoothEngine(engine.core)
    _drain_tail_engine(engine, sched)
    assert [
        o.total_num_scheduled_tokens for o in executor.commands
        if o.sawtooth_execution.is_first_round
    ] == [1024, 1024, 1024]


@pytest.mark.parametrize("last_rank", [False, True])
def test_split_tail_reuses_group_dependencies_without_sample_wait(
    admission_scheduler, last_rank
):
    from tests.v1.worker.test_sawtooth import ready_fixture

    engine, sched, executor, _ = _tail_engine(admission_scheduler, 513)
    _drain_tail_engine(engine, sched)
    handler, runner, waits, updates = ready_fixture(last=last_rank)
    handler.allow_prefill_overlap = True
    for output in executor.commands:
        execution, steps = output.sawtooth_execution, output.sawtooth_request_steps
        if execution.is_first_round:
            handler.begin_batch(execution.batch_id, steps, runner, steps)
        handler.begin_round(execution.round_id, steps)
        handler.complete_round(execution.round_id, steps)
        if execution.is_last_round:
            handler.batch_id = execution.batch_id
            handler.sample_steps = steps
            if not last_rank and any(s.produces_sample for s in steps.values()):
                handler.receive(SimpleNamespace(req_ids=list(steps)))
            handler.complete_sample()
    assert not waits and not updates


def test_failed_split_submission_does_not_publish_tokens_or_release_kv(
    admission_scheduler,
):
    engine, sched, executor, request = _tail_engine(admission_scheduler, 513)
    executor.fail_first = True
    for _ in range(4):
        engine.step()
    with pytest.raises(RuntimeError, match="stage submission failed"):
        engine.step()
    assert request.num_output_tokens == 0
    assert request.num_in_flight_tokens == 513
    assert len(sched.sawtooth_request_ledger.requests[request.request_id].live) == 2


@pytest.mark.parametrize(
    "length,counts", [(3072, [1024, 1024]), (4096, [1024, 1024, 1024])]
)
def test_even_tail_stays_whole_after_an_earlier_chunk_completed_alone(
    admission_scheduler, length, counts
):
    engine, sched, executor, request = _tail_engine(admission_scheduler, length)
    # An earlier chunk completed alone, e.g. before another slot became usable.
    first = sched.schedule()
    assert first.total_num_scheduled_tokens == 1024
    sched.sawtooth_mark_head_enqueued(first)
    sched.sawtooth_mark_tail_enqueued(first)
    sched.update_from_output(first, result(first, sample=False))
    _drain_tail_engine(engine, sched)
    assert [
        o.total_num_scheduled_tokens for o in executor.commands
        if o.sawtooth_execution.is_first_round
    ] == counts
    assert request.num_output_tokens == 1


def test_waiting_request_without_an_admission_slot_is_not_a_partner(
    admission_scheduler,
):
    engine, sched, executor, request = _tail_engine(
        admission_scheduler, 513, max_num_seqs=1
    )
    (waiting,) = create_requests(1, num_tokens=512, max_tokens=1, req_ids=["waiting"])
    sched.add_request(waiting)
    _drain_tail_engine(engine, sched)
    assert [
        o.num_scheduled_tokens[request.request_id]
        for o in executor.commands
        if o.sawtooth_execution.is_first_round
        and request.request_id in o.num_scheduled_tokens
    ] == [257, 256]


def test_tail_partner_keeps_original_batch_packing_during_refill(
    admission_scheduler,
):
    engine, sched, executor, _ = _tail_engine(
        admission_scheduler, 2048, capacity=2
    )
    for _ in range(4):
        engine.step()
    engine.step()  # A retires, B still occupies the other slot.
    for rid, length in [("E", 513), ("F", 511)]:
        (request,) = create_requests(1, num_tokens=length, max_tokens=1, req_ids=[rid])
        sched.add_request(request)
    _drain_tail_engine(engine, sched)
    heads = [
        o for o in executor.commands
        if o.sawtooth_execution.is_first_round and "0" not in o.num_scheduled_tokens
    ]
    assert [o.num_scheduled_tokens for o in heads] == [{"E": 513, "F": 511}]


@pytest.mark.parametrize("refill", [False, True])
@pytest.mark.parametrize("tail_split", [False, True])
def test_many_short_prefills_preserve_original_request_caps(
    admission_scheduler, refill, tail_split
):
    engine, sched, executor, _ = _tail_engine(
        admission_scheduler, 2048 if refill else 8, capacity=2, max_num_seqs=64
    )
    if not tail_split:
        sched._sawtooth_split_tail_tokens = lambda req, start, count, scheduled: count
    if refill:
        for _ in range(5):
            engine.step()
        count = 64
    else:
        count = 63
    for request in create_requests(
        count, num_tokens=8, max_tokens=1,
        req_ids=[f"short-{index}" for index in range(count)],
    ):
        sched.add_request(request)
    _drain_tail_engine(engine, sched)
    cohorts = [
        o.num_scheduled_tokens for o in executor.commands
        if o.sawtooth_execution.is_first_round
        and (not refill or "0" not in o.num_scheduled_tokens)
    ]
    assert [len(c) for c in cohorts] == ([63, 1] if refill else [32, 32])
    assert all(set(c.values()) == {8} for c in cohorts)


def _admit_wave(sched, slots):
    outputs = []
    for free_slots in range(slots, 0, -1):
        sched.sawtooth_available_batches = free_slots
        outputs.append(sched.schedule())
    return outputs


def test_prompt_successor_keeps_disjoint_token_leases_before_first_tail(
    admission_scheduler,
):
    sched = admission_scheduler(max_num_seqs=1)
    sched.sawtooth_request_ledger.allow_prefill_overlap = True
    (request,) = create_requests(1, num_tokens=12288)
    sched.add_request(request)
    first = sched.schedule()
    assert first.total_num_scheduled_tokens == 10240
    assert not sched.sawtooth_can_schedule(request)
    sched.sawtooth_mark_head_enqueued(first)
    assert sched.sawtooth_can_schedule(request)
    second = sched.schedule()
    assert second.total_num_scheduled_tokens == 2048
    a = first.sawtooth_request_steps[request.request_id]
    b = second.sawtooth_request_steps[request.request_id]
    assert (a.token_start, a.token_end, b.token_start, b.token_end) == (
        0, 10240, 10240, 12288
    )
    assert not a.produces_sample and b.produces_sample
    assert request.num_in_flight_tokens == 12288
    assert not sched.sawtooth_can_schedule(request)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("waiting_count", [0, 64])
def test_decode_cohorts_fill_slots_even_with_a_full_running_set(
    admission_scheduler, async_scheduling, waiting_count
):
    sched = admission_scheduler(max_num_seqs=64, async_scheduling=async_scheduling)
    requests = create_requests(64, num_tokens=8)
    for request in requests:
        sched.add_request(request)
    prefill = sched.schedule()
    sched.update_from_output(prefill, result(prefill))
    for request in create_requests(
        waiting_count,
        num_tokens=8,
        req_ids=[f"waiting-{i}" for i in range(waiting_count)],
    ):
        sched.add_request(request)

    outputs = _admit_wave(sched, 4)
    cohorts = [set(output.num_scheduled_tokens) for output in outputs]
    assert [len(cohort) for cohort in cohorts] == [16, 16, 16, 16]
    assert len(set.union(*cohorts)) == 64
    assert all(set(output.num_scheduled_tokens.values()) == {1} for output in outputs)
    assert len(sched.waiting) == waiting_count


def test_prefill_keeps_10240_token_chunk_while_short_requests_fill_other_slots(
    admission_scheduler,
):
    sched = admission_scheduler(max_num_seqs=4)
    lengths = [12288, 3, 5, 7]
    for index, length in enumerate(lengths):
        (request,) = create_requests(1, num_tokens=length, req_ids=[str(index)])
        sched.add_request(request)
    outputs = _admit_wave(sched, 4)
    assert [output.total_num_scheduled_tokens for output in outputs] == [
        10240,
        3,
        5,
        7,
    ]
    assert all(len(output.num_scheduled_tokens) == 1 for output in outputs)
    assert sched.requests["0"].is_prefill_chunk


def test_running_and_waiting_share_one_request_cap(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8)
    for request in create_requests(2, num_tokens=8):
        sched.add_request(request)
    prefill = sched.schedule()
    sched.update_from_output(prefill, result(prefill))
    for request in create_requests(
        6, num_tokens=16, req_ids=[f"new-{i}" for i in range(6)]
    ):
        sched.add_request(request)
    first, second = _admit_wave(sched, 2)
    assert len(first.num_scheduled_tokens) == len(second.num_scheduled_tokens) == 4
    assert first.num_scheduled_tokens["0"] == first.num_scheduled_tokens["1"] == 1
    assert len(first.scheduled_new_reqs) == 2
    assert set(first.num_scheduled_tokens).isdisjoint(second.num_scheduled_tokens)


def test_blocked_waiters_do_not_inflate_available_work(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8)
    requests = create_requests(8, num_tokens=8)
    for index, request in enumerate(requests):
        if index >= 4:
            request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        sched.add_request(request)
    outputs = _admit_wave(sched, 4)
    assert [len(output.num_scheduled_tokens) for output in outputs] == [1] * 4
    assert set.union(*(set(o.num_scheduled_tokens) for o in outputs)) == {
        request.request_id for request in requests[:4]
    }


def test_ready_kv_holding_waiters_count_toward_cohort_size(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8)
    for request in create_requests(8, num_tokens=8):
        sched.add_request(request)
    for _ in range(4):
        sched.kv_holding_waiting.add_request(sched.waiting.pop_request())
    outputs = _admit_wave(sched, 4)
    assert [len(output.num_scheduled_tokens) for output in outputs] == [2] * 4


def test_lora_skip_does_not_consume_successful_admission_capacity(
    admission_scheduler,
):
    sched = admission_scheduler(max_num_seqs=8)
    sched.lora_config = SimpleNamespace(max_loras=1)
    requests = create_requests(5, num_tokens=8)
    for index, request in enumerate(requests):
        adapter = 2 if index == 1 else 1
        request.lora_request = LoRARequest(str(adapter), adapter, "/unused")
        sched.add_request(request)
    sched.sawtooth_available_batches = 3
    output = sched.schedule()
    assert set(output.num_scheduled_tokens) == {"0", "2"}
    assert requests[1] in sched.deferred_waiting


def test_small_cohorts_do_not_preempt_another_live_batch(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=4, num_blocks=5)
    requests = create_requests(4, num_tokens=32, max_tokens=1)
    for request in requests:
        sched.add_request(request)
    sched.sawtooth_available_batches = 4
    first = sched.schedule()
    sched.sawtooth_available_batches = 3
    second = sched.schedule()
    sched.sawtooth_available_batches = 2
    blocked = sched.schedule()
    assert len(first.num_scheduled_tokens) == len(second.num_scheduled_tokens) == 1
    assert not blocked.num_scheduled_tokens
    assert all(request.num_preemptions == 0 for request in requests)
    assert requests[0].num_in_flight_tokens == requests[1].num_in_flight_tokens == 32
    sched.update_from_output(first, result(first))
    resumed = sched.schedule()
    assert len(resumed.num_scheduled_tokens) == 1
    assert requests[1].num_in_flight_tokens == 32


def test_feature_off_preserves_original_admission(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8, enabled=False)
    for request in create_requests(8, num_tokens=8):
        sched.add_request(request)
    sched.sawtooth_available_batches = 4
    assert len(sched.schedule().num_scheduled_tokens) == 8

@pytest.mark.parametrize('defer_decode', [False, True])
def test_pending_decode_does_not_block_ready_prompt_chunk(
    admission_scheduler, defer_decode
):
    sched = admission_scheduler(max_num_seqs=4, max_model_len=32768)
    sched.sawtooth_request_ledger.allow_prefill_overlap = True
    sched.sawtooth_allow_early_admission = True
    a, b = create_requests(2, num_tokens=16384, max_tokens=1024)
    sched.add_request(a)
    sched.add_request(b)
    first = sched.schedule()
    sched.sawtooth_mark_head_enqueued(first)
    sched.sawtooth_mark_tail_enqueued(first)
    sched.update_from_output(first, result(first, sample=False))
    mixed = sched.schedule()
    assert mixed.num_scheduled_tokens == {a.request_id: 6144, b.request_id: 4096}
    sched.sawtooth_mark_head_enqueued(mixed)
    sched.sawtooth_mark_tail_enqueued(mixed)
    sched.sawtooth_defer_dependent_decode = defer_decode
    next_batch = sched.schedule()
    if defer_decode:
        assert next_batch.num_scheduled_tokens == {b.request_id: 10240}
    else:
        assert next_batch.num_scheduled_tokens == {a.request_id: 1, b.request_id: 10239}


def test_ready_decode_is_not_filtered_by_prefill_policy(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=4)
    (a,) = create_requests(1, num_tokens=32, max_tokens=16)
    sched.add_request(a)
    first = sched.schedule()
    sched.update_from_output(first, result(first))
    (b,) = create_requests(1, num_tokens=16384, req_ids=['new-prefill'])
    sched.add_request(b)
    sched.sawtooth_defer_dependent_decode = True
    output = sched.schedule()
    assert output.num_scheduled_tokens == {a.request_id: 1, b.request_id: 10239}


@pytest.mark.parametrize("blocked", [False, True])
def test_prefill_opportunity_excludes_known_blocked_waiters(
    admission_scheduler, blocked
):
    sched = admission_scheduler(max_num_seqs=4)
    (request,) = create_requests(1, num_tokens=32)
    sched.add_request(request)
    if blocked:
        request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert sched.sawtooth_has_prefill_candidate(require_eligible=True) == (not blocked)


def test_prefill_opportunity_excludes_waiter_without_running_capacity(
    admission_scheduler,
):
    sched = admission_scheduler(max_num_seqs=1)
    (running,) = create_requests(1, num_tokens=32, max_tokens=16)
    sched.add_request(running)
    first = sched.schedule()
    sched.sawtooth_mark_head_enqueued(first)
    sched.sawtooth_mark_tail_enqueued(first)
    sched.sawtooth_allow_early_admission = True
    (waiting,) = create_requests(1, num_tokens=32, req_ids=["waiting"])
    sched.add_request(waiting)
    assert not sched.sawtooth_has_prefill_candidate(require_eligible=True)


@pytest.mark.parametrize("capacity", [2, 3])
@pytest.mark.parametrize("count,expected", [(1, 1), (3, 2), (4, 2), (5, 3)])
def test_pure_decode_refill_preserves_independent_request_cohorts(
    admission_scheduler, count, expected, capacity
):
    sched = admission_scheduler(max_num_seqs=8)
    sched.sawtooth_pipeline_capacity = capacity
    requests = create_requests(count, num_tokens=8, max_tokens=32)
    for request in requests:
        sched.add_request(request)
    prefill = sched.schedule()
    sched.update_from_output(prefill, result(prefill))
    sched.sawtooth_available_batches = capacity
    first = sched.schedule()
    assert len(first.num_scheduled_tokens) == expected
    assert set(first.num_scheduled_tokens.values()) == {1}
    if count > expected:
        second = sched.schedule()
        assert set(first.num_scheduled_tokens).isdisjoint(second.num_scheduled_tokens)
        assert len(second.num_scheduled_tokens) == count - expected


def test_pure_decode_cohorts_keep_advancing_across_refills(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8)
    sched.sawtooth_pipeline_capacity = 2
    requests = create_requests(4, num_tokens=8, max_tokens=32)
    for request in requests:
        sched.add_request(request)
    prefill = sched.schedule()
    sched.update_from_output(prefill, result(prefill))
    sched.sawtooth_available_batches = 1
    pending = [sched.schedule(), sched.schedule()]
    counts = {request.request_id: 0 for request in requests}
    for _ in range(12):
        completed = pending.pop(0)
        sched.update_from_output(completed, result(completed))
        for request_id in completed.num_scheduled_tokens:
            counts[request_id] += 1
        refill = sched.schedule()
        assert len(refill.num_scheduled_tokens) == 2
        assert set(refill.num_scheduled_tokens).isdisjoint(
            pending[0].num_scheduled_tokens
        )
        pending.append(refill)
    assert set(counts.values()) == {6}


def test_third_slot_admits_successor_before_old_results_retire(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8)
    sched.sawtooth_pipeline_capacity = 3
    requests = create_requests(4, num_tokens=8, max_tokens=32)
    for request in requests:
        sched.add_request(request)
    prefill = sched.schedule()
    sched.sawtooth_mark_head_enqueued(prefill)
    sched.sawtooth_mark_tail_enqueued(prefill)
    sched.update_from_output(prefill, result(prefill))
    sched.sawtooth_available_batches = 3
    first = sched.schedule()
    sched.sawtooth_available_batches = 2
    second = sched.schedule()
    assert len(first.num_scheduled_tokens) == len(second.num_scheduled_tokens) == 2
    assert set(first.num_scheduled_tokens).isdisjoint(second.num_scheduled_tokens)
    for batch in (first, second):
        sched.sawtooth_mark_head_enqueued(batch)
        sched.sawtooth_mark_tail_enqueued(batch)
    sched.sawtooth_allow_early_admission = True
    sched.sawtooth_available_batches = 1
    successor = sched.schedule()
    assert successor.num_scheduled_tokens == first.num_scheduled_tokens
    for rid, step in successor.sawtooth_request_steps.items():
        assert step.predecessor == first.sawtooth_request_steps[rid].key
        assert step.token_start == first.sawtooth_request_steps[rid].token_end


def test_short_tail_fills_remaining_budget_before_final_tail_split(
    admission_scheduler, tmp_path
):
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 131072
    config_path.write_text(json.dumps(config))
    def configured(**kwargs):
        kwargs["max_num_batched_tokens"] = 10240
        return admission_scheduler(**kwargs)

    engine, sched, executor, request = _tail_engine(
        configured, 65619, capacity=3, max_model_len=131072, num_blocks=20000
    )
    # One old cohort is frozen; the next cohort starts after a slot retires.
    for _ in range(5):
        engine.step()
    (partner,) = create_requests(
        1, num_tokens=65619, max_tokens=1, req_ids=["partner"]
    )
    sched.add_request(partner)
    _drain_tail_engine(engine, sched)
    heads = [
        o.num_scheduled_tokens for o in executor.commands
        if o.sawtooth_execution.is_first_round
    ]
    assert {request.request_id: 4179, "partner": 6061} in heads
    assert heads[-2:] == [{"partner": 4179}, {"partner": 4179}]
    assert sum(sum(h.values()) for h in heads) == 2 * 65619


@pytest.mark.parametrize("capacity", [2, 3])
@pytest.mark.parametrize("arrival_step", [0, 1, 2, 3, 4])
def test_two_50k_prefills_pack_boundary_before_splitting_unpaired_tail(
    admission_scheduler, tmp_path, capacity, arrival_step
):
    # A prompt boundary must not leave half of a prefill batch unused.
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 131072
    config_path.write_text(json.dumps(config))

    def configured(**kwargs):
        kwargs["max_num_batched_tokens"] = 20480
        return admission_scheduler(**kwargs)

    engine, sched, executor, request = _tail_engine(
        configured,
        51200,
        capacity=capacity,
        max_model_len=131072,
        num_blocks=20000,
    )
    for _ in range(arrival_step):
        engine.step()
    (partner,) = create_requests(
        1, num_tokens=51200, max_tokens=1, req_ids=["partner"]
    )
    sched.add_request(partner)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    batches = [o.num_scheduled_tokens for o in heads]
    # Live capacity must not turn three heads into a pair. After two full
    # pairs, split only the final unpaired batch, including at capacity three.
    expected = [20480] * 4 + [10240, 10240]
    assert [sum(b.values()) for b in batches] == expected, batches
    assert {request.request_id: 10240, partner.request_id: 10240} in batches
    for req in (request, partner):
        end = 0
        for out in heads:
            if req.request_id not in out.num_scheduled_tokens:
                continue
            step = out.sawtooth_request_steps[req.request_id]
            assert step.token_start == end
            end = step.token_end
        assert end == 51200
        assert req.num_in_flight_tokens == 0
        assert req.num_output_tokens == 1
        assert req.request_id not in sched.requests


@pytest.mark.parametrize("limits", [{"max_num_seqs": 1}, {"num_blocks": 3500}])
def test_prefill_tail_packing_preserves_resource_admission(
    admission_scheduler, tmp_path, limits
):
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 131072
    config_path.write_text(json.dumps(config))

    def configured(**kwargs):
        kwargs["max_num_batched_tokens"] = 20480
        kwargs.update(limits)
        return admission_scheduler(**kwargs)

    engine, sched, executor, request = _tail_engine(
        configured, 51200, capacity=3, max_model_len=131072
    )
    for _ in range(5):
        engine.step()
    (partner,) = create_requests(
        1, num_tokens=51200, max_tokens=1, req_ids=["partner"]
    )
    sched.add_request(partner)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert all(o.total_num_scheduled_tokens <= 20480 for o in heads)
    assert all(len(o.num_scheduled_tokens) == 1 for o in heads)
    for req in (request, partner):
        assert sum(
            o.num_scheduled_tokens.get(req.request_id, 0) for o in heads
        ) == 51200
        assert req.num_output_tokens == 1
        assert req.num_in_flight_tokens == 0
        assert req.request_id not in sched.requests


@pytest.mark.parametrize("max_tokens", [1, 4])
def test_prefill_boundary_reuses_only_a_finishing_requests_kv(
    admission_scheduler, tmp_path, max_tokens
):
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 131072
    config_path.write_text(json.dumps(config))

    def configured(**kwargs):
        kwargs["max_num_batched_tokens"] = 20480
        return admission_scheduler(**kwargs)

    engine, sched, executor, request = _tail_engine(
        configured, 51200, capacity=3, max_tokens=max_tokens,
        max_model_len=131072, num_blocks=4000,
    )
    sched.scheduler_reserve_full_isl = True
    # Arrive after the first pair tails, before the next head.
    while len(executor.commands) < 4:
        engine.step()
    (partner,) = create_requests(
        1, num_tokens=51200, max_tokens=max_tokens, req_ids=["partner"]
    )
    sched.add_request(partner)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    mixed = {request.request_id: 10240, partner.request_id: 10240}
    if max_tokens == 1:
        assert mixed in [o.num_scheduled_tokens for o in heads]
        assert [o.total_num_scheduled_tokens for o in heads] == [20480] * 5
    else:
        # A retains KV for decode: preserve full-input admission policy for B.
        assert all(len(o.num_scheduled_tokens) == 1 for o in heads)
    for req in (request, partner):
        assert req.num_output_tokens == max_tokens
        assert req.num_in_flight_tokens == 0
        assert req.num_preemptions == 0
        assert req.request_id not in sched.requests
    assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == 3999


@pytest.mark.parametrize("capacity", [2, 3])
@pytest.mark.parametrize("max_tokens,num_blocks", [(1, 270), (1, 250), (4, 270)])
def test_next_prefill_head_overlaps_a_full_finishing_batch(
    admission_scheduler, capacity, max_tokens, num_blocks
):
    # Only A's full prompt + B's first chunk fit. B must enter before A retires.
    engine, sched, executor, request = _tail_engine(
        admission_scheduler,
        3072,
        capacity=capacity,
        num_blocks=num_blocks,
        max_tokens=max_tokens,
    )
    sched.scheduler_reserve_full_isl = True
    for _ in range(20):
        engine.step()
        final = next(
            (
                o
                for o in executor.commands
                if o.sawtooth_execution.is_first_round
                and o.sawtooth_request_steps[request.request_id].produces_sample
            ),
            None,
        )
        if final is not None:
            break
    assert final is not None
    assert request.num_output_tokens == 0
    (partner,) = create_requests(1, num_tokens=3072, max_tokens=1, req_ids=["partner"])
    sched.add_request(partner)
    before = len(executor.commands)
    admitted = False
    for _ in range(20):
        engine.step()
        heads = [
            o for o in executor.commands[before:] if o.sawtooth_execution.is_first_round
        ]
        if any(o.num_scheduled_tokens == {"partner": 1024} for o in heads):
            admitted = request.num_output_tokens == 0
            break
        if request.num_output_tokens:
            break
    assert admitted == (max_tokens == 1 and num_blocks == 270)
    if admitted:
        a_blocks = sched.kv_cache_manager.get_blocks(request.request_id).blocks[0]
        b_blocks = sched.kv_cache_manager.get_blocks(partner.request_id).blocks[0]
        assert all(block.ref_cnt > 0 for block in a_blocks)
        assert {b.block_id for b in a_blocks}.isdisjoint(b.block_id for b in b_blocks)
    _drain_tail_engine(engine, sched)
    for req, expected in [(request, max_tokens), (partner, 1)]:
        assert req.num_output_tokens == expected
        assert req.num_preemptions == 0
        assert req.num_in_flight_tokens == 0
        assert req.request_id not in sched.requests
    assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == num_blocks - 1


@pytest.mark.parametrize("length", [51200, 61440, 71680])
def test_only_unpaired_final_full_batch_splits(admission_scheduler, tmp_path, length):
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 131072
    config_path.write_text(json.dumps(config))

    def configured(**kwargs):
        kwargs["max_num_batched_tokens"] = 20480
        return admission_scheduler(**kwargs)

    engine, sched, executor, request = _tail_engine(
        configured, length, capacity=3, max_model_len=131072, num_blocks=6000
    )
    sched.scheduler_reserve_full_isl = True
    (partner,) = create_requests(1, num_tokens=length, max_tokens=1, req_ids=["B"])
    sched.add_request(partner)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    totals = [o.total_num_scheduled_tokens for o in heads]
    # 60k x 2 already pairs all six full batches. 50k/70k end with a lone batch.
    expected = (
        [20480] * (length // 10240 - 1) + [10240, 10240]
        if length != 61440
        else [20480] * 6
    )
    assert totals == expected
    assert sum(totals) == length * 2
    for req in (request, partner):
        assert req.num_output_tokens == 1
        assert req.num_preemptions == 0
        assert req.num_in_flight_tokens == 0
        assert req.request_id not in sched.requests
    assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == 5999


@pytest.mark.parametrize("warmup_length", [1, 3072])
def test_tail_parity_resets_between_drained_request_groups(
    admission_scheduler, warmup_length
):
    engine, sched, executor, _ = _tail_engine(admission_scheduler, warmup_length)
    _drain_tail_engine(engine, sched)
    executor.commands.clear()
    (request,) = create_requests(1, num_tokens=6144, max_tokens=1, req_ids=["next"])
    sched.add_request(request)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [1024] * 6
    assert request.num_output_tokens == 1


@pytest.mark.parametrize("arrival_step", [0, 1, 2, 3, 4])
def test_two_three_chunk_prompts_keep_six_full_batches(
    admission_scheduler, arrival_step
):
    engine, sched, executor, request = _tail_engine(admission_scheduler, 3072)
    for _ in range(arrival_step):
        engine.step()
    (partner,) = create_requests(1, num_tokens=3072, max_tokens=1, req_ids=["B"])
    sched.add_request(partner)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [1024] * 6
    for req in (request, partner):
        assert req.num_output_tokens == 1
        assert req.num_in_flight_tokens == 0


@pytest.mark.parametrize(
    "length,counts",
    [
        (51200, [10240] * 4 + [5120, 5120]),
        (56320, [10240] * 5 + [5120]),
        (61440, [10240] * 6),
        (65536, [10240] * 6 + [2048, 2048]),
    ],
)
@pytest.mark.parametrize("request_count", [1, 2])
def test_mbt10k_pairs_heads_and_splits_only_unpaired_tail(
    admission_scheduler, tmp_path, length, counts, request_count
):
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 131072
    config_path.write_text(json.dumps(config))

    def configured(**kwargs):
        kwargs["max_num_batched_tokens"] = 10240
        return admission_scheduler(**kwargs)

    engine, sched, executor, request = _tail_engine(
        configured, length // request_count, capacity=3,
        max_model_len=131072, num_blocks=10000
    )
    if request_count == 2:
        (partner,) = create_requests(
            1, num_tokens=length // 2, max_tokens=1, req_ids=["B"]
        )
        sched.add_request(partner)
    peak_live = _drain_tail_engine(engine, sched)
    assert engine.capacity == 3
    if request_count == 2:
        assert peak_live == 3
        assert partner.num_output_tokens == 1
        assert partner.num_in_flight_tokens == 0
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == counts
    assert sum(counts) == length
    assert request.num_output_tokens == 1
    # With three live slots, preserve pairs of heads before their tails.
    # Exact token counts alone cannot prove the final two heads are paired.
    commands = [
        (o.sawtooth_execution.batch_id, o.sawtooth_execution.round_id)
        for o in executor.commands
    ]
    expected = [
        (batch, round_id)
        for first in range(0, len(counts), 2)
        for round_id in range(2)
        for batch in (first, first + 1)
    ]
    assert commands == expected



def _balanced_engine(admission_scheduler, tmp_path, length, *, mbt=16384,
                     capacity=3, **kwargs):
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 262144
    config_path.write_text(json.dumps(config))

    def configured(**options):
        options["max_num_batched_tokens"] = mbt
        return admission_scheduler(**options)

    engine, sched, executor, request = _tail_engine(
        configured, length, capacity=capacity, max_model_len=262144,
        num_blocks=kwargs.pop("num_blocks", 20000), **kwargs
    )
    sched.sawtooth_balance_prefill_pairs = True
    return engine, sched, executor, request


@pytest.mark.parametrize("capacity", [2, 3])
@pytest.mark.parametrize("length,expected", [
    (16467, [8234, 8233]),
    (18432, [9216, 9216]),
    (10240, [5120, 5120]),
    (32768, [16384, 16384]),
    (49152, [16384, 16384, 8192, 8192]),
    (51200, [16384, 16384, 9216, 9216]),
    (56320, [16384, 16384, 11776, 11776]),
    (61440, [16384, 16384, 14336, 14336]),
    (65536, [16384] * 4),
    (1, [1]),
])
def test_pair_balance_real_scheduler_single_request(
    admission_scheduler, tmp_path, capacity, length, expected
):
    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, length, capacity=capacity
    )
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == expected
    end = 0
    for output in heads:
        step = output.sawtooth_request_steps[request.request_id]
        assert step.token_start == end
        end = step.token_end
        assert step.produces_sample == (end == length)
    assert end == length
    assert request.num_output_tokens == 1
    assert request.num_in_flight_tokens == 0
    order = [(o.sawtooth_execution.batch_id, o.sawtooth_execution.round_id)
             for o in executor.commands]
    for a, other in zip(heads[::2], heads[1::2]):
        aid = a.sawtooth_execution.batch_id
        bid = other.sawtooth_execution.batch_id
        assert order.index((aid, 0)) < order.index((bid, 0))
        assert order.index((bid, 0)) < order.index((aid, 1))
        assert order.index((aid, 1)) < order.index((bid, 1))


def test_pair_balance_waiting_small_partner_is_included(
    admission_scheduler, tmp_path
):
    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, 16384
    )
    (other,) = create_requests(1, num_tokens=2048, max_tokens=1, req_ids=["B"])
    sched.add_request(other)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.num_scheduled_tokens for o in heads] == [
        {request.request_id: 9216}, {request.request_id: 7168, "B": 2048}
    ]
    assert request.num_output_tokens == other.num_output_tokens == 1


def test_pair_balance_late_arrival_keeps_reserved_second_budget(
    admission_scheduler, tmp_path
):
    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, 16467
    )
    engine.step()
    (other,) = create_requests(1, num_tokens=32768, max_tokens=1, req_ids=["B"])
    sched.add_request(other)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [8234, 8233, 16384, 16384]
    assert heads[1].num_scheduled_tokens == {request.request_id: 8233}
    assert request.num_output_tokens == other.num_output_tokens == 1



@pytest.mark.parametrize("mbt,length,expected", [
    (10240, 51200, [10240] * 4 + [5120, 5120]),
    (10240, 56320, [10240] * 4 + [7680, 7680]),
    (10240, 61440, [10240] * 6),
    (10240, 65536, [10240] * 6 + [2048, 2048]),
])
def test_pair_balance_preserves_full_pairs_mbt10k(
    admission_scheduler, tmp_path, mbt, length, expected
):
    engine, sched, executor, _ = _balanced_engine(
        admission_scheduler, tmp_path, length, mbt=mbt
    )
    _drain_tail_engine(engine, sched)
    assert [o.total_num_scheduled_tokens for o in executor.commands
            if o.sawtooth_execution.is_first_round] == expected


def test_pair_balance_uses_actual_prefix_hit_remaining_tokens(
    admission_scheduler, tmp_path
):
    engine, sched, executor, _ = _balanced_engine(
        admission_scheduler, tmp_path, 114689, enable_prefix_caching=True
    )
    _drain_tail_engine(engine, sched)
    executor.commands.clear()
    (request,) = create_requests(1, num_tokens=131155, max_tokens=1, req_ids=["hit"])
    assert sched.kv_cache_manager.get_computed_blocks(request)[1] == 114688
    sched.add_request(request)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [8234, 8233]
    a, b = [o.sawtooth_request_steps["hit"] for o in heads]
    assert (a.token_start, a.token_end, b.token_start, b.token_end) == (
        114688, 122922, 122922, 131155
    )


def test_pair_balance_does_not_count_kv_blocked_waiter(
    admission_scheduler, tmp_path
):
    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, 16467, num_blocks=2200
    )
    sched.scheduler_reserve_full_isl = True
    (other,) = create_requests(1, num_tokens=32768, max_tokens=1, req_ids=["B"])
    sched.add_request(other)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads[:2]] == [8234, 8233]
    assert request.num_output_tokens == other.num_output_tokens == 1
    assert request.num_preemptions == other.num_preemptions == 0
    assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == 2199


def test_pair_balance_failed_allocation_can_retry_without_consuming_plan(
    admission_scheduler, tmp_path, monkeypatch
):
    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, 16467
    )
    allocate = sched.kv_cache_manager.allocate_slots
    monkeypatch.setattr(sched.kv_cache_manager, "allocate_slots", lambda *a, **k: None)
    engine.step()
    assert not executor.commands
    assert sched._sawtooth_prefill_batches == 0
    assert sched._sawtooth_pair_next_budget is None
    monkeypatch.setattr(sched.kv_cache_manager, "allocate_slots", allocate)
    _drain_tail_engine(engine, sched)
    assert [o.total_num_scheduled_tokens for o in executor.commands
            if o.sawtooth_execution.is_first_round] == [8234, 8233]
    assert request.num_output_tokens == 1


@pytest.mark.parametrize("admit_second", [False, True])
def test_pair_balance_abort_keeps_inflight_kv_and_resets_next_wave(
    admission_scheduler, tmp_path, admit_second
):
    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, 16467
    )
    engine.step()
    if admit_second:
        engine.step()
    blocks = [
        b
        for group in sched.kv_cache_manager.get_blocks(request.request_id).blocks
        for b in group
    ]
    sched.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert all(b.ref_cnt > 0 for b in blocks)
    _drain_tail_engine(engine, sched)
    assert request.num_output_tokens == 0
    assert all(b.ref_cnt == 0 for b in blocks)
    executor.commands.clear()
    (other,) = create_requests(1, num_tokens=18432, max_tokens=1, req_ids=["B"])
    sched.add_request(other)
    _drain_tail_engine(engine, sched)
    assert [o.total_num_scheduled_tokens for o in executor.commands
            if o.sawtooth_execution.is_first_round] == [9216, 9216]


def test_pair_balance_does_not_change_decode_steps(admission_scheduler, tmp_path):
    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, 16467, max_tokens=3
    )
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [8234, 8233, 1, 1]
    assert [o.sawtooth_has_decode for o in heads] == [False, False, True, True]
    assert request.num_output_tokens == 3


def test_pair_balance_configuration_reaches_engine(admission_scheduler, tmp_path):
    from vllm.distributed.sawtooth import SawtoothConfig
    from vllm.v1.engine.sawtooth import SawtoothEngine

    engine, sched, _, _ = _balanced_engine(admission_scheduler, tmp_path, 16467)
    options = engine.core.vllm_config.additional_config["sawtooth_pipeline"]
    config = SawtoothConfig.from_vllm_config(engine.core.vllm_config)
    assert not config.enable_prefill_pair_balance
    options["enable_prefill_pair_balance"] = True
    config = SawtoothConfig.from_vllm_config(engine.core.vllm_config)
    assert config.enable_prefill_pair_balance
    sched.sawtooth_balance_prefill_pairs = False
    SawtoothEngine(engine.core)
    assert sched.sawtooth_balance_prefill_pairs
    options["enable_prefill_pair_balance"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        SawtoothConfig.from_vllm_config(engine.core.vllm_config)


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_pair_balance_matches_waiting_queue_order(
    admission_scheduler, tmp_path, policy
):
    from vllm.v1.core.sched.request_queue import (
        SchedulingPolicy,
        create_request_queue,
    )

    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, 32768
    )
    sched.max_num_running_reqs = 1
    sched.max_num_active_reqs = 1
    sched.policy = SchedulingPolicy(policy)
    sched.waiting = create_request_queue(sched.policy)
    sched.kv_holding_waiting = create_request_queue(sched.policy)
    request.priority = 1
    sched.waiting.add_request(request)
    (other,) = create_requests(1, num_tokens=10240, max_tokens=1, req_ids=["B"])
    other.priority = 0
    sched.add_request(other)
    sched.waiting.remove_request(other)
    sched.kv_holding_waiting.add_request(other)
    _drain_tail_engine(engine, sched)
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert [o.num_scheduled_tokens for o in heads] == [
        {"B": 5120}, {"B": 5120},
        {request.request_id: 16384}, {request.request_id: 16384},
    ]


def test_pair_balance_lookahead_does_not_emit_cache_events(
    admission_scheduler, tmp_path, monkeypatch
):
    engine, sched, _, _ = _balanced_engine(
        admission_scheduler, tmp_path, 1025, enable_prefix_caching=True
    )
    _drain_tail_engine(engine, sched)
    (request,) = create_requests(1, num_tokens=17491, max_tokens=1, req_ids=["hit"])
    request.kv_cache_report_mode = "full"
    sched.add_request(request)
    manager = sched.kv_cache_manager
    manager.enable_kv_cache_events = True
    events = []
    monkeypatch.setattr(manager.coordinator, "emit_cached_block_events",
                        lambda *args: events.append(args))
    sched.sawtooth_split_prefill_tail = True
    assert sched._sawtooth_pair_budget(16384) == 8234
    assert events == []
    # The real lookup still emits its event.
    assert manager.get_computed_blocks(request)[1] == 1024
    assert len(events) == 1


def test_pair_balance_trims_retained_hits_for_kv_estimate(
    admission_scheduler, tmp_path, monkeypatch
):
    engine, sched, _, request = _balanced_engine(
        admission_scheduler, tmp_path, 16483, enable_prefix_caching=True
    )
    manager = sched.kv_cache_manager
    sched.sawtooth_request_ledger.allow_prefill_overlap = True
    manager.retained_hit_group_ids = {0}
    blocks = [manager.block_pool.null_block] * 4
    monkeypatch.setattr(manager.coordinator, "find_longest_cache_hit",
                        lambda *args: ((blocks,), 16, 48))
    estimate = manager.coordinator.get_num_blocks_to_allocate
    seen = []

    def check(**kwargs):
        seen.append(len(kwargs["new_computed_blocks"][0]))
        return estimate(**kwargs)

    monkeypatch.setattr(manager.coordinator, "get_num_blocks_to_allocate", check)
    sched.sawtooth_split_prefill_tail = True
    assert sched._sawtooth_pair_budget(16384) == 8234
    assert seen == [1]
    assert len(blocks) == 4  # Never mutate the cached lookup result.


@pytest.mark.parametrize("length", [49152, 65536, 65536 + 2490])
def test_pair_overlap_submits_second_head_before_previous_tail_retirement(
    admission_scheduler, tmp_path, length
):
    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, length, capacity=3
    )
    admissions = []
    original = executor.execute_model

    def execute(output, non_block):
        command = output.sawtooth_execution
        if command is not None and command.is_first_round:
            state = sched.sawtooth_request_ledger.requests[request.request_id]
            admissions.append((command.batch_id, state.completed_end,
                               len(state.live)))
        return original(output, non_block)

    executor.execute_model = execute
    _drain_tail_engine(engine, sched)
    # Heads 2 and 3 must share the remaining slots with unretired tail 1.
    assert admissions[3] == (3, 16384, 3)
    assert request.num_in_flight_tokens == 0
    assert request.num_output_tokens == 1


def test_pair_overlap_late_request_keeps_already_submitted_tail_immutable(
    admission_scheduler, tmp_path
):
    engine, sched, executor, request = _balanced_engine(
        admission_scheduler, tmp_path, 49152, capacity=3
    )
    for _ in range(5):
        engine.step()
    submitted = deepcopy(executor.commands)
    heads = [o for o in submitted if o.sawtooth_execution.is_first_round]
    assert [o.total_num_scheduled_tokens for o in heads] == [16384, 16384, 8192]
    (partner,) = create_requests(1, num_tokens=2048, max_tokens=1, req_ids=["late"])
    sched.add_request(partner)
    _drain_tail_engine(engine, sched)
    for before, after in zip(submitted, executor.commands):
        assert before.sawtooth_request_steps == after.sawtooth_request_steps
        assert before.num_scheduled_tokens == after.num_scheduled_tokens
    heads = [o for o in executor.commands if o.sawtooth_execution.is_first_round]
    assert heads[3].total_num_scheduled_tokens == 8192
    for req in (request, partner):
        end = 0
        for output in heads:
            step = output.sawtooth_request_steps.get(req.request_id)
            if step is not None:
                assert step.token_start == end
                end = step.token_end
        assert end == req.num_prompt_tokens
        assert req.num_output_tokens == 1
        assert req.num_in_flight_tokens == 0
        assert req.request_id not in sched.requests
    assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == 19999

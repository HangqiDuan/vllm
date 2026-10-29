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
    for _ in range(100):
        if not engine and not sched.has_requests():
            return
        engine.step()
    pytest.fail("Tail pipeline failed to drain")


@pytest.mark.parametrize("capacity", [2, 3])
@pytest.mark.parametrize(
    "length,counts",
    [
        (7168, [1024] * 7),
        (3072, [1024] * 3),
        (4096, [1024] * 4),
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
    "length,counts", [(3072, [1024, 1024]), (4096, [1024] * 3)]
)
def test_tail_pairing_uses_live_cohort_instead_of_original_prompt_parity(
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


def test_ready_skipped_waiters_count_toward_cohort_size(admission_scheduler):
    sched = admission_scheduler(max_num_seqs=8)
    for request in create_requests(8, num_tokens=8):
        sched.add_request(request)
    for _ in range(4):
        sched.skipped_waiting.add_request(sched.waiting.pop_request())
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
    assert requests[1] in sched.skipped_waiting


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
def test_pending_decode_does_not_block_ready_prompt_chunk(admission_scheduler, defer_decode):
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
def test_prefill_opportunity_excludes_known_blocked_waiters(admission_scheduler, blocked):
    sched = admission_scheduler(max_num_seqs=4)
    (request,) = create_requests(1, num_tokens=32)
    sched.add_request(request)
    if blocked:
        request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert sched.sawtooth_has_prefill_candidate(require_eligible=True) == (not blocked)


def test_prefill_opportunity_excludes_waiter_without_running_capacity(admission_scheduler):
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
@pytest.mark.parametrize("arrival_step", [0, 1, 2, 3, 4, 5])
def test_two_50k_prefills_pack_five_full_batches(
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
    assert [sum(b.values()) for b in batches] == [20480] * 5, batches
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

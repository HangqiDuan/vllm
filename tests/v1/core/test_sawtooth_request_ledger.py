# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.sawtooth import SawtoothRequestLedger


def test_unsampled_prefill_can_overlap_before_tail_but_not_release_its_lease():
    ledger = SawtoothRequestLedger()
    ledger.allow_prefill_overlap = True
    first = ledger.admit("r", 0, 10240, False)
    assert not ledger.can_advance("r")
    ledger.mark_head_enqueued(first)
    assert ledger.can_advance("r")
    second = ledger.admit("r", 10240, 6144, True)
    ledger.mark_head_enqueued(second)
    assert not ledger.can_advance("r")
    with pytest.raises(RuntimeError, match="live"):
        ledger.forget("r")
    ledger.mark_tail_enqueued(first)
    ledger.settle(first, 6144)
    assert not ledger.can_advance("r")
    ledger.mark_tail_enqueued(second)
    assert ledger.can_advance("r")
    ledger.settle(second, 0)


def test_head_submission_does_not_bypass_sample_dependency_or_disabled_overlap():
    for enabled, sampled in [(False, False), (True, True)]:
        ledger = SawtoothRequestLedger()
        ledger.allow_prefill_overlap = enabled
        first = ledger.admit("r", 0, 10240, sampled)
        ledger.mark_head_enqueued(first)
        assert not ledger.can_advance("r")


def test_tail_submission_only_opens_one_successor_not_resource_reclamation():
    ledger = SawtoothRequestLedger()
    first = ledger.admit("r", 8, 16, False)
    assert not ledger.can_advance("r")
    with pytest.raises(RuntimeError, match="unsent"):
        ledger.admit("r", 24, 16, False)
    ledger.mark_tail_enqueued(first)
    second = ledger.admit("r", 24, 16, False)
    assert second.predecessor == first.key
    ledger.mark_tail_enqueued(second)
    assert not ledger.can_advance("r")
    with pytest.raises(RuntimeError, match="token lease"):
        ledger.admit("r", 40, 1, True)
    with pytest.raises(RuntimeError, match="live"):
        ledger.forget("r")
    ledger.settle(first, 16)
    assert ledger.requests["r"].completed_end == 24
    assert ledger.requests["r"].scheduled_end == 40
    assert ledger.can_advance("r")
    ledger.settle(second, 0)
    ledger.forget("r")
    reused = ledger.admit("r", 0, 1, True)
    assert reused.step_id == 0
    assert reused.request_generation != first.request_generation


@pytest.mark.parametrize("error", ["out_of_order", "wrong_count", "duplicate"])
def test_completion_cannot_advance_past_an_unsettled_interval(error):
    ledger = SawtoothRequestLedger()
    first = ledger.admit("r", 0, 3, True)
    ledger.mark_tail_enqueued(first)
    second = ledger.admit("r", 3, 1, True)
    if error == "out_of_order":
        step, count = second, 3
    elif error == "wrong_count":
        step, count = first, 0
    else:
        ledger.settle(first, 1)
        step, count = first, 1
    with pytest.raises(RuntimeError, match="Non-contiguous"):
        ledger.settle(step, count)


def test_wrong_or_repeated_tail_identity_is_rejected():
    ledger = SawtoothRequestLedger()
    step = ledger.admit("r", 0, 1, True)
    ledger.mark_tail_enqueued(step)
    with pytest.raises(RuntimeError, match="terminal submission"):
        ledger.mark_tail_enqueued(step)


def test_final_prefill_admission_credit_requires_a_live_submitted_head():
    ledger = SawtoothRequestLedger()
    ledger.allow_prefill_overlap = True
    step = ledger.admit("a", 0, 1024, True)
    assert not ledger.has_enqueued_final_prefill("a", 1024)
    ledger.mark_head_enqueued(step)
    assert ledger.has_enqueued_final_prefill("a", 1024)
    assert not ledger.has_enqueued_final_prefill("a", 2048)
    ledger.allow_prefill_overlap = False
    assert not ledger.has_enqueued_final_prefill("a", 1024)
    ledger.allow_prefill_overlap = True
    ledger.mark_tail_enqueued(step)
    ledger.settle(step, 0)
    assert not ledger.has_enqueued_final_prefill("a", 1024)


def test_prefill_window_keeps_old_tail_and_new_pair_live():
    ledger = SawtoothRequestLedger()
    ledger.allow_prefill_overlap = True
    ledger.max_live_prefill_steps = 3
    old = ledger.admit("r", 0, 16384, False)
    ledger.mark_head_enqueued(old)
    ledger.mark_tail_enqueued(old)
    first = ledger.admit("r", 16384, 1245, False)
    ledger.mark_head_enqueued(first)
    assert ledger.can_overlap_prefill("r")
    second = ledger.admit("r", 17629, 1245, True)
    ledger.mark_head_enqueued(second)
    assert not ledger.can_advance("r")
    with pytest.raises(RuntimeError, match="token lease"):
        ledger.admit("r", 18874, 1, True)
    with pytest.raises(RuntimeError, match="live"):
        ledger.forget("r")
    with pytest.raises(RuntimeError, match="Non-contiguous"):
        ledger.settle(first, 1245)
    ledger.settle(old, 2490)
    ledger.mark_tail_enqueued(first)
    ledger.settle(first, 1245)
    assert not ledger.can_advance("r")
    ledger.mark_tail_enqueued(second)
    ledger.settle(second, 0)
    ledger.forget("r")


@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_window_requires_old_tails_before_third_head(enabled):
    ledger = SawtoothRequestLedger()
    ledger.allow_prefill_overlap = enabled
    ledger.max_live_prefill_steps = 3
    first = ledger.admit("r", 0, 16, False)
    ledger.mark_head_enqueued(first)
    if not enabled:
        ledger.mark_tail_enqueued(first)
    second = ledger.admit("r", 16, 16, False)
    ledger.mark_head_enqueued(second)
    assert not ledger.can_advance("r")
    if enabled:
        ledger.mark_tail_enqueued(first)
        assert ledger.can_overlap_prefill("r")
    else:
        ledger.mark_tail_enqueued(second)
        assert not ledger.can_advance("r")


def test_prefill_window_cannot_cross_unretired_sampled_step():
    ledger = SawtoothRequestLedger()
    ledger.allow_prefill_overlap = True
    ledger.max_live_prefill_steps = 3
    sampled = ledger.admit("r", 0, 16, True)
    ledger.mark_head_enqueued(sampled)
    ledger.mark_tail_enqueued(sampled)
    second = ledger.admit("r", 16, 1, False)
    ledger.mark_head_enqueued(second)
    assert not ledger.can_overlap_prefill("r")
    assert not ledger.can_advance("r")

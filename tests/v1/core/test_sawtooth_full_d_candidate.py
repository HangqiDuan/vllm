"""Exercise pure-Decode cohort sizing with three in-flight batch slots."""

from .test_sawtooth_scheduler import local_model_config, result, scheduler
from .utils import create_requests


def test_pure_decode_keeps_four_ready_requests_in_one_batch():
    sched = scheduler(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256)
    requests = create_requests(4, num_tokens=16, max_tokens=8)
    for request in requests:
        sched.add_request(request)
    prefill = sched.schedule()
    assert len(prefill.num_scheduled_tokens) == 4
    sched.sawtooth_mark_tail_enqueued(prefill)
    sched.update_from_output(prefill, result(prefill))

    # The engine sets both fields when three pipeline slots are live.
    sched.sawtooth_pipeline_capacity = 3
    sched.sawtooth_available_batches = 3
    for _ in range(3):
        assert sched._sawtooth_request_cap(False) == 4
        decode = sched.schedule()
        assert decode.num_scheduled_tokens == {
            request.request_id: 1 for request in requests
        }
        sched.sawtooth_mark_tail_enqueued(decode)
        sched.update_from_output(decode, result(decode))

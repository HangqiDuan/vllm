from types import SimpleNamespace

import pytest
import torch

import vllm.v1.core.kv_cache_utils as utils
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.request import Request


class CountingHash:
    def __init__(self):
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return sha256(value)


def make(tokens, hasher, **kwargs):
    return Request("test", list(tokens), SamplingParams(max_tokens=4), None,
                   block_hasher=hasher, **kwargs)


def setup_cache(limit=32, block_size=2):
    utils.init_none_hash(sha256)
    counter = CountingHash()
    cached = utils.get_request_block_hasher(
        block_size, counter, max_cached_prompt_tokens=limit)
    plain = utils.get_request_block_hasher(block_size, sha256)
    return cached, plain, counter


@pytest.mark.parametrize("change", [0, 1, 3, 7, 15, None])
def test_exact_prefix_hashes_and_suffix_call_count(change):
    cached, plain, counter = setup_cache()
    tokens = list(range(16))
    first = make(tokens, cached)
    assert first.block_hashes == make(tokens, plain).block_hashes
    changed = tokens.copy()
    if change is not None:
        changed[change] += 100
    before = counter.calls
    second = make(changed, cached)
    assert second.block_hashes == make(changed, plain).block_hashes
    assert counter.calls-before == (8-change//2 if change is not None else 0)


def test_cache_has_token_bound_and_does_not_alias_request_lists():
    cached, plain, counter = setup_cache(limit=8)
    tokens = list(range(20))
    first = make(tokens, cached)
    first.block_hashes[:] = [b"wrong"] * 10
    first._all_token_ids[0] = -1
    first.prompt_token_ids[1] = -2
    before = counter.calls
    assert make(tokens, cached).block_hashes == make(tokens, plain).block_hashes
    assert counter.calls-before == 6


def test_partial_blocks_longer_and_shorter_prompts():
    cached, plain, counter = setup_cache()
    for tokens in [list(range(7)), list(range(9)), list(range(3)), list(range(10))]:
        assert make(tokens, cached).block_hashes == make(tokens, plain).block_hashes
    assert counter.calls == 3+1+0+4


def test_salt_and_root_hash_are_part_of_cache_identity(monkeypatch):
    cached, plain, counter = setup_cache()
    tokens = list(range(8))
    for salt in [None, "one", "two", "two"]:
        before = counter.calls
        assert make(tokens, cached, cache_salt=salt).block_hashes == make(
            tokens, plain, cache_salt=salt).block_hashes
        assert counter.calls-before == (0 if salt == "two" and before == 12 else 4)
    monkeypatch.setattr(utils, "NONE_HASH", b"another-root")
    before = counter.calls
    assert make(tokens, cached, cache_salt="two").block_hashes == make(
        tokens, plain, cache_salt="two").block_hashes
    assert counter.calls-before == 4


def test_incremental_generation_and_input_prefix_remain_identical():
    cached, plain, _ = setup_cache()
    a, b = make(range(7), cached), make(range(7), plain)
    for token in [100, 101, 102, 103]:
        a.append_output_token_ids(token)
        b.append_output_token_ids(token)
        assert a.block_hashes == b.block_hashes
    assert make(range(7), cached).block_hashes == make(range(7), plain).block_hashes


@pytest.mark.parametrize("kind", ["lora", "multimodal", "embeds", "mixed"])
def test_special_inputs_keep_original_hashing(kind):
    cached, plain, counter = setup_cache()
    tokens = list(range(8))
    make(tokens, cached)
    kwargs = {}
    if kind == "lora":
        kwargs['lora_request'] = SimpleNamespace(lora_name="adapter")
    elif kind == "multimodal":
        kwargs['mm_features'] = [SimpleNamespace(identifier="image-a",
            mm_position=SimpleNamespace(offset=0, length=4))]
    else:
        kwargs['prompt_embeds'] = torch.arange(16,dtype=torch.float32).reshape(8,2)
        if kind == "mixed":
            kwargs['prompt_is_token_ids'] = [True,False]*4
    for _ in range(2):
        before = counter.calls
        assert make(tokens,cached,**kwargs).block_hashes == make(tokens,plain,**kwargs).block_hashes
        assert counter.calls-before == 4


def test_default_disabled_and_limit_smaller_than_block():
    utils.init_none_hash(sha256)
    for limit in [None,0,1]:
        counter = CountingHash()
        kwargs = {} if limit is None else {'max_cached_prompt_tokens':limit}
        hasher = utils.get_request_block_hasher(2,counter,**kwargs)
        make(range(8),hasher); make(range(8),hasher)
        assert counter.calls == 8


@pytest.mark.parametrize("change", [4095,4096,4097,8195,None])
def test_long_prefix_comparison_windows(change):
    cached,plain,counter = setup_cache(limit=12000)
    tokens=list(range(10000))
    make(tokens,cached)
    if change is not None:
        tokens[change] = -1
    before=counter.calls
    assert make(tokens,cached).block_hashes == make(tokens,plain).block_hashes
    assert counter.calls-before == (5000-change//2 if change is not None else 0)


def test_nondivisible_cap_and_partial_matching_block():
    cached,plain,counter=setup_cache(limit=10,block_size=3)
    tokens=list(range(15));make(tokens,cached)
    tokens[8]=-1
    before=counter.calls
    assert make(tokens,cached).block_hashes == make(tokens,plain).block_hashes
    assert counter.calls-before==3
    before=counter.calls
    assert make(tokens,cached).block_hashes == make(tokens,plain).block_hashes
    assert counter.calls-before==2


def test_short_prompt_first_generated_block_is_not_cached():
    cached,plain,counter=setup_cache(block_size=4)
    a,b=make([1],cached),make([1],plain)
    for tok in [2,3,4]:
        a.append_output_token_ids(tok);b.append_output_token_ids(tok)
    assert a.block_hashes==b.block_hashes
    before=counter.calls
    assert make([1,2,3,4],cached).block_hashes==make([1,2,3,4],plain).block_hashes
    assert counter.calls-before==1


def test_concurrent_snapshot_replacements_preserve_exact_hashes():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    utils.init_none_hash(sha256)
    gate=threading.Barrier(2);local=threading.local()
    def hash_fn(value):
        if getattr(local,'wait',False):
            local.wait=False;gate.wait(timeout=10)
        return sha256(value)
    cached=utils.get_request_block_hasher(2,hash_fn,max_cached_prompt_tokens=10000)
    plain=utils.get_request_block_hasher(2,sha256)
    base=list(range(10000));make(base,cached)
    def run(last):
        local.wait=True
        tokens=base[:-1]+[last]
        return make(tokens,cached).block_hashes
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(run,-1);b=pool.submit(run,-2)
        assert a.result()==make(base[:-1]+[-1],plain).block_hashes
        assert b.result()==make(base[:-1]+[-2],plain).block_hashes
    assert make(base,cached).block_hashes==make(base,plain).block_hashes


def test_cbor_hash_algorithm_keeps_identical_values():
    from vllm.utils.hashing import sha256_cbor
    utils.init_none_hash(sha256_cbor)
    cached=utils.get_request_block_hasher(2,sha256_cbor,max_cached_prompt_tokens=32)
    plain=utils.get_request_block_hasher(2,sha256_cbor)
    for tokens in [list(range(16)),list(range(14))+[99,100]]:
        assert make(tokens,cached).block_hashes==make(tokens,plain).block_hashes

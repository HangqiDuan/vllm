# Sawtooth Pipeline

Status: experimental

[TOC]

!!! warning
    Sawtooth pipeline execution is an opt-in experimental feature. Numerical
    accuracy and complete pipeline-bubble masking have not been accepted. Do
    not infer a performance guarantee from the architecture described here.

## Overview

Sawtooth divides each physical pipeline-parallel rank into fixed virtual
stages. A scheduler admission creates one **logical batch**. The engine then
dispatches that batch through every virtual round without scheduling its tokens
again.

For pipeline size 2 and virtual size 2, four globally ordered groups are
created:

```text
g0 on rank 0 -> g1 on rank 1 -> g2 on rank 0 -> g3 on rank 1
      round 0                         round 1
```

The `g1 -> g2` edge wraps execution back to physical rank 0. Each group owns a
fixed, non-empty, contiguous layer range. Embedding runs only in `g0`, and the
final normalization and language-model head run only after `g3`.

This layout can interleave independent logical batches across physical ranks.
It cannot remove true autoregressive dependencies: a decode successor still
waits for its predecessor's sampled token.

## Components

The core implementation is split across six modules:

* `vllm/distributed/sawtooth.py` parses configuration, validates capabilities,
  and builds the immutable stage plan and plan identity.
* `vllm/model_executor/models/sawtooth.py` creates fixed stage callables,
  stage-specific compilation identities, MoE layer mappings, and residual
  carry schemas.
* `vllm/v1/sawtooth.py` defines logical execution identities and the
  per-request FIFO token-lease ledger.
* `vllm/v1/engine/sawtooth.py` admits logical batches, dispatches virtual
  rounds, bounds the live window, and retires terminal results.
* `vllm/v1/worker/gpu/sawtooth.py` restores stage inputs and matches sampled
  state to the exact request generation and step that produced it.
* `vllm/v1/sawtooth_trace.py` provides optional, bounded CPU JSONL tracing for
  pipeline attribution.

Shared scheduler, executor, model-runner, and model files connect these
modules. Backend implementations provide the stage transport and execution
needed by their devices.

## Execution Flow

1. Model initialization builds the same immutable stage plan on each process.
2. The scheduler admits tokens and allocates KV cache once for a logical batch.
   It records each request's generation, step, token interval, and whether the
   step produces a sample.
3. The Sawtooth engine assigns a live slot and dispatches the saved scheduler
   output through virtual rounds. A virtual round is not another admission.
4. Every worker selects its fixed local stage for the round. Intermediate
   tensors move through ordinary PP edges and the backend's wrap edge.
5. After the final round is submitted, the engine enqueues sampling. Only then
   can the scheduler certify that a request predecessor has all terminal
   commands in the worker FIFO.
6. The engine waits for the oldest terminal future before publishing output,
   committing the completed prefix boundary, releasing KV ownership, and
   returning the live slot.

Submission permits further safe work but never permits resource reclamation.
KV cache and request identity remain owned until terminal settlement, including
after abort or stop.

## Windowing and Early Admission

The live logical-batch window is bounded by `max_live_batches`, or by the
engine's normal concurrent-batch capacity when the option is omitted. A slot is
released only after terminal output has been applied by the scheduler.

Independent ready requests may fill free slots. With early admission enabled,
one successor step of the same request may also use a free slot when all of the
following hold:

* the predecessor's final execute and sample RPCs have been enqueued;
* the request is live, is not structured or resumable, and has the exact
  predecessor identity recorded in the ledger;
* no second successor is already outstanding;
* the current bounded cohort and virtual-round ordering allow a new head.

At most two unsettled steps of one request can exist. Early admission does not
release the predecessor's slot, KV blocks, or prefix-cache ownership.

The engine drains a finite admitted decode cohort and then gives prefill one
bounded admission opportunity. This prevents an unbounded stream of decode
heads from extending the drain. It does not guarantee that every host or device
gap is covered.

## Communication and Sampling Dependencies

Sawtooth execute and sample RPCs collect responses from every worker. A
non-final round must return an acknowledgement containing the exact execution
identity. The final execute round must return no model output; sampled tokens
come from the physical output rank. A missing, failed, or mismatched rank
poisons the executor so unread responses cannot be mistaken for a later RPC.

The worker reuses native PP sample broadcasts but keys pending state by request
ID, request generation, and step ID. A decode successor waits for the exact
sample receive and writeback event. A known non-final prefill successor has no
sampled token dependency and waits only for its local state-ready event.

Backends must preserve one global command order across ranks and provide the
wrap transport between the last physical rank and the next virtual group. The
core design does not add an all-rank device-completion barrier between rounds.

## Configuration

Sawtooth is disabled by default and is configured through
`additional_config.sawtooth_pipeline`. For example:

```bash
vllm serve <model> \
  --pipeline-parallel-size 2 \
  --additional-config '{
    "sawtooth_pipeline": {
      "enabled": true,
      "virtual_size": 2,
      "max_live_batches": 2,
      "enable_early_admission": true
    }
  }'
```

`layer_boundaries` can optionally provide the `PP * virtual_size + 1` layer
offsets. They must be strictly increasing, cover the complete layer interval,
and leave at least one layer in every stage. Unknown keys and invalid types are
rejected.

## Supported Scope and Failure Boundaries

The core path requires Model Runner V2, generation models, pipeline size greater
than one, and virtual size of at least two. Core model adapters are provided for
Qwen2, Qwen3, and Qwen3-MoE. A DeepSeek V4 architecture entry is accepted when
the installed backend supplies its model and transport adapter.

The feature fails closed for speculative decoding, data parallelism, prefill or
decode context parallelism, DBO, LoRA, KV or encoder connectors, KV or weight
offloading, pooling runners, and non-vLLM model implementations. Backends may
reject additional combinations such as KVPP, sparse KV offload, or DSA context
parallelism. Structured-output requests are rejected before engine admission.

Runtime checks reject invalid plan, round, slot-generation, stage-ownership,
request-predecessor, token-interval, carry-shape, and all-rank response
identities. Live virtual steps cannot be preempted or freed. When Sawtooth is
disabled, engine dispatch, scheduler admission, model forward, and executor
reply selection follow their native paths.

## Validation Status

An earlier CPU and control-plane regression run reported 216 passing tests in
19.56 seconds. It covered the stage plan, scheduling and KV ownership, logical
engine ordering, all-rank RPC behavior, tracing, and worker sample dependency
handling. It used fake device constructors and did not validate device model
execution.

The original PCP nesting in the shared GPU model runner was restored after that
run. At the user's request, no runtime tests were rerun after the restoration.
Ruff 0.14.0 formatting and static checks were run only on the four Python files
adjusted during final static cleanup. This does not update the runtime evidence,
so the historical result must not be presented as validation of the final
committed tree.

Device serving and lifecycle probes have exercised the experimental path, but
the semantic accuracy gate remains incomplete and mixed-batch output variation
has not been accepted. Existing traces also retain uncovered pipeline gaps.
The implementation must not be described as numerically accepted, generally
performance-improving, or fully masking the pipeline.

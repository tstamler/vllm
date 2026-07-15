# FT NCCL EP Dispatch/Combine Plan

## Goal

Add a single-node vLLM expert-parallel backend that uses FT NCCL's coalesced
variable all-to-all transport for routed MoE dispatch and combine. The first
implementation evaluates correctness and eager-mode performance with all ranks
active. Fault injection, expert remapping, and replay are follow-up work.

This plan supersedes the transport portions of the earlier NIXL EP and
active-mask notes. It does not attempt to replace NIXL's multi-node RDMA path.

## Verified FT NCCL Prerequisite

Use FT NCCL branch `ft-all-to-allv` at or after commit `0f14744d5`.

The following capabilities are present:

| Requirement | Verified implementation |
| --- | --- |
| Public EP transport API | `FTProcessGroup.all_to_allv_multi()` |
| Reusable registered workspace | `create_a2av_multi_workspace()` and `A2AVMultiWorkspace` |
| Device send counts | Contiguous CUDA `int32[ep_size]` tensors are consumed directly |
| Device receive counts | Discovered counts are returned as CUDA `int32[ep_size]` |
| Automatic receive sizing | Count is carried in the completion flag; no count exchange |
| Device displacement generation | `send_displs=None` runs a device prefix sum |
| Token-oriented routing | Counts and capacity are rows/tokens |
| Heterogeneous coalesced payloads | Up to eight byte-oriented buffers share one exchange |
| Stream ordering | FT stream waits for caller work; returned `FTWork` orders consumers |
| Stable allocation | Workspace scratch and metadata are reusable across calls |
| Reverse exchange test | Forward discovered counts drive a reverse all-to-allv |
| Capacity shape checks | C entry point checks capacity, row widths, and scratch strides |

The new Python test covers BF16 activations, integer IDs, float metadata, zero
counts, skew, repeated workspace reuse, reverse exchange, and optional rank
failure.

### Remaining FT NCCL Caveat

The C entry point validates the host-provided `max_send_count` hint against
`recv_capacity`, but it does not inspect every value in device `send_counts`.
The public API currently passes `recv_capacity` as that hint. Therefore the
router must guarantee:

```text
0 <= send_counts[peer] <= recv_capacity
sum(send_counts) <= number of allocated send rows
```

The PoC routing kernel should set a device overflow flag and skip the exchange
if either invariant fails. FT NCCL should eventually add device-side validation
before any peer write so an incorrect caller cannot overrun a scratch slot.

## Implemented Stage 1

The standalone PoC uses vLLM's Standard activation format instead of immediately
building an expert-major format. A token is sent once to every rank owning at
least one selected expert, together with its full top-k IDs and weights. The
existing expert map filters computation to local experts and the existing
top-k reduction produces one contribution per received token. Reverse FT A2AV
returns those contributions, and the source rank adds them using saved token
indices.

```text
top-k routes
  -> pack one token copy per distinct destination rank
  -> coalesced FT A2AV(hidden, topk_ids, topk_weights)
  -> compact source slots into Standard-format tensors
  -> existing local expert compute and top-k weight/reduce
  -> reverse FT A2AV(local contribution)
  -> source-side index_add by original token
```

This implementation is in
`vllm/model_executor/layers/fused_moe/prepare_finalize/ft_nccl_ep.py`. It keeps
symmetric buffers and registered workspaces persistent, but its initial routing
uses dynamic PyTorch indexing and copies device receive counts to the host.
Consequently, Stage 1 requires `--enforce-eager`. The expert-major and
fixed-address GPU-routing sections below describe Stage 2 optimization work,
not the current implementation.

Stage 1 supports single-node linear expert placement, modular MoE kernels,
unquantized FP16/BF16/FP32 activation transport, and fixed all-active membership.
It rejects elastic EP, round-robin placement, MoE LoRA, and quantized activation
transport rather than silently falling back.

## Backend Structure

Add `ft_nccl_ep` as a separate vLLM all-to-all backend. Do not modify the
existing `nixl_ep` behavior. NIXL exposes routing and transport through a fused
external `Buffer.dispatch()` call, so the PoC cannot replace only its transport
without first changing NIXL's public API.

The FT backend should reuse vLLM's modular MoE prepare/finalize interface and
match the tensor contract produced by NIXL and DeepEP:

```text
router output
  -> local route and pack
  -> FT all_to_allv_multi dispatch
  -> local expert-major reorder
  -> fused experts
  -> inverse expert reorder
  -> FT all_to_allv combine
  -> local weighted scatter/reduce
```

## Persistent State

Create one `FTNcclEPState` per compatible MoE shape. It should own:

- FT process group for the EP rank set.
- `A2AVMultiWorkspace` for dispatch.
- A smaller workspace for reverse combine, or the dispatch workspace when its
  buffer layout is compatible.
- Symmetric registered dispatch send buffers.
- Capacity-padded local dispatch receive buffers.
- Symmetric registered combine send buffer.
- Capacity-padded local combine receive buffer.
- Device send and receive count tensors.
- Routing histograms, prefix sums, permutations, and overflow status.
- At least one `FTMoERouteHandle`; use two slots when DBO overlap is enabled.

All allocations and registrations occur during backend initialization. There
must be no allocation, registration, host count conversion, or host
synchronization in steady state.

Suggested dispatch buffers are:

1. Activation rows: BF16 initially, optional FP8 later.
2. Destination-local expert ID: `int32`.
3. Source assignment ID: `int32`.
4. Optional per-assignment scale or metadata.

Routing weights can remain local when combine applies them on the source rank.
This avoids transporting weights unless an expert-side operation requires them.

## Route Handle

`FTMoERouteHandle` should contain only stable tensors and scalar configuration:

- Forward device receive counts.
- Original forward device send counts.
- Local assignment-to-token map.
- Local assignment routing weights.
- Dispatch destination-order permutation.
- Receiver expert-order permutation and its inverse.
- Source-slot offsets or the information needed to derive them.
- Workspace slot index.
- Membership epoch placeholder for later fault-tolerant replay.

Forward receive counts are sufficient to size the reverse exchange, but are not
sufficient to restore token identity after expert-major reordering. The local
permutations remain necessary.

## Dispatch Implementation

### 1. Map and expand routes

Input tensors are:

```text
hidden_states: [num_tokens, hidden_size]
topk_ids:      [num_tokens, top_k]
topk_weights:  [num_tokens, top_k]
```

Map global expert IDs to physical expert IDs using the existing expert mapping.
Expand each token into one assignment per selected expert. For every assignment,
compute:

```text
destination_rank = physical_expert_id // experts_per_rank
local_expert_id  = physical_expert_id % experts_per_rank
source_assignment_id = token_id * top_k + topk_slot
```

### 2. Histogram and pack on the GPU

A routing kernel should:

- Histogram assignments by destination rank into device `send_counts`.
- Prefix-sum rank counts or reserve rank-local positions atomically.
- Pack activations and metadata contiguously in destination-rank order.
- Preserve a local map from packed assignment to source token and top-k slot.
- Set an overflow flag if any peer count exceeds `recv_capacity` or total packed
  rows exceed the send allocation.

FT NCCL can derive final displacements from counts, so vLLM does not need to
materialize `send_displs` unless the pack kernel already produces them cheaply.

### 3. Exchange all fields together

Call:

```python
work, recv_counts = ft_pg.all_to_allv_multi(
    send_bufs=[packed_hidden, packed_local_expert, packed_assignment_id],
    out_bufs=[recv_hidden, recv_local_expert, recv_assignment_id],
    send_counts=send_counts,
    workspace=dispatch_workspace,
)
work.wait()
```

Each output is source-slotted. Rank `s` starts at
`s * recv_capacity`, and `recv_counts[s]` rows are valid.

### 4. Build expert-major input

A receiver kernel should consume the source slots and device receive counts,
then:

- Histogram rows by local expert.
- Produce `expert_num_tokens` and expert offsets.
- Reorder activation rows into expert-major layout.
- Record the inverse permutation from expert-major rows to source-slot rows.

Return expert-major activations and `expert_num_tokens` through the modular MoE
prepare interface.

## Combine Implementation

### 1. Restore source-rank order

After expert execution, apply the inverse receiver permutation so outputs are in
the same per-source order in which dispatch received them. Pack those rows into
a contiguous symmetric send buffer grouped by original source rank.

The forward `recv_counts` are the reverse `send_counts`:

```text
reverse_send_counts[source_rank] = forward_recv_counts[source_rank]
```

Clone or use a separate stable count tensor because the workspace receive-count
tensor is overwritten by the reverse call.

### 2. Reverse exchange

Use a one-buffer `all_to_allv_multi` call initially so combine shares the same
public API and automatic receive sizing:

```python
work, returned_counts = ft_pg.all_to_allv_multi(
    send_bufs=[packed_expert_output],
    out_bufs=[returned_assignment_output],
    send_counts=reverse_send_counts,
    workspace=combine_workspace,
)
work.wait()
```

For an all-active execution, `returned_counts` must equal the original dispatch
`send_counts` elementwise.

### 3. Weighted scatter/reduce

Use the original local dispatch permutation to map each returned assignment to
its source token and top-k slot. Apply the original routing weight and sum all
top-k contributions into `[num_tokens, hidden_size]`.

This kernel should be fused where practical:

```text
output[token] += returned_assignment * topk_weight
```

The first implementation may use FP32 accumulation followed by conversion to
the model dtype. Compare this with the existing vLLM reference behavior.

## vLLM Changes

Expected files and responsibilities:

- `vllm/config/parallel.py`
  - Add `ft_nccl_ep` to `All2AllBackend` and user-facing documentation.
- `vllm/model_executor/layers/fused_moe/config.py`
  - Add `use_ft_nccl_ep_kernels` and include it in relevant format properties.
- `vllm/distributed/device_communicators/all2all.py`
  - Add `FTNcclEPAll2AllManager` to initialize the FT EP process group and own
    persistent workspaces.
- `vllm/distributed/device_communicators/cuda_communicator.py`
  - Select the manager for `--all2all-backend ft_nccl_ep`.
- `vllm/model_executor/layers/fused_moe/all2all_utils.py`
  - Construct the new prepare/finalize implementation.
- `vllm/model_executor/layers/fused_moe/prepare_finalize/ft_nccl_ep.py`
  - Implement dispatch preparation, expert reorder, reverse combine, and route
    handle lifecycle.
- A CUDA or Triton routing module
  - Implement histogram/pack, expert-major reorder, inverse reorder, overflow
    checking, and weighted scatter/reduce.

The initial backend should support:

- Single node, LSA/NVLink only.
- Linear expert placement.
- BF16 activation transport.
- Fixed all-active EP membership.
- CUDA graph capture with stable workspace addresses.
- `top_k > 1`, including DeepSeek-V2-Lite's routed configuration.

Add FP8 transport, round-robin placement, active-mask remapping, and multi-node
transport after the baseline is correct and measured.

## Correctness Tests

### FT NCCL prerequisite test

Run the new package test after rebuilding FT NCCL:

```bash
cd /workspace/nccl/contrib/fault_tolerant_collectives/ft_handle/python
/workspace/vllm/.venv/bin/python test_alltoallv_multi.py \
  --nranks 8 --base 32 --hidden 2048 --iters 100
```

Add a negative capacity test once FT NCCL validates actual device counts.

### Routing kernel tests

Test without communication using deterministic tensors:

- `top_k=1` and DeepSeek-like `top_k`.
- Balanced routing.
- All assignments targeting one rank or expert.
- Zero assignments for selected ranks and experts.
- Duplicate expert selections if the router permits them.
- Capacity boundary and overflow.
- Pack followed by inverse permutation is identity.
- Weighted scatter/reduce matches a PyTorch reference.

### Distributed dispatch/combine tests

For 2, 4, and 8 ranks where available:

- Identity expert: dispatch plus combine equals the weighted routing reference.
- Rank-dependent expert transform detects incorrect source/destination routing.
- Heterogeneous metadata remains aligned with activation rows.
- Forward receive counts equal the transpose of send counts.
- Reverse receive counts equal the original forward send counts.
- Repeated calls reuse identical tensor addresses.
- CUDA graph capture and at least 100 replays produce stable results.
- DBO uses separate workspace slots without cross-batch corruption.

Reuse the structure of `tests/kernels/moe/test_deepep_moe.py` for numerical
comparison and `tests/distributed/test_ft_nccl_communicator.py` for process-group
startup and CUDA graph coverage.

### End-to-end model tests

Launch `deepseek-ai/DeepSeek-V2-Lite-Chat` with `TP=1`, `DP=EP=8` and compare:

- Greedy token IDs for short deterministic prompts.
- Logits or selected-token log probabilities within BF16 tolerance.
- Eager and CUDA graph execution.
- Batch sizes from 1 through the configured maximum.

## Performance Tests

### Configurations

Measure five configurations at the same vLLM and model revision:

1. `allgather_reducescatter` baseline.
2. Native `nixl_ep`.
3. Existing FT NCCL staged Ag/Rs path.
4. FT all-to-allv with activation and metadata sent separately.
5. FT `all_to_allv_multi` with coalesced activation and metadata.

Configurations 4 and 5 isolate the value of coalescing. Configurations 2 and 5
compare the complete routed implementations. Configuration 1 shows the value of
sparse routed communication relative to dense Ag/Rs.

### Transport microbenchmarks

Use 8 H100 GPUs on one NVLink/NVSwitch domain. Sweep:

- Tokens per rank: `1, 8, 32, 128, 512, 2048, 4096`.
- Hidden size: `2048` BF16 for DeepSeek-V2-Lite, plus `4096` as a stress case.
- Metadata: activation only, then activation plus two and three metadata fields.
- Routing distribution: balanced, Zipf/skewed, one hot destination, and zeros.
- CUDA graphs off and on.

Record:

- Dispatch, combine, and round-trip latency at p50/p95/p99.
- Useful payload GB/s, excluding capacity padding.
- Total bytes moved and achieved link utilization.
- Separate routing/packing and communication times.
- Workspace memory and peak allocated GPU memory.

Use at least 100 warm-up iterations and 1,000 measured iterations for small
messages. Increase measured duration until each case runs for at least 10
seconds.

### MoE layer benchmark

Instrument these phases with CUDA events:

1. Route histogram and pack.
2. FT dispatch communication.
3. Expert-major reorder.
4. Expert compute.
5. Inverse reorder and combine pack.
6. FT reverse communication.
7. Weighted scatter/reduce.

Compare total prepare/finalize time against NIXL and Ag/Rs for the same synthetic
router output. Report both balanced and skewed routes.

### API server benchmark

Extend `benchmarks/ft_nccl_ep_communicator` with server configurations for
`nixl_ep`, separate-buffer FT A2AV, and coalesced FT A2AV. Retain the current
DeepSeek-V2-Lite defaults:

```text
TP=1
DP=8
EP=8
dtype=bfloat16
max_model_len=4096
max_num_batched_tokens=4096
CUDA graphs enabled
```

Run both existing workloads:

```text
decode-heavy:  input=128,  output=1024, prompts=512, warmups=64
prefill-heavy: input=2048, output=128,  prompts=512, warmups=64
```

Also add a low-concurrency decode test with concurrency `1, 8, 32`, because EP
collective latency is more visible there than at saturation.

For every configuration, run one untimed full warm-up followed by at least three
measured repetitions. Randomize configuration order when practical.

Save:

- Request throughput.
- Input, output, and total token throughput.
- TTFT and TPOT p50/p95/p99.
- End-to-end request latency p50/p95/p99.
- Server and client logs.
- Backend-selection log lines.
- vLLM, FT NCCL, NIXL, PyTorch, CUDA, driver, and model revisions.
- Environment variables, GPU clocks/power settings, and `nvidia-smi topo -m`.

## Acceptance Criteria

The PoC is ready for performance evaluation when:

1. Distributed identity dispatch/combine matches the PyTorch reference.
2. Stage 1 serves DeepSeek-V2-Lite requests in eager mode.
3. Stage 1 performs no steady-state symmetric allocation or registration.
4. Device count overflow fails before communication.
5. Stage 1 completes the eager benchmark matrix with reproducible logs and at
   least three measured repetitions.

CUDA graph execution, allocation-free routing, and removal of D2H count copies
are Stage 2 acceptance criteria.

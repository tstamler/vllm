# Proposal: Split NIXL EP Buffer API

## Motivation

NIXL EP currently exposes fused `Buffer.dispatch()` and `Buffer.combine()`
operations. This prevents an integrator from reusing NIXL's routing, packing,
expert layout, and reduction kernels while substituting another transport such
as FT NCCL.

Add a split API that keeps NIXL's route representation opaque but exposes the
packed device buffers and device-resident peer counts at the transport boundary.

## Proposed API

```python
send_buffers, send_counts, route = buffer.prepare_dispatch(
    hidden_states,
    physical_topk_ids,
    max_tokens_per_rank,
    num_experts,
    out=dispatch_send_buffers,
)

expert_x, expert_num_tokens = buffer.finish_dispatch(
    recv_buffers,
    recv_counts,
    route,
    out=expert_buffers,
)

combine_send, reverse_counts = buffer.prepare_combine(
    fused_expert_output,
    route,
    out=combine_send_buffer,
)

output = buffer.finish_combine(
    combine_recv,
    returned_counts,
    route,
    topk_weights,
    out=output,
)
```

`prepare_dispatch()` should perform expert mapping, token expansion, peer
histograms, and destination-major packing. `finish_dispatch()` should transform
source-major received records into expert-major input and retain the inverse
permutation. `prepare_combine()` should restore source-major order.
`finish_combine()` should map returned assignments to source tokens, apply
routing weights, and reduce top-k contributions.

## Contract

- Counts are contiguous CUDA `int32[num_ranks]` tensors in token rows.
- Packed buffers and counts remain valid until the corresponding finish call.
- The route handle is opaque and owns all permutations and token metadata.
- Caller-provided output buffers permit stable addresses and CUDA graph capture.
- Methods enqueue work on the current CUDA stream without host synchronization.
- Preparation reports required row capacity before transport begins.
- Existing fused `dispatch()` and `combine()` remain as convenience wrappers
  implemented using the split API and NIXL's native transport.

With this boundary, vLLM can place `FTProcessGroup.all_to_allv_multi()` between
the prepare and finish calls while retaining NIXL's optimized local kernels.

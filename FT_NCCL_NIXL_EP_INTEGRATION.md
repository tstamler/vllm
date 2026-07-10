# FT NCCL for NIXL EP Dispatch/Combine

## Goal

Evaluate what would be required to use FT NCCL for the expert-parallel dispatch and
combine path currently handled by the NIXL EP kernels.

## Key conclusion

FT NCCL would need a fault-tolerant `all_to_allv`-style primitive to replace NIXL EP
dispatch/combine directly.

The existing FT NCCL `all_reduce` and `all_gather` support is enough for the simpler
`allgather_reducescatter` EP backend, but not for NIXL EP's token-routed exchange.

## Why existing FT collectives are not enough

The `allgather_reducescatter` backend has this shape:

- dispatch: `all_gatherv`
- combine: `reduce_scatterv`

Those can be approximated with current FT NCCL primitives:

- `all_gatherv`: pad to the largest local shard, run FT `all_gather`, then trim.
- `reduce_scatterv`: run FT `all_reduce` over the full tensor, then return the local slice.

NIXL EP dispatch/combine has a different shape:

- each token is sent only to the rank that owns its selected expert
- each source/destination pair can have a different number of tokens
- dispatch produces expert-grouped local buffers plus token-count metadata
- combine sends expert outputs back to original token owners
- combine also restores original token order and applies top-k weights/reduction

That is a variable-size personalized exchange, i.e. `all_to_allv`, plus pack/unpack
metadata. Emulating it with all-gather would work only as a slow fallback because every
rank would receive far more data than it needs.

## Minimal integration shape

1. Add a gated backend or mode, for example `ft_nccl_ep` or
   `VLLM_USE_FT_NCCL_NIXL_EP=1`.

2. Create an FT NCCL process group for the EP group, separate from the TP group.

3. Preallocate or register symmetric EP workspaces during initialization:
   - dispatch send/recv payload buffers
   - combine send/recv payload buffers
   - per-peer counts and offsets
   - routing metadata needed to reverse the combine
   - optional buffers for top-k ids, top-k weights, scales, and LoRA mapping

4. Dispatch flow:
   - map `topk_ids` to physical expert/rank ids
   - count tokens per destination rank
   - pack hidden states and metadata by destination
   - exchange counts/offsets
   - run FT `all_to_allv`
   - unpack into `expert_x`, `expert_num_tokens`, and a handle for combine

5. Combine flow:
   - use the dispatch handle to map expert outputs back to source ranks/tokens
   - pack `fused_expert_output` by original source rank
   - run FT `all_to_allv` in reverse
   - unpack into original token order
   - apply top-k weights/reduction into the final output tensor

6. Failure handling:
   - for performance-only experiments, require a full active mask and error out on any
     missing peer
   - for a real FT path, propagate the FT active mask into the existing EP recovery flow,
     fail in-flight requests that touched masked collectives, and trigger EPLB recovery

## Practical recommendation

For short-term benchmarking, keep using FT NCCL for TP collectives and use the existing
staged FT `all_gatherv` / `reduce_scatterv` path only with the Ag/Rs EP backend.

For a faithful NIXL EP replacement, first add or expose FT NCCL `all_to_allv`. Without it,
the integration would either be incomplete or would become an all-gather-based fallback
that is unlikely to represent NIXL EP performance.

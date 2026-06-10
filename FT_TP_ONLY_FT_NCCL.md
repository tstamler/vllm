# TP-Only `ft_nccl` Integration

Goal: use `ft_collective` / `ft_nccl` for vLLM tensor-parallel (TP)
all-reduce only. EP dispatch/combine is out of scope for now.

## Current TP Path

TP all-reduce enters vLLM here:

- `GroupCoordinator.all_reduce` in `vllm/distributed/parallel_state.py`
- `CudaCommunicator.all_reduce` in
  `vllm/distributed/device_communicators/cuda_communicator.py`

`CudaCommunicator.all_reduce` normally tries several CUDA-side paths before
falling back to `torch.distributed.all_reduce`:

1. NCCL symmetric-memory custom op
2. quick reduce
3. flashinfer all-reduce
4. custom all-reduce
5. torch symmetric-memory communicator
6. PyNCCL
7. torch distributed fallback

`ft_nccl` only participates in step 7, so the earlier paths must be bypassed
for TP if the goal is to route TP collectives through `ft_nccl`.

## Minimal Changes

1. Import `ft_collective` before distributed initialization.

   Importing `ft_collective` registers the `"ft_nccl"` torch distributed backend.
   This must happen before vLLM creates process groups.

2. Add a small enable flag.

   Example env flag:

   ```text
   VLLM_USE_FT_NCCL_TP=1
   ```

   The FT all-reduce kernel also preallocates scratch buffers sized by element
   count. `ft_collective` defaults this to 4M elements and can override it
   with:

   ```text
   FT_NCCL_MAX_COUNT=16777216
   ```

   This must be at least as large as the largest TP all-reduce input numel
   expected during profiling or serving, for example
   `max_num_batched_tokens * hidden_size` for dense hidden-state reductions.

3. Create the TP device group with backend `"ft_nccl"`.

   The TP group is created in `initialize_model_parallel` when calling
   `init_model_parallel_group(..., group_name="tp")`. Under the new flag, pass
   `"ft_nccl"` for this group only. Leave world, DP, EP, PP, etc. on the normal
   backend.

4. Special-case TP all-reduce when the flag is enabled.

   `CudaCommunicator.all_reduce` checks `VLLM_USE_FT_NCCL_TP` first for the TP
   group and routes directly to the FT path before custom all-reduce, PyNCCL,
   flashinfer, or other TP all-reduce backends.

5. Require directly registered symmetric inputs.

   This is the key correctness caveat. `FTProcessGroup` only uses the FT kernel
   when the input tensor is in its registered symmetric-window set, either
   because it was allocated with `FTProcessGroup.empty()` or because it was
   externally allocated in symmetric memory and registered. A normal
   `input_.clone()` will fall back to regular NCCL inside `ft_nccl`.

   If a TP input is already registered with `FTProcessGroup`, or if it is a
   vLLM NCCL symmetric-memory tensor that can be registered with
   `register_symmetric_tensor()`, the communicator calls `torch.distributed`
   directly on that input. For tensors allocated from vLLM's NCCL
   symmetric-memory pool, register the pool segment as the primary FT window
   first, then register the tensor pointer as a derived alias:

   ```python
   ft_pg.register_symmetric_tensor(base_ptr=segment_ptr, size_bytes=segment_size)
   ft_pg.register_symmetric_tensor(input_)  # alias within the segment
   torch.distributed.all_reduce(input_, group=self.device_group)
   return input_
   ```

   This direct path is in-place. It avoids staging copies, but it depends on
   the TP producer treating the all-reduce input as disposable.

   If the TP input cannot be used directly with `ft_nccl`, the communicator
   raises an error instead of copying through a staging buffer.

   The existing `ft_collective.get_ft_process_group()` registry is rank-keyed,
   which is probably not precise enough once vLLM creates multiple groups.

6. Allocate known TP all-reduce inputs from symmetric memory.

   For `RowParallelLinear`, wrap the `quant_method.apply(...)` output
   allocation in `nccl_symm_mem_context(...)` when `VLLM_USE_FT_NCCL_TP=1`.
   For `VocabParallelEmbedding`, do the same for the embedding output
   allocation. This makes the all-reduce input itself a vLLM NCCL
   symmetric-memory tensor, so the FT communicator can register it directly
   with `register_symmetric_tensor()` before the in-place all-reduce.

   The FT TP flag enables this allocator path without enabling vLLM's existing
   NCCL symmetric-memory all-reduce copy path.

   The NCCL symmetric allocator is initialized when the TP CUDA communicator
   is created, before Dynamo traces model forwards. This avoids tracing the
   filesystem checks used by PyTorch extension loading.

   The allocation-producing `RowParallelLinear` and `VocabParallelEmbedding`
   FT helpers are marked with `torch.compiler.disable` so Dynamo does not
   trace the symmetric-memory context manager.

## Suggested First Milestone

Implement only this path:

- `VLLM_USE_FT_NCCL_TP=1`
- import/register `ft_collective`
- optionally configure `FT_NCCL_MAX_COUNT`
- TP group backend becomes `"ft_nccl"`
- `CudaCommunicator.all_reduce` special-cases `unique_name.split(":")[0] == "tp"`
- `RowParallelLinear` and `VocabParallelEmbedding` allocate TP all-reduce
  inputs from symmetric memory
- symmetric TP inputs are registered and reduced directly
- other TP inputs raise an error

Everything else should stay on the existing vLLM communication paths.

## Out of Scope

- EP `all_gatherv`
- EP `reduce_scatterv`
- native FT reduce-scatter
- variadic FT all-gather
- continuing execution with partial-rank result masks

For the first TP-only version, treat an FT timeout as a fail-fast condition
unless a higher-level recovery policy is added later.

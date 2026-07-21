# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.ubatching import dbo_enabled


@dataclass
class _FTNcclEPRoute:
    positions: torch.Tensor
    recv_counts: torch.Tensor
    num_input_tokens: int


@triton.jit
def _ft_nccl_ep_reset_counts_kernel(send_counts_ptr, ep_size: tl.constexpr):
    destination = tl.program_id(0)
    tl.store(send_counts_ptr + destination, 0, mask=destination < ep_size)


@triton.jit
def _ft_nccl_ep_prefix_displacements_kernel(
    counts_ptr,
    displacements_ptr,
    ep_size: tl.constexpr,
):
    offset = 0
    for destination in tl.static_range(ep_size):
        tl.store(displacements_ptr + destination, offset)
        offset += tl.load(counts_ptr + destination)


@triton.jit
def _ft_nccl_ep_route_kernel(
    topk_ids_ptr,
    send_counts_ptr,
    positions_ptr,
    num_tokens: tl.constexpr,
    ep_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    top_k: tl.constexpr,
):
    token = tl.program_id(0)
    destination = tl.program_id(1)
    selected = False
    for topk_idx in tl.static_range(top_k):
        expert = tl.load(topk_ids_ptr + token * top_k + topk_idx)
        selected |= expert // experts_per_rank == destination

    position = tl.atomic_add(send_counts_ptr + destination, 1, mask=selected)
    tl.store(
        positions_ptr + token * ep_size + destination,
        tl.where(selected, position, -1),
    )


@triton.jit
def _ft_nccl_ep_pack_kernel(
    hidden_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    positions_ptr,
    send_displacements_ptr,
    send_hidden_ptr,
    send_ids_ptr,
    send_weights_ptr,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    ep_size: tl.constexpr,
    capacity: tl.constexpr,
    packed: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    destination = tl.program_id(1)
    block = tl.program_id(2)
    position = tl.load(positions_ptr + token * ep_size + destination)
    selected = (position >= 0) & (position < capacity)
    if packed:
        destination_row = tl.load(send_displacements_ptr + destination) + position
    else:
        destination_row = destination * capacity + position

    hidden_offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = selected & (hidden_offsets < hidden_size)
    hidden = tl.load(
        hidden_ptr + token * hidden_size + hidden_offsets,
        mask=hidden_mask,
    )
    tl.store(
        send_hidden_ptr + destination_row * hidden_size + hidden_offsets,
        hidden,
        mask=hidden_mask,
    )

    metadata_mask = selected & (hidden_offsets < top_k)
    ids = tl.load(
        topk_ids_ptr + token * top_k + hidden_offsets,
        mask=metadata_mask,
    )
    weights = tl.load(
        topk_weights_ptr + token * top_k + hidden_offsets,
        mask=metadata_mask,
    )
    tl.store(
        send_ids_ptr + destination_row * top_k + hidden_offsets,
        ids,
        mask=metadata_mask,
    )
    tl.store(
        send_weights_ptr + destination_row * top_k + hidden_offsets,
        weights,
        mask=metadata_mask,
    )


@triton.jit
def _ft_nccl_ep_combine_kernel(
    recv_ptr,
    positions_ptr,
    output_ptr,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    ep_size: tl.constexpr,
    capacity: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for source in tl.static_range(ep_size):
        position = tl.load(positions_ptr + token * ep_size + source)
        selected = (position >= 0) & (position < capacity)
        source_row = source * capacity + position
        contribution = tl.load(
            recv_ptr + source_row * hidden_size + offsets,
            mask=mask & selected,
            other=0.0,
        )
        result += contribution
    tl.store(output_ptr + token * hidden_size + offsets, result, mask=mask)


@triton.jit
def _ft_nccl_ep_pack_combine_kernel(
    contributions_ptr,
    counts_ptr,
    displacements_ptr,
    packed_ptr,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    ep_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    destination = tl.program_id(0)
    position = tl.program_id(1)
    offsets = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    selected = position < tl.load(counts_ptr + destination)
    hidden_mask = selected & (offsets < hidden_size)
    source_row = destination * num_tokens + position
    destination_row = tl.load(displacements_ptr + destination) + position
    contribution = tl.load(
        contributions_ptr + source_row * hidden_size + offsets,
        mask=hidden_mask,
    )
    tl.store(
        packed_ptr + destination_row * hidden_size + offsets,
        contribution,
        mask=hidden_mask,
    )


def _ft_nccl_ep_pack(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    send_counts: torch.Tensor,
    positions: torch.Tensor,
    send_displacements: torch.Tensor,
    send_hidden: torch.Tensor,
    send_ids: torch.Tensor,
    send_weights: torch.Tensor,
    ep_size: int,
    experts_per_rank: int,
    capacity: int,
    packed: bool,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    _ft_nccl_ep_reset_counts_kernel[(ep_size,)](
        send_counts,
        ep_size=ep_size,
    )
    _ft_nccl_ep_route_kernel[(num_tokens, ep_size)](
        topk_ids,
        send_counts,
        positions,
        num_tokens=num_tokens,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        top_k=top_k,
    )
    if packed:
        _ft_nccl_ep_prefix_displacements_kernel[(1,)](
            send_counts,
            send_displacements,
            ep_size=ep_size,
        )
    block_size = min(triton.next_power_of_2(max(hidden_size, top_k)), 256)
    _ft_nccl_ep_pack_kernel[
        (num_tokens, ep_size, triton.cdiv(hidden_size, block_size))
    ](
        hidden_states,
        topk_ids,
        topk_weights,
        positions,
        send_displacements,
        send_hidden,
        send_ids,
        send_weights,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
        ep_size=ep_size,
        capacity=capacity,
        packed=packed,
        BLOCK_SIZE=block_size,
    )


def _ft_nccl_ep_pack_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    send_counts: torch.Tensor,
    positions: torch.Tensor,
    send_displacements: torch.Tensor,
    send_hidden: torch.Tensor,
    send_ids: torch.Tensor,
    send_weights: torch.Tensor,
    ep_size: int,
    experts_per_rank: int,
    capacity: int,
    packed: bool,
) -> None:
    return None


direct_register_custom_op(
    op_name="ft_nccl_ep_pack",
    op_func=_ft_nccl_ep_pack,
    mutates_args=[
        "send_counts",
        "positions",
        "send_displacements",
        "send_hidden",
        "send_ids",
        "send_weights",
    ],
    fake_impl=_ft_nccl_ep_pack_fake,
)


def _ft_nccl_ep_pack_combine(
    local_contributions: torch.Tensor,
    send_counts: torch.Tensor,
    send_displacements: torch.Tensor,
    packed: torch.Tensor,
    ep_size: int,
) -> None:
    num_tokens = local_contributions.shape[0] // ep_size
    hidden_size = local_contributions.shape[1]
    _ft_nccl_ep_prefix_displacements_kernel[(1,)](
        send_counts,
        send_displacements,
        ep_size=ep_size,
    )
    block_size = min(triton.next_power_of_2(hidden_size), 256)
    _ft_nccl_ep_pack_combine_kernel[
        (ep_size, num_tokens, triton.cdiv(hidden_size, block_size))
    ](
        local_contributions,
        send_counts,
        send_displacements,
        packed,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        ep_size=ep_size,
        BLOCK_SIZE=block_size,
    )


def _ft_nccl_ep_pack_combine_fake(
    local_contributions: torch.Tensor,
    send_counts: torch.Tensor,
    send_displacements: torch.Tensor,
    packed: torch.Tensor,
    ep_size: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="ft_nccl_ep_pack_combine",
    op_func=_ft_nccl_ep_pack_combine,
    mutates_args=["send_displacements", "packed"],
    fake_impl=_ft_nccl_ep_pack_combine_fake,
)


def _ft_nccl_ep_combine(
    recv: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    ep_size: int,
    capacity: int,
) -> None:
    num_tokens, hidden_size = output.shape
    block_size = min(triton.next_power_of_2(hidden_size), 256)
    _ft_nccl_ep_combine_kernel[(num_tokens, triton.cdiv(hidden_size, block_size))](
        recv,
        positions,
        output,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        ep_size=ep_size,
        capacity=capacity,
        BLOCK_SIZE=block_size,
    )


def _ft_nccl_ep_combine_fake(
    recv: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    ep_size: int,
    capacity: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="ft_nccl_ep_combine",
    op_func=_ft_nccl_ep_combine,
    mutates_args=["output"],
    fake_impl=_ft_nccl_ep_combine_fake,
)


class FTNcclEPHandle:
    """Persistent FT NCCL buffers for one routed MoE tensor configuration."""

    def __init__(
        self,
        ft_process_group: Any,
        ep_rank: int,
        ep_size: int,
        max_num_tokens_per_rank: int,
        token_hidden_size: int,
        num_global_experts: int,
        num_experts_per_token: int,
        input_dtype: torch.dtype,
        topk_weights_dtype: torch.dtype,
        use_multi_a2av: bool = True,
    ) -> None:
        if num_global_experts % ep_size != 0:
            raise ValueError("ft_nccl_ep requires an equal, linear expert partition")
        if input_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"ft_nccl_ep does not support {input_dtype=}")

        self.pg = ft_process_group
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.max_tokens = max_num_tokens_per_rank
        self.hidden_size = token_hidden_size
        self.num_experts = num_global_experts
        self.top_k = num_experts_per_token
        self.experts_per_rank = num_global_experts // ep_size
        self.padding_expert = ep_rank * self.experts_per_rank
        self.input_dtype = input_dtype
        self.weights_dtype = topk_weights_dtype
        self.use_multi_a2av = use_multi_a2av

        if not use_multi_a2av:
            from ft_collective import set_active_pg

            set_active_pg(self.pg)
            for method in ("_order_after_current", "_ft_work"):
                if not hasattr(self.pg, method):
                    raise RuntimeError(
                        "ft_nccl_a2av requires an FTProcessGroup with "
                        f"{method}() stream-ordering support"
                    )

        # A token is sent at most once to each rank, even when several selected
        # experts reside there. The full top-k metadata lets the standard MoE
        # kernel select only the receiver's local experts.
        max_send_rows = ep_size * max_num_tokens_per_rank
        max_recv_rows = ep_size * max_num_tokens_per_rank
        self.send_hidden = self.pg.empty(
            max_send_rows * token_hidden_size, dtype=input_dtype
        ).view(max_send_rows, token_hidden_size)
        self.send_ids = self.pg.empty(
            max_send_rows * num_experts_per_token, dtype=torch.int64
        ).view(max_send_rows, num_experts_per_token)
        self.send_weights = self.pg.empty(
            max_send_rows * num_experts_per_token, dtype=topk_weights_dtype
        ).view(max_send_rows, num_experts_per_token)
        self.recv_hidden_slots = torch.empty(
            max_recv_rows,
            token_hidden_size,
            dtype=input_dtype,
            device=self.send_hidden.device,
        )
        self.recv_ids_slots = torch.empty(
            max_recv_rows,
            num_experts_per_token,
            dtype=torch.int64,
            device=self.send_hidden.device,
        )
        self.recv_weights_slots = torch.empty(
            max_recv_rows,
            num_experts_per_token,
            dtype=topk_weights_dtype,
            device=self.send_hidden.device,
        )
        self.recv_hidden = torch.empty_like(self.recv_hidden_slots)
        self.recv_ids = torch.empty_like(self.recv_ids_slots)
        self.recv_weights = torch.empty_like(self.recv_weights_slots)
        self.send_counts = torch.empty(
            ep_size, dtype=torch.int32, device=self.send_hidden.device
        )
        self.send_displacements = (
            torch.arange(ep_size, dtype=torch.int64, device=self.send_hidden.device)
            * max_num_tokens_per_rank
        )
        self.route_positions = torch.empty(
            max_num_tokens_per_rank,
            ep_size,
            dtype=torch.int32,
            device=self.send_hidden.device,
        )

        self.dispatch_workspace = None
        if use_multi_a2av:
            dispatch_row_bytes = [
                token_hidden_size * input_dtype.itemsize,
                num_experts_per_token * torch.int64.itemsize,
                num_experts_per_token * topk_weights_dtype.itemsize,
            ]
            self.dispatch_workspace = self.pg.create_a2av_multi_workspace(
                dispatch_row_bytes, max_num_tokens_per_rank
            )
        else:
            self.single_recv_counts = torch.empty_like(self.send_counts)
            self.single_returned_counts = torch.empty_like(self.send_counts)

        self.combine_send = self.pg.empty(
            max_recv_rows * token_hidden_size, dtype=input_dtype
        ).view(max_recv_rows, token_hidden_size)
        self.combine_recv_slots = torch.empty_like(self.recv_hidden_slots)
        self.combine_workspace = None
        if use_multi_a2av:
            self.combine_workspace = self.pg.create_a2av_multi_workspace(
                [token_hidden_size * input_dtype.itemsize], max_num_tokens_per_rank
            )

    def _single_all_to_allv(
        self,
        send: torch.Tensor,
        output: torch.Tensor,
        send_counts: torch.Tensor,
        recv_counts: torch.Tensor,
    ) -> None:
        row_size = send.shape[-1] if send.dim() > 1 else 1
        torch.ops.ft_collective.alltoallv(
            send,
            output,
            send_counts,
            recv_counts,
            self.max_tokens,
            row_size,
        )

    def _check_status(self, operation: str) -> None:
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            return
        status = self.pg.check_and_clear_error()
        if status != 0:
            raise RuntimeError(f"FT NCCL EP {operation} failed with status {status}")

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, _FTNcclEPRoute]:
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_tokens:
            raise ValueError(
                f"ft_nccl_ep received {num_tokens} tokens, capacity is "
                f"{self.max_tokens}"
            )
        if hidden_states.shape != (num_tokens, self.hidden_size):
            raise ValueError("ft_nccl_ep hidden-state shape changed after setup")
        if hidden_states.dtype != self.input_dtype:
            raise ValueError("ft_nccl_ep hidden-state dtype changed after setup")
        if not hidden_states.is_contiguous():
            raise ValueError("ft_nccl_ep requires contiguous hidden states")
        expected_topk_shape = (num_tokens, self.top_k)
        if topk_ids.shape != expected_topk_shape or topk_ids.dtype != torch.int64:
            raise ValueError(f"ft_nccl_ep requires int64 topk_ids{expected_topk_shape}")
        if not topk_ids.is_contiguous():
            raise ValueError("ft_nccl_ep requires contiguous top-k IDs")
        if (
            topk_weights.shape != expected_topk_shape
            or topk_weights.dtype != self.weights_dtype
        ):
            raise ValueError("ft_nccl_ep top-k weight shape or dtype changed")
        if not topk_weights.is_contiguous():
            raise ValueError("ft_nccl_ep requires contiguous top-k weights")

        positions = self.route_positions[:num_tokens]
        torch.ops.vllm.ft_nccl_ep_pack(
            hidden_states,
            topk_ids,
            topk_weights,
            self.send_counts,
            positions,
            self.send_displacements,
            self.send_hidden,
            self.send_ids,
            self.send_weights,
            self.ep_size,
            self.experts_per_rank,
            self.max_tokens,
            not self.use_multi_a2av,
        )
        for source in range(self.ep_size):
            slot = source * self.max_tokens
            slot_end = slot + num_tokens
            self.recv_ids_slots[slot:slot_end].fill_(self.padding_expert)
            self.recv_weights_slots[slot:slot_end].zero_()

        if self.use_multi_a2av:
            work, device_recv_counts = self.pg.all_to_allv_multi(
                [self.send_hidden, self.send_ids, self.send_weights],
                [
                    self.recv_hidden_slots,
                    self.recv_ids_slots,
                    self.recv_weights_slots,
                ],
                self.send_counts,
                self.dispatch_workspace,
                self.send_displacements,
            )
            work.wait()
        else:
            self.pg._order_after_current()
            self._single_all_to_allv(
                self.send_hidden,
                self.recv_hidden_slots,
                self.send_counts,
                self.single_recv_counts,
            )
            self._single_all_to_allv(
                self.send_ids.view(torch.float32),
                self.recv_ids_slots.view(torch.float32),
                self.send_counts,
                self.single_recv_counts,
            )
            self._single_all_to_allv(
                self.send_weights,
                self.recv_weights_slots,
                self.send_counts,
                self.single_recv_counts,
            )
            self.pg._ft_work().wait()
            device_recv_counts = self.single_recv_counts
        self._check_status("dispatch")

        # Keep the routed shape fixed for a captured input shape. Padding rows
        # use a valid local expert ID because the standard Triton MoE assignment
        # indexes expert_map before filtering. Zero weights make them inert.
        recv_rows = self.ep_size * num_tokens
        for source in range(self.ep_size):
            output_start = source * num_tokens
            output_end = output_start + num_tokens
            slot = source * self.max_tokens
            slot_end = slot + num_tokens
            self.recv_hidden[output_start:output_end].copy_(
                self.recv_hidden_slots[slot:slot_end]
            )
            self.recv_ids[output_start:output_end].copy_(
                self.recv_ids_slots[slot:slot_end]
            )
            self.recv_weights[output_start:output_end].copy_(
                self.recv_weights_slots[slot:slot_end]
            )

        route = _FTNcclEPRoute(
            positions=positions,
            recv_counts=device_recv_counts,
            num_input_tokens=num_tokens,
        )
        return (
            self.recv_hidden[:recv_rows],
            self.recv_ids[:recv_rows],
            self.recv_weights[:recv_rows],
            route,
        )

    def combine(
        self,
        local_contributions: torch.Tensor,
        route: _FTNcclEPRoute,
        output: torch.Tensor,
    ) -> None:
        expected_rows = self.ep_size * route.num_input_tokens
        if local_contributions.shape != (expected_rows, self.hidden_size):
            raise ValueError(
                "ft_nccl_ep combine input does not match the dispatch route"
            )
        if output.shape != (route.num_input_tokens, self.hidden_size):
            raise ValueError("ft_nccl_ep output does not match the source tokens")

        if self.use_multi_a2av:
            for destination in range(self.ep_size):
                input_start = destination * route.num_input_tokens
                input_end = input_start + route.num_input_tokens
                slot = destination * self.max_tokens
                self.combine_send[slot : slot + route.num_input_tokens].copy_(
                    local_contributions[input_start:input_end]
                )
            work, device_returned_counts = self.pg.all_to_allv_multi(
                [self.combine_send],
                [self.combine_recv_slots],
                route.recv_counts,
                self.combine_workspace,
                self.send_displacements,
            )
            work.wait()
        else:
            torch.ops.vllm.ft_nccl_ep_pack_combine(
                local_contributions,
                route.recv_counts,
                self.send_displacements,
                self.combine_send,
                self.ep_size,
            )
            self.pg._order_after_current()
            self._single_all_to_allv(
                self.combine_send,
                self.combine_recv_slots,
                route.recv_counts,
                self.single_returned_counts,
            )
            self.pg._ft_work().wait()
            device_returned_counts = self.single_returned_counts
        self._check_status("combine")
        del device_returned_counts
        torch.ops.vllm.ft_nccl_ep_combine(
            self.combine_recv_slots,
            route.positions,
            output,
            self.ep_size,
            self.max_tokens,
        )


class FTNcclEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Standard-format routed EP using FT NCCL variable all-to-all."""

    def __init__(self, handle: FTNcclEPHandle, num_dispatchers: int) -> None:
        super().__init__()
        self.handle = handle
        self.num_dispatchers_ = num_dispatchers
        self.route: _FTNcclEPRoute | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if dbo_enabled():
            raise NotImplementedError("ft_nccl_ep does not support DBO yet")
        if num_experts != self.handle.num_experts:
            raise ValueError("ft_nccl_ep expert count changed after setup")
        if quant_config.quant_dtype is not None and not defer_input_quant:
            raise NotImplementedError(
                "ft_nccl_ep currently supports unquantized activation transport"
            )
        if apply_router_weight_on_input:
            if topk_ids.shape[1] != 1:
                raise ValueError(
                    "apply_router_weight_on_input requires top_k=1 for ft_nccl_ep"
                )
            a1 = a1 * topk_weights.to(a1.dtype)

        recv_hidden, recv_ids, recv_weights, self.route = self.handle.dispatch(
            a1, topk_ids, topk_weights
        )
        return recv_hidden, None, None, recv_ids, recv_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if self.route is None:
            raise RuntimeError("ft_nccl_ep combine called without a dispatch route")
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        local_contributions = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        self.handle.combine(local_contributions, self.route, output)
        self.route = None

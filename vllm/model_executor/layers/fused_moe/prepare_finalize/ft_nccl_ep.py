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
from vllm.v1.worker.ubatching import dbo_enabled


@dataclass
class _FTNcclEPRoute:
    send_token_indices: list[torch.Tensor]
    send_counts: list[int]
    recv_counts: list[int]
    num_input_tokens: int


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
        self.input_dtype = input_dtype
        self.weights_dtype = topk_weights_dtype

        # A token is sent at most once to each rank, even when several selected
        # experts reside there. The full top-k metadata lets the standard MoE
        # kernel select only the receiver's local experts.
        max_send_rows = ep_size * max_num_tokens_per_rank
        max_recv_rows = ep_size * max_num_tokens_per_rank
        self.send_hidden = self.pg.empty(
            max_send_rows, token_hidden_size, dtype=input_dtype
        )
        self.send_ids = self.pg.empty(
            max_send_rows, num_experts_per_token, dtype=torch.int64
        )
        self.send_weights = self.pg.empty(
            max_send_rows, num_experts_per_token, dtype=topk_weights_dtype
        )
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

        dispatch_row_bytes = [
            token_hidden_size * input_dtype.itemsize,
            num_experts_per_token * torch.int64.itemsize,
            num_experts_per_token * topk_weights_dtype.itemsize,
        ]
        self.dispatch_workspace = self.pg.create_a2av_multi_workspace(
            dispatch_row_bytes, max_num_tokens_per_rank
        )

        self.combine_send = self.pg.empty(
            max_recv_rows, token_hidden_size, dtype=input_dtype
        )
        self.combine_recv_slots = torch.empty_like(self.recv_hidden_slots)
        self.combine_workspace = self.pg.create_a2av_multi_workspace(
            [token_hidden_size * input_dtype.itemsize], max_num_tokens_per_rank
        )

    def _check_status(self, operation: str) -> None:
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
        expected_topk_shape = (num_tokens, self.top_k)
        if topk_ids.shape != expected_topk_shape or topk_ids.dtype != torch.int64:
            raise ValueError(f"ft_nccl_ep requires int64 topk_ids{expected_topk_shape}")
        if (
            topk_weights.shape != expected_topk_shape
            or topk_weights.dtype != self.weights_dtype
        ):
            raise ValueError("ft_nccl_ep top-k weight shape or dtype changed")

        send_counts: list[int] = []
        send_token_indices: list[torch.Tensor] = []
        send_offset = 0
        for destination in range(self.ep_size):
            first_expert = destination * self.experts_per_rank
            last_expert = first_expert + self.experts_per_rank
            selected = torch.any(
                (topk_ids >= first_expert) & (topk_ids < last_expert), dim=1
            )
            token_indices = torch.nonzero(selected, as_tuple=False).flatten()
            count = token_indices.numel()
            if count > self.max_tokens:
                raise RuntimeError(
                    f"ft_nccl_ep route to rank {destination} has {count} rows, "
                    f"capacity is {self.max_tokens}"
                )
            end = send_offset + count
            if count:
                torch.index_select(
                    hidden_states,
                    0,
                    token_indices,
                    out=self.send_hidden[send_offset:end],
                )
                torch.index_select(
                    topk_ids,
                    0,
                    token_indices,
                    out=self.send_ids[send_offset:end],
                )
                torch.index_select(
                    topk_weights,
                    0,
                    token_indices,
                    out=self.send_weights[send_offset:end],
                )
            send_counts.append(count)
            send_token_indices.append(token_indices)
            send_offset = end

        work, device_recv_counts = self.pg.all_to_allv_multi(
            [self.send_hidden, self.send_ids, self.send_weights],
            [
                self.recv_hidden_slots,
                self.recv_ids_slots,
                self.recv_weights_slots,
            ],
            send_counts,
            self.dispatch_workspace,
        )
        work.wait()
        recv_counts = [int(v) for v in device_recv_counts.cpu().tolist()]
        self._check_status("dispatch")

        recv_offset = 0
        for source, count in enumerate(recv_counts):
            if count > self.max_tokens:
                raise RuntimeError(
                    f"ft_nccl_ep received {count} rows from rank {source}, "
                    f"capacity is {self.max_tokens}"
                )
            end = recv_offset + count
            slot = source * self.max_tokens
            if count:
                self.recv_hidden[recv_offset:end].copy_(
                    self.recv_hidden_slots[slot : slot + count]
                )
                self.recv_ids[recv_offset:end].copy_(
                    self.recv_ids_slots[slot : slot + count]
                )
                self.recv_weights[recv_offset:end].copy_(
                    self.recv_weights_slots[slot : slot + count]
                )
            recv_offset = end

        route = _FTNcclEPRoute(
            send_token_indices=send_token_indices,
            send_counts=send_counts,
            recv_counts=recv_counts,
            num_input_tokens=num_tokens,
        )
        return (
            self.recv_hidden[:recv_offset],
            self.recv_ids[:recv_offset],
            self.recv_weights[:recv_offset],
            route,
        )

    def combine(
        self,
        local_contributions: torch.Tensor,
        route: _FTNcclEPRoute,
        output: torch.Tensor,
    ) -> None:
        expected_rows = sum(route.recv_counts)
        if local_contributions.shape != (expected_rows, self.hidden_size):
            raise ValueError(
                "ft_nccl_ep combine input does not match the dispatch route"
            )
        if output.shape != (route.num_input_tokens, self.hidden_size):
            raise ValueError("ft_nccl_ep output does not match the source tokens")

        # Dispatch compaction preserved source order, so the local contribution
        # is already packed by the rank to which it must be returned.
        self.combine_send[:expected_rows].copy_(local_contributions)
        work, device_returned_counts = self.pg.all_to_allv_multi(
            [self.combine_send],
            [self.combine_recv_slots],
            route.recv_counts,
            self.combine_workspace,
        )
        work.wait()
        returned_counts = [int(v) for v in device_returned_counts.cpu().tolist()]
        self._check_status("combine")
        if returned_counts != route.send_counts:
            raise RuntimeError(
                "ft_nccl_ep combine counts do not match the dispatch route: "
                f"returned={returned_counts}, expected={route.send_counts}"
            )

        output.zero_()
        for source, count in enumerate(returned_counts):
            if not count:
                continue
            slot = source * self.max_tokens
            output.index_add_(
                0,
                route.send_token_indices[source],
                self.combine_recv_slots[slot : slot + count],
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

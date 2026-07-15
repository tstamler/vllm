# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize.ft_nccl_ep import (
    FTNcclEPHandle,
)


class _CompletedWork:
    def wait(self) -> bool:
        return True


class _LoopbackFTProcessGroup:
    """Preserve all-to-allv source slots without requiring distributed CUDA."""

    def empty(self, *shape, dtype):
        return torch.empty(*shape, dtype=dtype)

    def create_a2av_multi_workspace(self, row_bytes, recv_capacity):
        return SimpleNamespace(row_bytes=row_bytes, recv_capacity=recv_capacity)

    def all_to_allv_multi(self, send_bufs, out_bufs, send_counts, workspace):
        counts = (
            send_counts.tolist()
            if isinstance(send_counts, torch.Tensor)
            else list(send_counts)
        )
        send_offset = 0
        for peer, count in enumerate(counts):
            for send, out in zip(send_bufs, out_bufs):
                out_offset = peer * workspace.recv_capacity
                out[out_offset : out_offset + count].copy_(
                    send[send_offset : send_offset + count]
                )
            send_offset += count
        return _CompletedWork(), torch.tensor(counts, dtype=torch.int32)

    def check_and_clear_error(self) -> int:
        return 0


def test_ft_nccl_ep_routes_once_per_destination_and_combines() -> None:
    handle = FTNcclEPHandle(
        ft_process_group=_LoopbackFTProcessGroup(),
        ep_rank=0,
        ep_size=2,
        max_num_tokens_per_rank=4,
        token_hidden_size=2,
        num_global_experts=4,
        num_experts_per_token=2,
        input_dtype=torch.bfloat16,
        topk_weights_dtype=torch.float32,
    )
    hidden = torch.tensor(
        [[1, 2], [3, 4], [5, 6]],
        dtype=torch.bfloat16,
    )
    # Token 0 stays on rank 0, token 1 spans ranks 0 and 1, token 2 stays on
    # rank 1. Selecting two experts on one rank still emits only one token row.
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32
    )

    recv_hidden, recv_ids, recv_weights, route = handle.dispatch(
        hidden, topk_ids, topk_weights
    )

    assert route.send_counts == [2, 2]
    assert route.recv_counts == [2, 2]
    torch.testing.assert_close(recv_hidden, hidden[[0, 1, 1, 2]])
    torch.testing.assert_close(recv_ids, topk_ids[[0, 1, 1, 2]])
    torch.testing.assert_close(recv_weights, topk_weights[[0, 1, 1, 2]])

    output = torch.empty_like(hidden)
    handle.combine(recv_hidden, route, output)
    expected = hidden.clone()
    expected[1].mul_(2)
    torch.testing.assert_close(output, expected)


def test_ft_nccl_ep_rejects_uneven_expert_partition() -> None:
    try:
        FTNcclEPHandle(
            ft_process_group=_LoopbackFTProcessGroup(),
            ep_rank=0,
            ep_size=2,
            max_num_tokens_per_rank=4,
            token_hidden_size=2,
            num_global_experts=3,
            num_experts_per_token=1,
            input_dtype=torch.bfloat16,
            topk_weights_dtype=torch.float32,
        )
    except ValueError as error:
        assert "equal, linear expert partition" in str(error)
    else:
        raise AssertionError("expected an uneven expert partition to fail")

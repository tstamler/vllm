# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.prepare_finalize.ft_nccl_ep import (
    FTNcclEPHandle,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _worker(
    local_rank: int,
    world_size: int,
    master_port: int,
    use_cuda_graph: bool,
    use_multi_a2av: bool,
    q: mp.Queue,
) -> None:
    try:
        from ft_collective import get_ft_process_group
    except ImportError:
        q.put("ft_collective is not importable")
        return

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=local_rank,
        world_size=world_size,
        device_id=device,
    )
    ft_group = None
    try:
        ft_group = dist.new_group(
            ranks=list(range(world_size)),
            backend="ft_nccl",
            use_local_synchronization=True,
        )
        pg = get_ft_process_group(local_rank) or get_ft_process_group()
        assert pg is not None
        pg.ft_converge()

        handle = FTNcclEPHandle(
            ft_process_group=pg,
            ep_rank=local_rank,
            ep_size=world_size,
            max_num_tokens_per_rank=4,
            token_hidden_size=2,
            num_global_experts=4,
            num_experts_per_token=2,
            input_dtype=torch.bfloat16,
            topk_weights_dtype=torch.float32,
            use_multi_a2av=use_multi_a2av,
        )
        hidden = (
            torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16, device=device)
            + local_rank * 10
        )
        topk_ids = torch.tensor(
            [[0, 1], [2, 3], [0, 2]], dtype=torch.int64, device=device
        )
        topk_weights = torch.full((3, 2), 0.5, dtype=torch.float32, device=device)

        output = torch.empty_like(hidden)

        def run_round_trip() -> None:
            recv_hidden, _, _, route = handle.dispatch(hidden, topk_ids, topk_weights)
            # Rank-dependent local work lets the reverse exchange identify
            # which destination produced every contribution.
            local_contributions = recv_hidden * (local_rank + 1)
            handle.combine(local_contributions, route, output)

        if use_cuda_graph:
            run_round_trip()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_round_trip()
            for _ in range(8):
                hidden.add_(1)
                graph.replay()
                torch.cuda.synchronize(device)
        else:
            run_round_trip()

        expected = hidden.clone()
        expected[1].mul_(2)
        expected[2].mul_(3)
        torch.testing.assert_close(output, expected)
        if use_cuda_graph:
            del graph
            torch.cuda.synchronize(device)
    finally:
        if ft_group is not None:
            dist.destroy_process_group(ft_group)
        dist.destroy_process_group()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="ft_nccl_ep requires CUDA")
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda", reason="Only test on CUDA")
@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["eager", "cuda_graph"])
@pytest.mark.parametrize(
    "use_multi_a2av", [True, False], ids=["multi", "single_buffer"]
)
def test_ft_nccl_ep_backend_round_trip(
    use_cuda_graph: bool, use_multi_a2av: bool
) -> None:
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip("Not enough GPUs")

    q = mp.get_context("spawn").Queue()
    mp.spawn(
        _worker,
        args=(world_size, get_open_port(), use_cuda_graph, use_multi_a2av, q),
        nprocs=world_size,
        join=True,
    )
    try:
        reason = q.get(timeout=1)
    except queue.Empty:
        reason = None
    if reason is not None:
        pytest.skip(reason)

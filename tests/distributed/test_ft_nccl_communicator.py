# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import typing

import pytest
import torch
import torch.multiprocessing as mp

import vllm.envs as envs
from tests.utils import ensure_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

TEST_SIZE_ELEMENTS = 1024


def _run_cuda_graph_replay_test(
    cuda_communicator: CudaCommunicator,
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    workspace_count = len(cuda_communicator._ft_staging_workspaces)
    ft_process_group = cuda_communicator._ft_process_group
    assert ft_process_group is not None
    windows = getattr(ft_process_group, "_windows", {})
    window_count = len(windows)

    reduce_input = torch.empty(TEST_SIZE_ELEMENTS, dtype=dtype, device=device)
    gather_input = torch.empty((4, 2), dtype=dtype, device=device)
    reduce_input.fill_(local_rank + 1)
    gather_input.fill_(local_rank + 1)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph):
        reduce_output = cuda_communicator.all_reduce(reduce_input)
        gather_output = cuda_communicator.all_gather(gather_input, dim=0)

    rank_sum = world_size * (world_size + 1) / 2
    for offset in range(3):
        reduce_input.fill_(local_rank + offset + 1)
        gather_input.fill_(local_rank + offset + 1)
        graph.replay()
        torch.cuda.synchronize(device)

        torch.testing.assert_close(
            reduce_output,
            torch.full_like(reduce_output, rank_sum + offset * world_size),
        )
        expected_gather = torch.cat(
            [
                torch.full_like(gather_input, rank + offset + 1)
                for rank in range(world_size)
            ],
            dim=0,
        )
        torch.testing.assert_close(gather_output, expected_gather)

    cuda_communicator._check_ft_nccl_status(ft_process_group)
    assert len(cuda_communicator._ft_staging_workspaces) == workspace_count
    assert len(windows) == window_count


def ft_nccl_communicator_worker(
    local_rank: int,
    world_size: int,
    master_port: int,
    q: mp.Queue,
    run_cuda_graph_test: bool,
):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        try:
            import ft_collective  # noqa: F401
        except ImportError:
            q.put("ft_collective is not importable.")
            return

        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_USE_FT_NCCL_COMMUNICATOR", "1")
        m.setenv("FT_NCCL_MAX_COUNT", "2048")
        m.setenv("NCCL_NVLS_ENABLE", "1")
        m.setenv("NCCL_CUMEM_ENABLE", "1")

        dtype = torch.float32
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(master_port),
            }
        )

        try:
            init_distributed_environment()
            with ensure_current_vllm_config():
                initialize_model_parallel(tensor_model_parallel_size=world_size)

            tp_group = get_tp_group()
            cuda_communicator = typing.cast(
                CudaCommunicator, tp_group.device_communicator
            )

            input_tensor = torch.full(
                (TEST_SIZE_ELEMENTS,),
                local_rank + 1,
                dtype=dtype,
                device=device,
            )
            output = tp_group.all_reduce(input_tensor)
            expected = torch.full_like(input_tensor, world_size * (world_size + 1) / 2)

            ft_process_group = cuda_communicator._ft_process_group
            if ft_process_group is None:
                q.put("FT NCCL communicator path did not initialize.")
                return
            if not hasattr(ft_process_group, "empty"):
                q.put("FTProcessGroup does not expose empty().")
                return

            windows = getattr(ft_process_group, "_windows", {})
            assert input_tensor.data_ptr() not in windows
            assert len(cuda_communicator._ft_staging_workspaces) == 1
            workspace = next(iter(cuda_communicator._ft_staging_workspaces.values()))
            assert workspace.numel() == 2048
            assert workspace.data_ptr() in windows
            assert output is not input_tensor
            torch.testing.assert_close(
                input_tensor, torch.full_like(input_tensor, 1 + local_rank)
            )
            torch.testing.assert_close(output, expected)

            gather_dim0_input = torch.full(
                (4, 2),
                local_rank + 1,
                dtype=dtype,
                device=device,
            )
            gather_dim0_output = cuda_communicator.all_gather(
                gather_dim0_input, dim=0
            )
            expected_gather_dim0 = torch.cat(
                [
                    torch.full_like(gather_dim0_input, rank + 1)
                    for rank in range(world_size)
                ],
                dim=0,
            )

            assert gather_dim0_input.data_ptr() not in windows
            assert gather_dim0_output.shape == expected_gather_dim0.shape
            torch.testing.assert_close(gather_dim0_output, expected_gather_dim0)

            gather_last_dim_input = torch.full(
                (2, 4),
                local_rank + 1,
                dtype=dtype,
                device=device,
            )
            gather_last_dim_output = cuda_communicator.all_gather(
                gather_last_dim_input, dim=-1
            )
            expected_gather_last_dim = torch.cat(
                [
                    torch.full_like(gather_last_dim_input, rank + 1)
                    for rank in range(world_size)
                ],
                dim=-1,
            )

            assert gather_last_dim_input.data_ptr() not in windows
            assert gather_last_dim_output.shape == expected_gather_last_dim.shape
            torch.testing.assert_close(
                gather_last_dim_output, expected_gather_last_dim
            )
            assert len(cuda_communicator._ft_staging_workspaces) == 1

            if run_cuda_graph_test:
                _run_cuda_graph_replay_test(
                    cuda_communicator,
                    local_rank,
                    world_size,
                    dtype,
                    device,
                )
        except RuntimeError as e:
            if "requires an FTProcessGroup with empty()" in str(e):
                q.put(str(e))
                return
            raise
        finally:
            cleanup_dist_env_and_memory()


def _run_ft_nccl_communicator_test(run_cuda_graph_test: bool) -> None:
    world_size = 2
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    q = mp.get_context("spawn").Queue()
    mp.spawn(
        ft_nccl_communicator_worker,
        args=(world_size, get_open_port(), q, run_cuda_graph_test),
        nprocs=world_size,
    )
    try:
        val = q.get(timeout=1)
    except queue.Empty:
        val = None
    finally:
        cleanup_dist_env_and_memory()
        if val is not None:
            pytest.skip(val)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="ft_nccl communicator smoke test is only available on CUDA.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_ft_nccl_communicator_staging_smoke():
    _run_ft_nccl_communicator_test(run_cuda_graph_test=False)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="ft_nccl communicator CUDA graph test is only available on CUDA.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_ft_nccl_communicator_cuda_graph_replay():
    _run_ft_nccl_communicator_test(run_cuda_graph_test=True)

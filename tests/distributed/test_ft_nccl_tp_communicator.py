# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import random
import typing

import pytest
import torch
import torch.multiprocessing as mp

import vllm.envs as envs
from tests.utils import ensure_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.pynccl_allocator import (
    get_nccl_mem_pool,
    get_symmetric_memory_region,
    is_symmetric_memory_tensor,
    nccl_symm_mem_context,
)
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

torch.manual_seed(42)
random.seed(44)

TEST_SIZE_ELEMENTS = 1024


def ft_nccl_tp_communicator_worker(
    local_rank: int,
    world_size: int,
    master_port: int,
    q: mp.Queue,
):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        try:
            import ft_collective  # noqa: F401
        except ImportError:
            q.put("ft_collective is not importable.")
            return

        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_USE_FT_NCCL_TP", "1")
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

        init_distributed_environment()
        with ensure_current_vllm_config():
            initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        ft_process_group = ft_collective.get_ft_process_group()
        if ft_process_group is None:
            q.put("ft_collective did not register an FTProcessGroup.")
            return
        if not hasattr(ft_process_group, "register_symmetric_tensor"):
            pg_cls = type(ft_process_group)
            q.put(
                "ft_collective was imported from "
                f"{getattr(ft_collective, '__file__', '<unknown>')}, but "
                f"{pg_cls.__module__}.{pg_cls.__qualname__} does not expose "
                "register_symmetric_tensor(). Use an ft_collective build with "
                "external symmetric tensor registration support."
            )
            return
        if getattr(ft_process_group, "_max_count", None) != 2048:
            q.put(
                "ft_nccl process group did not use FT_NCCL_MAX_COUNT; "
                f"got {getattr(ft_process_group, '_max_count', None)}."
            )
            return

        pynccl_comm = cuda_communicator.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            q.put("PyNCCL communicator is not available.")
            return
        if pynccl_comm.nccl_version < 22703:
            q.put("NCCL 2.27.3 or newer is required.")
            return
        if get_nccl_mem_pool() is None:
            q.put("NCCL allocator compilation failed.")
            return

        with nccl_symm_mem_context(pynccl_comm):
            input_tensor = torch.full(
                (TEST_SIZE_ELEMENTS,),
                local_rank + 1,
                dtype=dtype,
                device=device,
            )

        if not is_symmetric_memory_tensor(input_tensor):
            q.put("NCCL symmetric-memory allocation is not available.")
            return

        input_region = get_symmetric_memory_region(input_tensor)
        if input_region is None:
            q.put("Could not resolve NCCL symmetric-memory region.")
            return

        output = cuda_communicator.all_reduce(input_tensor)
        expected = torch.full_like(input_tensor, world_size * (world_size + 1) / 2)

        assert output is input_tensor
        assert input_region[0] in ft_process_group._windows
        assert input_tensor.data_ptr() in ft_process_group._windows
        torch.testing.assert_close(output, expected)

        vocab_size = world_size * 64
        embedding_dim = 8
        embedding = VocabParallelEmbedding(
            vocab_size,
            embedding_dim,
            params_dtype=dtype,
        )
        with torch.no_grad():
            embedding.weight.fill_(local_rank + 1)

        token_ids = torch.arange(world_size, dtype=torch.long, device=device) * 64
        embedding_output = embedding(token_ids)
        expected_embedding_output = torch.arange(
            1,
            world_size + 1,
            dtype=dtype,
            device=device,
        ).unsqueeze(-1).expand(world_size, embedding_dim)

        torch.testing.assert_close(embedding_output, expected_embedding_output)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="ft_nccl TP communicator smoke test is only available on CUDA.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_ft_nccl_tp_communicator_smoke():
    world_size = 2
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    q = mp.get_context("spawn").Queue()
    mp.spawn(
        ft_nccl_tp_communicator_worker,
        args=(world_size, get_open_port(), q),
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

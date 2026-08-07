# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
    _unpack_ft_rank_values,
)


def test_unpack_ft_rank_values_restores_inactive_slot():
    rank_values = _unpack_ft_rank_values(
        torch.tensor([8, 8, 4, 12, 12, 2, 2, -1], dtype=torch.int32),
        torch.tensor([1, 1, 1, 0, 1, 1, 1, 1], dtype=torch.int32),
    )

    torch.testing.assert_close(
        rank_values,
        torch.tensor([8, 8, 4, 0, 12, 12, 2, 2], dtype=torch.int32),
    )


def test_prepare_ft_convergence_allocates_legacy_buffers(monkeypatch):
    class FakeProcessGroup:
        _bar_send = None
        _bar_out = None

        @staticmethod
        def empty(numel, dtype):
            return torch.empty(numel, dtype=dtype)

    monkeypatch.setenv("VLLM_FT_SURVIVE_WORKER_FAILURE", "1")
    monkeypatch.setenv("FT_BARRIER_MODE", "collective")
    communicator = object.__new__(CudaCommunicator)
    communicator.world_size = 2
    communicator.device = torch.device("cpu")
    communicator.unique_name = "ep:0"
    process_group = FakeProcessGroup()

    communicator._prepare_ft_convergence(process_group)

    assert process_group._bar_send is not None
    assert process_group._bar_send.numel() == 4
    assert process_group._bar_out is not None
    assert process_group._bar_out.numel() == 4

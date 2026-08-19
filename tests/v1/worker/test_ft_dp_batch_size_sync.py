# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
    _unpack_ft_dp_metadata,
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


def test_unpack_ft_dp_metadata_pads_active_graph_ranks():
    tokens, cudagraph_mode = _unpack_ft_dp_metadata(
        torch.tensor(
            [8, 1, 8, 1, 12, 1, 12, 1, 4, 1, 4, 1],
            dtype=torch.int32,
        ),
        torch.tensor([2, 2, 0, 0, 2, 2, 2, 2], dtype=torch.int32),
        dp_size=4,
    )

    assert cudagraph_mode == 1
    torch.testing.assert_close(
        tokens,
        torch.tensor([12, 0, 12, 12], dtype=torch.int32),
    )


def test_unpack_ft_dp_metadata_preserves_eager_sizes():
    tokens, cudagraph_mode = _unpack_ft_dp_metadata(
        torch.tensor(
            [8, 0, 8, 0, 12, 1, 12, 1, 4, 1, 4, 1],
            dtype=torch.int32,
        ),
        torch.tensor([2, 2, 0, 0, 2, 2, 2, 2], dtype=torch.int32),
        dp_size=4,
    )

    assert cudagraph_mode == 0
    torch.testing.assert_close(
        tokens,
        torch.tensor([8, 0, 12, 4], dtype=torch.int32),
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


def test_set_ft_ep_active_mask_removes_entire_failed_dp_rank(monkeypatch):
    class FakeProcessGroup:
        active_mask = [True] * 8
        error_cleared = False

        def get_active_mask(self):
            return list(self.active_mask)

        def set_active_mask(self, mask):
            self.active_mask = list(mask)

        def clear_error(self):
            self.error_cleared = True

    monkeypatch.setenv("VLLM_FT_SURVIVE_WORKER_FAILURE", "1")
    communicator = object.__new__(CudaCommunicator)
    communicator.world_size = 8
    communicator.rank_in_group = 4
    communicator.unique_name = "ep:0"
    communicator._ft_process_group = FakeProcessGroup()

    active_mask = communicator.set_ft_ep_active_mask((1,), dp_size=4)

    assert active_mask == [True, True, False, False, True, True, True, True]
    assert communicator._ft_process_group.active_mask == active_mask
    assert communicator._ft_process_group.error_cleared

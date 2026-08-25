# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
    _unpack_ft_dp_metadata,
    _unpack_ft_rank_values,
)
from vllm.v1.worker.gpu_worker import Worker


def test_rejoin_ft_membership_refreshes_mask_without_rebuild(monkeypatch):
    class FakeStream:
        synchronized = False

        def synchronize(self):
            self.synchronized = True

    class FakeProcessGroup:
        cleared = False

        @staticmethod
        def ft_rejoin():
            return [True, True]

        def clear_error(self):
            self.cleared = True

    monkeypatch.setenv("VLLM_FT_SURVIVE_WORKER_FAILURE", "1")
    stream = FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    process_group = FakeProcessGroup()
    communicator = object.__new__(CudaCommunicator)
    communicator.device = torch.device("cuda")
    communicator.unique_name = "ep:0"
    communicator._ft_active_mask = [True, False]
    communicator._ft_result_mask = [True, False]
    communicator._get_ft_process_group = lambda: process_group

    assert communicator.rejoin_ft_membership() == [True, True]
    assert stream.synchronized
    assert communicator._ft_active_mask == [True, True]
    assert communicator._ft_result_mask is None
    assert process_group.cleared


def test_worker_executes_one_ft_rejoin_group(monkeypatch):
    class FakeCommunicator:
        @staticmethod
        def rejoin_ft_membership():
            return [True, True]

    group = type("Group", (), {"device_communicator": FakeCommunicator()})()
    monkeypatch.setattr("vllm.v1.worker.gpu_worker.get_tp_group", lambda: group)
    worker = object.__new__(Worker)
    worker.rank = 1

    assert worker.execute_ft_rejoin_group("TP") == [True, True]


def test_worker_publishes_ft_rejoin_ack(tmp_path, monkeypatch):
    group = type("Group", (), {"rank_in_group": 3})()
    monkeypatch.setattr("vllm.v1.worker.gpu_worker.get_ep_group", lambda: group)
    worker = object.__new__(Worker)
    worker._ft_rejoin_ack_dir = str(tmp_path)

    worker.complete_ft_rejoin_generation("generation-1")

    assert (tmp_path / "generation-1.rank-3.ack").read_text() == "rank=3\n"


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
    communicator._ft_active_mask = None
    communicator._ft_result_mask = [True, True, False, False, True, True, True, True]

    active_mask = communicator.set_ft_ep_active_mask((1,), dp_size=4)

    assert active_mask == [True, True, False, False, True, True, True, True]
    assert communicator._ft_process_group.active_mask == active_mask
    assert communicator._ft_process_group.error_cleared
    assert communicator._ft_active_mask == active_mask
    assert communicator._ft_result_mask is None


def test_ft_result_mask_records_collective_responders():
    class FakeProcessGroup:
        @staticmethod
        def get_result_mask():
            return [True, True, False, False]

    communicator = object.__new__(CudaCommunicator)
    communicator._ft_result_mask = None

    assert communicator._record_ft_result_mask(FakeProcessGroup()) == [
        True,
        True,
        False,
        False,
    ]
    assert communicator.get_ft_result_mask() == [True, True, False, False]


def test_ft_active_mask_cache_refreshes_on_demand():
    class FakeProcessGroup:
        active_mask = [True, True]
        reads = 0

        def get_active_mask(self):
            self.reads += 1
            return list(self.active_mask)

    communicator = object.__new__(CudaCommunicator)
    communicator._ft_active_mask = None
    process_group = FakeProcessGroup()

    assert communicator._get_ft_active_mask(process_group) == [True, True]
    process_group.active_mask = [True, False]
    assert communicator._get_ft_active_mask(process_group) == [True, True]
    assert process_group.reads == 1

    assert communicator._get_ft_active_mask(process_group, refresh=True) == [
        True,
        False,
    ]
    assert process_group.reads == 2


def test_ft_dp_metadata_buffers_are_reused_outside_inference_mode():
    communicator = object.__new__(CudaCommunicator)
    communicator.device = torch.device("cpu")
    communicator.world_size = 8
    communicator._ft_dp_metadata_host = None
    communicator._ft_dp_metadata_device = None
    communicator._ft_dp_metadata_packed = None

    with torch.inference_mode():
        first = communicator._get_ft_dp_metadata_buffers()
    second = communicator._get_ft_dp_metadata_buffers()

    assert all(
        first_tensor is second_tensor
        for first_tensor, second_tensor in zip(first, second)
    )
    assert not first[0].is_inference()
    assert not first[1].is_inference()
    assert not first[2].is_inference()
    first[0].fill_(1)


def test_ft_all_gatherv_transaction_skips_output_initialization(monkeypatch):
    class FakeWork:
        @staticmethod
        def wait():
            return True

    class FakeProcessGroup:
        @staticmethod
        def all_gatherv(send, output, send_counts, max_size):
            return FakeWork(), send_counts

    communicator = object.__new__(CudaCommunicator)
    communicator.world_size = 2
    communicator.rank_in_group = 0
    communicator._ft_all_gatherv_send_counts = {}
    communicator._get_ft_native_staging_view = lambda *args: torch.empty((1, 4))
    communicator._get_ft_process_group = lambda: FakeProcessGroup()
    communicator._copy_to_ft_staging = lambda *args: None
    communicator._check_ft_nccl_status = lambda *args: None

    allocations = []
    real_empty = torch.empty
    real_zeros = torch.zeros

    def tracked_empty(*args, **kwargs):
        allocations.append(("empty", args[0]))
        return real_empty(*args, **kwargs)

    def tracked_zeros(*args, **kwargs):
        allocations.append(("zeros", args[0]))
        return real_zeros(*args, **kwargs)

    monkeypatch.setenv("VLLM_FT_SURVIVE_WORKER_FAILURE", "1")
    monkeypatch.setattr(torch, "empty", tracked_empty)
    monkeypatch.setattr(torch, "zeros", tracked_zeros)
    input_tensor = real_empty((1, 4))

    communicator._ft_nccl_native_all_gatherv(
        input_tensor, sizes=[1, 1], check_status=False
    )
    assert ("empty", (2, 4)) in allocations
    assert ("zeros", (2, 4)) not in allocations

    allocations.clear()
    communicator._ft_nccl_native_all_gatherv(
        input_tensor, sizes=[1, 1], check_status=True
    )
    assert ("zeros", (2, 4)) in allocations


def test_ft_dispatch_success_does_not_clear_ok_status(monkeypatch):
    class FakeStream:
        @staticmethod
        def synchronize():
            return None

    class FakeProcessGroup:
        def __init__(self):
            self.clear_calls = 0

        @staticmethod
        def get_error():
            return 0

        def clear_error(self):
            self.clear_calls += 1

    process_group = FakeProcessGroup()
    communicator = object.__new__(CudaCommunicator)
    communicator.world_size = 2
    communicator.rank_in_group = 0
    communicator.unique_name = "ep:0"
    communicator._ft_ok_status = 0
    communicator._ft_active_mask = [True, True]
    communicator._get_ft_process_group = lambda: process_group
    communicator._ft_nccl_native_all_gatherv = (
        lambda input_, dim, sizes, check_status: torch.empty((2, 4))
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: FakeStream(),
    )

    outputs = communicator._ft_nccl_all_gatherv_transaction(
        [torch.empty((1, 4))], dim=0, sizes=[1, 1]
    )

    assert len(outputs) == 1
    assert process_group.clear_calls == 0

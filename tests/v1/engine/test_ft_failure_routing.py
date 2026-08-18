# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient


def _make_client(dead_engine_indices: set[int]) -> DPLBAsyncMPClient:
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.core_engines = [b"\x00\x00", b"\x01\x00"]
    client.lb_engines = [[0, 0], [5, 5]]
    client.eng_start_index = 0
    client.dead_engine_indices = dead_engine_indices
    return client


def _make_request(data_parallel_rank: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="request-0",
        data_parallel_rank=data_parallel_rank,
        pooling_params=None,
    )


def test_dplb_skips_degraded_engine():
    client = _make_client({0})

    chosen_engine = client.get_core_engine_for_request(_make_request())

    assert chosen_engine == client.core_engines[1]


def test_dplb_rejects_explicit_degraded_engine():
    client = _make_client({1})

    with pytest.raises(RuntimeError, match="no longer serving"):
        client.get_core_engine_for_request(_make_request(data_parallel_rank=1))


def test_dp_engine_withdraws_when_worker_dies_during_dummy_batch():
    core = object.__new__(DPEngineCoreProc)
    core.dp_rank = 1
    core.model_executor = SimpleNamespace(has_dead_workers=lambda: True)
    core._maybe_handle_own_tp_degradation = Mock()

    def fail_dummy_batch():
        raise RuntimeError("worker died during RPC")

    completed, result = core._run_with_ft_worker_death_guard(
        "dummy batch", fail_dummy_batch
    )

    assert not completed
    assert result is None
    core._maybe_handle_own_tp_degradation.assert_called_once_with()


def test_dp_engine_propagates_non_worker_dummy_batch_failure():
    core = object.__new__(DPEngineCoreProc)
    core.model_executor = SimpleNamespace(has_dead_workers=lambda: False)

    def fail_dummy_batch():
        raise RuntimeError("unrelated failure")

    with pytest.raises(RuntimeError, match="unrelated failure"):
        core._run_with_ft_worker_death_guard("dummy batch", fail_dummy_batch)


def test_ft_dp_sync_installs_globally_agreed_failures(monkeypatch):
    monkeypatch.setenv("VLLM_FT_SURVIVE_WORKER_FAILURE", "1")
    core = object.__new__(DPEngineCoreProc)
    core.dp_rank = 2
    core.dp_group = object()
    core.pending_pause = False
    core.step_counter = 0
    core._tp_degraded = False
    core._install_ft_membership_after_failure = Mock()

    sync = Mock(return_value=(True, False, (1,)))
    monkeypatch.setattr("vllm.v1.engine.core.ParallelConfig.sync_ft_dp_state", sync)

    assert core._has_global_unfinished_reqs(local_unfinished=True)
    sync.assert_called_once_with(
        core.dp_group,
        has_unfinished=True,
        pending_pause=False,
        failed_dp_rank=None,
    )
    core._install_ft_membership_after_failure.assert_called_once_with((1,))

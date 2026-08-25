# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from vllm.v1.engine import EngineCoreOutputs
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


def test_dplb_reintroduces_engine_after_rejoin():
    client = _make_client({1})
    client._abort_in_flight_for_dead_engine = Mock()

    client._update_degraded_engines(set())

    assert client.dead_engine_indices == set()
    assert (
        client.get_core_engine_for_request(_make_request(data_parallel_rank=1))
        == client.core_engines[1]
    )


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
    core._ft_observed_failures = ()
    core._install_ft_membership_after_failure = Mock()

    sync = Mock(return_value=(True, False, (1,)))
    monkeypatch.setattr("vllm.v1.engine.core.ParallelConfig.sync_ft_dp_state", sync)

    assert core._has_global_unfinished_reqs(local_unfinished=True)
    sync.assert_called_once_with(
        core.dp_group,
        has_unfinished=True,
        pending_pause=False,
        failed_dp_ranks=(),
    )
    core._install_ft_membership_after_failure.assert_called_once_with((1,))


def test_withdrawn_dp_engine_dispatches_rejoin_to_workers(tmp_path, monkeypatch):
    trigger = tmp_path / "rejoin.trigger"
    trigger.write_text("generation-1\n", encoding="utf-8")
    core = object.__new__(DPEngineCoreProc)
    core.dp_rank = 1
    core._tp_degraded = True
    core._ft_stall_withdrawn = False
    core._ft_observed_failures = ()
    core._ft_installed_failures = ()
    core._ft_rejoin_trigger = str(trigger)
    core._ft_rejoin_poll_interval = 0.0
    core._ft_rejoin_max_attempts = 1
    core._ft_rejoin_last_poll = 0.0
    core._ft_rejoin_generation = None
    collective_rpc = Mock(
        side_effect=[
            [[True, True], [True, True]],
            [[True, True], [True, True]],
            [None, None],
        ]
    )
    core.model_executor = SimpleNamespace(collective_rpc=collective_rpc)
    core.dp_group = object()
    core.barrier = Mock()
    monkeypatch.setattr("torch.distributed.all_reduce", lambda *args, **kwargs: None)

    core._maybe_execute_ft_rejoin()
    core._maybe_execute_ft_rejoin()

    assert collective_rpc.call_args_list == [
        call("execute_ft_rejoin_group", args=("TP",)),
        call("execute_ft_rejoin_group", args=("EP",)),
        call("complete_ft_rejoin_generation", args=("generation-1",)),
    ]
    assert core.barrier.call_count == 4
    assert core._ft_rejoin_generation == "generation-1"


def test_mask_failure_withdraws_dp_engine_until_rejoin():
    core = object.__new__(DPEngineCoreProc)
    core.dp_rank = 1
    core.dp_size = 4
    core._tp_degraded = False
    core._ft_stall_withdrawn = False
    core._ft_observed_failures = ()
    core._ft_installed_failures = ()
    core.output_queue = Mock()
    core.scheduler = SimpleNamespace(
        finish_requests=Mock(return_value=[("request-0", 0)])
    )
    core._send_error_outputs = Mock()
    core.model_executor = SimpleNamespace(collective_rpc=Mock())

    output = SimpleNamespace(
        ft_ep_active_mask=[True, True, False, True, True, True, True, True]
    )
    core._observe_model_runner_output(output)
    core._install_ft_membership_after_failure(core._ft_observed_failures)

    assert core._ft_observed_failures == (1,)
    assert core._ft_stall_withdrawn
    core.output_queue.put_nowait.assert_called_once_with(
        (-1, EngineCoreOutputs(dp_engine_available=(1, False)))
    )
    core._send_error_outputs.assert_called_once_with([("request-0", 0)])
    core.model_executor.collective_rpc.assert_not_called()


def test_successful_rejoin_returns_transiently_withdrawn_engine(tmp_path, monkeypatch):
    trigger = tmp_path / "rejoin.trigger"
    trigger.write_text("generation-2\n", encoding="utf-8")
    core = object.__new__(DPEngineCoreProc)
    core.dp_rank = 1
    core._tp_degraded = False
    core._ft_stall_withdrawn = True
    core._ft_observed_failures = (1,)
    core._ft_installed_failures = (1,)
    core._ft_rejoin_trigger = str(trigger)
    core._ft_rejoin_poll_interval = 0.0
    core._ft_rejoin_max_attempts = 1
    core._ft_rejoin_last_poll = 0.0
    core._ft_rejoin_generation = None
    core.model_executor = SimpleNamespace(
        collective_rpc=Mock(
            side_effect=[
                [[True, True], [True, True]],
                [[True] * 8, [True] * 8],
                [None, None],
            ]
        )
    )
    core.dp_group = object()
    core.barrier = Mock()
    core.output_queue = Mock()
    monkeypatch.setattr("torch.distributed.all_reduce", lambda *args, **kwargs: None)

    core._maybe_execute_ft_rejoin()

    assert not core._ft_stall_withdrawn
    assert core._ft_observed_failures == ()
    assert core._ft_installed_failures == ()
    core.output_queue.put_nowait.assert_called_once_with(
        (-1, EngineCoreOutputs(dp_engine_available=(1, True)))
    )

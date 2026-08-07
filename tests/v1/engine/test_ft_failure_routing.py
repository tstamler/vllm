# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

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

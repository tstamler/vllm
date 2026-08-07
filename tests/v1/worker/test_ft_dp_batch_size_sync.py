# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.device_communicators.cuda_communicator import (
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

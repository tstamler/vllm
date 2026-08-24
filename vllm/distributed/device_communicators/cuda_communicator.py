# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.all_reduce_utils import (
    NCCL_SYMM_MEM_ALL_REDUCE_CONFIG,
    should_nccl_symm_mem_allreduce,
)
from vllm.distributed.device_communicators.pynccl import register_nccl_symmetric_ops
from vllm.distributed.device_communicators.pynccl_allocator import (
    is_symmetric_memory_enabled,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..utils import StatelessProcessGroup
from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)

_FT_NCCL_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})
_FT_NCCL_ALL_GATHERV_DTYPES = _FT_NCCL_DTYPES | frozenset(
    {torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8}
)


def _unpack_ft_rank_values(
    packed: torch.Tensor, recv_counts: torch.Tensor
) -> torch.Tensor:
    """Restore one packed value per FT rank, leaving inactive ranks at zero."""
    counts_cpu = recv_counts.cpu()
    if torch.any((counts_cpu < 0) | (counts_cpu > 1)):
        raise RuntimeError(
            f"Expected zero or one FT metadata value per rank, got {counts_cpu}."
        )
    num_active = int(counts_cpu.sum().item())
    rank_values = torch.zeros_like(counts_cpu)
    rank_values[counts_cpu.bool()] = packed[:num_active].cpu()
    return rank_values


def _unpack_ft_dp_metadata(
    packed: torch.Tensor, recv_counts: torch.Tensor, dp_size: int
) -> tuple[torch.Tensor, int]:
    """Restore FT DP token counts and synchronize the CUDA graph mode."""
    counts_cpu = recv_counts.cpu()
    if torch.any((counts_cpu != 0) & (counts_cpu != 2)):
        raise RuntimeError(
            f"Expected zero or two FT metadata values per rank, got {counts_cpu}."
        )
    world_size = counts_cpu.numel()
    if world_size % dp_size != 0:
        raise RuntimeError(
            f"EP world size {world_size} is not divisible by DP size {dp_size}."
        )

    responding_ranks = counts_cpu == 2
    num_responders = int(responding_ranks.sum().item())
    if num_responders == 0:
        raise RuntimeError("FT DP metadata exchange received no responses.")
    rank_metadata = torch.zeros((world_size, 2), dtype=torch.int32)
    rank_metadata[responding_ranks] = (
        packed[: num_responders * 2].view(num_responders, 2).cpu()
    )

    replicas_per_dp = world_size // dp_size
    dp_responders = responding_ranks.view(dp_size, replicas_per_dp).any(dim=1)
    tokens_across_dp = rank_metadata[:, 0].view(dp_size, replicas_per_dp).amax(dim=1)
    synced_cudagraph_mode = int(rank_metadata[responding_ranks, 1].min().item())

    if synced_cudagraph_mode != 0:
        max_num_tokens = int(tokens_across_dp.max().item())
        tokens_across_dp[dp_responders] = max_num_tokens
    return tokens_across_dp, synced_cudagraph_mode


class CudaCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
        )
        if "tp" not in unique_name:
            # custom allreduce or torch symm mem can be used only by tp
            use_custom_allreduce = False
            use_torch_symm_mem = False
            use_flashinfer_allreduce = False
        else:
            from vllm.distributed.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE

            use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
            use_torch_symm_mem = envs.VLLM_ALLREDUCE_USE_SYMM_MEM
            use_flashinfer_allreduce = envs.VLLM_ALLREDUCE_USE_FLASHINFER

        self.use_custom_allreduce = use_custom_allreduce
        self.use_torch_symm_mem = use_torch_symm_mem
        self.use_flashinfer_allreduce = use_flashinfer_allreduce

        # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )
        from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator

        self.pynccl_comm: PyNcclCommunicator | None = None
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group if tcp_store_group is None else tcp_store_group,
                device=self.device,
            )
            if is_symmetric_memory_enabled():
                register_nccl_symmetric_ops(self.pynccl_comm)

        self.ca_comm: CustomAllreduce | None = None
        self.qr_comm: QuickAllReduce | None = None
        self.symm_mem_comm: SymmMemCommunicator | None = None
        self.fi_ar_comm: FlashInferAllReduce | None = None
        self._ft_process_group = None
        self._ft_torch_group = None
        self._ft_ok_status = None
        self._ft_external_stream = None
        self._ft_staging_workspaces: dict[
            tuple[torch.dtype, str, int | None], torch.Tensor
        ] = {}
        self._ft_all_gatherv_send_counts: dict[
            tuple[str, int | None, int], torch.Tensor
        ] = {}
        self._ft_active_mask: list[bool] | None = None
        self._ft_dp_metadata_host: torch.Tensor | None = None
        self._ft_dp_metadata_device: torch.Tensor | None = None
        self._ft_dp_metadata_packed: torch.Tensor | None = None

        if use_torch_symm_mem and current_platform.is_cuda():
            self.symm_mem_comm = SymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        if self.use_flashinfer_allreduce and self.world_size > 1:
            self.fi_ar_comm = FlashInferAllReduce(
                group=self.cpu_group,
                device=self.device,
            )

        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                symm_mem_enabled=(
                    self.symm_mem_comm is not None and not self.symm_mem_comm.disabled
                ),
            )

            if current_platform.is_rocm():
                # Initialize a custom quick all-reduce implementation for AMD.
                # Quick reduce is designed as a complement to custom allreduce.
                # Based on quickreduce (https://github.com/mk1-project/quickreduce).
                # If it's a rocm, 'use_custom_allreduce==True' means it must
                # currently be an MI300 series.
                self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)

        if self._should_init_ft_nccl_communicator():
            self._get_ft_process_group()
            if self._should_use_ft_nccl_ep_communicator():
                logger.info_once(
                    "Using FT NCCL native all_gatherv/reduce_scatterv with symmetric "
                    "staging for group '%s'.",
                    self.unique_name or "<unnamed>",
                    scope="global",
                )

        if self.world_size > 1:
            self._log_all_reduce_backend_selection()

        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend == "deepep_high_throughput":
                from .all2all import DeepEPHTAll2AllManager

                self.all2all_manager = DeepEPHTAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend == "deepep_low_latency":
                from .all2all import DeepEPLLAll2AllManager

                self.all2all_manager = DeepEPLLAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend == "mori":
                from .all2all import MoriAll2AllManager

                self.all2all_manager = MoriAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "nixl_ep":
                from .all2all import NixlEPAll2AllManager

                self.all2all_manager = NixlEPAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif (
                self.all2all_backend == "flashinfer_all2allv"
                or self.all2all_backend == "flashinfer_nvlink_two_sided"
            ):
                if self.all2all_backend == "flashinfer_all2allv":
                    logger.warning_once(
                        "'flashinfer_all2allv' is deprecated and has been renamed to"
                        "'flashinfer_nvlink_two_sided'. It will be removed in a future"
                        "release."
                    )
                from .all2all import FlashInferNVLinkTwoSidedManager

                self.all2all_manager = FlashInferNVLinkTwoSidedManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend == "flashinfer_nvlink_one_sided":
                from .all2all import FlashInferNVLinkOneSidedManager

                self.all2all_manager = FlashInferNVLinkOneSidedManager(self.cpu_group)
            else:
                raise ValueError(f"Unknown all2all backend: {self.all2all_backend}")

            logger.info_once(
                "Using %s all2all manager.",
                self.all2all_manager.__class__.__name__,
                scope="global",
            )

    def _log_all_reduce_backend_selection(self) -> None:
        """Log the all-reduce backends that are active for this group.

        The dispatch chain in ``all_reduce`` tries backends in this order and
        falls through to the next one if the current backend rejects the
        input (size/dtype gates) or is disabled. The list of "enabled"
        backends below is the subset of potential backends that may be
        chosen at dispatch time for this group; the actual per-call choice
        depends on the input tensor.
        """
        all_potential_ar_backends = [
            "FT_NCCL",
            "NCCL_SYMM_MEM",
            "QUICK_REDUCE",
            "FLASHINFER",
            "CUSTOM",
            "SYMM_MEM",
            "PYNCCL",
        ]
        enabled_ar_backends: list[str] = []
        # Mirror the static preconditions of `should_nccl_symm_mem_allreduce`:
        # VLLM_BATCH_INVARIANT off, NCCL symm mem enabled, world_size meets
        # min_world_size, and world_size either has a tuned entry in
        # `custom_ar_preferred_ranges` or is greater than
        # `always_use_above_world_size`. World sizes that fail the latter (e.g.
        # 5/6/7 with the default config) never dispatch NCCL symm mem
        # regardless of input. The per-tensor-size check inside the function
        # stays as a runtime decision.
        nccl_symm_ws_ok = self.world_size >= NCCL_SYMM_MEM_ALL_REDUCE_CONFIG[
            "min_world_size"
        ] and (
            self.world_size
            in NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["custom_ar_preferred_ranges"]
            or self.world_size
            > NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["always_use_above_world_size"]
        )
        if self._should_use_ft_nccl_communicator():
            enabled_ar_backends.append("FT_NCCL")
        if (
            self.pynccl_comm is not None
            and not self.pynccl_comm.disabled
            and is_symmetric_memory_enabled()
            and not envs.VLLM_BATCH_INVARIANT
            and nccl_symm_ws_ok
        ):
            enabled_ar_backends.append("NCCL_SYMM_MEM")
        if self.qr_comm is not None and not self.qr_comm.disabled:
            enabled_ar_backends.append("QUICK_REDUCE")
        if self.fi_ar_comm is not None and not self.fi_ar_comm.disabled:
            enabled_ar_backends.append("FLASHINFER")
        if self.ca_comm is not None and not self.ca_comm.disabled:
            enabled_ar_backends.append("CUSTOM")
        if self.symm_mem_comm is not None and not self.symm_mem_comm.disabled:
            enabled_ar_backends.append("SYMM_MEM")
        if self.pynccl_comm is not None and not self.pynccl_comm.disabled:
            enabled_ar_backends.append("PYNCCL")

        logger.info_once(
            "Using %s all-reduce backends (in dispatch order) for group "
            "'%s' out of potential backends: %s.",
            "[" + ", ".join(f"'{b}'" for b in enabled_ar_backends) + "]",
            self.unique_name or "<unnamed>",
            "[" + ", ".join(f"'{b}'" for b in all_potential_ar_backends) + "]",
            scope="global",
        )

    def _should_use_ft_nccl_communicator(self) -> bool:
        return self._should_use_ft_nccl_tp_communicator()

    def _ft_group_kind(self) -> str:
        return self.unique_name.split(":", 1)[0] if self.unique_name else ""

    def _should_use_ft_nccl_tp_communicator(self) -> bool:
        return (
            envs.VLLM_USE_FT_NCCL_COMMUNICATOR
            and self._ft_group_kind() == "tp"
            and self.world_size > 1
            and current_platform.is_cuda()
        )

    def _should_use_ft_nccl_ep_communicator(self) -> bool:
        return (
            envs.VLLM_USE_FT_NCCL_EP
            and self._ft_group_kind() in ("dp", "ep")
            and self.world_size > 1
            and current_platform.is_cuda()
        )

    def _should_init_ft_nccl_communicator(self) -> bool:
        return (
            self._should_use_ft_nccl_tp_communicator()
            or self._should_use_ft_nccl_ep_communicator()
        )

    def _get_ft_process_group(self):
        if self._ft_process_group is not None:
            return self._ft_process_group

        try:
            import ft_collective
            from ft_collective import FT_OK, get_ft_process_group
        except ImportError as e:
            raise RuntimeError(
                "VLLM_USE_FT_NCCL_COMMUNICATOR=1 requires ft_collective to be "
                "importable."
            ) from e

        try:
            self._ft_torch_group = torch.distributed.new_group(
                ranks=self.ranks,
                backend="ft_nccl",
                use_local_synchronization=True,
            )
        except TypeError:
            self._ft_torch_group = torch.distributed.new_group(
                ranks=self.ranks,
                backend="ft_nccl",
            )

        ft_process_group = (
            get_ft_process_group(self.rank_in_group)
            or get_ft_process_group()
            or get_ft_process_group(self.global_rank)
        )
        if ft_process_group is None:
            raise RuntimeError(
                "VLLM_USE_FT_NCCL_COMMUNICATOR=1 created an ft_nccl group, "
                "but ft_collective did not expose an FTProcessGroup."
            )
        if not hasattr(ft_process_group, "empty"):
            raise RuntimeError(
                "VLLM_USE_FT_NCCL_COMMUNICATOR=1 requires an FTProcessGroup "
                "with empty() staging-buffer allocation support. Imported "
                f"ft_collective from {getattr(ft_collective, '__file__', '<unknown>')}."
            )
        if self._should_use_ft_nccl_ep_communicator():
            missing = [
                method
                for method in ("all_gatherv", "reduce_scatterv")
                if not hasattr(ft_process_group, method)
            ]
            if missing:
                raise RuntimeError(
                    "VLLM_USE_FT_NCCL_EP=1 requires an FTProcessGroup with native "
                    f"{', '.join(missing)} support. Imported ft_collective from "
                    f"{getattr(ft_collective, '__file__', '<unknown>')}."
                )

        self._ft_process_group = ft_process_group
        self._ft_ok_status = FT_OK
        self._prepare_ft_convergence(ft_process_group)
        return ft_process_group

    def _prepare_ft_convergence(self, ft_process_group) -> None:
        """Collectively allocate convergence buffers before a peer can fail."""
        if (
            not envs.VLLM_FT_SURVIVE_WORKER_FAILURE
            or self.world_size <= 1
            or self._ft_group_kind() not in ("tp", "ep")
            or os.environ.get("FT_BARRIER_MODE", "collective") != "collective"
        ):
            return

        prepare = getattr(ft_process_group, "prepare_convergence", None)
        if callable(prepare):
            with torch.inference_mode(False):
                prepare()
            return

        # Compatibility with FTProcessGroup versions that allocate these
        # lazily in _ft_barrier_collective(). empty() and window registration
        # are collective, so this must remain in communicator initialization.
        with torch.inference_mode(False):
            n = self.world_size
            bar_send = getattr(ft_process_group, "_bar_send", None)
            if bar_send is None or bar_send.numel() != n * n:
                ft_process_group._bar_send = ft_process_group.empty(
                    n * n, dtype=torch.float32
                )
                ft_process_group._bar_out = torch.zeros(
                    n * n, dtype=torch.float32, device=self.device
                )

    def _ft_fallback_reason(self, input_: torch.Tensor, op: str) -> str | None:
        return self._ft_staging_fallback_reason(
            input_.dtype,
            input_.device,
            input_.numel(),
            input_.element_size(),
            op,
        )

    def _ft_staging_fallback_reason(
        self,
        dtype: torch.dtype,
        device: torch.device,
        numel: int,
        element_size: int,
        op: str,
    ) -> str | None:
        if dtype not in _FT_NCCL_DTYPES:
            return f"dtype {dtype} is not supported by FT NCCL {op}"
        if device.type != "cuda":
            return "input is not on CUDA"
        if numel == 0:
            return "zero-sized input is not supported by FT NCCL staging"
        if op == "all-gather" and numel * element_size % 4 != 0:
            return "all-gather input byte size is not divisible by 4"

        ft_process_group = self._get_ft_process_group()
        max_count = getattr(ft_process_group, "_max_count", None)
        if max_count is not None and numel > max_count:
            return f"numel {numel} exceeds max_count {max_count}"
        return None

    def _get_ft_staging_buffer(
        self, input_: torch.Tensor, op: str
    ) -> torch.Tensor | None:
        return self._get_ft_staging_view(
            input_.shape,
            input_.dtype,
            input_.device,
            op,
            input_,
        )

    def _get_ft_staging_view(
        self,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        op: str,
        input_: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if input_ is not None:
            input_numel = input_.numel()
            reason = self._ft_fallback_reason(input_, op)
        else:
            input_numel = 1
            for dim in shape:
                input_numel *= dim
            element_size = 4 if dtype == torch.float32 else 2
            reason = self._ft_staging_fallback_reason(
                dtype,
                device,
                input_numel,
                element_size,
                op,
            )

        if reason is not None:
            logger.warning_once(
                "FT NCCL communicator %s fallback for group '%s': %s.",
                op,
                self.unique_name or "<unnamed>",
                reason,
            )
            return None

        workspace = self._get_ft_staging_workspace(dtype, device, input_numel)
        return workspace[:input_numel].view(shape)

    def _get_ft_staging_workspace(
        self,
        dtype: torch.dtype,
        device: torch.device,
        required_numel: int,
    ) -> torch.Tensor:
        key = (dtype, device.type, device.index)
        workspace = self._ft_staging_workspaces.get(key)
        ft_process_group = self._get_ft_process_group()
        max_count = getattr(ft_process_group, "_max_count", None)
        if max_count is None:
            raise RuntimeError(
                "FT NCCL communicator requires FTProcessGroup._max_count "
                "to preallocate its staging workspace."
            )
        capacity = max(max_count, required_numel)
        if workspace is None or workspace.numel() < capacity:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "FT NCCL communicator staging workspace must be initialized "
                    "before CUDA graph capture. Run one eager collective for "
                    f"dtype={dtype} on device={device} first."
                )
            workspace = ft_process_group.empty(capacity, dtype=dtype)
            if workspace.device != device:
                raise RuntimeError(
                    "FT NCCL communicator staging buffer was allocated on "
                    f"{workspace.device}, but input is on {device}."
                )
            self._ft_staging_workspaces[key] = workspace
        return workspace

    def _get_ft_native_staging_view(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        op: str,
        max_segment_numel: int,
        allocation_numel: int,
    ) -> torch.Tensor | None:
        supported_dtypes = (
            _FT_NCCL_ALL_GATHERV_DTYPES if op == "all-gatherv" else _FT_NCCL_DTYPES
        )
        reason = None
        if dtype not in supported_dtypes:
            reason = f"dtype {dtype} is not supported by FT NCCL {op}"
        elif device.type != "cuda":
            reason = "input is not on CUDA"
        elif max_segment_numel == 0:
            reason = f"zero-sized input is not supported by FT NCCL {op}"
        else:
            ft_process_group = self._get_ft_process_group()
            max_count = getattr(ft_process_group, "_max_count", None)
            if max_count is not None:
                segment_bytes = (max_segment_numel * dtype.itemsize + 15) // 16 * 16
                slot_bytes = (max_count * 4 + 15) // 16 * 16
                if segment_bytes > slot_bytes:
                    reason = (
                        f"largest segment requires {segment_bytes} scratch bytes, "
                        f"exceeding the {slot_bytes}-byte FT NCCL slot"
                    )

        if reason is not None:
            logger.warning_once(
                "FT NCCL communicator %s fallback for group '%s': %s.",
                op,
                self.unique_name or "<unnamed>",
                reason,
            )
            return None

        workspace = self._get_ft_staging_workspace(
            dtype, device, max(allocation_numel, max_segment_numel)
        )
        view_numel = 1
        for dim in shape:
            view_numel *= dim
        return workspace[:view_numel].view(shape)

    def _check_ft_nccl_status(self, ft_process_group) -> None:
        if torch.cuda.is_current_stream_capturing():
            return
        if envs.VLLM_FT_SURVIVE_WORKER_FAILURE:
            status = ft_process_group.get_error()
            if status == self._ft_ok_status:
                # FT Work.wait() only orders CUDA streams. The kernel may still
                # be running, so an observed FT_OK is not safe to clear yet.
                return
            if not hasattr(ft_process_group, "get_result_mask"):
                raise RuntimeError(
                    "FT failure survival requires FTProcessGroup.get_result_mask()."
                )
            result_mask = ft_process_group.get_result_mask()
            self._get_ft_active_mask(ft_process_group, refresh=True)
            logger.warning(
                "FT NCCL collective for group '%s' completed after a peer "
                "failure; observed responders: %s.",
                self.unique_name or "<unnamed>",
                result_mask,
            )
            return
        if hasattr(ft_process_group, "check_and_clear_error"):
            status = ft_process_group.check_and_clear_error()
        else:
            status = ft_process_group.get_error()
        if status != self._ft_ok_status:
            raise RuntimeError(f"FT NCCL communicator collective failed: {status}.")

    def converge_ft_membership(self) -> list[bool] | None:
        """Converge this communicator's FT membership between model steps."""
        if not envs.VLLM_FT_SURVIVE_WORKER_FAILURE:
            return None
        ft_process_group = self._get_ft_process_group()
        if not hasattr(ft_process_group, "ft_converge"):
            raise RuntimeError(
                "FT failure survival requires FTProcessGroup.ft_converge()."
            )
        # This can first run from Worker.execute_model(), which is decorated
        # with torch.inference_mode(). FTProcessGroup lazily allocates barrier
        # buffers in ft_converge(); keep them as normal tensors so later calls
        # from execute_dummy_batch() can update them outside inference mode.
        with torch.inference_mode(False):
            old_mask = self._get_ft_active_mask(ft_process_group)
            ft_process_group.ft_converge()
            new_mask = self._get_ft_active_mask(ft_process_group, refresh=True)
        if new_mask != old_mask:
            logger.warning(
                "FT NCCL membership for group '%s' changed from %s to %s.",
                self.unique_name or "<unnamed>",
                old_mask,
                new_mask,
            )
        return new_mask

    def rejoin_ft_membership(self) -> list[bool] | None:
        """Re-admit responsive ranks without rebuilding captured CUDA graphs."""
        if not envs.VLLM_FT_SURVIVE_WORKER_FAILURE:
            return None
        ft_process_group = self._get_ft_process_group()
        if not hasattr(ft_process_group, "ft_rejoin"):
            raise RuntimeError(
                "FT rank rejoin requires FTProcessGroup.ft_rejoin()."
            )

        # Rejoin is an infrequent control-plane transition between model steps.
        # Drain the application stream before changing the device-resident mask;
        # captured collectives retain the same handle and mask address.
        torch.cuda.current_stream(self.device).synchronize()
        with torch.inference_mode(False):
            old_mask = self._get_ft_active_mask(ft_process_group)
            new_mask = list(ft_process_group.ft_rejoin())
            self._ft_active_mask = new_mask
            ft_process_group.clear_error()
        if new_mask != old_mask:
            logger.warning(
                "FT NCCL membership for group '%s' rejoined from %s to %s.",
                self.unique_name or "<unnamed>",
                old_mask,
                new_mask,
            )
        return new_mask

    def _get_ft_active_mask(
        self, ft_process_group, refresh: bool = False
    ) -> list[bool]:
        if refresh or self._ft_active_mask is None:
            self._ft_active_mask = list(ft_process_group.get_active_mask())
        return self._ft_active_mask

    def _get_ft_dp_metadata_buffers(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._ft_dp_metadata_device is None:
            with torch.inference_mode(False), torch.device("cpu"):
                self._ft_dp_metadata_host = torch.empty(
                    2,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=self.device.type == "cuda",
                )
                self._ft_dp_metadata_device = torch.empty(
                    2, dtype=torch.int32, device=self.device
                )
                self._ft_dp_metadata_packed = torch.empty(
                    self.world_size * 2,
                    dtype=torch.int32,
                    device=self.device,
                )
        assert self._ft_dp_metadata_host is not None
        assert self._ft_dp_metadata_device is not None
        assert self._ft_dp_metadata_packed is not None
        return (
            self._ft_dp_metadata_host,
            self._ft_dp_metadata_device,
            self._ft_dp_metadata_packed,
        )

    def set_ft_ep_active_mask(
        self, failed_dp_ranks: tuple[int, ...], dp_size: int
    ) -> list[bool]:
        """Install framework-owned EP membership after a DP failure."""
        if not envs.VLLM_FT_SURVIVE_WORKER_FAILURE:
            raise RuntimeError("FT EP membership requires worker-failure survival.")
        if dp_size <= 0 or self.world_size % dp_size != 0:
            raise ValueError(
                f"EP world size {self.world_size} is not divisible by DP size "
                f"{dp_size}."
            )
        failed = set(failed_dp_ranks)
        if any(rank < 0 or rank >= dp_size for rank in failed):
            raise ValueError(f"Invalid failed DP ranks: {sorted(failed)}")

        replicas_per_dp = self.world_size // dp_size
        active_mask = [
            ep_rank // replicas_per_dp not in failed
            for ep_rank in range(self.world_size)
        ]
        if not active_mask[self.rank_in_group]:
            raise RuntimeError(
                "A withdrawn DP rank cannot install membership for subsequent "
                "EP collectives."
            )

        ft_process_group = self._get_ft_process_group()
        if not hasattr(ft_process_group, "set_active_mask"):
            raise RuntimeError(
                "Framework-managed FT EP membership requires "
                "FTProcessGroup.set_active_mask()."
            )
        old_mask = self._get_ft_active_mask(ft_process_group)
        ft_process_group.set_active_mask(active_mask)
        self._ft_active_mask = active_mask
        # This RPC runs between model steps, after all previous FT work has
        # completed. Do not let a sticky timeout from the interrupted step
        # poison the first collective under the newly installed membership.
        ft_process_group.clear_error()
        if active_mask != old_mask:
            logger.warning(
                "Installed framework-managed FT NCCL membership for group "
                "'%s': %s -> %s.",
                self.unique_name or "<unnamed>",
                old_mask,
                active_mask,
            )
        return active_mask

    def ft_sync_dp_batch_sizes(
        self, local_num_tokens: int, dp_size: int, cudagraph_mode: int
    ) -> tuple[torch.Tensor, int]:
        """Exchange DP token counts and CUDA graph mode over the FT EP group."""
        if not self._should_use_ft_nccl_ep_communicator():
            raise RuntimeError("FT DP batch-size sync requires the FT NCCL EP path.")
        if self.world_size % dp_size != 0:
            raise RuntimeError(
                f"EP world size {self.world_size} is not divisible by DP size "
                f"{dp_size}."
            )

        ft_process_group = self._get_ft_process_group()
        for _ in range(self.world_size):
            metadata_host, local_metadata, packed = self._get_ft_dp_metadata_buffers()
            metadata_host[0] = local_num_tokens
            metadata_host[1] = cudagraph_mode
            local_metadata.copy_(metadata_host, non_blocking=True)
            staging = self._get_ft_native_staging_view(
                (2,),
                local_metadata.dtype,
                local_metadata.device,
                "all-gatherv",
                2,
                2,
            )
            if staging is None:
                raise RuntimeError("FT NCCL could not exchange DP batch metadata.")
            self._copy_to_ft_staging(staging, local_metadata, ft_process_group)
            work, recv_counts = ft_process_group.all_gatherv(
                staging,
                packed,
                2,
                2,
            )
            work.wait()

            # The count exchange defines the shape of the whole forward pass.
            # Complete it before accepting the metadata, but only once per step.
            torch.cuda.current_stream(local_metadata.device).synchronize()
            status = ft_process_group.get_error()
            if status == self._ft_ok_status:
                # FT_OK is already the desired sticky state. Avoid a redundant
                # mapped-host write on every model step.
                tokens_across_dp, synced_cudagraph_mode = _unpack_ft_dp_metadata(
                    packed, recv_counts, dp_size
                )
                if synced_cudagraph_mode != 0 and bool(
                    (recv_counts.cpu() == 0).any().item()
                ):
                    logger.warning_once(
                        "FT EP membership is incomplete for group '%s'; "
                        "retaining CUDA graph replay with active DP ranks "
                        "padded to %d tokens.",
                        self.unique_name or "<unnamed>",
                        int(tokens_across_dp.max().item()),
                    )
                return tokens_across_dp, synced_cudagraph_mode

            result_mask = ft_process_group.get_result_mask()
            old_mask = self._get_ft_active_mask(ft_process_group)
            with torch.inference_mode(False):
                ft_process_group.ft_converge()
            new_mask = self._get_ft_active_mask(ft_process_group, refresh=True)
            logger.warning(
                "Retrying FT NCCL DP batch-size exchange for group '%s' after "
                "status %s; observed responders: %s; membership changed from "
                "%s to %s.",
                self.unique_name or "<unnamed>",
                status,
                result_mask,
                old_mask,
                new_mask,
            )
            if new_mask == old_mask:
                raise RuntimeError(
                    "FT NCCL DP batch-size exchange failed, but convergence "
                    "did not remove a rank from the active mask."
                )

        raise RuntimeError(
            "FT NCCL DP batch-size exchange exhausted its membership-change "
            "retry limit."
        )

    def _get_ft_external_stream(self, ft_process_group):
        stream_ptr = getattr(ft_process_group, "_stream", None)
        if stream_ptr is None:
            return None
        if self._ft_external_stream is None:
            try:
                self._ft_external_stream = torch.cuda.ExternalStream(
                    stream_ptr,
                    device=self.device,
                )
            except TypeError:
                self._ft_external_stream = torch.cuda.ExternalStream(stream_ptr)
        return self._ft_external_stream

    def _copy_to_ft_staging(
        self,
        staging: torch.Tensor,
        input_: torch.Tensor,
        ft_process_group,
    ) -> None:
        ft_stream = self._get_ft_external_stream(ft_process_group)
        if ft_stream is None:
            staging.copy_(input_, non_blocking=True)
            torch.cuda.current_stream(input_.device).synchronize()
            return

        current_stream = torch.cuda.current_stream(input_.device)
        ft_stream.wait_stream(current_stream)
        with torch.cuda.stream(ft_stream):
            staging.copy_(input_, non_blocking=True)

    def _wait_for_ft_stream(self, input_: torch.Tensor, ft_process_group) -> None:
        ft_stream = self._get_ft_external_stream(ft_process_group)
        if ft_stream is not None:
            torch.cuda.current_stream(input_.device).wait_stream(ft_stream)

    def _ft_nccl_staged_all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        staging = self._get_ft_staging_buffer(input_, "all-reduce")
        if staging is None:
            return None

        ft_process_group = self._get_ft_process_group()
        self._copy_to_ft_staging(staging, input_, ft_process_group)
        assert self._ft_torch_group is not None
        torch.distributed.all_reduce(staging, group=self._ft_torch_group)
        self._wait_for_ft_stream(input_, ft_process_group)
        self._check_ft_nccl_status(ft_process_group)

        output = torch.empty_like(input_)
        output.copy_(staging, non_blocking=True)
        return output

    def _ft_nccl_staged_all_gather(
        self, input_: torch.Tensor, dim: int = -1
    ) -> torch.Tensor | None:
        staging = self._get_ft_staging_buffer(input_, "all-gather")
        if staging is None:
            return None

        ft_process_group = self._get_ft_process_group()
        self._copy_to_ft_staging(staging, input_, ft_process_group)
        return self._ft_nccl_all_gather_from_staging(staging, dim)

    def _ft_nccl_all_gather_from_staging(
        self, staging: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        ft_process_group = self._get_ft_process_group()

        if dim < 0:
            dim += staging.dim()
        input_size = staging.size()
        output_factory = (
            torch.zeros if envs.VLLM_FT_SURVIVE_WORKER_FAILURE else torch.empty
        )

        if dim == 0:
            output_tensor = output_factory(
                (self.world_size * input_size[0],) + input_size[1:],
                dtype=staging.dtype,
                device=staging.device,
            )
            output_chunks = output_tensor.reshape(
                (self.world_size,) + input_size
            ).unbind(0)
        else:
            output_tensor = output_factory(
                (self.world_size,) + input_size,
                dtype=staging.dtype,
                device=staging.device,
            )
            output_chunks = output_tensor.unbind(0)

        assert self._ft_torch_group is not None
        torch.distributed.all_gather(
            list(output_chunks), staging, group=self._ft_torch_group
        )
        self._wait_for_ft_stream(staging, ft_process_group)
        self._check_ft_nccl_status(ft_process_group)

        if dim == 0:
            return output_tensor
        return torch.cat(list(output_chunks), dim=dim)

    def _ft_nccl_native_all_gatherv(
        self,
        input_: torch.Tensor,
        dim: int = 0,
        sizes: list[int] | None = None,
        check_status: bool = True,
    ) -> torch.Tensor | None:
        if dim != 0:
            return None
        sizes = sizes or [input_.shape[0]] * self.world_size
        assert len(sizes) == self.world_size
        local_size = sizes[self.rank_in_group]
        assert input_.shape[0] == local_size, f"{input_.shape[0]} != {local_size}"
        max_size = max(sizes)
        output_shape = (sum(sizes),) + input_.shape[1:]
        if max_size == 0:
            return torch.empty(output_shape, dtype=input_.dtype, device=input_.device)

        row_width = 1
        for size in input_.shape[1:]:
            row_width *= size
        staging = self._get_ft_native_staging_view(
            (max_size, row_width),
            input_.dtype,
            input_.device,
            "all-gatherv",
            max_size * row_width,
            # Reserve the reverse-path volume before CUDA graph capture.
            sum(sizes) * row_width,
        )
        if staging is None:
            return None

        ft_process_group = self._get_ft_process_group()
        if local_size > 0:
            self._copy_to_ft_staging(
                staging.narrow(0, 0, local_size),
                input_.reshape(local_size, row_width),
                ft_process_group,
            )
        count_key = (input_.device.type, input_.device.index, local_size)
        device_send_counts = self._ft_all_gatherv_send_counts.get(count_key)
        if device_send_counts is None:
            device_send_counts = torch.empty(
                (self.world_size,),
                dtype=torch.int32,
                device=input_.device,
            )
            self._ft_all_gatherv_send_counts[count_key] = device_send_counts
        # Record count initialization in every CUDA graph that uses this cache.
        device_send_counts.fill_(local_size)
        # A dispatch transaction checks completion before exposing its outputs,
        # so failed-attempt buffers do not need initialization. Direct gathers
        # retain zero filling to keep absent-rank data safe after a timeout.
        output_factory = (
            torch.zeros
            if envs.VLLM_FT_SURVIVE_WORKER_FAILURE and check_status
            else torch.empty
        )
        output = output_factory(output_shape, dtype=input_.dtype, device=input_.device)
        work, _ = ft_process_group.all_gatherv(
            staging,
            output.view(sum(sizes), row_width),
            device_send_counts,
            max_size,
        )
        work.wait()
        if check_status:
            self._check_ft_nccl_status(ft_process_group)
        return output

    def _ft_nccl_all_gatherv_transaction(
        self,
        inputs: list[torch.Tensor],
        dim: int,
        sizes: list[int] | None,
    ) -> list[torch.Tensor]:
        """Gather an EP dispatch payload atomically across membership changes."""
        if not inputs:
            return []
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FT NCCL dispatch retry requires eager execution because it "
                "checks collective status between dispatch attempts."
            )

        ft_process_group = self._get_ft_process_group()
        original_sizes = (
            list(sizes)
            if sizes is not None
            else [inputs[0].shape[dim]] * self.world_size
        )
        if len(original_sizes) != self.world_size:
            raise ValueError(
                f"Expected {self.world_size} all-gatherv sizes, got "
                f"{len(original_sizes)}."
            )

        active_mask = self._get_ft_active_mask(ft_process_group)
        for attempt in range(self.world_size):
            if not active_mask[self.rank_in_group]:
                raise RuntimeError(
                    "The local rank was removed from the FT NCCL active mask."
                )
            active_sizes = [
                size if active else 0
                for size, active in zip(original_sizes, active_mask)
            ]
            outputs = []
            for input_tensor in inputs:
                output = self._ft_nccl_native_all_gatherv(
                    input_tensor,
                    dim,
                    active_sizes,
                    check_status=False,
                )
                if output is None:
                    raise RuntimeError(
                        "FT NCCL dispatch retry requires every payload to use "
                        "the native all-gatherv path."
                    )
                outputs.append(output)

            # Every work item has ordered this stream after the private FT stream.
            # One host synchronization therefore completes the whole dispatch.
            torch.cuda.current_stream(inputs[0].device).synchronize()
            status = ft_process_group.get_error()
            if status == self._ft_ok_status:
                # A successful transaction leaves the sticky flag at FT_OK.
                return outputs

            result_mask = ft_process_group.get_result_mask()
            old_mask = active_mask
            with torch.inference_mode(False):
                ft_process_group.ft_converge()
            active_mask = self._get_ft_active_mask(ft_process_group, refresh=True)
            logger.warning(
                "Retrying FT NCCL dispatch for group '%s' after status %s; "
                "observed responders: %s; membership changed from %s to %s.",
                self.unique_name or "<unnamed>",
                status,
                result_mask,
                old_mask,
                active_mask,
            )
            if active_mask == old_mask:
                raise RuntimeError(
                    "FT NCCL dispatch failed, but convergence did not remove "
                    "a rank from the active mask."
                )

        raise RuntimeError(
            "FT NCCL dispatch exhausted its membership-change retry limit."
        )

    def _ft_nccl_native_reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor | None:
        if dim < 0:
            dim += input_.dim()

        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == self.world_size, f"{len(sizes)} == {self.world_size}"
            assert input_tensor.shape[0] == sum(sizes)
        else:
            assert input_tensor.shape[0] % self.world_size == 0
            sizes = [input_tensor.shape[0] // self.world_size] * self.world_size

        row_width = 1
        for size in input_tensor.shape[1:]:
            row_width *= size
        # vLLM partitions rows; FT reduce_scatterv partitions flattened elements.
        element_counts = [size * row_width for size in sizes]
        staging = self._get_ft_native_staging_view(
            (input_tensor.numel(),),
            input_tensor.dtype,
            input_tensor.device,
            "reduce-scatterv",
            max(element_counts),
            input_tensor.numel(),
        )
        if staging is None:
            return None
        ft_process_group = self._get_ft_process_group()
        self._copy_to_ft_staging(staging, input_tensor.reshape(-1), ft_process_group)
        chunk_size = sizes[self.rank_in_group]
        output = torch.empty(
            (chunk_size,) + input_tensor.shape[1:],
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        work = ft_process_group.reduce_scatterv(
            staging, output.reshape(-1), element_counts
        )
        work.wait()
        self._check_ft_nccl_status(ft_process_group)
        return output.movedim(0, dim).contiguous()

    def all_reduce(self, input_):
        if self._should_use_ft_nccl_tp_communicator():
            out = self._ft_nccl_staged_all_reduce(input_)
            if out is not None:
                return out

        # since currently we perform copy input -> symm_input -> out-of-place AR
        # return symm_output, we don't need to check if input is symmetric
        if self.pynccl_comm is not None and should_nccl_symm_mem_allreduce(
            self.pynccl_comm.world_size, input_
        ):
            out = torch.ops.vllm.all_reduce_symmetric_with_copy(input_)
            if out is not None:
                return out
        # always try quick reduce first, then flashinfer, then custom allreduce,
        # and then pynccl. (quick reduce just for ROCM MI3*)
        qr_comm = self.qr_comm
        if (
            qr_comm is not None
            and not qr_comm.disabled
            and qr_comm.should_quick_allreduce(input_)
        ):
            out = qr_comm.quick_all_reduce(input_)
            assert out is not None
            return out
        fi_ar_comm = self.fi_ar_comm
        if (
            fi_ar_comm is not None
            and not fi_ar_comm.disabled
            and fi_ar_comm.should_use_fi_ar(input_)
        ):
            out = fi_ar_comm.all_reduce(input_)
            assert out is not None
            return out
        ca_comm = self.ca_comm
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input_)
        ):
            out = ca_comm.custom_all_reduce(input_)
            assert out is not None
            return out
        symm_mem_comm = self.symm_mem_comm
        if symm_mem_comm is not None and symm_mem_comm.should_use_symm_mem(input_):
            out = symm_mem_comm.all_reduce(input_)
            assert out is not None
            return out
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group)
            return out
        assert pynccl_comm is not None
        out = pynccl_comm.all_reduce(input_)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group)
        return out

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self._should_use_ft_nccl_tp_communicator():
            out = self._ft_nccl_staged_all_gather(input_, dim)
            if out is not None:
                return out
        return super().all_gather(input_, dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        if self._should_use_ft_nccl_ep_communicator():
            ft_sizes = sizes
            if envs.VLLM_FT_SURVIVE_WORKER_FAILURE and sizes is not None:
                ft_process_group = self._get_ft_process_group()
                active_mask = self._get_ft_active_mask(ft_process_group)
                ft_sizes = [
                    size if active else 0 for size, active in zip(sizes, active_mask)
                ]
            out = self._ft_nccl_native_reduce_scatterv(input_, dim, ft_sizes)
            if out is not None:
                return out

        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size, f"{len(sizes)} == {world_size}"
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
            pynccl_comm.reduce_scatterv(output, input_tensor, sizes=sizes)
        else:
            pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        if self.world_size == 1:
            return tensor

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.broadcast(tensor, src)
            return tensor
        else:
            raise ValueError("No PyNCCL communicator found")

    def destroy(self):
        self._ft_staging_workspaces.clear()
        self._ft_all_gatherv_send_counts.clear()
        self._ft_active_mask = None
        self._ft_dp_metadata_host = None
        self._ft_dp_metadata_device = None
        self._ft_dp_metadata_packed = None
        self._ft_external_stream = None
        self._ft_process_group = None
        if self._ft_torch_group is not None:
            try:
                torch.distributed.destroy_process_group(self._ft_torch_group)
            except Exception:
                logger.exception("Failed to destroy FT NCCL communicator group.")
            self._ft_torch_group = None
        if self.pynccl_comm is not None:
            self.pynccl_comm.destroy()
            self.pynccl_comm = None
        if self.ca_comm is not None:
            self.ca_comm = None
        if self.fi_ar_comm is not None:
            self.fi_ar_comm.destroy()
            self.fi_ar_comm = None
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None  # type: ignore[assignment]

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            if self._should_use_ft_nccl_ep_communicator():
                out = self._ft_nccl_native_all_gatherv(input_, dim, sizes)
                if out is not None:
                    return out

            if pynccl_comm is None or pynccl_comm.disabled:
                raise ValueError(
                    "No PyNCCL communicator found for all-gatherv fallback."
                )

            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )
            if sizes is not None:
                pynccl_comm.all_gatherv(output_tensor, input_, sizes=sizes)
            else:
                pynccl_comm.all_gather(output_tensor, input_)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        if self._should_use_ft_nccl_ep_communicator():
            if envs.VLLM_FT_SURVIVE_WORKER_FAILURE:
                return self._ft_nccl_all_gatherv_transaction(input_, dim, sizes)
            return [_all_gather_single(inp, sizes=sizes) for inp in input_]

        assert pynccl_comm is not None and not pynccl_comm.disabled
        output_list = []
        pynccl_comm.group_start()
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        pynccl_comm.group_end()

        return output_list

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )

    def batch_isend_irecv(self, p2p_ops: list):
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.batch_isend_irecv(p2p_ops)
        else:
            raise ValueError("No PyNCCL communicator found")

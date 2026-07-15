# FT NCCL EP Communicator Benchmark

This benchmark compares dense Ag/Rs, NIXL EP, the FT NCCL staged Ag/Rs path,
and the standalone routed FT NCCL all-to-allv EP backend.

Default model and layout:

- Model: `deepseek-ai/DeepSeek-V2-Lite-Chat`
- `TP=1`
- `DP=8`
- EP enabled via `--enable-expert-parallel`
- EP backends: `allgather_reducescatter`, `nixl_ep`, and `ft_nccl_ep`

Run both configs:

```bash
export FT_COLLECTIVE_PYTHON=/workspace/nccl/contrib/fault_tolerant_collectives/ft_handle/python
export PYTHONPATH="$FT_COLLECTIVE_PYTHON:${PYTHONPATH:-}"

FT_NCCL_MAX_COUNT=33554432 \
RESULT_DIR=bench-results-ft-communicator-ep-8gpu \
benchmarks/ft_nccl_ep_communicator/run_all.sh
```

Run only the routed FT NCCL backend:

```bash
FT_NCCL_MAX_COUNT=33554432 \
RESULT_DIR=bench-results-ft-communicator-ep-8gpu \
benchmarks/ft_nccl_ep_communicator/run_all.sh ft-a2av
```

The initial `ft_nccl_ep` PoC supports single-node, linear expert placement and
unquantized FP16/BF16/FP32 activation transport. It defaults to eager execution
because routing currently uses dynamic PyTorch packing and host-visible receive
counts. Replace those steps with fixed-address GPU routing kernels before using
CUDA graphs for the final performance comparison.

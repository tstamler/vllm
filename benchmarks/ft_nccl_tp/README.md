<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-->

# FT NCCL TP Benchmark Scripts

These scripts compare three TP all-reduce configurations with the same vLLM
API server and `vllm bench serve` workloads:

- `custom-tp`: vLLM's original custom TP all-reduce path.
- `torch-nccl`: PyTorch `torch.distributed.all_reduce` over NCCL.
- `ft-nccl`: FT NCCL TP all-reduce through the `ft_collective` process group.

The defaults target an 8-GPU dense 70B-class run:

```bash
MODEL=Qwen/Qwen2.5-72B-Instruct
TP=8
DTYPE=bfloat16
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
```

Run all configurations and both benchmark shapes:

```bash
benchmarks/ft_nccl_tp/run_all.sh
```

Run only one configuration:

```bash
benchmarks/ft_nccl_tp/run_all.sh ft-nccl
```

Run only the decode-heavy workload:

```bash
FT_TP_BENCH_WORKLOADS=decode benchmarks/ft_nccl_tp/run_all.sh
```

Important environment overrides:

```bash
MODEL=meta-llama/Llama-3.1-70B-Instruct
TP=8
PORT=8000
RESULT_DIR=/path/to/results
FT_COLLECTIVE_PYTHON=/Users/tstamler/nccl/contrib/fault_tolerant_collectives/ft_handle/python
FT_NCCL_MAX_COUNT=67108864
ENFORCE_EAGER=1
VLLM_EXTRA_ENGINE_ARGS="--trust-remote-code"
BENCH_EXTRA_ARGS="--save-detailed"
```

Manual server and client flow:

```bash
benchmarks/ft_nccl_tp/serve_custom_tp.sh
benchmarks/ft_nccl_tp/serve_torch_nccl.sh
benchmarks/ft_nccl_tp/serve_ft_nccl.sh
```

In another shell:

```bash
benchmarks/ft_nccl_tp/bench_decode.sh custom-tp
benchmarks/ft_nccl_tp/bench_prefill.sh custom-tp
```

Results are saved under `bench-results-ft-tp-${TP}gpu` by default. Compare the
median `output_throughput` across repeated runs. Also check `failed == 0` before
using a result.

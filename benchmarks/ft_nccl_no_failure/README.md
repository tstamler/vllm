# FT NCCL No-Failure Performance Evaluation

This harness measures steady-state overhead without injecting a failure. It
compares matched DeepSeek-V2-Lite TP2/DP4 servers:

- `nccl`: regular NCCL tensor parallelism and vLLM's default AG/RS EP path.
- `ft-nccl`: the FT NCCL CUDA communicator, native FT AG/RS EP path, worker
  survival integration, and active-mask handling.

Both configurations use the branch's breakable piecewise CUDA-graph path by
default. This keeps compilation and graph segmentation matched so the measured
delta reflects FT communication and failure-survival overhead rather than a
comparison between two different graph implementations.

Each server runs one decode-oriented and one prefill-oriented workload before
being restarted. The full sequence is repeated three times by default. Results
include server and benchmark logs, raw vLLM JSON, GPU topology, repository
state, and a CSV of median throughput and FT overhead.

On the 8-GPU benchmark host:

```bash
export FT_COLLECTIVE_PYTHON=/workspace/nccl/contrib/fault_tolerant_collectives/ft_handle/python
export HF_TOKEN_FILE=/workspace/secrets/huggingface-token  # optional

RESULT_DIR=bench-results-ft-no-failure \
  benchmarks/ft_nccl_no_failure/run_all.sh
```

Run only one configuration or workload:

```bash
WORKLOADS=decode benchmarks/ft_nccl_no_failure/run_all.sh ft-nccl
WORKLOADS=prefill benchmarks/ft_nccl_no_failure/run_all.sh nccl
```

Useful overrides include `REPETITIONS`, `MODEL`, `TP_SIZE`, `DP_SIZE`,
`GPU_MEMORY_UTILIZATION`, `FT_NCCL_MAX_COUNT`, `BENCH_EXTRA_ARGS`, and
`VLLM_EXTRA_ENGINE_ARGS`. Set `ENFORCE_EAGER=1` only for an explicit eager-mode
diagnostic. Set `USE_BREAKABLE_CUDAGRAPH=0` only to diagnose vLLM's standard
compile/graph path; the default matched comparison evaluates breakable CUDA
graphs for both configurations.

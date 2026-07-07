# FT NCCL Communicator Benchmarks

These scripts compare TP all-reduce configurations for the communicator-only
FT NCCL staging path:

- `custom-tp`: vLLM custom TP all-reduce.
- `torch-nccl`: PyTorch NCCL fallback, with custom all-reduce and PyNccl disabled.
- `ft-nccl`: `VLLM_USE_FT_NCCL_COMMUNICATOR=1`, with custom all-reduce and
  PyNccl disabled so the FT communicator path is the first TP backend.

Run all configs and workloads:

```bash
benchmarks/ft_nccl_communicator/run_all.sh
```

Run only one config:

```bash
benchmarks/ft_nccl_communicator/run_all.sh ft-nccl
```

Run only decode:

```bash
FT_COMM_BENCH_WORKLOADS=decode benchmarks/ft_nccl_communicator/run_all.sh
```

Common overrides:

```bash
MODEL=Qwen/Qwen2.5-32B-Instruct \
TP=8 \
RESULT_DIR=bench-results-ft-communicator-8gpu \
benchmarks/ft_nccl_communicator/run_all.sh
```

Results are written to `RESULT_DIR`, including:

- benchmark JSON files named `<config>-<workload>-<run_id>.json`
- server logs under `logs/server-*.log`
- benchmark logs under `logs/bench-*.log`
- `run-info-<run_id>.txt`
- `nvidia-smi-q-<run_id>.txt`
- `nvidia-smi-topo-<run_id>.txt`
- `backend-summary-<run_id>.txt`

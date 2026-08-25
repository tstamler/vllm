# FT NCCL failure-recovery demos

This directory contains three single-node, eight-GPU experiments for the
experimental FT NCCL integration. The server uses `TP=2`, `DP=4`, `EP=8`, the
default vLLM all-gather/reduce-scatter MoE backend, and breakable piecewise
CUDA graphs unless a command overrides those defaults.

The experiments require an FT NCCL build that provides the PyTorch process
group used by this branch. Stalled-rank rejoin additionally requires
`ft_rejoin()` to update the existing device active mask without replacing the
communicator, buffers, or mask address.

Set the FT NCCL Python path before running any experiment:

```bash
export FT_COLLECTIVE_PYTHON=/workspace/nccl/contrib/fault_tolerant_collectives/ft_handle/python
```

If the model is not already cached, also export `HF_TOKEN` or point the
no-failure harness at a token file with `HF_TOKEN_FILE`.

## 1. No-failure performance

This comparison starts matched regular NCCL and FT NCCL servers and runs one
decode-oriented and one prefill-oriented benchmark. The harness owns the
server lifecycle, so it is the only command needed:

```bash
RESULT_DIR=/tmp/ft-nccl-no-failure \
REPETITIONS=3 \
benchmarks/ft_nccl_no_failure/run_all.sh
```

For a quick single repetition:

```bash
RESULT_DIR=/tmp/ft-nccl-no-failure-r1 \
REPETITIONS=1 \
benchmarks/ft_nccl_no_failure/run_all.sh
```

See `benchmarks/ft_nccl_no_failure/README.md` for workload and topology
overrides. The retained reference run is in `results/no_failure/`.

## Shared recovery server

The killed-rank and stalled-rank experiments use the same server. Start it in
one terminal and wait for `/health` to return HTTP 200:

```bash
FT_TIMEOUT_US=5000000 \
FT_BARRIER_ROUNDS=2 \
FT_BARRIER_MODE=collective \
ENFORCE_EAGER=0 \
examples/fault_tolerance/ft_nccl_ep/run_server.sh \
  2>&1 | tee /tmp/ft-nccl-server.log
```

`ENFORCE_EAGER=0` enables the branch's breakable piecewise CUDA-graph path.
Use `ENFORCE_EAGER=1` only as a diagnostic.

## 2. Killed-rank survival

The reference workload kills `Worker_DP1_TP1` with `SIGKILL`. Its complete DP
replica is withdrawn permanently, in-flight requests assigned to the replica
fail, and new traffic is routed to the three surviving replicas.

Run this command in a second terminal:

```bash
OUTPUT_DIR=/tmp/ft-nccl-killed-rank \
DURATION_SECONDS=360 \
FAILURE_AT_SECONDS=120 \
CONCURRENCY=768 \
REQUEST_TIMEOUT_SECONDS=30 \
MIN_TOKENS=64 \
MAX_TOKENS=256 \
STARTUP_RAMP_SECONDS=20 \
CLIENT_SMOOTHING_SECONDS=30 \
examples/fault_tolerance/ft_nccl_ep/run_recovery_experiment.sh
```

The retained reference run is in `results/killed_rank/`.

## 3. Stalled-rank rejoin

This experiment sends `SIGSTOP` to `Worker_DP1_TP1`, resumes it after 20
seconds, serves temporarily with DP1 withdrawn, and requests rejoin 60 seconds
after resume. Rejoin is admitted at a globally agreed DP step boundary. Each
EngineCore drains outstanding model outputs, rejoins TP groups first and the
world-wide EP group second, verifies full membership, and returns DP1 to
request routing. Existing CUDA graphs remain valid because communicator state
and graph-visible addresses do not change.

Run this command in a second terminal:

```bash
OUTPUT_DIR=/tmp/ft-nccl-stall-rejoin \
DURATION_SECONDS=600 \
STALL_AT_SECONDS=180 \
STALL_SECONDS=20 \
REJOIN_DELAY_SECONDS=60 \
CONCURRENCY=256 \
REQUEST_TIMEOUT_SECONDS=120 \
MIN_TOKENS=192 \
MAX_TOKENS=320 \
STARTUP_RAMP_SECONDS=10 \
REQUEST_JITTER_SECONDS=0.5 \
SERVER_SMOOTHING_SECONDS=10 \
CLIENT_SMOOTHING_SECONDS=30 \
examples/fault_tolerance/ft_nccl_ep/run_rejoin_experiment.sh
```

The retained reference run is in `results/stall_rejoin/`.

## Outputs

The recovery runners produce:

```text
metrics.csv       server counters and scheduler gauges sampled once per second
client.jsonl      one record per successful or failed request
events.csv        failure, resume, and rejoin timestamps
throughput.png    server and delivered-client throughput over time
summary.json      recovery latency and steady-state capacity statistics
```

## Scope and limitations

- The prototype is single-node and assumes all ranks share `/tmp`.
- Killed workers are not recreated; only resumed stalled workers can rejoin.
- Requests in flight on a withdrawn DP replica fail instead of being replayed.
- FT membership preserves liveness but does not reconstruct missing expert
  outputs while a rank is absent. Evaluate reduced-membership output quality
  separately from throughput and recovery behavior.
- Rejoin is an experiment trigger, not a production health-management API.

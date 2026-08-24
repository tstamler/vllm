# FT NCCL EP recovery experiments

These scripts exercise the experimental FT NCCL communicator with the default
vLLM all-gather/reduce-scatter MoE backend.

## Stalled-rank rejoin

The rejoin experiment requires an FT NCCL build where `ft_rejoin()` commits the
new active mask directly to the existing device mask before returning. Updating
membership must not require new buffers, a new communicator, or CUDA graph
recapture.

Start the piecewise CUDA graph server:

```bash
ENFORCE_EAGER=0 \
FT_TIMEOUT_US=5000000 \
FT_BARRIER_ROUNDS=2 \
examples/fault_tolerance/ft_nccl_ep/run_server.sh
```

In another shell, run the experiment:

```bash
STALL_AT_SECONDS=120 \
STALL_SECONDS=15 \
REJOIN_DELAY_SECONDS=30 \
DURATION_SECONDS=360 \
CONCURRENCY=64 \
examples/fault_tolerance/ft_nccl_ep/run_rejoin_experiment.sh
```

The experiment sends `SIGSTOP` to one worker, sends `SIGCONT` after
`STALL_SECONDS`, leaves the reduced membership installed for
`REJOIN_DELAY_SECONDS`, and then writes a generation token to the rejoin trigger.
Every worker calls `ft_rejoin()` once for that generation between model steps.
TP groups rejoin first and the world-wide EP rejoin runs last, acting as the
final rendezvous before workers resume model execution.
The original captured graphs continue to be used because communicator state,
buffer addresses, graph shapes, and the device-mask address remain stable.

The output directory contains server metrics, per-request client results, an
event timeline, a throughput plot, and a JSON summary. The summary reports
pre-stall, degraded, and restored throughput when the corresponding windows are
long enough.

This prototype assumes all ranks are on one host and share `/tmp`. It supports a
stalled process that resumes; it does not recreate or rejoin a killed process.

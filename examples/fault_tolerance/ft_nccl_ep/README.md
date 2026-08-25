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
REJOIN_DELAY_SECONDS=60 \
DURATION_SECONDS=360 \
CONCURRENCY=64 \
examples/fault_tolerance/ft_nccl_ep/run_rejoin_experiment.sh
```

The experiment sends `SIGSTOP` to one worker, sends `SIGCONT` after
`STALL_SECONDS`, leaves the reduced membership installed for
`REJOIN_DELAY_SECONDS`, and then writes a generation token to the rejoin trigger.
Every worker calls `ft_rejoin()` once for that generation between model steps.
Each persistent DP EngineCore polls the trigger and explicitly invokes both of
its TP workers, including workers withdrawn from model execution. Active
workers do not initiate rejoin from the model path.
EngineCore ranks rendezvous on their existing CPU DP group before TP rejoin,
before EP rejoin, and before resuming. A CPU all-reduce commits an attempt only
when every local TP and EP worker reports full membership; filesystem markers
are used only for experiment acknowledgements.
TP groups rejoin first and the world-wide EP rejoin runs last, acting as the
final rendezvous before workers resume model execution.
An incomplete rejoin round is retried in-place rather than escaping through the
active `execute_model` RPC. `VLLM_FT_REJOIN_MAX_ATTEMPTS` bounds these retries;
each attempt can take up to the configured FT timeout.
After a rank observes full membership it publishes a generation-specific
readiness marker but continues calling `ft_rejoin()`. Workers resume model and
CUDA graph execution only after every expected rank has published that marker.
Before each attempt, workers use generation-specific arrival markers to enter
`ft_rejoin()` together. This prevents a resumed rank and the reduced group from
executing different rejoin rounds when the interrupted model step drains late.
Workers rendezvous again after their independent TP rejoin and before the
full-world EP rejoin, since the recovering TP pair can finish later than the
healthy TP pairs.
The original captured graphs continue to be used because communicator state,
buffer addresses, graph shapes, and the device-mask address remain stable.

The output directory contains server metrics, per-request client results, an
event timeline, a throughput plot, and a JSON summary. The summary reports
pre-stall, degraded, and restored throughput when the corresponding windows are
long enough.

This prototype assumes all ranks are on one host and share `/tmp`. It supports a
stalled process that resumes; it does not recreate or rejoin a killed process.

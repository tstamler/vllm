# FT NCCL Serving Failure-Survival Experiment

## Scope

This branch is an initial single-node liveness experiment built on
`ft-nccl-communicator-ep`. It uses FT NCCL for TP collectives and for the
default MoE `allgather_reducescatter` dispatch/combine path. It does not use
NIXL EP.

The experiment targets `TP=2, DP=4` on eight GPUs. When one worker process is
killed, vLLM keeps the worker's DP engine alive but marks it degraded. The
degraded engine stops accepting real requests, fails its in-flight requests,
and uses its surviving worker for dummy MoE steps while a DP wave is active.
The other DP engines remain available for new requests.

Enable this behavior with:

```bash
export VLLM_USE_FT_NCCL_COMMUNICATOR=1
export VLLM_USE_FT_NCCL_EP=1
export VLLM_FT_SURVIVE_WORKER_FAILURE=1
```

## What Changed

1. The multiprocess executor records a dead worker instead of shutting down the
   entire engine when at least one local worker remains.
2. The shared-memory broadcast queue excludes the dead reader from ring-buffer
   flow control.
3. RPC response collection stops waiting for a worker that dies in flight and
   sends later RPCs only to live workers.
4. After a worker death, the still-alive DP engine processes rendezvous on
   their CPU group and issue one aligned FT membership convergence RPC to all
   surviving TP and EP workers. A collective timeout is treated as a membership
   event under the experiment flag rather than an immediate fatal exception.
5. After convergence, worker-side DP batch-size coordination moves from the
   original fixed-membership Gloo group to a small FT EP `all_gatherv`. This
   keeps the per-DP token sizes needed by AG/RS dispatch and combine without
   contacting the dead worker. This fallback currently requires eager mode.
6. A degraded DP engine is relayed through the DP coordinator to front-end load
   balancers. New requests avoid it and its in-flight requests finish with an
   error so clients can retry.
7. During active DP waves, the remaining worker in a degraded engine executes
   dummy batches so the healthy engines can continue their EP collectives.

## Run

Use three terminals from the repository root.

Terminal 1:

```bash
examples/fault_tolerance/ft_nccl_ep/run_server.sh 2>&1 | tee /tmp/ft-server.log
```

Terminal 2, after `/health` returns HTTP 200:

```bash
examples/fault_tolerance/ft_nccl_ep/send_requests.sh | tee /tmp/ft-client.log
```

Terminal 3, while requests are active:

```bash
examples/fault_tolerance/ft_nccl_ep/kill_worker.sh Worker_DP1_TP1
```

Expected liveness signals in `/tmp/ft-server.log`:

```text
FT NCCL worker ... died
DP rank ... lost TP worker
DP engine ... reported a worker failure
FT NCCL membership ... changed
```

The API server should remain alive, `/health` should continue returning 200,
and requests after the failure window should be routed to intact DP engines.
Requests assigned to the degraded engine at failure time may return an error
and should be retried by the client.

## Current Correctness Limit

This first experiment demonstrates process and collective liveness only. In an
MoE model, a failed EP rank may own the only loaded copy of one or more logical
experts. Active masks let `all_gatherv` and `reduce_scatterv` complete without
that rank, but they cannot reconstruct its expert outputs. Therefore, token
accuracy after the kill is not guaranteed yet, even for requests routed to an
otherwise intact DP engine.

## Next Milestone: Correct MoE Recovery

The minimum correctness extension is:

1. Build one authoritative, EP-wide dead-rank set from converged FT masks.
2. Pause admission and fail requests that overlap the membership transition.
3. Remove dead physical expert slots from the EPLB placement.
4. Ensure every logical expert has a surviving physical slot. Reload any
   missing expert weights from the checkpoint into available survivor slots.
5. Atomically publish the new logical-to-physical and physical-to-logical maps
   to every surviving rank before resuming requests.
6. Make EPLB load reduction and placement barriers operate on the survivor EP
   group rather than a fixed group containing the failed process.
7. Resume serving only after all surviving ranks report the same placement
   generation and map checksum.

Until that is implemented, use this branch to measure detection time, request
failure duration, absence of deadlock, and post-failure availability, not model
correctness.

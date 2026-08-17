# FT NCCL Integration and Serving Failure-Survival Experiment

## Integration Status

This branch is a single-node research prototype built on
`ft-nccl-communicator-ep`. It routes the following vLLM communication through
the FT NCCL PyTorch process group:

| Parallel path | vLLM operation | FT NCCL operation | Status |
|---|---|---|---|
| TP | activation reduction | `all_reduce` | Integrated with symmetric staging buffers |
| TP | tensor gathering | `all_gather` | Integrated with symmetric staging buffers |
| EP dispatch metadata | per-DP token-count exchange | `all_gatherv` | Integrated with membership retry |
| EP dispatch | hidden states, top-k IDs, and top-k weights | three native `all_gatherv` calls | Integrated as one checked transaction |
| EP combine | return expert contributions | native `reduce_scatterv` | Integrated; mid-combine replay is not implemented |

The EP path is vLLM's default `allgather_reducescatter` backend, not NIXL EP.
The prototype currently requires eager execution for failure survival because
dispatch checks status and may relaunch collectives after a membership change.

The primary experiment uses DeepSeek-V2-Lite with `TP=2, DP=4` on eight GPUs.
A `SIGKILL` of one TP worker permanently withdraws its entire DP replica. The
remaining three DP replicas continue serving; the prototype does not restart
the worker or restore the original capacity.

Enable the integration with:

```bash
export VLLM_USE_FT_NCCL_COMMUNICATOR=1
export VLLM_USE_FT_NCCL_EP=1
export VLLM_FT_SURVIVE_WORKER_FAILURE=1
export FT_BARRIER_MODE=collective
```

The communicator allocates symmetric staging workspaces and prepares the
collective convergence barrier's symmetric windows during initialization,
before a worker can be removed from the communicator.

## Recovery Design

### Worker and executor survival

1. The multiprocess worker monitor records an individual worker death instead
   of terminating the complete DP executor while another local worker remains.
2. The shared-memory broadcast queue removes the dead reader from ring-buffer
   flow control, preventing a producer from waiting forever for its
   acknowledgement.
3. An RPC captures the live worker set at launch. A newly dead participant
   interrupts the complete RPC even when another rank is the designated reply
   rank. This avoids waiting for the normal model-execution timeout.
4. Once degraded, the DP engine issues no more model or recovery RPCs to its
   surviving worker. This prevents stale responses from an interrupted RPC from
   being consumed as responses to later commands.

### Request routing and failure semantics

1. The affected EngineCore marks its DP replica degraded and finishes all of
   that replica's in-flight requests with `FINISHED_ERROR`.
2. The DP coordinator relays the degraded-engine set to every API frontend.
3. Frontend load balancers remove the degraded engine from request selection
   and synthesize error completion for requests already assigned to it.
4. New requests are routed only to surviving DP replicas. Failed in-flight
   requests are not replayed automatically; retry remains a client policy.

### EP membership transition

1. Worker-side DP batch-size coordination uses a small FT EP `all_gatherv`
   rather than the original fixed-membership Gloo all-reduce.
2. On failure, the count exchange checks FT status, converges membership, zeros
   inactive rank slots, and retries before accepting the new forward-pass
   shape.
3. Dispatch treats its three payload gathers as one transaction. It launches
   all three, performs one synchronization and status check, converges the
   active mask on failure, and retries all three with inactive sizes set to
   zero.
4. Combine uses `reduce_scatterv` counts with inactive rank entries set to zero.
5. The surviving worker in the withdrawn DP replica remains idle. It does not
   execute dummy MoE batches or rejoin EP after withdrawal.

Process-death records in the shared DP store are authoritative for control-plane
withdrawal. FT operation result masks remain authoritative for the in-band EP
transaction. Keeping these roles separate avoids permanently withdrawing a
healthy but temporarily delayed rank based only on one transient kernel mask.

## Current Evidence

The following behaviors have been demonstrated:

- normal eager-mode serving under load with FT TP and FT AG/RS enabled
- FT communicator smoke tests for TP and EP collectives
- continued API availability after killing one worker while request submission
  is paused, followed by successful requests on surviving DP replicas
- in-band EP mask contraction that removes the failed worker and the remaining
  worker of its withdrawn DP replica
- frontend routing around a degraded DP engine

The sustained-load failure experiment remains in progress. An earlier run kept
the server alive and contracted EP membership, but the client completed only a
small number of requests because the EngineCore waited on a reply from the
surviving TP worker until the model RPC timeout. Commit `30645d5df3` changes the
RPC to fail when any participant dies and avoids a follow-up RPC on the
withdrawn executor. The full under-load experiment must be repeated to validate
recovery latency and client progress with this fix.

Therefore, the current evidence supports a claim of collective and service
process survival with reduced capacity. It does not yet support transparent
preservation of in-flight requests or complete post-failure model correctness.

## Experiment

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

Expected recovery signals are:

```text
FT NCCL worker ... died
Model step was interrupted by a worker death
DP rank ... lost TP worker
DP engine ... reported a worker failure
Retrying FT NCCL DP batch-size exchange ...
Retrying FT NCCL dispatch ...
```

The experiment should record:

- worker-death timestamp
- first converged EP-mask timestamp
- degraded-engine routing timestamp
- number and latency of requests completed before, during, and after recovery
- number of failed in-flight requests
- first successful post-failure completion
- post-failure steady-state throughput with three surviving DP replicas

### Automated throughput graph

With the server already running, execute:

```bash
DURATION_SECONDS=600 \
FAILURE_AT_SECONDS=180 \
CONCURRENCY=64 \
examples/fault_tolerance/ft_nccl_ep/run_recovery_experiment.sh
```

The runner maintains closed-loop completion traffic with bounded per-request
timeouts, samples `/metrics` once per second, injects the worker kill, and
produces:

```text
metrics.csv       cumulative server counters and scheduler gauges
client.jsonl      one record per successful, failed, or timed-out request
events.csv        exact worker-kill timestamp
throughput.png    raw and smoothed recovery throughput
summary.json      recovery latency and pre/post capacity statistics
```

The graph uses the derivative of `vllm:generation_tokens_total` for server
output throughput and overlays output tokens from successfully completed client
requests. The latter distinguishes useful delivered throughput from tokens
computed for requests that are later failed during the membership transition.

## Correctness Limits

This prototype primarily demonstrates liveness. A failed EP rank may own the
only loaded copy of one or more logical experts. Active masks allow
`all_gatherv` and `reduce_scatterv` to terminate without that rank, but cannot
reconstruct its expert outputs. Consequently:

- output accuracy after a failure is not guaranteed
- dispatch retries only communication; it does not replace missing experts
- a failure during combine is detected, but complete MoE transaction replay is
  not implemented
- requests spanning a membership transition are failed rather than replayed
- CUDA-graph failure recovery is not supported by this eager-mode experiment
- the prototype permanently loses the complete DP replica and does not recover
  capacity

These limitations must be stated separately from the demonstrated service
availability result.

## Next Milestones

### Complete the liveness evaluation

1. Repeat the sustained-load kill experiment after `30645d5df3`.
2. Verify that DP withdrawal occurs on the worker monitor's polling interval,
   not the model-execution RPC timeout.
3. Confirm that failed streaming requests terminate at the client and that new
   requests continue without an unbounded queue.
4. Sweep request rate and FT timeout to distinguish true failures from
   load-induced false-positive membership changes.

### Add transactional MoE replay

1. Assign a membership epoch to dispatch and combine.
2. Treat any result-mask change during dispatch or combine as invalidating the
   complete MoE transaction.
3. Preserve the original hidden states and routing result until combine commits.
4. Replay dispatch, expert execution, and combine under the new epoch.

### Restore model correctness and capacity

1. Build an authoritative EP-wide failed-rank set from process-death records and
   converged FT membership.
2. Pause admission while expert placement changes.
3. Remove failed physical expert slots from EPLB placement.
4. Select existing replicas or load missing expert weights onto survivor slots.
5. Publish one placement generation and map checksum to all surviving ranks.
6. Make EPLB reductions and barriers operate on the survivor group.
7. Optionally launch replacement workers and rebuild the original DP/EP degree.

Until those milestones are complete, evaluate this branch for failure detection
latency, bounded request loss, absence of collective deadlock, and post-failure
availability rather than transparent recovery or exact post-failure accuracy.

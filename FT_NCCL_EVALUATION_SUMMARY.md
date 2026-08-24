# FT NCCL Evaluation Summary

This branch retains two result sets: the latest fault-recovery experiment and
the latest matched no-failure performance comparison. Older diagnostic and
superseded benchmark outputs have been removed from version control.

## Failure recovery

Results: `clean-results2/ft-recovery-c768/`

- Model: `deepseek-ai/DeepSeek-V2-Lite`
- Parallelism: TP=2, DP=4, EP=8
- Execution: breakable FULL_AND_PIECEWISE CUDA graphs
- Injected failure: `SIGKILL` of `Worker_DP1_TP1`
- Median pre-failure output throughput: 11,127.5 tok/s
- Median post-failure output throughput: 9,405.8 tok/s
- Retained post-failure capacity: 84.5%
- Time to first successful response after failure: 12.1 seconds
- Time to 90% of post-failure throughput: 18.7 seconds
- Requests failed after failure injection: 188

The server detected the worker loss, withdrew the affected DP engine, updated
FT collective membership, and continued serving on the surviving engines.
CUDA graphs remained enabled through the experiment.

## No-failure performance

Results: `bench-results-ft-fresh2/`

- Model: `deepseek-ai/DeepSeek-V2-Lite-Chat`
- Parallelism: TP=1, DP=8, EP=8
- GPU memory utilization: 0.90
- Workloads: 512-request decode and prefill benchmarks
- Repetitions: three per configuration and workload
- Execution: matched breakable FULL_AND_PIECEWISE CUDA graphs

| Workload | Regular NCCL | FT NCCL | FT difference |
| --- | ---: | ---: | ---: |
| Decode output throughput | 7,988.9 tok/s | 5,415.8 tok/s | -32.2% |
| Prefill total throughput | 37,306.5 tok/s | 32,421.6 tok/s | -13.1% |

All 6,144 measured requests completed successfully with no pre-shutdown worker
failures, collective errors, membership changes, CUDA faults, or fallbacks.
Run-to-run variation was low: coefficient of variation was between 0.75% and
1.70% across the four configuration/workload combinations. TCPStore warnings
in some logs occurred only during intentional server shutdown.

FT NCCL used native fault-tolerant `all_gatherv` and `reduce_scatterv` for the
AG/RS expert-parallel path. This comparison includes the current communicator,
active-mask handling, and worker-survival integration, rather than measuring
only the collective implementation.

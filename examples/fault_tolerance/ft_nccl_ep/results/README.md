# Retained reference results

These are the latest clean result sets generated on the prototype branch.
Large server logs and superseded diagnostic runs are intentionally omitted.
The raw benchmark JSON, recovery metrics, client records, event timelines, and
plots needed to inspect the reported measurements are retained.

## No-failure performance

Directory: `no_failure/`

- Model: `deepseek-ai/DeepSeek-V2-Lite-Chat`
- Topology: TP=1, DP=8, EP=8
- Execution: breakable FULL_AND_PIECEWISE CUDA graphs
- Repetitions: one

| Workload | Regular NCCL | FT NCCL | FT difference |
| --- | ---: | ---: | ---: |
| Decode output throughput | 8,022.1 tok/s | 5,451.0 tok/s | -32.1% |
| Prefill total throughput | 37,024.8 tok/s | 32,521.3 tok/s | -12.2% |

## Killed-rank survival

Directory: `killed_rank/`

- Model: `deepseek-ai/DeepSeek-V2-Lite`
- Topology: TP=2, DP=4, EP=8
- Injected failure: `SIGKILL` of `Worker_DP1_TP1`
- Pre-failure median: 11,127.5 output tok/s
- Post-failure median: 9,405.8 output tok/s
- Retained capacity: 84.5%
- First successful response after injection: 12.1 seconds
- Failed in-flight requests: 188

## Stalled-rank rejoin

Directory: `stall_rejoin/`

- Model: `deepseek-ai/DeepSeek-V2-Lite`
- Topology: TP=2, DP=4, EP=8
- Injected stall: `SIGSTOP`/`SIGCONT` of `Worker_DP1_TP1`
- Pre-stall median: 5,409.1 output tok/s
- Restored median: 5,391.6 output tok/s
- Restored capacity: 99.7%
- Rejoin completion: 90.3 seconds after stall injection
- Post-rejoin requests: 7,180 successful, zero failed

The `degraded_output_tok_s_median` in `stall_rejoin/summary.json` includes the
zero-throughput detection interval. After reduced serving resumed and before
rejoin began, measured throughput was approximately 5,199 tok/s.

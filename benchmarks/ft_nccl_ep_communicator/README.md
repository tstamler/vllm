# FT NCCL EP Communicator Benchmark

This benchmark compares dense Ag/Rs, NIXL EP, the FT NCCL staged Ag/Rs path,
and the standalone routed FT NCCL all-to-allv EP backend.

Default model and layout:

- Model: `deepseek-ai/DeepSeek-V2-Lite-Chat`
- `TP=1`
- `DP=8`
- EP enabled via `--enable-expert-parallel`
- EP backends: `allgather_reducescatter`, `nixl_ep`, `ft_nccl_a2av`, and
  `ft_nccl_ep`
- The NIXL EP configuration uses the Ray data-parallel backend and enables
  elastic EP and EPLB, as required by its TCPStore-based rank-management
  interface.

For authenticated Hugging Face downloads, export the token directly:

```bash
export HF_TOKEN="hf_..."
```

Alternatively, keep it in a protected file and pass only the path:

```bash
chmod 600 /workspace/secrets/huggingface-token
export HF_TOKEN_FILE=/workspace/secrets/huggingface-token
```

All server configurations inherit the token. Its value is not written to the
benchmark results; `run-info` records only whether authentication was configured.

Run both configs:

```bash
export FT_COLLECTIVE_PYTHON=/workspace/nccl/contrib/fault_tolerant_collectives/ft_handle/python
export PYTHONPATH="$FT_COLLECTIVE_PYTHON:${PYTHONPATH:-}"

FT_NCCL_MAX_COUNT=134217728 \
RESULT_DIR=bench-results-ft-communicator-ep-8gpu \
benchmarks/ft_nccl_ep_communicator/run_all.sh
```

Run only the routed FT NCCL backend:

```bash
FT_NCCL_MAX_COUNT=134217728 \
RESULT_DIR=bench-results-ft-communicator-ep-8gpu \
benchmarks/ft_nccl_ep_communicator/run_all.sh ft-a2av
```

The two routed FT configurations differ only in how they invoke all-to-allv:

- `ft-a2av-single` launches three independent single-buffer collectives for
  dispatch and one for combine. It does not create application-level A2AV
  workspaces.
- `ft-a2av` launches one multi-buffer collective for dispatch and one for
  combine using persistent workspaces.

Run only the single-buffer comparison with:

```bash
FT_NCCL_MAX_COUNT=134217728 \
RESULT_DIR=bench-results-ft-a2av-single-8gpu \
benchmarks/ft_nccl_ep_communicator/run_all.sh ft-a2av-single
```

The `ft_nccl_ep` PoC supports single-node, linear expert placement and
unquantized FP16/BF16/FP32 activation transport. Device-side fixed-address
routing permits CUDA graph execution when the installed FT NCCL package exposes
graph-safe opaque all-to-allv operations.

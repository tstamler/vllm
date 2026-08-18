# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run bounded-timeout closed-loop completion traffic during FT recovery."""

import argparse
import asyncio
import contextlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--startup-ramp",
        type=float,
        default=10.0,
        help="Seconds over which client workers begin submitting requests.",
    )
    parser.add_argument(
        "--request-jitter",
        type=float,
        default=0.25,
        help="Maximum delay in seconds before each subsequent request.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-template",
        default="Count upward from one. Recovery request {request_id}.",
        help="May contain {request_id} and {worker_id} placeholders.",
    )
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--progress-interval", type=float, default=5.0)
    return parser.parse_args()


async def run_load(args: argparse.Namespace) -> None:
    if args.duration <= 0 or args.concurrency <= 0:
        raise ValueError("duration and concurrency must be positive")
    if args.startup_ramp < 0 or args.request_jitter < 0:
        raise ValueError("startup ramp and request jitter must be non-negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    start_mono = time.monotonic()
    deadline = start_mono + args.duration
    write_lock = asyncio.Lock()
    counters = {"success": 0, "failure": 0, "output_tokens": 0}
    request_sequence = 0

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        output_file = args.output.open("w", encoding="utf-8")

        async def write_record(record: dict[str, Any]) -> None:
            async with write_lock:
                output_file.write(json.dumps(record, sort_keys=True) + "\n")
                output_file.flush()

        async def worker(worker_id: int) -> None:
            nonlocal request_sequence
            rng = random.Random(args.seed + worker_id)
            if args.startup_ramp > 0 and args.concurrency > 1:
                await asyncio.sleep(
                    args.startup_ramp * worker_id / (args.concurrency - 1)
                )
            first_request = True
            while time.monotonic() < deadline:
                if not first_request and args.request_jitter > 0:
                    await asyncio.sleep(rng.uniform(0.0, args.request_jitter))
                    if time.monotonic() >= deadline:
                        break
                first_request = False
                request_sequence += 1
                request_id = f"recovery-{worker_id}-{request_sequence}"
                payload = {
                    "model": args.model,
                    "prompt": args.prompt_template.format(
                        request_id=request_id, worker_id=worker_id
                    ),
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "stream": False,
                }
                started = time.time()
                status = 0
                success = False
                prompt_tokens = 0
                output_tokens = 0
                error_text = ""
                try:
                    async with session.post(args.url, json=payload) as response:
                        status = response.status
                        body = await response.text()
                    try:
                        response_data = json.loads(body)
                    except json.JSONDecodeError:
                        response_data = {}
                    usage = response_data.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    output_tokens = int(usage.get("completion_tokens") or 0)
                    success = status == 200 and bool(response_data.get("choices"))
                    if not success:
                        error_value = response_data.get("error", body)
                        error_text = str(error_value)[:1000]
                except Exception as error:
                    error_text = f"{type(error).__name__}: {error}"

                ended = time.time()
                counters["success" if success else "failure"] += 1
                if success:
                    counters["output_tokens"] += output_tokens
                await write_record(
                    {
                        "request_id": request_id,
                        "worker_id": worker_id,
                        "started_unix": started,
                        "ended_unix": ended,
                        "latency_s": ended - started,
                        "status": status,
                        "success": success,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "error": error_text,
                    }
                )

        async def report_progress() -> None:
            previous = dict(counters)
            previous_time = time.monotonic()
            while time.monotonic() < deadline:
                await asyncio.sleep(args.progress_interval)
                now = time.monotonic()
                successes = counters["success"] - previous["success"]
                failures = counters["failure"] - previous["failure"]
                tokens = counters["output_tokens"] - previous["output_tokens"]
                rate = tokens / max(now - previous_time, 1e-9)
                elapsed = time.monotonic() - start_mono
                print(
                    f"elapsed={elapsed:.1f}s success={successes} "
                    f"failure={failures} client_output_tok_s={rate:.1f}",
                    flush=True,
                )
                previous = dict(counters)
                previous_time = now

        try:
            workers = [
                asyncio.create_task(worker(worker_id))
                for worker_id in range(args.concurrency)
            ]
            reporter = asyncio.create_task(report_progress())
            await asyncio.gather(*workers)
            reporter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reporter
        finally:
            output_file.close()

    print(
        f"completed success={counters['success']} failure={counters['failure']} "
        f"output_tokens={counters['output_tokens']}",
        flush=True,
    )


def main() -> None:
    asyncio.run(run_load(parse_args()))


if __name__ == "__main__":
    main()

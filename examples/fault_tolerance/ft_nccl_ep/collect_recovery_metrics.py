# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample vLLM Prometheus counters for a recovery throughput timeline."""

import argparse
import csv
import re
import time
import urllib.request
from pathlib import Path

SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def parse_metrics(text: str) -> dict[str, float]:
    totals = {
        "prompt_tokens_total": 0.0,
        "generation_tokens_total": 0.0,
        "requests_running": 0.0,
        "requests_waiting": 0.0,
        "requests_finished_total": 0.0,
        "requests_error_total": 0.0,
    }
    found: set[str] = set()

    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue

        if name == "vllm:prompt_tokens_total":
            totals["prompt_tokens_total"] += value
            found.add("prompt_tokens_total")
        elif name == "vllm:generation_tokens_total":
            totals["generation_tokens_total"] += value
            found.add("generation_tokens_total")
        elif name == "vllm:num_requests_running":
            totals["requests_running"] += value
            found.add("requests_running")
        elif name == "vllm:num_requests_waiting":
            totals["requests_waiting"] += value
            found.add("requests_waiting")
        elif name == "vllm:request_success_total":
            totals["requests_finished_total"] += value
            found.add("requests_finished_total")
            labels = dict(LABEL_RE.findall(match.group("labels") or ""))
            finish_reason = labels.get("finished_reason", "").lower()
            if "error" in finish_reason:
                totals["requests_error_total"] += value
                found.add("requests_error_total")

    required = {"prompt_tokens_total", "generation_tokens_total"}
    if not required.issubset(found):
        missing = ", ".join(sorted(required - found))
        raise ValueError(f"Prometheus response is missing: {missing}")
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0 or args.duration <= 0:
        raise ValueError("interval and duration must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_unix",
        "elapsed_s",
        "scrape_ok",
        "prompt_tokens_total",
        "generation_tokens_total",
        "requests_running",
        "requests_waiting",
        "requests_finished_total",
        "requests_error_total",
        "error",
    ]
    start_wall = time.time()
    start_mono = time.monotonic()
    deadline = start_mono + args.duration
    next_sample = start_mono

    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        while time.monotonic() < deadline:
            now_wall = time.time()
            row: dict[str, object] = {
                "timestamp_unix": f"{now_wall:.6f}",
                "elapsed_s": f"{now_wall - start_wall:.6f}",
                "scrape_ok": 0,
                "error": "",
            }
            try:
                with urllib.request.urlopen(  # noqa: S310
                    args.url, timeout=args.request_timeout
                ) as response:
                    metrics = parse_metrics(response.read().decode("utf-8"))
                row.update(metrics)
                row["scrape_ok"] = 1
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            writer.writerow(row)
            output_file.flush()

            next_sample += args.interval
            time.sleep(max(0.0, next_sample - time.monotonic()))


if __name__ == "__main__":
    main()

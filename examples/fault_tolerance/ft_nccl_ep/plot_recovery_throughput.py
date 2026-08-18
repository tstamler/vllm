# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot server and useful-client throughput across an FT recovery event."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict, deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--client-log", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--smoothing-seconds", type=float, default=5.0)
    parser.add_argument(
        "--client-smoothing-seconds",
        type=float,
        default=15.0,
        help="Longer window for completion-based client token throughput.",
    )
    parser.add_argument("--steady-window", type=float, default=60.0)
    parser.add_argument("--pre-guard", type=float, default=5.0)
    parser.add_argument("--include-prompt", action="store_true")
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if row.get("scrape_ok") != "1":
                continue
            try:
                rows.append(
                    {
                        "timestamp": float(row["timestamp_unix"]),
                        "prompt": float(row["prompt_tokens_total"]),
                        "generation": float(row["generation_tokens_total"]),
                        "running": float(row.get("requests_running") or 0),
                    }
                )
            except (KeyError, ValueError):
                continue
    if len(rows) < 2:
        raise ValueError("At least two successful metric samples are required")
    return rows


def counter_rates(
    rows: list[dict[str, float]], counter: str
) -> tuple[list[float], list[float]]:
    timestamps = []
    rates = []
    for previous, current in zip(rows, rows[1:]):
        elapsed = current["timestamp"] - previous["timestamp"]
        delta = current[counter] - previous[counter]
        if elapsed <= 0 or delta < 0:
            continue
        timestamps.append(current["timestamp"])
        rates.append(delta / elapsed)
    return timestamps, rates


def rolling_mean(
    timestamps: list[float], values: list[float], window_seconds: float
) -> list[float]:
    result = []
    window: deque[tuple[float, float]] = deque()
    running_sum = 0.0
    for timestamp, value in zip(timestamps, values):
        window.append((timestamp, value))
        running_sum += value
        while window and timestamp - window[0][0] > window_seconds:
            running_sum -= window.popleft()[1]
        result.append(running_sum / len(window))
    return result


def read_events(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as input_file:
        return [
            {
                "timestamp": float(row["timestamp_unix"]),
                "event": row["event"],
                "detail": row.get("detail", ""),
            }
            for row in csv.DictReader(input_file)
        ]


def read_client_throughput(
    path: Path | None, start_time: float
) -> tuple[list[float], list[float], list[dict[str, object]]]:
    if path is None or not path.exists():
        return [], [], []
    tokens_by_second: dict[int, int] = defaultdict(int)
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
            if record.get("success"):
                second = math.floor(float(record["ended_unix"]) - start_time)
                tokens_by_second[second] += int(record.get("output_tokens") or 0)
    if not tokens_by_second:
        return [], [], records
    first = min(tokens_by_second)
    last = max(tokens_by_second)
    timestamps = [start_time + second for second in range(first, last + 1)]
    values = [float(tokens_by_second[second]) for second in range(first, last + 1)]
    return timestamps, values, records


def median_in_range(
    timestamps: list[float], values: list[float], start: float, end: float
) -> float | None:
    selected = [
        value
        for timestamp, value in zip(timestamps, values)
        if start <= timestamp <= end
    ]
    return statistics.median(selected) if selected else None


def main() -> None:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib") from error

    rows = read_metrics(args.metrics)
    generation_times, generation_rates = counter_rates(rows, "generation")
    generation_smooth = rolling_mean(
        generation_times, generation_rates, args.smoothing_seconds
    )
    prompt_times, prompt_rates = counter_rates(rows, "prompt")
    prompt_smooth = rolling_mean(prompt_times, prompt_rates, args.smoothing_seconds)
    events = read_events(args.events)
    kill_event = next(
        (event for event in events if event["event"] == "worker_killed"), None
    )
    origin = (
        float(kill_event["timestamp"])
        if kill_event is not None
        else rows[0]["timestamp"]
    )
    client_times, client_rates, client_records = read_client_throughput(
        args.client_log, rows[0]["timestamp"]
    )
    client_smooth = rolling_mean(
        client_times, client_rates, args.client_smoothing_seconds
    )

    relative_generation = [timestamp - origin for timestamp in generation_times]
    relative_prompt = [timestamp - origin for timestamp in prompt_times]
    relative_client = [timestamp - origin for timestamp in client_times]

    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(
        relative_generation,
        generation_rates,
        color="#76b900",
        alpha=0.22,
        linewidth=1,
        label="Server output tok/s (1 s)",
    )
    axis.plot(
        relative_generation,
        generation_smooth,
        color="#2f7d00",
        linewidth=2.2,
        label=f"Server output tok/s ({args.smoothing_seconds:g} s mean)",
    )
    if relative_client:
        axis.plot(
            relative_client,
            client_smooth,
            color="#1f77b4",
            linewidth=1.8,
            linestyle="--",
            label=(
                "Successful client output tok/s "
                f"({args.client_smoothing_seconds:g} s mean)"
            ),
        )
    if args.include_prompt:
        axis.plot(
            relative_prompt,
            prompt_smooth,
            color="#9467bd",
            linewidth=1.5,
            label="Server prompt tok/s",
        )

    for event in events:
        event_x = float(event["timestamp"]) - origin
        axis.axvline(event_x, color="#333333", linestyle=":", linewidth=1.2)
        axis.annotate(
            str(event["event"]).replace("_", " "),
            xy=(event_x, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -6),
            textcoords="offset points",
            rotation=90,
            va="top",
            fontsize=8,
        )

    summary: dict[str, object] = {
        "origin_unix": origin,
        "smoothing_seconds": args.smoothing_seconds,
        "client_smoothing_seconds": args.client_smoothing_seconds,
    }
    if kill_event is not None:
        kill_time = float(kill_event["timestamp"])
        pre_start = max(generation_times[0], kill_time - args.steady_window)
        pre_end = kill_time - args.pre_guard
        post_end = generation_times[-1]
        post_start = max(kill_time, post_end - args.steady_window)
        pre_median = median_in_range(
            generation_times, generation_smooth, pre_start, pre_end
        )
        post_median = median_in_range(
            generation_times, generation_smooth, post_start, post_end
        )
        summary["pre_failure_output_tok_s_median"] = pre_median
        summary["post_failure_output_tok_s_median"] = post_median
        summary["post_to_pre_capacity_ratio"] = (
            post_median / pre_median if pre_median and post_median is not None else None
        )
        pre_values = [
            value
            for timestamp, value in zip(generation_times, generation_rates)
            if pre_start <= timestamp <= pre_end
        ]
        summary["pre_failure_output_tok_s_p95"] = (
            sorted(pre_values)[math.ceil(0.95 * len(pre_values)) - 1]
            if pre_values
            else None
        )
        if pre_median is not None:
            axis.axhline(
                pre_median,
                color="#555555",
                linestyle="--",
                linewidth=1,
                label=f"Pre-failure median: {pre_median:.0f} tok/s",
            )
        if post_median is not None:
            axis.axhline(
                post_median,
                color="#888888",
                linestyle="-.",
                linewidth=1,
                label=f"Post-failure median: {post_median:.0f} tok/s",
            )
        zero_timestamp = next(
            (
                timestamp
                for timestamp, value in zip(generation_times, generation_rates)
                if timestamp >= kill_time and value == 0
            ),
            None,
        )
        summary["seconds_to_first_zero_throughput"] = (
            zero_timestamp - kill_time if zero_timestamp is not None else None
        )
        recovered = None
        if post_median is not None and post_median > 0 and zero_timestamp is not None:
            threshold = 0.9 * post_median
            recovered = next(
                (
                    timestamp
                    for timestamp, value in zip(generation_times, generation_smooth)
                    if timestamp >= zero_timestamp and value >= threshold
                ),
                None,
            )
        summary["seconds_to_90pct_post_failure_throughput"] = (
            recovered - kill_time if recovered is not None else None
        )

        successful_after_kill = [
            float(record["ended_unix"])
            for record in client_records
            if record.get("success") and float(record["ended_unix"]) >= kill_time
        ]
        summary["seconds_to_first_successful_response"] = (
            min(successful_after_kill) - kill_time if successful_after_kill else None
        )
        summary["failed_requests_after_kill"] = sum(
            not record.get("success") and float(record["ended_unix"]) >= kill_time
            for record in client_records
        )

    axis.set_title("vLLM Throughput During FT NCCL Worker-Failure Recovery")
    axis.set_xlabel("Seconds relative to worker failure" if kill_event else "Seconds")
    axis.set_ylabel("Tokens per second")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

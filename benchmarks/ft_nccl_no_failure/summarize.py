# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize repeated regular-NCCL and FT-NCCL serving benchmarks."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    pattern = f"*-*-r*-{args.run_id}.json"
    for path in sorted(args.result_dir.glob(pattern)):
        with path.open(encoding="utf-8") as result_file:
            result = json.load(result_file)
        metadata = result.get("metadata") or {}
        config = str(result.get("config") or metadata.get("config"))
        workload = str(result.get("workload") or metadata.get("workload"))
        grouped[(config, workload)].append(result)

    rows: list[dict[str, object]] = []
    medians: dict[tuple[str, str], float] = {}
    for (config, workload), results in sorted(grouped.items()):
        metric = "output_throughput" if workload == "decode" else "total_token_throughput"
        values = [float(result[metric]) for result in results]
        median = statistics.median(values)
        medians[(config, workload)] = median
        rows.append(
            {
                "config": config,
                "workload": workload,
                "metric": metric,
                "runs": len(values),
                "median_tok_s": f"{median:.3f}",
                "min_tok_s": f"{min(values):.3f}",
                "max_tok_s": f"{max(values):.3f}",
                "ft_delta_percent": "",
            }
        )

    for row in rows:
        if row["config"] != "ft-nccl":
            continue
        workload = str(row["workload"])
        baseline = medians.get(("nccl", workload))
        if baseline:
            delta = 100.0 * (medians[("ft-nccl", workload)] / baseline - 1.0)
            row["ft_delta_percent"] = f"{delta:.2f}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "workload",
        "metric",
        "runs",
        "median_tok_s",
        "min_tok_s",
        "max_tok_s",
        "ft_delta_percent",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        raise SystemExit(f"No benchmark JSON files matched {pattern}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

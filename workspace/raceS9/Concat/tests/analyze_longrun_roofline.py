#!/usr/bin/env python3
"""Summarize the long-run Concat matrix against measured 910B4 limits."""

import argparse
import csv
import math
import statistics
from pathlib import Path


D2D_PAYLOAD_BYTES_PER_US = 762_392.0
MIN_WARMED_CHAIN_US = 3.44


def repeated_lengths(count):
    return [16 * (1 << (index % 10)) for index in range(count)]


CASES = {
    "official": ([(128, n) for n in (27, 40, 63, 24, 50, 26, 19, 2, 5)], 2),
    "many64": ([(128, 1 + i % 3) for i in range(64)], 2),
    "many128": ([(64, 1 + i % 3) for i in range(128)], 2),
    "tile16": ([(64, 8192), (64, 8201)], 2),
    "tile32": ([(32, 16384), (32, 16393)], 2),
    "tile64": ([(16, 32768), (16, 32777)], 2),
    "over64": ([(16, 65537), (16, 65539)], 2),
    "dim0_large": ([(4, 262144), (5, 262144)], 2),
    "dim0_unaligned": ([(4, 262145), (5, 262145)], 2),
    "dim0_fp32": ([(4, 131072), (5, 131072)], 4),
    "dim0_int8": ([(4, 524289), (5, 524289)], 1),
    "segment128k": ([(1, 65536), (1, 65536)], 2),
    "segment256k": ([(1, 131072), (1, 131072)], 2),
    "segment512k": ([(1, 262144), (1, 262144)], 2),
    "aligned_last_many64": ([(1024, 16) for _ in range(64)], 2),
    "aligned_middle_many64": ([(64, 1 + i % 4, 16) for i in range(64)], 2),
    "virtual_uneven9": ([(128, n) for n in (16, 32, 64, 128, 256, 512, 2048, 8192, 32768)], 2),
    "virtual_uneven64": ([(64, n) for n in repeated_lengths(64)], 2),
    "whole_32x64": ([(2048, 32) for _ in range(32)], 2),
    "whole_64x64": ([(512, 32) for _ in range(64)], 2),
    "whole_128x32": ([(512, 16) for _ in range(128)], 2),
    "whole_32x256": ([(512, 128) for _ in range(32)], 2),
    "whole_48_mixed": ([(512, n) for n in ([16, 32, 64] * 16)], 2),
    "whole_below1m": ([(128, 16) for _ in range(64)], 2),
}


def product(shape):
    value = 1
    for extent in shape:
        value *= extent
    return value


def value(row, field, default=0.0):
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return default


def median(rows, field, default=0.0):
    values = [value(row, field, default) for row in rows]
    return statistics.median(values) if values else default


def read_rows(root, label, case_name):
    matches = list((root / f"{label}_{case_name}").rglob("op_summary*.csv"))
    if not matches:
        return []
    with matches[0].open(newline="", encoding="utf-8-sig") as source:
        return [row for row in csv.DictReader(source) if row.get("Op Name") == "Concat"]


def summarize(root, label):
    output = []
    for case_name, (shapes, element_bytes) in CASES.items():
        all_rows = read_rows(root, label, case_name)
        if not all_rows:
            continue
        rows = all_rows[10:] if len(all_rows) > 10 else all_rows
        durations = [value(row, "Task Duration(us)") for row in rows]
        payload = sum(product(shape) for shape in shapes) * element_bytes
        d2d_floor = payload / D2D_PAYLOAD_BYTES_PER_US
        chain_floor = max(d2d_floor, MIN_WARMED_CHAIN_US)
        task_median = statistics.median(durations)
        block_dim = int(median(rows, "Block Dim", 1))
        aiv_time = median(rows, "aiv_time(us)")
        cycles = median(rows, "aiv_total_cycles")
        frequency = cycles / aiv_time / block_dim if aiv_time and block_dim else 0.0
        output.append({
            "label": label,
            "case": case_name,
            "inputs": len(shapes),
            "payload_bytes": payload,
            "samples": len(rows),
            "block_dim": block_dim,
            "task_min_us": round(min(durations), 4),
            "task_median_us": round(task_median, 4),
            "task_mean_us": round(statistics.mean(durations), 4),
            "task_max_us": round(max(durations), 4),
            "d2d_floor_us": round(d2d_floor, 4),
            "warmed_chain_floor_us": round(chain_floor, 4),
            "actual_over_d2d": round(task_median / d2d_floor, 2),
            "actual_over_chain_floor": round(task_median / chain_floor, 2),
            "payload_gbps": round(payload / task_median / 1000.0, 2),
            "d2d_payload_util_pct": round(payload / task_median / D2D_PAYLOAD_BYTES_PER_US * 100.0, 2),
            "aiv_time_us": round(aiv_time, 4),
            "aiv_total_cycles": round(cycles, 1),
            "estimated_aiv_mhz": round(frequency, 2),
            "scalar_ratio": round(median(rows, "aiv_scalar_ratio"), 4),
            "mte2_ratio": round(median(rows, "aiv_mte2_ratio"), 4),
            "mte3_ratio": round(median(rows, "aiv_mte3_ratio"), 4),
            "icache_miss_rate": round(median(rows, "aiv_icache_miss_rate"), 6),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for label in args.labels:
        rows.extend(summarize(args.profile_root, label))
    if not rows:
        raise SystemExit("no Concat profiling rows found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

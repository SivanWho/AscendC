#!/usr/bin/env python3
"""Summarize Concat msprof data against transparent transfer-only limits."""

import argparse
import csv
import math
import statistics
from pathlib import Path


GM_PEAK_BYTES_PER_US = 1.6e6
MTE_BYTES_PER_CYCLE_PER_DIRECTION = 64.0
FALLBACK_AIV_MHZ = 800.0

CASES = {
    "official": {
        "shapes": [(128, n) for n in (27, 40, 63, 24, 50, 26, 19, 2, 5)],
        "dtype": "fp16", "bytes": 2, "dim": -1, "path": 0,
    },
    "tile16": {
        "shapes": [(64, 8192), (64, 8201)],
        "dtype": "fp16", "bytes": 2, "dim": 1, "path": 0,
    },
    "tile32": {
        "shapes": [(32, 16384), (32, 16393)],
        "dtype": "fp16", "bytes": 2, "dim": 1, "path": 0,
    },
    "tile64": {
        "shapes": [(16, 32768), (16, 32777)],
        "dtype": "fp16", "bytes": 2, "dim": 1, "path": 2,
    },
    "over64": {
        "shapes": [(16, 65537), (16, 65539)],
        "dtype": "fp16", "bytes": 2, "dim": 1, "path": 2,
    },
    "dim0_large": {
        "shapes": [(4, 262144), (5, 262144)],
        "dtype": "fp16", "bytes": 2, "dim": 0, "path": 1,
    },
    "dim0_unaligned": {
        "shapes": [(4, 262145), (5, 262145)],
        "dtype": "fp16", "bytes": 2, "dim": 0, "path": 1,
    },
    "dim0_fp32": {
        "shapes": [(4, 131072), (5, 131072)],
        "dtype": "fp32", "bytes": 4, "dim": 0, "path": 1,
    },
    "dim0_int8": {
        "shapes": [(4, 524289), (5, 524289)],
        "dtype": "int8", "bytes": 1, "dim": 0, "path": 1,
    },
    "segment128k": {
        "shapes": [(1, 65536), (1, 65536)],
        "dtype": "fp16", "bytes": 2, "dim": 0, "path": 2,
    },
    "segment256k": {
        "shapes": [(1, 131072), (1, 131072)],
        "dtype": "fp16", "bytes": 2, "dim": 0, "path": 1,
    },
    "segment512k": {
        "shapes": [(1, 262144), (1, 262144)],
        "dtype": "fp16", "bytes": 2, "dim": 0, "path": 1,
    },
    "many64": {
        "shapes": [(128, 1 + index % 3) for index in range(64)],
        "dtype": "fp16", "bytes": 2, "dim": 1, "path": 0,
    },
    "many128": {
        "shapes": [(64, 1 + index % 3) for index in range(128)],
        "dtype": "fp16", "bytes": 2, "dim": 1, "path": 0,
    },
}

SYNC_BY_PATH = {
    0: "HardEvent MTE2_MTE3; HardEvent MTE3_MTE2",
    1: "TQueBind<2>: Alloc-MTE2-EnQue-DeQue-MTE3-Free",
    2: "TQueBind<2>: Alloc-MTE2-EnQue-DeQue-MTE3-Free",
}


def output_elements(shapes):
    return sum(math.prod(shape) for shape in shapes)


def find_concat_rows(profile_root, version, case_name):
    matches = list((profile_root / f"{version}_{case_name}").rglob("op_summary*.csv"))
    if not matches:
        return []
    rows = []
    with matches[0].open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row.get("Op Name") == "Concat":
                rows.append(row)
    return rows


def number(row, name, default=0.0):
    value = row.get(name, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def median_metric(rows, name, default=0.0):
    values = [number(row, name, default) for row in rows]
    return statistics.median(values) if values else default


def summarize(profile_root, version):
    output = []
    for case_name, spec in CASES.items():
        all_rows = find_concat_rows(profile_root, version, case_name)
        if not all_rows:
            continue
        # The official Python extension performs 30 benchmark launches per
        # custom_op call. Match the judge convention by using the first group
        # and discarding its ten warm-up launches.
        benchmark_group = all_rows[:30]
        rows = benchmark_group[10:] if len(benchmark_group) > 10 else benchmark_group
        durations = [number(row, "Task Duration(us)") for row in rows]
        duration_us = statistics.median(durations)
        block_dim = int(median_metric(rows, "Block Dim", 1))
        aiv_time_us = median_metric(rows, "aiv_time(us)")
        cycles = median_metric(rows, "aiv_total_cycles")
        frequency_mhz = (
            cycles / aiv_time_us / block_dim
            if cycles > 0 and aiv_time_us > 0 and block_dim > 0
            else FALLBACK_AIV_MHZ
        )
        payload_bytes = output_elements(spec["shapes"]) * spec["bytes"]
        min_gm_bytes = payload_bytes * 2
        direction_capacity = (
            block_dim * MTE_BYTES_PER_CYCLE_PER_DIRECTION * frequency_mhz
        )
        # Path 0 deliberately serializes MTE2 and MTE3. Queue paths use the
        # ideal fully-overlapped read/write bound.
        mte_ideal_us = (
            min_gm_bytes / direction_capacity
            if spec["path"] == 0
            else payload_bytes / direction_capacity
        )
        gm_ideal_us = min_gm_bytes / GM_PEAK_BYTES_PER_US
        theoretical_floor_us = max(mte_ideal_us, gm_ideal_us)
        effective_gbps = min_gm_bytes / duration_us / 1000.0
        output.append({
            "case": case_name,
            "input_shapes": ";".join("x".join(map(str, shape)) for shape in spec["shapes"]),
            "dtype": spec["dtype"],
            "dim": spec["dim"],
            "path": spec["path"],
            "block_dim": block_dim,
            "payload_bytes": payload_bytes,
            "min_gm_bytes": min_gm_bytes,
            "profiler_rows_total": len(all_rows),
            "benchmark_group_size": len(benchmark_group),
            "samples_used": len(rows),
            "task_min_us": round(min(durations), 4),
            "task_median_us": round(duration_us, 4),
            "task_mean_us": round(statistics.mean(durations), 4),
            "task_max_us": round(max(durations), 4),
            "aiv_time_us": round(aiv_time_us, 4),
            "estimated_aiv_mhz": round(frequency_mhz, 2),
            "mte_ideal_us": round(mte_ideal_us, 4),
            "gm_ideal_us": round(gm_ideal_us, 4),
            "theoretical_floor_us": round(theoretical_floor_us, 4),
            "headroom_us": round(duration_us - theoretical_floor_us, 4),
            "actual_over_floor": round(duration_us / theoretical_floor_us, 2),
            "effective_gbps": round(effective_gbps, 2),
            "gm_peak_percent": round(effective_gbps / 1600.0 * 100.0, 2),
            "aiv_scalar_time_us": median_metric(rows, "aiv_scalar_time(us)"),
            "aiv_scalar_ratio": median_metric(rows, "aiv_scalar_ratio"),
            "aiv_mte2_time_us": median_metric(rows, "aiv_mte2_time(us)"),
            "aiv_mte2_ratio": median_metric(rows, "aiv_mte2_ratio"),
            "aiv_mte3_time_us": median_metric(rows, "aiv_mte3_time(us)"),
            "aiv_mte3_ratio": median_metric(rows, "aiv_mte3_ratio"),
            "aiv_icache_miss_rate": median_metric(rows, "aiv_icache_miss_rate"),
            "sync_events": SYNC_BY_PATH[spec["path"]],
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--version", default="hybrid_v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = summarize(args.profile_root, args.version)
    if not rows:
        raise SystemExit("no Concat profiling rows found")
    fieldnames = list(rows[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        target = args.output.open("w", newline="", encoding="utf-8")
    else:
        import sys
        target = sys.stdout
    with target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""Reproducible Concat validation suite based on the official ACLNN contract.

The official S9 example constructs an ND tensor list and invokes
``aclnnConcat(inputs, dim, result)``.  This suite uses the same call ABI while
covering supported dtypes, ranks, legal dimensions, empty slices, input counts,
alignment boundaries, and large tiled transfers.  It intentionally does not
specialize to a known judge shape.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch_npu

import concat_direct_test_ops


SEED = 20260730
MAX_DYNAMIC_INPUTS = 256  # ACLNN rejects larger dynamic tensor lists.
MAX_OUTPUT_BYTES = 8 * 1024 * 1024

DTYPES = (torch.float16, torch.float32, torch.int32, torch.int8)
DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.int32: "int32",
    torch.int8: "int8",
}


def dtype_from_name(name):
    for dtype, dtype_name in DTYPE_NAMES.items():
        if name == dtype_name:
            return dtype
    raise ValueError(name)


def element_bytes(dtype):
    return torch.empty((), dtype=dtype).element_size()


def values_for(shape, dtype, value_mode, seed):
    generator = torch.Generator().manual_seed(seed)
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, dtype=dtype, generator=generator)
    if dtype == torch.int32:
        if value_mode == "small_positive":
            return torch.randint(1, 11, shape, dtype=dtype, generator=generator)
        return torch.randint(-1000, 1001, shape, dtype=dtype, generator=generator)
    if value_mode == "wide":
        return torch.empty(shape, dtype=dtype).uniform_(-1000.0, 1000.0, generator=generator)
    return torch.empty(shape, dtype=dtype).uniform_(-1.0, 1.0, generator=generator)


def bytes_for(shapes, dtype):
    elements = sum(math.prod(shape) for shape in shapes)
    return elements * element_bytes(dtype)


def output_shape(shapes, dim):
    result = list(shapes[0])
    axis = dim if dim >= 0 else dim + len(result)
    result[axis] = sum(shape[axis] for shape in shapes)
    return result


def make_spec(name, dtype, shapes, dim, value_mode="unit"):
    rank = len(shapes[0])
    assert 1 <= rank <= 6
    axis = dim if dim >= 0 else dim + rank
    assert 0 <= axis < rank
    assert 1 <= len(shapes) <= MAX_DYNAMIC_INPUTS
    assert all(len(shape) == rank for shape in shapes)
    assert all(
        all(shape[index] == shapes[0][index] for index in range(rank) if index != axis)
        for shape in shapes
    )
    assert bytes_for(shapes, dtype) <= MAX_OUTPUT_BYTES
    return {
        "name": name,
        "dtype": DTYPE_NAMES[dtype],
        "shapes": [list(shape) for shape in shapes],
        "dim": dim,
        "value_mode": value_mode,
    }


def curated_specs():
    specs = []
    alignment_lengths = (0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65)

    # dtype/rank/first-last axis combinations, including negative dim aliases.
    for dtype in DTYPES:
        for rank in range(1, 7):
            axes = sorted({0, rank - 1})
            for axis in axes:
                base = [2 + (index % 3) for index in range(rank)]
                for sign, label in ((axis, "pos"), (axis - rank, "neg")):
                    shapes = []
                    for length in (1, 3, 7, 0):
                        shape = list(base)
                        shape[axis] = length
                        shapes.append(tuple(shape))
                    specs.append(
                        make_spec(
                            f"rank{rank}_axis{axis}_{label}_{DTYPE_NAMES[dtype]}",
                            dtype,
                            shapes,
                            sign,
                            "wide" if dtype.is_floating_point else "small_positive",
                        )
                    )

    # Byte-alignment boundaries on the concat axis for every dtype.
    for dtype in DTYPES:
        for offset in range(0, len(alignment_lengths), 3):
            lengths = alignment_lengths[offset : offset + 3]
            specs.append(
                make_spec(
                    f"alignment_{DTYPE_NAMES[dtype]}_{offset}",
                    dtype,
                    [(11, length) for length in lengths],
                    1,
                    "wide",
                )
            )

    # Input-list cardinality and empty-slice combinations.
    for dtype in DTYPES:
        for input_count in (1, 2, 3, 8, 16, 64, 128, 129, 256):
            widths = [index % 5 for index in range(input_count)]
            if not any(widths):
                widths[-1] = 1
            specs.append(
                make_spec(
                    f"input_count_{input_count}_{DTYPE_NAMES[dtype]}",
                    dtype,
                    [(4, width) for width in widths],
                    1,
                    "small_positive",
                )
            )

    # Row-count and 2-D DMA boundaries.
    for dtype in DTYPES:
        for rows in (1, 2, 31, 32, 33, 4094, 4095, 4096):
            specs.append(
                make_spec(
                    f"rows_{rows}_{DTYPE_NAMES[dtype]}",
                    dtype,
                    [(rows, 3), (rows, 5), (rows, 1)],
                    1,
                    "wide",
                )
            )

    # Long tile chains and strided output rows. Values are kept under 8 MiB.
    for dtype in DTYPES:
        scale = max(1, 2 // element_bytes(dtype))
        for width in (8192 * scale, 16384 * scale, 32768 * scale, 40003 * scale):
            specs.append(
                make_spec(
                    f"tile_{width}_{DTYPE_NAMES[dtype]}",
                    dtype,
                    [(3, width), (3, width + 7)],
                    1,
                    "wide",
                )
            )
        specs.append(
            make_spec(
                f"dim0_large_{DTYPE_NAMES[dtype]}",
                dtype,
                [(3, 131072 // element_bytes(dtype)), (4, 131072 // element_bytes(dtype))],
                0,
                "wide",
            )
        )

    return specs


def random_shape_spec(rng, index):
    dtype = DTYPES[index % len(DTYPES)]
    rank = rng.randint(1, 6)
    axis = rng.randrange(rank)
    dim = axis if rng.random() < 0.5 else axis - rank
    input_count = rng.choice((2, 3, 4, 5, 8, 16, 32))

    base = [rng.choice((1, 2, 3, 4, 5, 7, 8, 11, 16)) for _ in range(rank)]
    non_axis_product = math.prod(
        size for position, size in enumerate(base) if position != axis
    )
    while non_axis_product > 256:
        largest = max(
            (base[position], position)
            for position in range(rank)
            if position != axis
        )[1]
        base[largest] = 1
        non_axis_product = math.prod(
            size for position, size in enumerate(base) if position != axis
        )

    boundary_lengths = (0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65)
    lengths = [rng.choice(boundary_lengths) for _ in range(input_count)]
    if not any(lengths):
        lengths[rng.randrange(input_count)] = 1

    shapes = []
    for length in lengths:
        shape = list(base)
        shape[axis] = length
        shapes.append(tuple(shape))

    while bytes_for(shapes, dtype) > MAX_OUTPUT_BYTES:
        for shape_index, shape in enumerate(shapes):
            shape_list = list(shape)
            shape_list[axis] = max(0, shape_list[axis] // 2)
            shapes[shape_index] = tuple(shape_list)
        if not any(shape[axis] for shape in shapes):
            shape_list = list(shapes[0])
            shape_list[axis] = 1
            shapes[0] = tuple(shape_list)

    return make_spec(
        f"random_{index:04d}",
        dtype,
        shapes,
        dim,
        "wide" if index % 5 == 0 else "unit",
    )


def generate_specs(total):
    specs = curated_specs()
    if len(specs) > total:
        return specs[:total]
    rng = random.Random(SEED)
    while len(specs) < total:
        specs.append(random_shape_spec(rng, len(specs)))
    return specs


def run_case(index, spec):
    dtype = dtype_from_name(spec["dtype"])
    shapes = [tuple(shape) for shape in spec["shapes"]]
    cpu_inputs = [
        values_for(shape, dtype, spec["value_mode"], SEED + index * 257 + input_index)
        for input_index, shape in enumerate(shapes)
    ]
    expected = torch.cat(cpu_inputs, dim=spec["dim"])
    npu_inputs = [tensor.npu() for tensor in cpu_inputs]
    actual = concat_direct_test_ops.concat_once(
        npu_inputs, spec["dim"], list(expected.shape)
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    if not torch.equal(actual_cpu, expected):
        mismatch = int((actual_cpu != expected).sum().item())
        raise AssertionError(f"{spec['name']}: {mismatch} values differ")
    return expected.numel() * expected.element_size()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    if args.count != 1000:
        raise ValueError("The checked-in suite is fixed at 1000 valid cases")

    specs = generate_specs(args.count)
    stop = args.count if args.stop is None else min(args.stop, args.count)
    if not 0 <= args.start <= stop <= args.count:
        raise ValueError("invalid start/stop range")

    result_dir = Path(args.results)
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = result_dir / "manifest_1000.json"
    manifest_path.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    result_path = result_dir / f"results_{args.start:04d}_{stop:04d}.csv"

    passed = 0
    bytes_checked = 0
    started = time.monotonic()
    with result_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=("index", "name", "dtype", "rank", "input_count", "dim", "bytes", "status", "error"),
        )
        writer.writeheader()
        for index in range(args.start, stop):
            spec = specs[index]
            try:
                checked = run_case(index, spec)
                passed += 1
                bytes_checked += checked
                writer.writerow(
                    {
                        "index": index,
                        "name": spec["name"],
                        "dtype": spec["dtype"],
                        "rank": len(spec["shapes"][0]),
                        "input_count": len(spec["shapes"]),
                        "dim": spec["dim"],
                        "bytes": checked,
                        "status": "pass",
                        "error": "",
                    }
                )
            except Exception as error:
                writer.writerow(
                    {
                        "index": index,
                        "name": spec["name"],
                        "dtype": spec["dtype"],
                        "rank": len(spec["shapes"][0]),
                        "input_count": len(spec["shapes"]),
                        "dim": spec["dim"],
                        "bytes": 0,
                        "status": "fail",
                        "error": repr(error),
                    }
                )
                result_file.flush()
                raise
            result_file.flush()
            if (index + 1) % 25 == 0 or index + 1 == stop:
                elapsed = time.monotonic() - started
                print(
                    f"PROGRESS {index + 1}/{stop} passed={passed} "
                    f"checked_bytes={bytes_checked} elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    elapsed = time.monotonic() - started
    print(
        f"ALL PASS: {passed}/{stop - args.start} cases, "
        f"checked_bytes={bytes_checked}, elapsed_s={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

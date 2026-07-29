"""CPU property tests for the index formulas used by the S9 V1 kernels.

These tests do not replace CANN compilation or on-device precision tests.  They
make the flatten/unflatten formulas independently executable on a Windows host.
"""

from __future__ import annotations

import numpy as np


def greater_v1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out_shape = np.broadcast_shapes(a.shape, b.shape)
    rank = len(out_shape)
    aa = (1,) * (rank - a.ndim) + a.shape
    bb = (1,) * (rank - b.ndim) + b.shape
    sa = tuple(0 if n == 1 else s // a.itemsize for n, s in zip(aa, (0,) * (rank - a.ndim) + a.strides))
    sb = tuple(0 if n == 1 else s // b.itemsize for n, s in zip(bb, (0,) * (rank - b.ndim) + b.strides))
    af, bf = a.ravel(), b.ravel()
    out = np.empty(np.prod(out_shape), dtype=np.bool_)
    for flat in range(out.size):
        rem, oa, ob = flat, 0, 0
        for d in range(rank - 1, -1, -1):
            c, rem = rem % out_shape[d], rem // out_shape[d]
            oa += c * sa[d]
            ob += c * sb[d]
        out[flat] = af[oa] > bf[ob]
    return out.reshape(out_shape)


def transpose_v1(x: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    out_shape = tuple(x.shape[d] for d in perm)
    src_strides = tuple(x.strides[d] // x.itemsize for d in perm)
    flat_x, out = x.ravel(), np.empty(x.size, dtype=x.dtype)
    for flat in range(out.size):
        rem, src = flat, 0
        for d in range(len(out_shape) - 1, -1, -1):
            c, rem = rem % out_shape[d], rem // out_shape[d]
            src += c * src_strides[d]
        out[flat] = flat_x[src]
    return out.reshape(out_shape)


def square_sum_v1(x: np.ndarray, axes: tuple[int, ...], keepdims: bool) -> np.ndarray:
    axes = tuple(d if d >= 0 else d + x.ndim for d in axes)
    reduced = [d in axes for d in range(x.ndim)]
    out_shape = tuple(1 if reduced[d] else x.shape[d] for d in range(x.ndim)) if keepdims else tuple(x.shape[d] for d in range(x.ndim) if not reduced[d])
    logical_shape = tuple(x.shape[d] for d in range(x.ndim) if not reduced[d])
    red_shape = tuple(x.shape[d] for d in range(x.ndim) if reduced[d])
    out = np.empty(max(1, int(np.prod(logical_shape))), dtype=np.float32)
    for of in range(out.size):
        out_coord = np.unravel_index(of, logical_shape) if logical_shape else ()
        total = 0.0
        for rf in range(max(1, int(np.prod(red_shape)))):
            red_coord = np.unravel_index(rf, red_shape) if red_shape else ()
            oi = ri = 0
            coord = []
            for is_red in reduced:
                if is_red: coord.append(red_coord[ri]); ri += 1
                else: coord.append(out_coord[oi]); oi += 1
            v = float(x[tuple(coord)])
            total += v * v
        out[of] = total
    return out.reshape(out_shape)


def index_add_v1(self_: np.ndarray, index: np.ndarray, source: np.ndarray, dim: int) -> np.ndarray:
    dim %= self_.ndim
    inner = int(np.prod(self_.shape[dim + 1 :]))
    m = index.size
    out = self_.copy().ravel()
    for s, value in enumerate(source.ravel()):
        group, rem = divmod(s, m * inner)
        mi, inner_pos = divmod(rem, inner)
        dst_index = int(index[mi]) % self_.shape[dim]
        dst = (group * self_.shape[dim] + dst_index) * inner + inner_pos
        out[dst] += value
    return out.reshape(self_.shape)


def main() -> None:
    rng = np.random.default_rng(9)
    a = rng.normal(size=(3, 1, 5)).astype(np.float32)
    b = rng.normal(size=(1, 4, 1)).astype(np.float32)
    assert np.array_equal(greater_v1(a, b), np.greater(a, b))
    special = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    assert np.array_equal(greater_v1(special, special), np.greater(special, special))

    x = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    assert np.array_equal(transpose_v1(x, (2, 0, 1)), x.transpose(2, 0, 1))

    f = rng.normal(size=(2, 3, 5)).astype(np.float32)
    for axes, keep in [((2,), True), ((0, 2), False), ((-1,), False)]:
        assert np.allclose(square_sum_v1(f, axes, keep), np.sum(np.square(f), axis=axes, keepdims=keep))

    self_ = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(np.int32)
    idx = np.array([2, 0, 2], dtype=np.int32)
    src = np.arange(2 * 3 * 3).reshape(2, 3, 3).astype(np.int32)
    expected = self_.copy()
    for i, dst in enumerate(idx): expected[:, dst, :] += src[:, i, :]
    assert np.array_equal(index_add_v1(self_, idx, src, 1), expected)
    print("S9 V1 CPU index-mapping tests: PASS")


if __name__ == "__main__":
    main()

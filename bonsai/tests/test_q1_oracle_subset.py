"""Parity gates for the pure-NumPy packed-byte Q1 decode oracle."""
from __future__ import annotations

import numpy as np
import pytest

from trinote.infer_int.reference_bonsai import (
    _ORACLE_Q1_WORKERS,
    _unpack_q1_signs,
    q1_linear_ref,
)


def _expanded_oracle(x, bits, scales, frac, *, out_chunk=37):
    """The pre-optimization sign-expansion equation, retained as a test oracle."""
    x = np.atleast_2d(np.asarray(x, dtype=np.int64))
    bits = np.asarray(bits, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.int64)
    out_f, n_blocks = scales.shape
    xg = x.reshape(x.shape[0], n_blocks, 128)
    out = np.empty((x.shape[0], out_f), dtype=np.int64)
    with np.errstate(over="ignore"):
        for lo in range(0, out_f, out_chunk):
            hi = min(lo + out_chunk, out_f)
            signs = _unpack_q1_signs(bits[lo:hi]).astype(np.int64)
            acc = np.einsum("tbi,obi->tob", xg, signs, optimize=True)
            out[:, lo:hi] = (
                ((acc * scales[lo:hi][None, :, :]) >> frac)
                .sum(axis=2, dtype=np.int64)
            )
    return out


@pytest.mark.parametrize("out_f,n_blocks", [(1, 1), (19, 17), (257, 2)])
@pytest.mark.parametrize("frac", [0, 16, 31])
@pytest.mark.parametrize("out_chunk", [1, 7, 256, 1024])
def test_subset_decode_matches_expanded_sign_oracle(out_f, n_blocks, frac, out_chunk):
    rng = np.random.default_rng(out_f * 100_000 + n_blocks * 100 + frac)
    x = rng.integers(-(1 << 28), 1 << 28, size=(1, n_blocks * 128), dtype=np.int64)
    bits = rng.integers(0, 256, size=(out_f, n_blocks, 16), dtype=np.uint8)
    scales = rng.integers(-(1 << 24), 1 << 24, size=(out_f, n_blocks), dtype=np.int64)
    with np.errstate(over="ignore"):
        actual = q1_linear_ref(x, bits, scales, frac, out_chunk=out_chunk)
    expected = _expanded_oracle(x, bits, scales, frac)
    assert np.array_equal(actual, expected)


def test_subset_decode_preserves_modulo_int64_extrema():
    extrema = np.array(
        [np.iinfo(np.int64).min, np.iinfo(np.int64).max, -1, 0, 1],
        dtype=np.int64,
    )
    x = np.resize(extrema, 3 * 128).reshape(1, -1)
    bits = np.empty((41, 3, 16), dtype=np.uint8)
    bits[0::4] = 0x00
    bits[1::4] = 0xFF
    bits[2::4] = 0x55
    bits[3::4] = 0xAA
    scales = np.resize(extrema, 41 * 3).reshape(41, 3)
    with np.errstate(over="ignore"):
        actual = q1_linear_ref(x, bits, scales, 16, out_chunk=11)
    expected = _expanded_oracle(x, bits, scales, 16, out_chunk=13)
    assert np.array_equal(actual, expected)


def test_multirow_prefill_equation_remains_unchanged():
    rng = np.random.default_rng(23)
    x = rng.integers(-(1 << 18), 1 << 18, size=(4, 2 * 128), dtype=np.int64)
    bits = rng.integers(0, 256, size=(29, 2, 16), dtype=np.uint8)
    scales = rng.integers(-(1 << 12), 1 << 12, size=(29, 2), dtype=np.int64)
    assert np.array_equal(q1_linear_ref(x, bits, scales, 16), _expanded_oracle(x, bits, scales, 16))


def test_subset_decode_rejects_nonpositive_chunk():
    with pytest.raises(ValueError, match="out_chunk must be positive"):
        q1_linear_ref(
            np.zeros((1, 128), dtype=np.int64),
            np.zeros((1, 1, 16), dtype=np.uint8),
            np.ones((1, 1), dtype=np.int64),
            16,
            out_chunk=0,
        )


def test_oracle_worker_bound_is_deterministic_and_small():
    assert 1 <= _ORACLE_Q1_WORKERS <= 32

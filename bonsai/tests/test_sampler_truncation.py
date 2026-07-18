from __future__ import annotations

import numpy as np
import pytest

from trinote.infer_int.sampler import _truncate


def _truncate_stable_sort_reference(
    probs: np.ndarray, top_k: int, top_p_fp_val: int, frac_bits: int
) -> np.ndarray:
    """The receipt-bound pre-optimization implementation, kept as an oracle."""

    result = np.asarray(probs, dtype=np.int64).copy()
    order = np.argsort(result, kind="stable")[::-1]
    keep = np.zeros(result.shape[0], dtype=bool)
    keep[order[: max(1, top_k)] if top_k and top_k > 0 else order] = True
    if top_p_fp_val < (1 << frac_bits):
        csum = np.cumsum(result[order])
        n = int(np.searchsorted(csum, int(top_p_fp_val), side="left")) + 1
        nucleus = np.zeros(result.shape[0], dtype=bool)
        nucleus[order[: max(1, min(n, result.shape[0]))]] = True
        keep &= nucleus
    result[~keep] = 0
    if not result.any():
        result[int(order[0])] = 1
    return result


@pytest.mark.parametrize("size", [1, 2, 3, 7, 20, 21, 127, 257])
def test_bounded_top_k_truncation_matches_stable_sort_property(size: int) -> None:
    rng = np.random.default_rng(0xB035 + size)
    frac = 16
    # A deliberately tiny value range creates frequent ties at and around the
    # top-k boundary; injected zeros also exercise the all-zero fallback.
    rows = [
        np.zeros(size, dtype=np.int64),
        np.ones(size, dtype=np.int64),
        np.arange(size, dtype=np.int64) % 5,
    ]
    rows.extend(rng.integers(0, 33, size=size, dtype=np.int64) for _ in range(40))
    ks = sorted({0, 1, 2, 3, 7, 20, max(1, size - 1), size, size + 3})
    thresholds = [0, 1, 7, 1 << 8, (1 << frac) - 1, 1 << frac]
    for probs in rows:
        for top_k in ks:
            for top_p_fp in thresholds:
                expected = _truncate_stable_sort_reference(probs, top_k, top_p_fp, frac)
                actual = _truncate(probs, top_k, top_p_fp, frac)
                assert np.array_equal(actual, expected), (
                    f"size={size} k={top_k} top_p_fp={top_p_fp} probs={probs.tolist()}"
                )


def test_bounded_top_k_preserves_high_token_id_tie_break() -> None:
    probs = np.array([9, 10, 10, 10, 8, 10], dtype=np.int64)
    actual = _truncate(probs, top_k=3, top_p_fp_val=1 << 16, frac_bits=16)
    # Stable ascending argsort followed by reversal historically retained the
    # highest token ids when a tie straddled the k boundary.
    assert np.array_equal(actual, np.array([0, 0, 10, 10, 0, 10], dtype=np.int64))


@pytest.mark.parametrize(
    ("top_p_fp", "expected_ids"),
    [
        (7, [5]),
        (13, [4, 5]),
        (18, [3, 4, 5]),
        # The threshold lies beyond the retained top-k mass. The top-p nucleus
        # continues into discarded ranks, so its intersection is all top-k.
        (100, [3, 4, 5]),
    ],
)
def test_bounded_top_k_nucleus_crossing_is_exact(
    top_p_fp: int, expected_ids: list[int]
) -> None:
    probs = np.array([1, 2, 3, 5, 6, 7], dtype=np.int64)
    actual = _truncate(probs, top_k=3, top_p_fp_val=top_p_fp, frac_bits=8)
    assert np.flatnonzero(actual).tolist() == expected_ids
    assert np.array_equal(
        actual, _truncate_stable_sort_reference(probs, 3, top_p_fp, 8)
    )


def test_bounded_top_k_all_zero_fallback_is_exact() -> None:
    probs = np.zeros(32, dtype=np.int64)
    actual = _truncate(probs, top_k=7, top_p_fp_val=1, frac_bits=16)
    expected = np.zeros_like(probs)
    expected[-1] = 1
    assert np.array_equal(actual, expected)

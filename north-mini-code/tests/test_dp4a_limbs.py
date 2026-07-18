"""Regression (review-3 HIGH): the DP4A balanced-base-256 limb count must be large enough that the greedy
signed-low-byte decomposition reconstructs x EXACTLY. The old bound (2^(8L-1)) over-claimed a high band and
silently corrupted GPU dot products by 256^L per in-band element (e.g. L=2 on [32640, 32767]).

Pure-numeric mirror of make_limbs (nmc_qk_cuda.cu / bonsai_q1_gpu.cu) — no GPU needed."""
import random

import pytest

import numpy as np

from nmc.qk_cuda import (
    _DP4A_SAFE_LIMBS,
    _activation_absmax,
    _balanced_capacity,
    _dp4a_limb_count,
    limbs_needed,
)


def _greedy_limbs(x: int, L: int) -> int:
    """Reconstruct x from L greedy balanced base-256 digits, discarding the final carry — exactly what the
    make_limbs kernel does. Returns the reconstructed value (== x iff L is sufficient)."""
    r = x
    val = 0
    for l in range(L):
        lb = r & 0xFF
        d = lb - 256 if lb >= 128 else lb
        val += d * (256 ** l)
        r = (r - d) >> 8
    return val


def test_capacity_matches_closed_form():
    # cap(L) = 127*(256^L - 1)/255, and the finding's equivalent 2^(8L-1) - 1 - 128*(256^(L-1)-1)/255
    for L in range(1, 9):
        assert _balanced_capacity(L) == 127 * (256 ** L - 1) // 255
    assert _balanced_capacity(2) == 32639          # the boundary the old code got wrong
    assert _balanced_capacity(4) == 2139062143     # matches bonsai's committed L=4 envelope


def test_limbs_needed_reconstructs_exactly_across_boundaries():
    # Every |x| up to cap(L) must reconstruct; the old L=2 bound wrongly claimed up to 32767.
    for x in [0, 1, 127, 128, 32639, 32640, 32767, 32768, 8355711, 8355712, 2139062143, 2139062144]:
        for s in (1, -1):
            v = s * x
            L = limbs_needed(abs(v))
            assert _greedy_limbs(v, L) == v, (v, L)


def test_old_bound_band_now_gets_more_limbs():
    # The exact band the old bound corrupted: [32640, 32767] claimed L=2 but needs L=3.
    assert limbs_needed(32639) == 2
    assert limbs_needed(32640) == 3
    assert limbs_needed(32767) == 3
    # and L=2 provably fails on that band (documents why the fix matters)
    assert _greedy_limbs(32640, 2) == 32640 - 65536


def test_random_reconstruction_fuzz():
    rng = random.Random(1234567)
    for _ in range(20000):
        v = rng.randint(-(1 << 40), 1 << 40)
        L = limbs_needed(abs(v))
        assert _greedy_limbs(v, L) == v, (v, L)


def test_overflow_raises_beyond_8_digits():
    with pytest.raises(OverflowError):
        limbs_needed(_balanced_capacity(8) + 1)


def test_dp4a_dispatch_falls_back_above_int64_safe_envelope():
    cap = _balanced_capacity(_DP4A_SAFE_LIMBS)
    assert _dp4a_limb_count(np.array([cap, -cap], dtype=np.int64)) == _DP4A_SAFE_LIMBS
    # The decomposition remains mathematically exact with five limbs, but the CUDA kernels' int64 weighted
    # recombination is only proven through four. None tells the engine to use its exact int128 kernel.
    assert limbs_needed(cap + 1) == _DP4A_SAFE_LIMBS + 1
    assert _dp4a_limb_count(np.array([cap + 1], dtype=np.int64)) is None


def test_dp4a_explicit_limb_count_is_validated_and_int64_min_is_safe():
    x = np.array([32640], dtype=np.int64)                         # needs 3 limbs, not the old claimed 2
    with pytest.raises(ValueError, match="too small"):
        _dp4a_limb_count(x, 2)
    for bad in (0, _DP4A_SAFE_LIMBS + 1):
        with pytest.raises(ValueError, match="limb count"):
            _dp4a_limb_count(x, bad)
    extreme = np.array([np.iinfo(np.int64).min], dtype=np.int64)
    assert _activation_absmax(extreme) == 1 << 63
    assert _dp4a_limb_count(extreme) is None

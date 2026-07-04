"""Stage-2 gates for the Q4_K/Q6_K integer codec: (1) byte-exact integer determinism, (2) float fidelity."""
import numpy as np
import pytest

from nmc import qk_codec as qk

FRAC = 24                       # block-scale fixed-point bits; ~5e-8 fidelity (see test_*_fidelity)
SEEDS = range(200)


@pytest.mark.parametrize("s", SEEDS)
def test_q4k_int_parity(s):
    b = qk.random_q4k(s)
    assert np.array_equal(qk.dequant_q4k_int(**b, frac=FRAC), qk.dequant_q4k_int_vec(**b, frac=FRAC))


@pytest.mark.parametrize("s", SEEDS)
def test_q6k_int_parity(s):
    b = qk.random_q6k(s)
    assert np.array_equal(qk.dequant_q6k_int(**b, frac=FRAC), qk.dequant_q6k_int_vec(**b, frac=FRAC))


def _max_rel(intv, fltv, frac):
    return np.max(np.abs(intv.astype(np.float64) / (1 << frac) - fltv)) / max(np.max(np.abs(fltv)), 1e-9)


def test_q4k_fidelity():
    worst = max(_max_rel(qk.dequant_q4k_int(**(b := qk.random_q4k(s)), frac=FRAC),
                         qk.dequant_q4k_float(**b), FRAC) for s in SEEDS)
    assert worst < 1e-6, worst


def test_q6k_fidelity():
    worst = max(_max_rel(qk.dequant_q6k_int(**(b := qk.random_q6k(s)), frac=FRAC),
                         qk.dequant_q6k_float(**b), FRAC) for s in SEEDS)
    assert worst < 1e-6, worst


def test_matmul_fixed_vs_float():
    rows = 32
    W_int = np.stack([qk.dequant_q4k_int(**qk.random_q4k(1000 + r), frac=FRAC) for r in range(rows)])
    W_flt = np.stack([qk.dequant_q4k_float(**qk.random_q4k(1000 + r)) for r in range(rows)])
    xf = np.random.default_rng(7).uniform(-1, 1, size=qk.QK_K).astype(np.float64)
    x_int = np.round(xf * (1 << FRAC)).astype(np.int64)
    out_int = qk.matmul_fixed(W_int, x_int, FRAC).astype(np.float64) / (1 << FRAC)
    out_flt = W_flt.astype(np.float64) @ xf
    assert np.max(np.abs(out_int - out_flt)) / max(np.max(np.abs(out_flt)), 1e-9) < 1e-6


def test_matmul_contraction_order_invariant():
    """The integer GEMM is byte-identical under any contraction order — the determinism property that makes
    the engine reproducible across threads/SIMD/GPU tilings. Permute columns of W and x together; result equal."""
    rng = np.random.default_rng(3)
    W = np.stack([qk.dequant_q4k_int(**qk.random_q4k(2000 + r), frac=FRAC) for r in range(8)])
    x = rng.integers(-(1 << FRAC), 1 << FRAC, size=qk.QK_K, dtype=np.int64)
    base = qk.matmul_fixed(W, x, FRAC)
    for _ in range(5):
        p = rng.permutation(qk.QK_K)
        assert np.array_equal(qk.matmul_fixed(W[:, p], x[p], FRAC), base)


def _q4k_raw(b):
    """Pack a synthetic Q4_K block dict back to its 144 raw bytes."""
    return (np.float16(b["d"]).tobytes() + np.float16(b["dmin"]).tobytes()
            + np.asarray(b["scales"], np.uint8).tobytes() + np.asarray(b["qs"], np.uint8).tobytes())


def _q6k_raw(b):
    return (np.asarray(b["ql"], np.uint8).tobytes() + np.asarray(b["qh"], np.uint8).tobytes()
            + np.asarray(b["scales"], np.int8).tobytes() + np.float16(b["d"]).tobytes())


def test_q4k_tensor_vec_matches_perblock():
    """Vectorized whole-tensor Q4_K dequant is byte-identical (int) to the per-block reference over many blocks."""
    blocks = [qk.random_q4k(s) for s in range(40)]
    raw = b"".join(_q4k_raw(b) for b in blocks)
    want = np.concatenate([qk.dequant_q4k_int(**b, frac=FRAC) for b in blocks])
    assert np.array_equal(qk.dequant_q4k_tensor(raw, 40 * 256, frac=FRAC), want)
    fwant = np.concatenate([qk.dequant_q4k_float(**b) for b in blocks])
    assert np.max(np.abs(qk.dequant_q4k_tensor(raw, 40 * 256) - fwant)) == 0.0


def test_q6k_tensor_vec_matches_perblock():
    blocks = [qk.random_q6k(s) for s in range(40)]
    raw = b"".join(_q6k_raw(b) for b in blocks)
    want = np.concatenate([qk.dequant_q6k_int(**b, frac=FRAC) for b in blocks])
    assert np.array_equal(qk.dequant_q6k_tensor(raw, 40 * 256, frac=FRAC), want)
    fwant = np.concatenate([qk.dequant_q6k_float(**b) for b in blocks])
    assert np.max(np.abs(qk.dequant_q6k_tensor(raw, 40 * 256) - fwant)) == 0.0

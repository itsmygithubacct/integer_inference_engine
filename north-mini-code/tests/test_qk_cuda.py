"""CUDA kernel parity gate: the GPU `qk_linear_cuda` MUST be byte-identical to the CPU kernel AND the numpy
oracle, for Q4_K and Q6_K, across shapes/seeds/token-counts. Synthetic blocks — no model/weights needed.
Skips when no GPU / no .so (CPU-only or unbuilt host)."""
import numpy as np
import pytest

from nmc import qk_codec as qk
from nmc import qk_cuda
from tests.test_qk_codec import _q4k_raw, _q6k_raw

FW = 24
pytestmark = pytest.mark.skipif(not qk_cuda.available(),
                                reason="CUDA kernel/GPU unavailable (build tools/build_nmc_cuda.sh on a GPU host)")


def _lin(x, W, fw):                                                           # numpy big-int reference matmul
    return ((np.asarray(x, dtype=object) @ np.asarray(W, dtype=object).T) >> fw).astype(np.int64)


def _oracle(raw, out_f, n_blocks, qtype, x):
    deq = qk.dequant_q4k_tensor if qtype == qk_cuda.Q4_K else qk.dequant_q6k_tensor
    W = deq(raw, out_f * n_blocks * 256, FW).reshape(out_f, n_blocks * 256)
    return _lin(x, W, FW)                                                     # kernel-test is self-contained


@pytest.mark.parametrize("qtype", [qk_cuda.Q4_K, qk_cuda.Q6_K])
@pytest.mark.parametrize("out_f,n_blocks,T", [(8, 1, 1), (16, 2, 3), (5, 3, 4), (64, 8, 2), (262, 8, 1)])
def test_cuda_matches_oracle(qtype, out_f, n_blocks, T):
    gen, pack = ((qk.random_q4k, _q4k_raw) if qtype == qk_cuda.Q4_K else (qk.random_q6k, _q6k_raw))
    raw = b"".join(pack(gen(s)) for s in range(out_f * n_blocks))
    x = np.random.default_rng(out_f + n_blocks + T + qtype).integers(
        -(1 << (FW - 6)), 1 << (FW - 6), size=(T, n_blocks * 256), dtype=np.int64)
    got = qk_cuda.qk_linear(raw, x, out_f, n_blocks, FW, qtype)
    assert got is not None, "CUDA launch returned None"
    assert np.array_equal(got, _oracle(raw, out_f, n_blocks, qtype, x)), np.abs(got - _oracle(raw, out_f, n_blocks, qtype, x)).max()


def test_cuda_matches_cpu_kernel():
    """GPU producer == CPU producer, byte-for-byte (the determinism contract across backends)."""
    from nmc import qk_native
    if not qk_native.available():
        pytest.skip("CPU kernel not built")
    raw = b"".join(_q6k_raw(qk.random_q6k(s)) for s in range(32 * 8))
    x = np.random.default_rng(5).integers(-(1 << 16), 1 << 16, size=(3, 8 * 256), dtype=np.int64)
    g = qk_cuda.qk_linear(raw, x, 32, 8, FW, qk_cuda.Q6_K)
    c = qk_native.qk_linear(raw, x, 32, 8, FW, qk_native.Q6_K)
    assert np.array_equal(g, c)


def test_cuda_negative_floor():
    raw = b"".join(_q4k_raw(qk.random_q4k(s)) for s in range(4 * 4))
    x = -np.abs(np.random.default_rng(2).integers(1, 1 << 14, size=(2, 4 * 256), dtype=np.int64))
    got = qk_cuda.qk_linear(raw, x, 4, 4, FW, qk_cuda.Q4_K)
    assert np.array_equal(got, _oracle(raw, 4, 4, qk_cuda.Q4_K, x))


def test_cuda_resident_matches_oracle_and_percall():
    """Register API: apply_resident byte-identical to the numpy oracle AND to per-call qk_linear (Q4_K+Q6_K)."""
    if not qk_cuda.resident_available():
        pytest.skip("resident register API not in the .so")
    rng = np.random.default_rng(7)
    for qtype in (qk_cuda.Q4_K, qk_cuda.Q6_K):
        out_f, nb, T = 16, 4, 3
        gen, pack = ((qk.random_q4k, _q4k_raw) if qtype == qk_cuda.Q4_K else (qk.random_q6k, _q6k_raw))
        raw = b"".join(pack(gen(s)) for s in range(out_f * nb))
        x = rng.integers(-(1 << 16), 1 << 16, size=(T, nb * 256), dtype=np.int64)
        h = qk_cuda.register_weight(raw, out_f, nb, qtype)
        assert h is not None
        got = qk_cuda.apply_resident(h, x, out_f, FW)
        assert np.array_equal(got, _oracle(raw, out_f, nb, qtype, x))                 # resident == oracle
        assert np.array_equal(got, qk_cuda.qk_linear(raw, x, out_f, nb, FW, qtype))    # resident == per-call
    qk_cuda.free_all()


def test_cuda_moe_ffn_matches_cpu():
    """Fused batched MoE expert-FFN kernel == the per-expert CPU path (matmul + integer SiLU + gu + combine)."""
    if not qk_cuda.moe_ffn_available():
        pytest.skip("qk_moe_ffn not in the .so")
    from nmc._bonsai.fixedpoint import fixed_point_sigmoid
    rng = np.random.default_rng(11)
    n_e, d_model, e_ffn, fa, fw = 4, 512, 256, 16, 24
    nb_in, nb_dn = d_model // 256, e_ffn // 256
    Q = qk_cuda.Q4_K

    def mkw(out_f, nb, base):
        return b"".join(_q4k_raw(qk.random_q4k(base + i)) for i in range(out_f * nb))

    gate_raw = [mkw(e_ffn, nb_in, 1000 + 50 * e) for e in range(n_e)]
    up_raw = [mkw(e_ffn, nb_in, 4000 + 50 * e) for e in range(n_e)]
    down_raw = [mkw(d_model, nb_dn, 8000 + 50 * e) for e in range(n_e)]
    h = rng.integers(-(1 << 16), 1 << 16, size=d_model, dtype=np.int64)
    gates = rng.integers(0, 1 << 16, size=n_e, dtype=np.int64)

    gh = [qk_cuda.register_weight(gate_raw[e], e_ffn, nb_in, Q) for e in range(n_e)]
    uh = [qk_cuda.register_weight(up_raw[e], e_ffn, nb_in, Q) for e in range(n_e)]
    dh = [qk_cuda.register_weight(down_raw[e], d_model, nb_dn, Q) for e in range(n_e)]
    got = qk_cuda.moe_ffn(gh, uh, dh, h, gates, d_model, e_ffn, fa, fw)

    def silu(x):
        s = fixed_point_sigmoid(np.asarray(x, np.int64), fa)
        return ((np.asarray(x, object) * np.asarray(s, object)) >> fa).astype(np.int64)

    ref = np.zeros(d_model, dtype=object)
    for e in range(n_e):
        g = silu(qk_cuda.qk_linear(gate_raw[e], h[None], e_ffn, nb_in, fw, Q)[0])
        u = qk_cuda.qk_linear(up_raw[e], h[None], e_ffn, nb_in, fw, Q)[0]
        gu = ((g.astype(object) * u.astype(object)) >> fa).astype(np.int64)
        d = qk_cuda.qk_linear(down_raw[e], gu[None], d_model, nb_dn, fw, Q)[0]
        ref += (d.astype(object) * int(gates[e])) >> fa
    assert np.array_equal(got, ref.astype(np.int64))
    qk_cuda.free_all()


def test_cuda_q6k_dp4a_matches_resident():
    """DP4A Q6_K apply == the __int128 resident apply == the numpy oracle (byte-exact), across x magnitudes/T."""
    if not qk_cuda.dp4a_available():
        pytest.skip("DP4A path not in the .so")
    rng = np.random.default_rng(5)
    out_f, nb = 24, 8                                          # in_f = 2048 (head-like)
    raw = b"".join(_q6k_raw(qk.random_q6k(s)) for s in range(out_f * nb))
    h = qk_cuda.register_weight(raw, out_f, nb, qk_cuda.Q6_K)
    for T, mag in [(1, 1 << 20), (3, 1 << 20), (2, 1 << 27)]:  # mag>2^24 forces L=5
        x = rng.integers(-mag, mag, size=(T, nb * 256), dtype=np.int64)
        ref = qk_cuda.apply_resident(h, x, out_f, FW)         # int128 path
        got = qk_cuda.apply_resident_q6k_dp4a(h, x, out_f, FW)
        assert got is not None and np.array_equal(got, ref), (T, mag)
        assert np.array_equal(got, _oracle(raw, out_f, nb, qk_cuda.Q6_K, x))
    qk_cuda.free_all()


def test_cuda_q4k_dp4a_matches_resident():
    """DP4A Q4_K apply (the affine dq·sc·q - dmq·m case) == __int128 resident == numpy oracle, byte-exact."""
    if not qk_cuda.dp4a_available():
        pytest.skip("DP4A path not in the .so")
    rng = np.random.default_rng(6)
    out_f, nb = 24, 8                                          # in_f = 2048
    raw = b"".join(_q4k_raw(qk.random_q4k(s)) for s in range(out_f * nb))
    h = qk_cuda.register_weight(raw, out_f, nb, qk_cuda.Q4_K)
    for T, mag in [(1, 1 << 20), (3, 1 << 20), (2, 1 << 27)]:
        x = rng.integers(-mag, mag, size=(T, nb * 256), dtype=np.int64)
        ref = qk_cuda.apply_resident(h, x, out_f, FW)
        got = qk_cuda.apply_resident_dp4a(h, x, out_f, FW, qk_cuda.Q4_K)
        assert got is not None and np.array_equal(got, ref), (T, mag)
        assert np.array_equal(got, _oracle(raw, out_f, nb, qk_cuda.Q4_K, x))
    qk_cuda.free_all()


def test_cuda_moe_ffn_dp4a_matches():
    """Fused-MoE DP4A == the __int128 fused MoE, byte-exact — incl. mixed experts (Q4_K gate/up, Q6_K down)."""
    if not qk_cuda.moe_ffn_dp4a_available():
        pytest.skip("fused-MoE DP4A not in the .so")
    rng = np.random.default_rng(13)
    n_e, d_model, e_ffn, fa, fw = 4, 512, 256, 16, 24
    nb_in, nb_dn = d_model // 256, e_ffn // 256
    mk4 = lambda of, nb, base: b"".join(_q4k_raw(qk.random_q4k(base + i)) for i in range(of * nb))
    mk6 = lambda of, nb, base: b"".join(_q6k_raw(qk.random_q6k(base + i)) for i in range(of * nb))
    h = rng.integers(-(1 << 16), 1 << 16, size=d_model, dtype=np.int64)
    gates = rng.integers(0, 1 << 16, size=n_e, dtype=np.int64)
    for down_q6k in (False, True):
        gh = [qk_cuda.register_weight(mk4(e_ffn, nb_in, 1000 + 50 * e), e_ffn, nb_in, qk_cuda.Q4_K) for e in range(n_e)]
        uh = [qk_cuda.register_weight(mk4(e_ffn, nb_in, 4000 + 50 * e), e_ffn, nb_in, qk_cuda.Q4_K) for e in range(n_e)]
        if down_q6k:
            dh = [qk_cuda.register_weight(mk6(d_model, nb_dn, 8000 + 50 * e), d_model, nb_dn, qk_cuda.Q6_K) for e in range(n_e)]
        else:
            dh = [qk_cuda.register_weight(mk4(d_model, nb_dn, 8000 + 50 * e), d_model, nb_dn, qk_cuda.Q4_K) for e in range(n_e)]
        ref = qk_cuda.moe_ffn(gh, uh, dh, h, gates, d_model, e_ffn, fa, fw, dp4a=False)
        got = qk_cuda.moe_ffn(gh, uh, dh, h, gates, d_model, e_ffn, fa, fw, dp4a=True)
        assert got is not None and np.array_equal(got, ref), f"down_q6k={down_q6k}"
        qk_cuda.free_all()


def test_cuda_moe_ffn_batched_matches():
    """Batched MoE over m·k (token,expert) pairs == per-token qk_moe_ffn, byte-exact (mixed Q4_K gate/up + Q6_K down)."""
    if not qk_cuda.moe_ffn_batched_available():
        pytest.skip("batched MoE not in the .so")
    rng = np.random.default_rng(21)
    n_pool, m, k, d_model, e_ffn, fa, fw = 6, 4, 2, 512, 256, 16, 24
    nb_in, nb_dn = d_model // 256, e_ffn // 256
    mk4 = lambda of, nb, base: b"".join(_q4k_raw(qk.random_q4k(base + i)) for i in range(of * nb))
    mk6 = lambda of, nb, base: b"".join(_q6k_raw(qk.random_q6k(base + i)) for i in range(of * nb))
    gate = [qk_cuda.register_weight(mk4(e_ffn, nb_in, 1000 + 50 * e), e_ffn, nb_in, qk_cuda.Q4_K) for e in range(n_pool)]
    up = [qk_cuda.register_weight(mk4(e_ffn, nb_in, 4000 + 50 * e), e_ffn, nb_in, qk_cuda.Q4_K) for e in range(n_pool)]
    down = [qk_cuda.register_weight(mk6(d_model, nb_dn, 8000 + 50 * e), d_model, nb_dn, qk_cuda.Q6_K) for e in range(n_pool)]
    h = rng.integers(-(1 << 16), 1 << 16, size=(m, d_model), dtype=np.int64)
    sel = [rng.choice(n_pool, k, replace=False) for _ in range(m)]
    gts = [rng.integers(0, 1 << 16, size=k).astype(np.int64) for _ in range(m)]
    gh = [gate[e] for t in range(m) for e in sel[t]]
    uh = [up[e] for t in range(m) for e in sel[t]]
    dh = [down[e] for t in range(m) for e in sel[t]]
    gflat = [int(g) for t in range(m) for g in gts[t]]
    got = qk_cuda.moe_ffn_batched(gh, uh, dh, m, k, h, gflat, d_model, e_ffn, fa, fw)
    ref = np.empty((m, d_model), np.int64)
    for t in range(m):
        ref[t] = qk_cuda.moe_ffn([gate[e] for e in sel[t]], [up[e] for e in sel[t]], [down[e] for e in sel[t]],
                                 h[t], gts[t], d_model, e_ffn, fa, fw)
    assert got is not None and np.array_equal(got, ref)
    qk_cuda.free_all()


def test_cuda_free_all_then_reregister():
    """free_all() resets the registry; re-registering the same weight yields the same byte-exact result —
    the mechanism engine.free() relies on (it clears its handle cache so reuse re-registers, not stale-handle)."""
    if not qk_cuda.resident_available():
        pytest.skip("resident API not in the .so")
    raw = b"".join(_q6k_raw(qk.random_q6k(s)) for s in range(8 * 4))
    x = np.random.default_rng(0).integers(-(1 << 16), 1 << 16, size=(1, 4 * 256), dtype=np.int64)
    h1 = qk_cuda.register_weight(raw, 8, 4, qk_cuda.Q6_K)
    r1 = qk_cuda.apply_resident(h1, x, 8, FW)
    qk_cuda.free_all()
    h2 = qk_cuda.register_weight(raw, 8, 4, qk_cuda.Q6_K)     # fresh handle after free
    r2 = qk_cuda.apply_resident(h2, x, 8, FW)
    assert np.array_equal(r1, r2)
    qk_cuda.free_all()

"""CUDA kernel parity gate: the GPU `qk_linear_cuda` MUST be byte-identical to the CPU kernel AND the numpy
oracle, for Q4_K and Q6_K, across shapes/seeds/token-counts. Synthetic blocks — no model/weights needed.
Skips when no GPU / no .so (CPU-only or unbuilt host)."""
from concurrent.futures import ThreadPoolExecutor

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


def test_cuda_resident_rmsnorm_router_topk_matches_integer_oracle():
    """Dense gain/router residency and compact route metadata are byte-exact."""
    if not qk_cuda.resident_preprocess_available():
        pytest.skip("resident RMSNorm/router API not in the .so")
    from nmc import cohere2 as c2

    qk_cuda.free_all()
    rng = np.random.default_rng(1907)
    rows, width, experts, used, fa, fw = 3, 256, 8, 3, 16, 24
    x = rng.integers(-(1 << 18), 1 << 18, size=(rows, width), dtype=np.int64)
    gain = rng.integers(1 << 15, 1 << 17, size=width, dtype=np.int64)
    router = rng.integers(-(1 << 13), 1 << 13, size=(experts, width), dtype=np.int64)
    gain_h = qk_cuda.register_i64(gain)
    router_h = qk_cuda.register_i64(router)

    got = qk_cuda.rmsnorm_router(gain_h, router_h, x, used, fa, fw, eps=1)
    assert got is not None
    h, ids, gates = got
    ref_h = c2.fixed_point_rmsnorm(x, fa, 1, gain_q=gain)
    logits = c2.linear(ref_h, router, fw)
    ref_ids = np.stack([c2._topk_lowidx(row, used) for row in logits])
    ref_gates = np.stack([
        c2.fixed_point_sigmoid(logits[t][ref_ids[t]].astype(np.int64), fa)
        for t in range(rows)
    ])
    assert np.array_equal(h, ref_h)
    assert np.array_equal(ids, ref_ids)
    assert np.array_equal(gates, ref_gates)

    # Equal logits must retain ascending expert IDs, the committed routing tie rule.
    zero_router_h = qk_cuda.register_i64(np.zeros_like(router))
    tied = qk_cuda.rmsnorm_router(gain_h, zero_router_h, x[:1], used, fa, fw)
    assert tied is not None and tied[1].tolist() == [list(range(used))]
    expected_gate = int(c2.fixed_point_sigmoid(np.array([0], dtype=np.int64), fa)[0])
    assert tied[2].tolist() == [[expected_gate] * used]

    # Dense handles are not accepted by the quantized resident projection ABI.
    assert qk_cuda.apply_resident(gain_h, x[:1], 1, fw) is None

    # Two INT64_MIN lanes exceed signed-i128 sum-of-squares and fall back before native mutation.
    assert qk_cuda.profile_reset(enabled=True)
    unsafe = np.zeros((1, width), dtype=np.int64)
    unsafe[0, :2] = np.iinfo(np.int64).min
    assert qk_cuda.rmsnorm_router(gain_h, router_h, unsafe, used, fa, fw) is None
    assert qk_cuda.profile_snapshot()["native_calls"] == 0
    qk_cuda.profile_set_enabled(False)
    qk_cuda.free_all()


def test_cuda_block_parallel_rmsnorm_router_matches_real_shape_oracle():
    """The guarded block kernels are exact at the shipped 2048x128 shape."""
    if not qk_cuda.resident_preprocess_available():
        pytest.skip("resident RMSNorm/router API not in the .so")
    from nmc import cohere2 as c2

    qk_cuda.free_all()
    rng = np.random.default_rng(1928)
    rows, width, experts, used, fa, fw = 2, 2048, 128, 8, 16, 24
    x = rng.integers(-(1 << 18), 1 << 18, size=(rows, width), dtype=np.int64)
    gain = rng.integers(1 << 15, 1 << 17, size=width, dtype=np.int64)
    router = rng.integers(-(1 << 13), 1 << 13, size=(experts, width), dtype=np.int64)
    gain_h = qk_cuda.register_i64(gain)
    router_h = qk_cuda.register_i64(router)

    got = qk_cuda.rmsnorm_router(gain_h, router_h, x, used, fa, fw, eps=1)
    assert got is not None
    h, ids, gates = got
    ref_h = c2.fixed_point_rmsnorm(x, fa, 1, gain_q=gain)
    logits = c2.linear(ref_h, router, fw)
    ref_ids = np.stack([c2._topk_lowidx(row, used) for row in logits])
    ref_gates = np.stack([
        c2.fixed_point_sigmoid(logits[row][ref_ids[row]].astype(np.int64), fa)
        for row in range(rows)
    ])
    assert np.array_equal(h, ref_h)
    assert np.array_equal(ids, ref_ids)
    assert np.array_equal(gates, ref_gates)
    qk_cuda.free_all()


def test_cuda_block_parallel_preprocess_exact_fallback_and_error():
    """Conservative fast-path misses fall back exactly; true gain overflow still fails loud."""
    if not qk_cuda.resident_preprocess_available():
        pytest.skip("resident RMSNorm/router API not in the .so")
    from nmc import cohere2 as c2

    qk_cuda.free_all()
    width, experts, used, fa, fw = 2048, 128, 8, 16, 24
    # The RMS sum exceeds uint64, while each router contraction has enormous
    # cancelling terms. Both are valid for the serial i128 kernels but outside
    # the conservative block-kernel proofs.
    x = np.full((1, width), 1 << 35, dtype=np.int64)
    gain = np.full(width, 1 << fa, dtype=np.int64)
    alternating = np.empty(width, dtype=np.int64)
    alternating[0::2] = 1 << 50
    alternating[1::2] = -(1 << 50)
    router = np.tile(alternating, (experts, 1))
    gain_h = qk_cuda.register_i64(gain)
    router_h = qk_cuda.register_i64(router)

    got = qk_cuda.rmsnorm_router(gain_h, router_h, x, used, fa, fw, eps=1)
    assert got is not None
    h, ids, gates = got
    ref_h = c2.fixed_point_rmsnorm(x, fa, 1, gain_q=gain)
    logits = c2.linear(ref_h, router, fw)
    ref_ids = np.stack([c2._topk_lowidx(row, used) for row in logits])
    ref_gates = np.stack([
        c2.fixed_point_sigmoid(logits[row][ref_ids[row]].astype(np.int64), fa)
        for row in range(x.shape[0])
    ])
    assert np.array_equal(h, ref_h)
    assert np.array_equal(ids, ref_ids)
    assert np.array_equal(gates, ref_gates)

    # Fast rejection is merely a fallback request, but the exact kernel must
    # still publish the established error when the gain product is truly bad.
    bad_gain_h = qk_cuda.register_i64(np.full(width, np.iinfo(np.int64).max, dtype=np.int64))
    zero_router_h = qk_cuda.register_i64(np.zeros((experts, width), dtype=np.int64))
    assert qk_cuda.rmsnorm_router(
        bad_gain_h, zero_router_h, np.ones((1, width), dtype=np.int64), used, fa, fw, eps=1,
    ) is None
    qk_cuda.free_all()


def test_cuda_grouped_resident_matches_separate_and_profiles_boundaries():
    """Same-input Q/K/V grouping is exact and performs one activation upload/result download."""
    if not qk_cuda.grouped_available():
        pytest.skip("grouped resident apply not in the .so")
    qk_cuda.free_all()
    rng = np.random.default_rng(701)
    nb, T = 3, 4
    specs = ((7, qk_cuda.Q4_K, 100), (11, qk_cuda.Q6_K, 200), (5, qk_cuda.Q4_K, 300))
    handles, out_features, refs = [], [], []
    x = rng.integers(-(1 << 18), 1 << 18, size=(T, nb * 256), dtype=np.int64)
    for out_f, qtype, seed in specs:
        gen, pack = ((qk.random_q4k, _q4k_raw) if qtype == qk_cuda.Q4_K else (qk.random_q6k, _q6k_raw))
        raw = b"".join(pack(gen(seed + i)) for i in range(out_f * nb))
        handles.append(qk_cuda.register_weight(raw, out_f, nb, qtype))
        out_features.append(out_f)
        refs.append(qk_cuda.apply_resident(handles[-1], x, out_f, FW))

    assert qk_cuda.profile_reset(enabled=True)
    got = qk_cuda.apply_resident_grouped(
        handles, x, out_features, FW,
        (qk_cuda.PROFILE_Q, qk_cuda.PROFILE_K, qk_cuda.PROFILE_V),
    )
    assert got is not None and all(np.array_equal(a, b) for a, b in zip(got, refs))
    stats = qk_cuda.profile_snapshot()
    assert stats["grouped_apply_calls"] == 1
    assert stats["h2d_calls"] == stats["d2h_calls"] == 1
    assert stats["q_projection_calls"] == stats["k_projection_calls"] == stats["v_projection_calls"] == 1
    assert stats["h2d_bytes"] == x.nbytes
    assert stats["d2h_bytes"] == sum(a.nbytes for a in refs)
    # A deliberately undersized caller declaration is rejected before native D2H writes the host allocation.
    assert qk_cuda.apply_resident_grouped(handles, x, [1, 1, 1], FW) is None
    qk_cuda.profile_set_enabled(False)
    qk_cuda.free_all()


@pytest.mark.parametrize(
    ("d_model", "heads"),
    ((256, 8), (512, 4)),
    ids=("query-wider-than-hidden", "hidden-wider-than-query"),
)
def test_cuda_resident_attention_decode_matches_host_cache_exact(d_model, heads):
    """Device Q/K/V->RoPE->KV append->attention->O equals the established host orchestration byte-for-byte."""
    if not qk_cuda.resident_attention_available():
        pytest.skip("resident attention context not in the .so")
    from nmc import cohere2 as c2

    qk_cuda.free_all()
    rng = np.random.default_rng(1701)
    # The shipped model has a wider Q projection than its residual stream
    # (4096 versus 2048). Keep that architectural distinction in this oracle
    # test so the resident bank cannot regress to conflating the two widths.
    fa, fw, n_kv, head_dim, max_length = 16, 24, 2, 64, 8
    cfg = c2.Cfg(d_model=d_model, n_heads=heads, n_kv=n_kv, head_dim=head_dim,
                 ffn=512, vocab=512, sliding_window=3, fa=fa, fw=fw)
    nb = d_model // 256
    q_width, q_nb = heads * head_dim, heads * head_dim // 256

    def make_weight(out_f, qtype, seed, in_blocks=nb):
        gen, pack = ((qk.random_q4k, _q4k_raw) if qtype == qk_cuda.Q4_K else (qk.random_q6k, _q6k_raw))
        raw = b"".join(pack(gen(seed + i)) for i in range(out_f * in_blocks))
        return raw, qk_cuda.register_weight(raw, out_f, in_blocks, qtype)

    qraw, qh = make_weight(q_width, qk_cuda.Q4_K, 20000)
    kraw, kh = make_weight(n_kv * head_dim, qk_cuda.Q6_K, 21000)
    vraw, vh = make_weight(n_kv * head_dim, qk_cuda.Q4_K, 22000)
    oraw, oh = make_weight(d_model, qk_cuda.Q6_K, 23000, q_nb)
    cos, sin = c2.build_rope_tables(max_length, head_dim, base=50000, frac_bits=fa)
    bank = qk_cuda.ResidentAttentionCache(
        1, max_length, d_model, heads, n_kv, head_dim, fa, cos, sin,
    )
    start = 3
    ck = rng.integers(-(1 << 17), 1 << 17, size=(n_kv, start, head_dim), dtype=np.int64)
    cv = rng.integers(-(1 << 17), 1 << 17, size=(n_kv, start, head_dim), dtype=np.int64)
    bank.import_layer(0, ck, cv)

    # Both sides of the split are part of the ABI contract: Q consumes the
    # hidden width but emits q_width, while O consumes q_width and emits the
    # hidden width. Reject stale v4-style declarations before mutating K/V.
    _wrong_q_raw, wrong_qh = make_weight(d_model, qk_cuda.Q4_K, 24000)
    _wrong_o_raw, wrong_oh = make_weight(d_model, qk_cuda.Q6_K, 25000, nb)
    _wrong_k_raw, wrong_kh = make_weight(n_kv * head_dim, qk_cuda.Q6_K, 26000, q_nb)
    _wrong_v_raw, wrong_vh = make_weight(n_kv * head_dim, qk_cuda.Q4_K, 27000, q_nb)
    probe = np.zeros((1, d_model), dtype=np.int64)
    with pytest.raises(ValueError, match="hidden shape"):
        bank.apply(0, qh, kh, vh, oh, np.zeros((1, q_width), dtype=np.int64), fw, None, False)
    with pytest.raises(qk_cuda.CudaContextError, match="native status"):
        bank.apply(0, wrong_qh, kh, vh, oh, probe, fw, None, False)
    with pytest.raises(qk_cuda.CudaContextError, match="native status"):
        bank.apply(0, qh, wrong_kh, vh, oh, probe, fw, None, False)
    with pytest.raises(qk_cuda.CudaContextError, match="native status"):
        bank.apply(0, qh, kh, wrong_vh, oh, probe, fw, None, False)
    with pytest.raises(qk_cuda.CudaContextError, match="native status"):
        bank.apply(0, qh, kh, vh, wrong_oh, probe, fw, None, False)
    assert bank.length(0) == start

    def linear(raw, qtype, x, out_f, in_blocks=nb):
        return qk_cuda.qk_linear(raw, x, out_f, in_blocks, fw, qtype)

    for rope, window in ((True, 3), (False, None)):
        x = rng.integers(-(1 << 17), 1 << 17, size=(1, d_model), dtype=np.int64)
        q = linear(qraw, qk_cuda.Q4_K, x, q_width).reshape(1, heads, head_dim)
        k = linear(kraw, qk_cuda.Q6_K, x, n_kv * head_dim).reshape(1, n_kv, head_dim)
        v = linear(vraw, qk_cuda.Q4_K, x, n_kv * head_dim).reshape(1, n_kv, head_dim)
        if rope:
            q = c2._rope_int(q, cos, sin, fa, start)
            k = c2._rope_int(k, cos, sin, fa, start)
        ck = np.concatenate((ck, np.transpose(k, (1, 0, 2))), axis=1)
        cv = np.concatenate((cv, np.transpose(v, (1, 0, 2))), axis=1)
        attended = c2.attention_cached(q, ck, cv, start, cfg, window)
        ref = linear(oraw, qk_cuda.Q6_K, attended, d_model, q_nb)
        got = bank.apply(0, qh, kh, vh, oh, x, fw, window, rope)
        assert np.array_equal(got, ref)
        start += 1
        assert bank.length(0) == start
    bank.reset()
    assert bank.length(0) == 0
    bank.close()
    bank.close()                                           # idempotent request cleanup

    # Match the host oracle's conservative probability×V envelope: the
    # resident context must fail loudly rather than returning an int64 wrap.
    bad = qk_cuda.ResidentAttentionCache(
        1, max_length, d_model, heads, n_kv, head_dim, fa, cos, sin,
    )
    bad_k = np.zeros((n_kv, 1, head_dim), dtype=np.int64)
    bad_v = np.full((n_kv, 1, head_dim), np.iinfo(np.int64).max, dtype=np.int64)
    bad.import_layer(0, bad_k, bad_v)
    x = np.zeros((1, d_model), dtype=np.int64)
    with pytest.raises(qk_cuda.CudaContextError, match="overflow guard"):
        bad.apply(0, qh, kh, vh, oh, x, fw, None, False)
    assert bad.length(0) == 1                         # failed append is never published to Python
    bad.close()
    qk_cuda.free_all()


def test_cuda_resident_moe_layer_cold_continuation_retains_exact_residual():
    """Only cold IDs cross out; normalized h/routes/MoE/residual chain stay in the request bank."""
    if not qk_cuda.resident_layer_available():
        pytest.skip("resident MoE layer continuation not in the .so")
    from nmc import cohere2 as c2

    qk_cuda.free_all()
    rng = np.random.default_rng(8142)
    fa, fw = 16, 24
    layers, d_model, heads, n_kv, head_dim = 2, 256, 8, 2, 64
    experts, used, expert_ffn, max_length = 4, 2, 256, 4
    nb = d_model // 256
    q_width, q_nb = heads * head_dim, heads * head_dim // 256
    cfg = c2.Cfg(d_model=d_model, n_heads=heads, n_kv=n_kv, head_dim=head_dim,
                 ffn=512, vocab=512, sliding_window=3, fa=fa, fw=fw)

    def make_raw(out_f, qtype, seed, in_blocks=nb):
        gen, pack = ((qk.random_q4k, _q4k_raw) if qtype == qk_cuda.Q4_K else (qk.random_q6k, _q6k_raw))
        return b"".join(pack(gen(seed + i)) for i in range(out_f * in_blocks))

    qraw = make_raw(q_width, qk_cuda.Q4_K, 31000)
    kraw = make_raw(n_kv * head_dim, qk_cuda.Q6_K, 32000)
    vraw = make_raw(n_kv * head_dim, qk_cuda.Q4_K, 33000)
    oraw = make_raw(d_model, qk_cuda.Q6_K, 34000, q_nb)
    qh = qk_cuda.register_weight(qraw, q_width, nb, qk_cuda.Q4_K)
    kh = qk_cuda.register_weight(kraw, n_kv * head_dim, nb, qk_cuda.Q6_K)
    vh = qk_cuda.register_weight(vraw, n_kv * head_dim, nb, qk_cuda.Q4_K)
    oh = qk_cuda.register_weight(oraw, d_model, q_nb, qk_cuda.Q6_K)
    gain = rng.integers(1 << 15, 1 << 17, size=d_model, dtype=np.int64)
    router = rng.integers(-(1 << 13), 1 << 13, size=(experts, d_model), dtype=np.int64)
    gain_h, router_h = qk_cuda.register_i64(gain), qk_cuda.register_i64(router)
    gate_raw = [make_raw(expert_ffn, qk_cuda.Q4_K, 36000 + 1000 * e) for e in range(experts)]
    up_raw = [make_raw(expert_ffn, qk_cuda.Q4_K, 40000 + 1000 * e) for e in range(experts)]
    down_raw = [make_raw(d_model, qk_cuda.Q6_K, 44000 + 1000 * e, expert_ffn // 256)
                for e in range(experts)]
    expert_handles = {}

    def register_selected(expert):
        expert = int(expert)
        if expert not in expert_handles:
            expert_handles[expert] = (
                qk_cuda.register_weight(gate_raw[expert], expert_ffn, nb, qk_cuda.Q4_K),
                qk_cuda.register_weight(up_raw[expert], expert_ffn, nb, qk_cuda.Q4_K),
                qk_cuda.register_weight(down_raw[expert], d_model, expert_ffn // 256, qk_cuda.Q6_K),
            )
        return expert_handles[expert]

    def reference_layer(x, router_weights=router):
        h = c2.fixed_point_rmsnorm(x, fa, 1, gain_q=gain)
        logits = c2.linear(h, router_weights, fw)
        ids = c2._topk_lowidx(logits[0], used)
        gates = c2.fixed_point_sigmoid(logits[0][ids].astype(np.int64), fa)
        q = qk_cuda.qk_linear(qraw, h, q_width, nb, fw, qk_cuda.Q4_K).reshape(1, heads, head_dim)
        k = qk_cuda.qk_linear(kraw, h, n_kv * head_dim, nb, fw, qk_cuda.Q6_K).reshape(1, n_kv, head_dim)
        v = qk_cuda.qk_linear(vraw, h, n_kv * head_dim, nb, fw, qk_cuda.Q4_K).reshape(1, n_kv, head_dim)
        attended = c2.attention_cached(q, np.transpose(k, (1, 0, 2)),
                                       np.transpose(v, (1, 0, 2)), 0, cfg, None)
        attention = qk_cuda.qk_linear(oraw, attended, d_model, q_nb, fw, qk_cuda.Q6_K)
        moe = np.zeros(d_model, dtype=object)
        for rank, expert in enumerate(ids):
            gate = qk_cuda.qk_linear(gate_raw[expert], h, expert_ffn, nb, fw, qk_cuda.Q4_K)
            up = qk_cuda.qk_linear(up_raw[expert], h, expert_ffn, nb, fw, qk_cuda.Q4_K)
            gu = ((c2.silu_int(gate, fa).astype(object) * up.astype(object)) >> fa).astype(np.int64)
            down = qk_cuda.qk_linear(down_raw[expert], gu, d_model, expert_ffn // 256,
                                     fw, qk_cuda.Q6_K)[0]
            moe += (down.astype(object) * int(gates[rank])) >> fa
        result = np.asarray(x, np.int64) + attention + moe.astype(np.int64)[None]
        return ids, result

    cos, sin = c2.build_rope_tables(max_length, head_dim, base=50000, frac_bits=fa)
    bank = qk_cuda.ResidentAttentionCache(
        layers, max_length, d_model, heads, n_kv, head_dim, fa, cos, sin,
    )
    for layer in range(layers):
        bank.configure_moe_layer(layer, experts, used, d_model, expert_ffn)

    x0 = rng.integers(-(1 << 17), 1 << 17, size=(1, d_model), dtype=np.int64)
    ids0, ref0 = reference_layer(x0)
    warm0 = int(ids0[0])
    bank.bind_moe_expert(0, warm0, *register_selected(warm0))
    # Projection handles are preflighted before normalization/router kernels
    # are queued. A rejected begin must leave both the logical cache length and
    # pending-layer state untouched so the same layer can start successfully.
    with pytest.raises(qk_cuda.CudaContextError, match="native status 1"):
        bank.begin_moe_layer(0, gain_h, router_h, gain_h, kh, vh, oh, x0, fw, 1, None, False)
    assert bank.length(0) == 0
    resident_before_begin = qk_cuda.resident_count()
    assert qk_cuda.profile_reset(enabled=True)
    cold0 = bank.begin_moe_layer(0, gain_h, router_h, qh, kh, vh, oh, x0, fw, 1, None, False)
    assert cold0 == (int(ids0[1]),)
    assert qk_cuda.resident_count() == resident_before_begin       # discovery itself uploads no expert
    begin_stats = qk_cuda.profile_snapshot()
    assert begin_stats["d2h_bytes"] < x0.nbytes                   # no normalized h/routes/attention row D2H
    with pytest.raises(qk_cuda.CudaContextError, match="cold experts remain"):
        bank.continue_moe_layer(0, fw)
    for expert in cold0:
        bank.bind_moe_expert(0, expert, *register_selected(expert))
    retained = bank.continue_moe_layer(0, fw, publish=False)
    assert isinstance(retained, qk_cuda.ResidentLayerHidden)

    ids1, ref1 = reference_layer(ref0)
    # Bind every already-resident selected expert before routing. Only a truly
    # missing slice is exposed by begin; unselected expert slices stay absent.
    for expert in ids1:
        if int(expert) in expert_handles:
            bank.bind_moe_expert(1, int(expert), *expert_handles[int(expert)])
    cold1 = bank.begin_moe_layer(1, gain_h, router_h, qh, kh, vh, oh,
                                 retained, fw, 1, None, False)
    assert cold1 == tuple(int(expert) for expert in ids1 if int(expert) not in expert_handles)
    for expert in cold1:
        bank.bind_moe_expert(1, expert, *register_selected(expert))
    got = bank.continue_moe_layer(1, fw, publish=True)
    assert np.array_equal(got, ref1)
    assert np.array_equal(bank.export_moe_hidden(), ref1)
    assert set(expert_handles).issubset(set(map(int, ids0)) | set(map(int, ids1)))
    assert len(expert_handles) < experts or len(set(map(int, ids0)) | set(map(int, ids1))) == experts
    qk_cuda.profile_set_enabled(False)
    bank.close()

    # Exercise the reusable orchestrator across three consecutive layers. A
    # tied zero router makes the selected IDs permanently {0,1}, so the second
    # token is a true warm-shape allocator gate rather than a route-coverage
    # accident. The test still registers/binds only selected expert slices.
    chain_layers = 3
    zero_router = np.zeros_like(router)
    zero_router_h = qk_cuda.register_i64(zero_router)
    chain = qk_cuda.ResidentAttentionCache(
        chain_layers, max_length, d_model, heads, n_kv, head_dim, fa, cos, sin,
    )
    expert_handles.clear()
    loads = []

    def lookup_expert(_layer, expert):
        return expert_handles.get(int(expert))

    def load_expert(layer, expert):
        loads.append((int(layer), int(expert)))
        return register_selected(expert)

    executor = qk_cuda.ResidentMoeTokenExecutor(
        chain, first_layer=0, layer_count=chain_layers,
        n_experts=experts, n_used=used, d_model=d_model, expert_ffn=expert_ffn,
        fw=fw, eps=1, lookup_expert=lookup_expert, load_expert=load_expert,
    )
    specs = tuple(qk_cuda.ResidentMoeLayerSpec(
        layer, gain_h, zero_router_h, qh, kh, vh, oh, None, False,
    ) for layer in range(chain_layers))
    request_bytes_before = chain.workspace_bytes()
    assert qk_cuda.profile_reset(enabled=True)
    chain_input = rng.integers(-(1 << 17), 1 << 17, size=(1, d_model), dtype=np.int64)
    got_chain = executor.run(chain_input, specs)
    ref_chain = chain_input
    for _layer in range(chain_layers):
        selected, ref_chain = reference_layer(ref_chain, zero_router)
        assert selected.tolist() == [0, 1]
    assert np.array_equal(got_chain, ref_chain)
    first_stats = qk_cuda.profile_snapshot()
    request_bytes_warm = chain.workspace_bytes()
    assert request_bytes_warm > request_bytes_before
    assert first_stats["allocation_calls"] > 0
    assert set(expert_handles) == {0, 1}
    assert loads == [(layer, expert) for layer in range(chain_layers) for expert in (0, 1)]

    executor.run(chain_input, specs)
    second_stats = qk_cuda.profile_snapshot()
    assert second_stats["allocation_calls"] == first_stats["allocation_calls"]
    # Route/cold discovery shares the attention guard synchronization, and
    # continuation reuses begin's host-tracked unresolved set. Each warm layer
    # therefore publishes only the combined attention guard, cold count, and
    # MoE guard; the final layer adds one residual-row D2H. The older two-stage
    # recheck used seven D2H calls and five synchronizations per layer.
    assert second_stats["d2h_calls"] - first_stats["d2h_calls"] == chain_layers * 3 + 1
    assert chain.workspace_bytes() == request_bytes_warm
    qk_cuda.profile_set_enabled(False)
    chain.close()

    # A preprocessing overflow is inherited by the combined attention guard.
    # It cannot publish a cache append and poisons the native request until it
    # is explicitly discarded/reset, preventing reuse of partially computed
    # resident state.
    overflow = qk_cuda.ResidentAttentionCache(
        1, max_length, d_model, heads, n_kv, head_dim, fa, cos, sin,
    )
    overflow.configure_moe_layer(0, experts, used, d_model, expert_ffn)
    unsafe = np.zeros((1, d_model), dtype=np.int64)
    unsafe[0, :2] = np.iinfo(np.int64).min
    with pytest.raises(qk_cuda.CudaContextError, match="overflow guard"):
        overflow.begin_moe_layer(
            0, gain_h, router_h, qh, kh, vh, oh, unsafe, fw, 1, None, False,
        )
    assert overflow.length(0) == 0
    with pytest.raises(qk_cuda.CudaContextError, match="native status 1"):
        overflow.begin_moe_layer(
            0, gain_h, router_h, qh, kh, vh, oh, x0, fw, 1, None, False,
        )
    overflow.close()
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
    workspace_grows = qk_cuda.moe_workspace_allocations()
    got_reused = qk_cuda.moe_ffn(gh, uh, dh, h, gates, d_model, e_ffn, fa, fw)
    assert np.array_equal(got_reused, got)
    if workspace_grows is not None:
        assert workspace_grows > 0
        assert qk_cuda.moe_workspace_allocations() == workspace_grows  # same shape performs no cudaMalloc

    # ctypes releases the GIL.  The Python bridge must serialize calls that
    # share the process-global CUDA workspace; every queued result remains
    # byte-identical under real concurrent callers.
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(
            lambda _: qk_cuda.moe_ffn(gh, uh, dh, h, gates, d_model, e_ffn, fa, fw),
            range(8),
        ))
    assert all(np.array_equal(result, got) for result in concurrent)

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
    workspace_grows = qk_cuda.moe_workspace_allocations()
    got_reused = qk_cuda.moe_ffn_batched(gh, uh, dh, m, k, h, gflat, d_model, e_ffn, fa, fw)
    assert np.array_equal(got_reused, got)
    if workspace_grows is not None:
        assert qk_cuda.moe_workspace_allocations() == workspace_grows  # same shape performs no cudaMalloc
    retained_bytes = qk_cuda.moe_workspace_bytes()
    resident_weights = qk_cuda.resident_count()
    if retained_bytes is not None:
        assert retained_bytes > 0
        assert qk_cuda.release_moe_workspace()
        assert qk_cuda.moe_workspace_bytes() == 0
        assert qk_cuda.resident_count() == resident_weights  # scratch release must not invalidate weights
    ref = np.empty((m, d_model), np.int64)
    for t in range(m):
        ref[t] = qk_cuda.moe_ffn([gate[e] for e in sel[t]], [up[e] for e in sel[t]], [down[e] for e in sel[t]],
                                 h[t], gts[t], d_model, e_ffn, fa, fw)
    assert got is not None and np.array_equal(got, ref)
    qk_cuda.free_all()


def test_cuda_moe_ffn_batched_dp4a_matches_exact_and_guard_fallback():
    """Batched prefill DP4A is byte-exact; an out-of-envelope input takes the int128 ABI instead."""
    if not qk_cuda.moe_ffn_batched_dp4a_available():
        pytest.skip("batched MoE DP4A not in the .so")
    qk_cuda.free_all()
    rng = np.random.default_rng(37)
    n_pool, m, k, d_model, e_ffn, fa, fw = 4, 3, 2, 512, 256, 16, 24
    nb_in, nb_dn = d_model // 256, e_ffn // 256
    mk4 = lambda of, nb, base: b"".join(_q4k_raw(qk.random_q4k(base + i)) for i in range(of * nb))
    mk6 = lambda of, nb, base: b"".join(_q6k_raw(qk.random_q6k(base + i)) for i in range(of * nb))
    gate = [qk_cuda.register_weight(mk4(e_ffn, nb_in, 11000 + 50 * e), e_ffn, nb_in, qk_cuda.Q4_K)
            for e in range(n_pool)]
    up = [qk_cuda.register_weight(mk4(e_ffn, nb_in, 14000 + 50 * e), e_ffn, nb_in, qk_cuda.Q4_K)
          for e in range(n_pool)]
    down = [qk_cuda.register_weight(mk6(d_model, nb_dn, 18000 + 50 * e), d_model, nb_dn, qk_cuda.Q6_K)
            for e in range(n_pool)]
    sel = [rng.choice(n_pool, k, replace=False) for _ in range(m)]
    gh = [gate[e] for t in range(m) for e in sel[t]]
    uh = [up[e] for t in range(m) for e in sel[t]]
    dh = [down[e] for t in range(m) for e in sel[t]]
    gates = rng.integers(0, 1 << fa, size=m * k, dtype=np.int64)
    h = rng.integers(-(1 << 18), 1 << 18, size=(m, d_model), dtype=np.int64)

    ref = qk_cuda.moe_ffn_batched(gh, uh, dh, m, k, h, gates, d_model, e_ffn, fa, fw)
    got = qk_cuda.moe_ffn_batched(gh, uh, dh, m, k, h, gates, d_model, e_ffn, fa, fw, dp4a=True)
    assert got is not None and np.array_equal(got, ref)

    outside = h.copy()
    outside[0, 0] = qk_cuda._balanced_capacity(qk_cuda._DP4A_SAFE_LIMBS) + 1
    assert qk_cuda.profile_reset(enabled=True)
    ref_outside = qk_cuda.moe_ffn_batched(gh, uh, dh, m, k, outside, gates, d_model, e_ffn, fa, fw)
    got_outside = qk_cuda.moe_ffn_batched(
        gh, uh, dh, m, k, outside, gates, d_model, e_ffn, fa, fw, dp4a=True,
    )
    assert np.array_equal(got_outside, ref_outside)
    stats = qk_cuda.profile_snapshot()
    assert stats["moe_batched_dp4a_calls"] == 0     # Python guard rejected it before native dispatch
    assert stats["moe_batched_calls"] == 2
    qk_cuda.profile_set_enabled(False)
    qk_cuda.free_all()


def test_cuda_registry_metadata_grows_beyond_old_limit():
    """The host registry crosses the removed 16,384-entry ceiling without allocating one VRAM block per slot."""
    if not qk_cuda.resident_available():
        pytest.skip("resident API not in the .so")
    qk_cuda.free_all()
    assert qk_cuda.reserve_resident_capacity(16_385)
    assert qk_cuda.resident_capacity() >= 16_385
    assert qk_cuda.resident_count() == 0

    raw = _q4k_raw(qk.random_q4k(991))
    handle = qk_cuda.register_weight(raw, 1, 1, qk_cuda.Q4_K)
    x = np.arange(256, dtype=np.int64)
    assert qk_cuda.apply_resident(handle, x, 1, FW) is not None
    qk_cuda.free_all()


def test_cuda_free_all_then_reregister():
    """free_all() resets the registry; re-registering the same weight yields the same byte-exact result —
    the mechanism engine.free() relies on (it clears its handle cache so reuse re-registers, not stale-handle)."""
    if not qk_cuda.resident_available():
        pytest.skip("resident API not in the .so")
    raw = b"".join(_q6k_raw(qk.random_q6k(s)) for s in range(8 * 4))
    x = np.random.default_rng(0).integers(-(1 << 16), 1 << 16, size=(1, 4 * 256), dtype=np.int64)
    h1 = qk_cuda.register_weight(raw, 8, 4, qk_cuda.Q6_K)
    assert qk_cuda.resident_capacity() >= qk_cuda.resident_count() >= 1
    r1 = qk_cuda.apply_resident(h1, x, 8, FW)
    qk_cuda.free_all()
    assert qk_cuda.resident_capacity() == 0
    h2 = qk_cuda.register_weight(raw, 8, 4, qk_cuda.Q6_K)     # fresh handle after free
    r2 = qk_cuda.apply_resident(h2, x, 8, FW)
    assert np.array_equal(r1, r2)
    qk_cuda.free_all()

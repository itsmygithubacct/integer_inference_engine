"""Regression (review-3 MEDIUM): the nmc int64 decode hot paths must FAIL LOUD if an activation leaves the
fixed-point envelope, instead of silently wrapping (which would diverge from the big-int oracle
deterministically — the receipt-lethal failure). Guards are no-ops in-envelope (covered by the decode/attn
suites); here we prove they trip out-of-envelope."""
import numpy as np
import pytest

from nmc import cohere2 as c2


def test_contraction_guard_trips_out_of_envelope():
    # in-envelope: no raise
    c2._assert_i64_contraction(1 << 20, 1 << 20, 128, "ok")            # 2^40 * 128 = 2^47, fine
    # out-of-envelope: raise before a silent wrap
    with pytest.raises(OverflowError):
        c2._assert_i64_contraction(1 << 40, 1 << 40, 128, "boom")      # 2^80+ > 2^63


def test_absmax_handles_int64_min():
    a = np.array([np.iinfo(np.int64).min, 5], dtype=np.int64)          # np.abs would wrap; helper must not
    assert c2._absmax_int(a) == abs(int(np.iinfo(np.int64).min))


def test_require_weight_bytes_rejects_short_buffer():
    """review-3 MEDIUM: the ctypes bridges must reject a weight buffer shorter than out_f*n_blocks*block_bytes
    (Q4_K=144, Q6_K=210) instead of letting the C/CUDA kernel read out of bounds on a truncated GGUF."""
    from nmc.qk_native import _require_weight_bytes, Q4_K, Q6_K
    from nmc import qk_cuda  # must import the same helper without error
    assert qk_cuda._require_weight_bytes is _require_weight_bytes
    assert _require_weight_bytes(b"\x00" * (2 * 3 * 144), 2, 3, Q4_K) == 2 * 3 * 144
    assert _require_weight_bytes(b"\x00" * (2 * 3 * 210), 2, 3, Q6_K) == 2 * 3 * 210
    with pytest.raises(ValueError):
        _require_weight_bytes(b"\x00" * 100, 2, 3, Q4_K)          # 100 < 864
    with pytest.raises(ValueError):
        _require_weight_bytes(b"\x00" * 999, 2, 3, 7)             # unknown qtype
    for out_f, n_blocks in ((0, 3), (2, 0), (-1, 3)):
        with pytest.raises(ValueError, match="must be positive"):
            _require_weight_bytes(b"", out_f, n_blocks, Q4_K)


def test_attention_cached_raises_on_extreme_activations():
    cfg = c2.Cfg(d_model=16, n_heads=2, n_kv=1, head_dim=8, ffn=32, vocab=32)
    big = 1 << 40
    q = np.full((1, cfg.n_heads, cfg.head_dim), big, dtype=np.int64)
    ck = np.full((cfg.n_kv, 1, cfg.head_dim), big, dtype=np.int64)
    cv = np.zeros((cfg.n_kv, 1, cfg.head_dim), dtype=np.int64)
    with pytest.raises(OverflowError):
        c2.attention_cached(q, ck, cv, 0, cfg, None)


def test_attention_probability_mass_bound_accepts_safe_long_context_and_rejects_real_overflow():
    cfg = c2.Cfg(d_model=2, n_heads=1, n_kv=1, head_dim=2, ffn=4, vocab=8)
    q = np.zeros((1, 1, 2), dtype=np.int64)
    ck = np.zeros((1, 4, 2), dtype=np.int64)
    # Equal scores spread a single unit of fixed-point probability mass over four values. The old bound
    # multiplied by Lc a second time and rejected this safe 2^61 accumulation.
    safe_v = np.full((1, 4, 2), 1 << 45, dtype=np.int64)
    out = c2.attention_cached(q, ck, safe_v, 3, cfg, None)
    assert out.shape == (1, 2)
    # Here 2^fa * max|V| really exceeds int64, so the matmul must fail before NumPy can wrap.
    unsafe_v = np.full((1, 4, 2), 1 << 48, dtype=np.int64)
    with pytest.raises(OverflowError, match="probs·v"):
        c2.attention_cached(q, ck, unsafe_v, 3, cfg, None)


def test_engine_context_guard_covers_every_rope_row_the_model_hash_commits():
    from nmc.engine import Engine
    eng = object.__new__(Engine)
    eng.context_length = 4
    assert eng._require_context(4) == 4
    with pytest.raises(ValueError, match="at least one"):
        eng._require_context(0)
    with pytest.raises(ValueError, match="exceeds the committed model context"):
        eng._require_context(5)

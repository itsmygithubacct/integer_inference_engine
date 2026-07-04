"""Stage-4 gates: interleaved sliding-window / full attention (the cohere2 per-layer pattern)."""
import numpy as np
import pytest

from nmc import cohere2 as c2

CFG = c2.Cfg(d_model=256, n_heads=4, n_kv=2, head_dim=64, ffn=512, vocab=512, sliding_window=3)


def _qkv(seq, seed=0):
    Wf = c2.random_weights_float(CFG, seed); Wq = c2.weights_to_fixed(Wf, CFG)
    x = np.random.default_rng(seed + 1).standard_normal((seq, CFG.d_model)) * 0.6
    cos, sin = c2.build_rope_tables(seq, CFG.head_dim, base=int(CFG.rope_base), frac_bits=CFG.fa)
    h = c2.fixed_point_rmsnorm(c2.to_fixed(x, CFG.fa), CFG.fa, CFG.eps, gain_q=Wq["attn_norm"])
    q = c2.linear(h, Wq["wq"], CFG.fw); k = c2.linear(h, Wq["wk"], CFG.fw); v = c2.linear(h, Wq["wv"], CFG.fw)
    return q, k, v, cos, sin, Wf, Wq, x


def test_layer_pattern_matches_gguf():
    """is_full_layer reproduces the 49-entry GGUF pattern (0=full): full at 0,4,…,48 → 13 full / 36 sliding."""
    assert len(c2.NORTH_SWA_PATTERN) == 49
    for idx, flag in enumerate(c2.NORTH_SWA_PATTERN):
        assert c2.is_full_layer(idx) == (flag == 0)
    full = sum(c2.is_full_layer(i) for i in range(49))
    assert full == 13 and 49 - full == 36
    assert c2.window_for_layer(CFG, 0) is None and c2.window_for_layer(CFG, 1) == CFG.sliding_window


def test_attn_mask_semantics():
    """Sliding window w: query i attends exactly to keys (i-w, i]; full causal attends to [0, i]."""
    full = ~c2._attn_mask(5, None)
    assert np.array_equal(full, np.tril(np.ones((5, 5), bool)))
    sw = ~c2._attn_mask(5, 2)                     # window=2 -> {i-1, i}
    assert np.array_equal(sw[2], [False, True, True, False, False])
    assert np.array_equal(sw[4], [False, False, False, True, True])
    assert np.array_equal(sw[0], [True, False, False, False, False])


def test_swa_equals_full_when_window_covers_seq():
    """seq <= window ⇒ no key is 'too old' ⇒ SWA is byte-identical to full causal."""
    q, k, v, cos, sin, *_ = _qkv(seq=6)
    full = c2.attention_int(q, k, v, CFG, cos, sin, window=None)
    swa = c2.attention_int(q, k, v, CFG, cos, sin, window=64)     # 64 >= seq
    assert np.array_equal(full, swa)


def test_swa_actually_changes_result():
    """With window < seq the mask must change the output (guards against a no-op mask)."""
    q, k, v, cos, sin, *_ = _qkv(seq=8)
    full = c2.attention_int(q, k, v, CFG, cos, sin, window=None)
    swa = c2.attention_int(q, k, v, CFG, cos, sin, window=3)
    assert not np.array_equal(full, swa)


@pytest.mark.parametrize("seed", range(6))
def test_swa_fidelity_int_vs_float(seed):
    """Integer sliding-window attention ≈ the float reference (same window)."""
    q, k, v, cos, sin, Wf, Wq, x = _qkv(seq=8, seed=seed)
    out_i = c2.from_fixed(c2.dense_block_int(c2.to_fixed(x, CFG.fa), Wq, CFG, cos, sin, window=3), CFG.fa)
    out_f = c2.dense_block_float(x, Wf, CFG, window=3)
    rel = np.max(np.abs(out_i - out_f)) / max(np.max(np.abs(out_f)), 1e-9)
    assert rel < 5e-3, rel


def test_swa_deterministic():
    q, k, v, cos, sin, *_ = _qkv(seq=8)
    a = c2.attention_int(q, k, v, CFG, cos, sin, window=3)
    b = c2.attention_int(q, k, v, CFG, cos, sin, window=3)
    assert np.array_equal(a, b) and a.dtype == np.int64

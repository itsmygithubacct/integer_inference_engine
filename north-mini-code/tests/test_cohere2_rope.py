"""Pin the cohere2 RoPE convention + NoPE — the two fidelity bugs found by the architecture audit.

cohere2/cohere2moe uses NORM/INTERLEAVED RoPE (rotate adjacent lane pairs 2i,2i+1), NOT NeoX (half-split
i,i+d/2), and applies RoPE ONLY on sliding-window + dense-prefix layers (full-attention layers are NoPE).
These regress silently if someone swaps the rope import back, so pin them against independent references."""
import numpy as np

from nmc import cohere2 as c2
from nmc._bonsai.rope_v2 import apply_rope_fixed_neox


def _float_rope(t, cos_f, sin_f, interleaved):
    """Float RoPE reference. interleaved -> pairs (2i,2i+1); else NeoX -> pairs (i, i+half)."""
    seq, n, hd = t.shape; half = hd // 2
    c = cos_f[:seq, None, :]; s = sin_f[:seq, None, :]
    if interleaved:
        x0, x1 = t[..., 0::2], t[..., 1::2]
        out = np.empty_like(t)
        out[..., 0::2] = x0 * c - x1 * s
        out[..., 1::2] = x0 * s + x1 * c
        return out
    x0, x1 = t[..., :half], t[..., half:]
    return np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)


def test_rope_is_interleaved_not_neox():
    fa, hd, seq, n = 16, 8, 6, 2
    rng = np.random.default_rng(0)
    t = rng.integers(-(1 << 16), 1 << 16, size=(seq, n, hd), dtype=np.int64)
    cos, sin = c2.build_rope_tables(seq, hd, base=50000, frac_bits=fa)
    got = c2._rope_int(t, cos, sin, fa)                              # the engine's RoPE (must be interleaved)
    neox = np.transpose(apply_rope_fixed_neox(np.transpose(t, (1, 0, 2)), cos, sin, fa), (1, 0, 2))

    assert np.array_equal(got[0], neox[0])                           # pos 0: no rotation -> conventions coincide
    assert not np.array_equal(got[1:], neox[1:])                     # pos>=1: interleaved != NeoX

    cos_f, sin_f = cos / (1 << fa), sin / (1 << fa)
    intl_f = _float_rope(t.astype(float), cos_f, sin_f, interleaved=True)
    neox_f = _float_rope(t.astype(float), cos_f, sin_f, interleaved=False)
    # integer RoPE matches the INTERLEAVED float reference (rounding), and is far from the NeoX one
    assert np.max(np.abs(got.astype(float) - intl_f)) < 4.0         # ~rounding in fixed-point units
    assert np.max(np.abs(got.astype(float) - neox_f)) > 1000.0      # clearly NOT NeoX


def test_nope_on_full_attention_layers():
    """Full-attention MoE layers apply NO RoPE (cohere2 NoPE) -> output invariant to the cos/sin tables;
    sliding-window layers DO apply RoPE -> output depends on them."""
    cfg = c2.Cfg(d_model=64, n_heads=4, n_kv=2, head_dim=16, ffn=128, vocab=64,
                 n_experts=4, n_used=2, expert_ffn=64, sliding_window=2)
    rng = np.random.default_rng(1)
    W = c2.weights_to_fixed(c2.random_weights_float(cfg, 1), cfg)
    We = c2.expert_weights_to_fixed(c2.random_expert_weights_float(cfg, 2), cfg)
    x = rng.integers(-(1 << 16), 1 << 16, size=(5, cfg.d_model), dtype=np.int64)
    cos, sin = c2.build_rope_tables(5, cfg.head_dim, base=int(cfg.rope_base), frac_bits=cfg.fa)
    zero = np.zeros_like(cos)

    full = c2.moe_block_int(x, W, We, cfg, cos, sin, window=None)
    full_z = c2.moe_block_int(x, W, We, cfg, zero, zero, window=None)
    assert np.array_equal(full, full_z)                              # NoPE: RoPE skipped, tables unused

    swa = c2.moe_block_int(x, W, We, cfg, cos, sin, window=2)
    swa_z = c2.moe_block_int(x, W, We, cfg, zero, zero, window=2)
    assert not np.array_equal(swa, swa_z)                            # SWA: RoPE applied, tables matter

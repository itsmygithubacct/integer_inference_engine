"""NeoX (rotate-half) RoPE for Atlas-v2 — the flagship BitNet b1.58 2B4T convention.

The flagship GGUF uses `rope type 2` = GGML_ROPE_TYPE_NEOX: each head vector of width `head_dim` is split
into two HALVES and rotated as pairs `(x[i], x[i+head_dim/2])`, unlike v1's interleaved-pairwise
`(x[2i], x[2i+1])` (`model/rope.py`). The rotation **frequencies are identical** —
`theta_i = pos * base^(-2i/head_dim)` for i in [0, head_dim/2) — so the committed fixed-point cos/sin
tables are exactly `build_rope_tables(...)`; ONLY the pairing of the apply changes. Re-exported here for
v2 so the reference engine and the torch graphs share one NeoX implementation, and so a from-scratch v2
train and an imported flagship (docs/ATLAS-V2.md "Notarizing the flagship") use the SAME rotation —
imported Q/K then rotate exactly as the flagship expects.

Clean-separated from `model/rope.py` (v1's interleaved path is untouched). The tables come from the
shared (version-agnostic) `build_rope_tables`; this module adds only the NeoX apply.
"""
from __future__ import annotations

import math

import numpy as np

from .rope import build_rope_tables, rope_table_hash, _assert_rope_no_overflow   # frequencies identical to NeoX; reused read-only

__all__ = ["build_rope_tables", "build_yarn_rope_tables", "rope_table_hash", "apply_rope_fixed_neox"]


def _yarn_corr_dim(n_dims: int, n_ctx_orig: int, n_rot: float, base: float) -> float:
    # Mirrors ggml_rope_yarn_corr_dim in llama.cpp.
    return n_dims * math.log(n_ctx_orig / (n_rot * 2 * math.pi)) / (2 * math.log(base))


def _yarn_corr_dims(n_dims: int, n_ctx_orig: int, base: float,
                    beta_fast: float, beta_slow: float) -> tuple[float, float]:
    start = math.floor(_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, base))
    end = math.ceil(_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, base))
    return max(0.0, float(start)), min(float(n_dims - 1), float(end))


def _yarn_ramp(low: float, high: float, i0: int) -> float:
    y = (i0 / 2 - low) / max(0.001, high - low)
    return 1.0 - min(1.0, max(0.0, y))


def build_yarn_rope_tables(seq_len: int, head_dim: int, *,
                           base: int | float = 10000,
                           freq_scale: float,
                           n_ctx_orig: int,
                           ext_factor: float = 1.0,
                           attn_factor: float = 1.0,
                           beta_fast: float = 32.0,
                           beta_slow: float = 1.0,
                           frac_bits: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Precompute llama.cpp/ggml YaRN RoPE tables for NeoX-style application.

    The implementation mirrors `ggml_rope_cache_init` + `rope_yarn` from llama.cpp: for each pair index
    it blends the interpolated angle (`freq_scale * theta`) with the extrapolated angle (`theta`) using
    the YaRN correction ramp, then applies the same magnitude scale to both cos and sin. The resulting
    tables are committed into the artifact, so inference still performs only integer multiply-adds.

    CROSS-MACHINE CAVEAT (same as `rope.py::build_rope_tables`): this build uses libm `math.log/cos/sin`,
    `math.pi`, and a float `base**(-2/head_dim)` power feeding `round()` to fixed-point, whose last-ULP
    results can differ across platforms; a borderline value could round to a different integer. This does
    NOT affect inference determinism — the tables are committed in the artifact and the reference path only
    reads them back — but it means RE-IMPORTING the GGUF on a different platform is not guaranteed to
    reproduce the exact `modelHash` bit-for-bit. The committed artifact is canonical; treat re-import as a
    fresh build to be re-validated, not a hash-stable regeneration.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE pairs"
    half = head_dim // 2
    scale = 1 << frac_bits
    corr = _yarn_corr_dims(head_dim, n_ctx_orig, float(base), beta_fast, beta_slow)
    theta_scale = float(base) ** (-2.0 / head_dim)
    yarn_mscale = 1.0 + 0.1 * math.log(1.0 / freq_scale) if ext_factor != 0.0 else 1.0
    cos = np.empty((seq_len, half), dtype=np.int64)
    sin = np.empty((seq_len, half), dtype=np.int64)
    for p in range(seq_len):
        theta_extrap = float(p)
        for j in range(half):
            i0 = 2 * j
            theta_interp = freq_scale * theta_extrap
            theta = theta_interp
            mscale = attn_factor
            if ext_factor != 0.0:
                ramp_mix = _yarn_ramp(corr[0], corr[1], i0) * ext_factor
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix
                mscale *= yarn_mscale
            cos[p, j] = round(math.cos(theta) * mscale * scale)
            sin[p, j] = round(math.sin(theta) * mscale * scale)
            theta_extrap *= theta_scale
    return cos, sin


def apply_rope_fixed_neox(x_fp: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                          frac_bits: int = 16) -> np.ndarray:
    """Rotate fixed-point activations x_fp (..., seq_len, head_dim) using the committed tables, NeoX style.

    Rotate-half convention: x is split into halves x0 = x[..., :half], x1 = x[..., half:], and the pair
    (x0[i], x1[i]) is rotated by the angle at (pos, i):

        out0 = (x0*cos - x1*sin) >> frac
        out1 = (x0*sin + x1*cos) >> frac

    Same committed cos/sin tables (shape (seq_len, head_dim//2)) as v1 — only the pairing differs from
    `apply_rope_fixed`. All integer multiply-add + one >> frac → deterministic, no contraction.
    """
    x = np.asarray(x_fp, dtype=np.int64)
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    half = head_dim // 2
    assert cos.shape[-2] >= seq_len and cos.shape[-1] == half
    x0 = x[..., :half]                      # first half  -> pairs with...
    x1 = x[..., half:]                      # ...second half
    c = cos[:seq_len, :]
    s = sin[:seq_len, :]
    _assert_rope_no_overflow(x, c, s)
    out = np.empty_like(x)
    out[..., :half] = (x0 * c - x1 * s) >> frac_bits
    out[..., half:] = (x0 * s + x1 * c) >> frac_bits
    return out

"""RoPE (rotary position embeddings) via COMMITTED fixed-point tables — deterministic positions.

RoPE rotates each (x[2i], x[2i+1]) pair by an angle theta(pos, i) = pos * base^(-2i/d). The angles'
cos/sin are transcendental, so computing them live on the reference path would be a determinism
hazard (libm differs across platforms). We remove the hazard the same way the design removes the
softmax hazard: **precompute cos/sin ONCE into fixed-point integer tables, commit them (their hash
is part of the model/env contract), and the reference path only does integer multiply-add against
the committed table.** No trig runs at inference time -> bit-exact across machines.

This is what makes the canonical-BitNet + RoPE choice (decision #arch) compatible with the
determinism keystone. The rotation itself is a fixed-point matmul-free multiply-add:

    x0' = (x0*cos - x1*sin) >> frac
    x1' = (x0*sin + x1*cos) >> frac      # x in fixed-point, cos/sin from the committed table
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

_INT64_MAX = (1 << 63) - 1


def _assert_rope_no_overflow(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> None:
    """Fail loud if the RoPE multiply-add would wrap int64 (consistent with fixed_point_matmul / the
    sampler guard). Each output term is `x0*c ± x1*s`, so |result| <= 2*max|x|*max|cos,sin| before the
    >>frac. RoPE has no native C peer, so a wrap here would be a SILENT (deterministic-but-wrong) fixed-point
    error baked into a committed logit. No-op for the committed model (worst term ~2^41, vs 2^63). Python
    ints on the extremes avoid np.abs(INT64_MIN) wrapping."""
    if x.size == 0 or cos.size == 0:
        return
    mx_x = max(abs(int(x.min())), abs(int(x.max())))
    mx_cs = max(abs(int(cos.min())), abs(int(cos.max())), abs(int(sin.min())), abs(int(sin.max())))
    if 2 * mx_x * mx_cs > _INT64_MAX:
        raise OverflowError(
            f"RoPE fixed-point apply overflows int64: 2*max|x|*max|cos,sin| = {2 * mx_x * mx_cs} > 2^63-1 "
            "— activation left the fixed-point envelope")


def build_rope_tables(seq_len: int, head_dim: int, base: int = 10000,
                      frac_bits: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Precompute fixed-point cos/sin tables, shape (seq_len, head_dim//2), int64 at scale 2**frac.

    Built ONCE with float trig, then frozen to fixed-point and committed. head_dim must be even.

    CROSS-MACHINE CAVEAT: this build uses libm `math.cos/sin`, whose last-ULP results can differ across
    platforms; the round() to fixed-point could therefore land on a different integer on a borderline
    value. This does NOT affect inference determinism — the tables are committed in the artifact and the
    reference path only reads them — but it means RE-IMPORTING the GGUF on a different platform is not
    guaranteed to reproduce the exact `modelHash` bit-for-bit. The committed artifact (shipped + hashed)
    is canonical; treat re-import as a fresh build to be re-validated, not a hash-stable regeneration.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE pairs"
    half = head_dim // 2
    scale = 1 << frac_bits
    inv_freq = [base ** (-(2 * i) / head_dim) for i in range(half)]  # base^(-2i/d)
    cos = np.empty((seq_len, half), dtype=np.int64)
    sin = np.empty((seq_len, half), dtype=np.int64)
    for p in range(seq_len):
        for i in range(half):
            ang = p * inv_freq[i]
            cos[p, i] = round(math.cos(ang) * scale)
            sin[p, i] = round(math.sin(ang) * scale)
    return cos, sin


def rope_table_hash(cos: np.ndarray, sin: np.ndarray) -> str:
    """SHA-256 over the canonical LITTLE-ENDIAN bytes of the committed tables (part of the model/env
    contract). `<i8` (not native `int64`) pins the hash across endianness — a no-op on little-endian
    hardware, but it keeps the committed identity machine-independent as the design claims."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(cos, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(sin, dtype="<i8").tobytes())
    return h.hexdigest()


def apply_rope_fixed(x_fp: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                     frac_bits: int = 16) -> np.ndarray:
    """Rotate fixed-point activations x_fp (..., seq_len, head_dim) using the committed tables.

    Pairwise convention: (x[...,2i], x[...,2i+1]) rotated by angle at (pos, i). All integer
    multiply-add + one >> frac -> deterministic, reduction-order trivial (no contraction).
    """
    x = np.asarray(x_fp, dtype=np.int64)
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    half = head_dim // 2
    assert cos.shape[-2] >= seq_len and cos.shape[-1] == half
    x0 = x[..., 0::2]                       # even indices -> first of each pair
    x1 = x[..., 1::2]                       # odd indices  -> second of each pair
    c = cos[:seq_len, :]
    s = sin[:seq_len, :]
    _assert_rope_no_overflow(x, c, s)
    out0 = (x0 * c - x1 * s) >> frac_bits
    out1 = (x0 * s + x1 * c) >> frac_bits
    out = np.empty_like(x)
    out[..., 0::2] = out0
    out[..., 1::2] = out1
    return out

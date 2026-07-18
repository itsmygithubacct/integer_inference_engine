"""Fixed-point RMSNorm and softmax — decision #8 (the bulletproof determinism bet).

LayerNorm/softmax are the #1 bit-exactness hazard (docs/DESIGN.md §2.2): in float they reduce in a
library/GPU-dependent order and use transcendental kernels that differ across platforms. We remove
that hazard by computing both reductions in **pure integer / fixed-point arithmetic**:

  * RMSNorm: sum-of-squares is an exact integer sum (order-independent); the reciprocal RMS is an
    integer square root (`math.isqrt`, bit-exact on every machine).
  * softmax: the max-shift and the normalizing sum are integer; `exp` is a fixed integer polynomial
    for 2^-f composed with a power-of-two shift — no libm, no float reduction.

Everything here is reduction-order invariant and identical across CPUs/GPUs/libraries, which is what
makes a forward pass re-executable to verify a TEA receipt (docs/DESIGN.md §5.4).

Scope: accuracy is deliberately modest (a cubic 2^-f approximation); the *property we need is
determinism*, not high precision. Training-time norms may stay float; only the committed inference
reference path must be fixed-point.
"""
from __future__ import annotations

import math

import numpy as np

# log2(e) in fixed-point (frac bits applied by caller); base constant at 2^16.
_LOG2E_Q16 = 94548          # round(log2(e) * 2**16)
# Cubic Taylor coefficients for 2^-f, f in [0,1), at Q16 (scale 2**16).
_C0, _C1, _C2, _C3 = 65536, 45426, 15743, 3638


def fixed_point_rmsnorm(x_fp: np.ndarray, frac_bits: int = 16, eps: int = 1,
                        gain_q: np.ndarray | None = None) -> np.ndarray:
    """RMSNorm of FIXED-POINT activations, returned in fixed-point (same scale 2**frac_bits).

    For input x_fp at scale 2**f, RMSNorm is scale-invariant in direction:
        ssq    = Σ x_fp_i²                       # scale 2**(2f), exact integer sum (order-free)
        rms_fp = isqrt(ssq // n + eps)           # scale 2**f  (RMS of the fixed-point values)
        out_i  = (x_fp_i << f) // rms_fp          # normalized, scale 2**f
    Optional integer gain `gain_q` (fixed-point, scale 2**f) applied as (out * gain_q) >> f.
    All integer; `isqrt` and `//` are bit-exact on every machine.
    """
    x = np.atleast_2d(np.asarray(x_fp, dtype=np.int64))
    n = x.shape[1]
    out = np.empty(x.shape, dtype=np.int64)
    for r in range(x.shape[0]):
        # EXACT sum-of-squares in Python big-ints (object dtype), NOT int64: the residual stream is an
        # unbounded fixed-point value (reference.py adds it without clamping across layers), so x_fp² over
        # d_model exceeds int64 once activations reach ~2048 (frac=16, d=512) — int64 np.dot would wrap
        # SILENTLY → garbage/crash, breaking the determinism keystone. Big-int is arbitrary-precision and
        # order-free; where int64 would NOT overflow the value is identical, so working inputs are unchanged.
        rowo = x[r].astype(object)
        ssq = int(np.dot(rowo, rowo))               # exact integer sum-of-squares (order-free)
        rms_fp = math.isqrt(ssq // n + eps)         # scale 2**f
        out[r] = ((rowo << frac_bits) // rms_fp).astype(np.int64)   # normalized, scale 2**f (exact)
    if gain_q is not None:
        g = np.asarray(gain_q, dtype=np.int64)
        # Cheap int64-overflow guard on the gain multiply (consistent with the matmul "fail loud" policy):
        # out is normalized to ~sqrt(head_dim)*2**f (|out| ~ a dozen * 2**f). The committed RMSNorm gains are
        # NOT unit-ish — k_norm reaches ~34x, attn_norm ~28x, ffn_norm ~19x — so the post-gain activation is
        # ~a few hundred * 2**f, still ~22 bits inside int64. This is a no-op for the committed model and only
        # catches a pathological gain that would silently wrap the int64 product (a determinism bug
        # masquerading as agreement). The native kernel (bonsai_rmsnorm_i64) mirrors this exact envelope.
        if out.size and g.size:
            mx_out = max(abs(int(out.min())), abs(int(out.max())))
            mx_g = max(abs(int(g.min())), abs(int(g.max())))
            if mx_out * mx_g > _INT64_MAX:
                raise OverflowError(
                    f"fixed_point_rmsnorm gain multiply would overflow int64: "
                    f"max|out|*max|gain| = {mx_out * mx_g} > 2^63-1 — gain left the fixed-point envelope")
        out = (out * g) >> frac_bits
    return out


def _exp2_neg_fixed(u: np.ndarray, frac_bits: int) -> np.ndarray:
    """Compute 2**(-u_real) in fixed-point, where u is fixed-point (scale 2**frac_bits), u >= 0.

    u_real = u / 2**frac = k + f_real,  f_real in [0,1).  2^-u = (poly(f) ) >> k.  Integer only.
    """
    FP = 1 << frac_bits
    mask = FP - 1
    k = u >> frac_bits                # integer part
    f = u & mask                      # fractional part, fixed-point in [0,FP)
    # Rescale the Q16 polynomial constants to this frac if needed.
    shift = 16 - frac_bits
    if shift >= 0:
        c0, c1, c2, c3 = (_C0 >> shift, _C1 >> shift, _C2 >> shift, _C3 >> shift)
    else:
        s = -shift
        c0, c1, c2, c3 = (_C0 << s, _C1 << s, _C2 << s, _C3 << s)
    f2 = (f * f) >> frac_bits
    f3 = (f2 * f) >> frac_bits
    poly = c0 - ((c1 * f) >> frac_bits) + ((c2 * f2) >> frac_bits) - ((c3 * f3) >> frac_bits)
    poly = np.maximum(poly, 0)
    # divide by 2**k, clamping huge k to 63 to avoid negative shifts
    kk = np.minimum(k, 63).astype(np.int64)
    return (poly.astype(np.int64) >> kk)


def fixed_point_softmax(logits_q: np.ndarray, frac_bits: int = 16) -> np.ndarray:
    """Deterministic fixed-point softmax over the last axis.

    `logits_q` are fixed-point logits (scale 2**frac_bits). Returns fixed-point probabilities
    (scale 2**frac_bits) that sum to ~2**frac_bits per row. Pure integer arithmetic.
    """
    # Restrict to a frac range where every fixed-point intermediate provably stays inside int64. At
    # frac>=30 the post-clamp `d * log2e` product (~(frac+2)*2^(2*frac)) wraps int64; the committed model
    # uses frac=16, so this is a no-op for it and only forbids an out-of-envelope configuration.
    assert 1 <= frac_bits <= 29, f"fixed_point_softmax frac_bits must be in [1, 29], got {frac_bits}"
    z = np.atleast_2d(np.asarray(logits_q, dtype=np.int64))
    FP = 1 << frac_bits
    log2e = _scaled_log2e(frac_bits)
    # exp(-(max-logit)) already floors to 0 in fixed-point once (max-logit) reaches this distance
    # (the cubic 2^-f result, magnitude <= 2**frac, is shifted right by >= frac+1). Clamp the distance
    # BEFORE the `* log2e` multiply: the causal-mask sentinel makes (max-logit) ~ 2**(frac+30), and
    # `d * log2e` would otherwise overflow int64. Clamping changes no output — those entries are 0 either
    # way — it only removes the overflow. The first term, (frac+2)*2^(2*frac)/log2e, is the distance past
    # which 2^-d floors to 0; the second, (2^62)//log2e, is a HARD cap guaranteeing `d * log2e < 2^62`
    # cannot wrap at any in-range frac. At frac=16 the first term (~8.2e5) dominates, so this min() is a
    # no-op for the committed model and the d_clip value is unchanged.
    d_clip = np.int64(min(((frac_bits + 2) << (2 * frac_bits)) // log2e, (1 << 62) // log2e))
    # NOTE (determinism contract): the distance `m - row` is NOT fail-loud-bounded like the matmul path.
    # For a well-formed row (a bounded causal-mask sentinel ~-2^(frac+30), which the engine always uses) the
    # clamp `np.minimum(m-row, d_clip)` gives the correct 0-weight for masked entries. If a caller instead
    # passes an int64-min-scale sentinel, `m-row` wraps int64 and that entry gets ~0.5 weight — WRONG, but
    # still deterministic AND byte-identical to the native/GPU kernels (they wrap the same way; pinned by
    # test_bonsai_native_silu_matches_oracle_if_present at the int64 extremes). This is the same
    # wrap-by-construction exception as the Q1 apply, not a fail-loud multiply. Use a bounded mask sentinel.
    out = np.empty(z.shape, dtype=np.int64)
    for r in range(z.shape[0]):
        row = z[r]
        m = int(row.max())
        d = np.minimum((m - row).astype(np.int64), d_clip)   # >= 0, clamped distance below max
        u = (d * log2e) >> frac_bits                # base-2 exponent magnitude, fixed-point
        e = _exp2_neg_fixed(u, frac_bits)           # exp(z-m) in fixed-point
        Z = int(e.sum())                            # exact integer normalizer (order-free)
        if Z == 0:
            out[r] = 0
        else:
            out[r] = (e << frac_bits) // Z          # fixed-point probabilities
    return out


def fixed_point_sigmoid(logits_q: np.ndarray, frac_bits: int = 16) -> np.ndarray:
    """Deterministic fixed-point sigmoid, byte-identical to softmax([0, x])[:, 1].

    This exists because Bonsai/Qwen3 SiLU applies sigmoid to every FFN activation. Calling the generic
    row-wise softmax on millions of two-column rows is mathematically simple but dominated by Python loop
    overhead. This vectorized form keeps the exact same integer exponent, clamp, and normalization rules.
    """
    # Same frac envelope as fixed_point_softmax and the native bonsai_silu_i64 kernel (which returns rc 1
    # outside [1,29]): beyond frac=29 the `d * log2e` product wraps int64. No-op for the committed frac=16.
    assert 1 <= frac_bits <= 29, f"fixed_point_sigmoid frac_bits must be in [1, 29], got {frac_bits}"
    x = np.asarray(logits_q, dtype=np.int64)
    log2e = _scaled_log2e(frac_bits)
    # Mirror fixed_point_softmax's d_clip EXACTLY, including the (1<<62)//log2e HARD cap that keeps
    # `d * log2e` below 2**62 at every in-envelope frac. At the committed frac=16 the first term (~8.2e5)
    # dominates the cap (~4.9e13), so the min() is a no-op and this is byte-identical to the uncapped form
    # for committed receipts (verified over a wide input sweep); the two forms only diverge near frac=29
    # (outside the committed envelope), where the cap brings sigmoid into lockstep with softmax.
    d_clip = np.int64(min(((frac_bits + 2) << (2 * frac_bits)) // log2e, (1 << 62) // log2e))
    # Same wrap-by-construction contract as fixed_point_softmax (see its note): `m - x` is intentionally not
    # fail-loud-bounded — for bounded activations it is exact, and at int64 extremes it wraps byte-identically
    # to the native SiLU kernel (pinned by test_bonsai_native_silu_matches_oracle_if_present). Determinism is
    # preserved by consistent wrap, not by raising (a raise would desync the oracle from the C/GPU producers).
    zero = np.zeros((), dtype=np.int64)
    m = np.maximum(x, zero)
    d0 = np.minimum((m - zero).astype(np.int64), d_clip)
    d1 = np.minimum((m - x).astype(np.int64), d_clip)
    e0 = _exp2_neg_fixed((d0 * log2e) >> frac_bits, frac_bits)
    e1 = _exp2_neg_fixed((d1 * log2e) >> frac_bits, frac_bits)
    z = e0 + e1
    out = np.zeros(x.shape, dtype=np.int64)
    np.floor_divide(e1 << frac_bits, z, out=out, where=(z != 0), casting="unsafe")
    return out


def _scaled_log2e(frac_bits: int) -> int:
    shift = 16 - frac_bits
    return _LOG2E_Q16 >> shift if shift >= 0 else _LOG2E_Q16 << (-shift)


# ---------------------------------------------------------------------------
# Fixed-point matmul + activations (the attention path: decision #attn = fixed-point).
#
# Attention's Q@K^T and probs@V are activation x activation (NOT ternary x int8), so they are
# computed in fixed-point. The determinism property is the SAME as the integer matmul: a fixed-point
# value is an integer, the product of two is an exact integer, and the contraction is an exact
# integer SUM -> reduction-order invariant and bit-exact across machines. The only extra step is a
# single arithmetic right-shift by `frac_bits` to return from scale 2^(2f) to scale 2^f (floor,
# deterministic for negatives too). Higher precision than INT8 attention, same trustless guarantee.
# ---------------------------------------------------------------------------

def to_fixed_point(x: np.ndarray, frac_bits: int = 16) -> np.ndarray:
    """Quantize float activations to fixed-point int64 (scale 2**frac_bits), round-half-to-even."""
    x = np.asarray(x, dtype=np.float64)
    return np.rint(x * (1 << frac_bits)).astype(np.int64)


def from_fixed_point(q: np.ndarray, frac_bits: int = 16) -> np.ndarray:
    """Dequantize fixed-point int64 back to float64."""
    return np.asarray(q, dtype=np.int64).astype(np.float64) / (1 << frac_bits)


_INT64_MAX = (1 << 63) - 1


def _assert_no_int64_overflow(a: np.ndarray, b: np.ndarray) -> None:
    """Guard the int64 accumulation in a fixed-point matmul: max|a|·max|b|·K must fit int64, else the
    integer sum WRAPS — silently-wrong-but-still-deterministic, the worst failure for a receipt (both
    producer and verifier would wrap identically and "agree" on a wrong logit). The companion
    fixed_point_rmsnorm uses Python big-ints because the residual stream is unbounded; the matmul path
    instead asserts the magnitude stays inside the fixed-point envelope (it does, because RMSNorm bounds
    every activation feeding these matmuls). Cheap vs the matmul; the reference engine is the slow
    canonical oracle, so correctness > speed here."""
    if a.size == 0 or b.size == 0:
        return
    k = a.shape[-1]
    # Compute the magnitude bound from the raw int64 extremes via Python int(): np.abs(int64-min) wraps
    # back to int64-min (a NEGATIVE value), which would defeat the guard, so a single -2^63 entry could
    # slip an overflow through. max(|min|,|max|) on the (now arbitrary-precision) Python ints is correct.
    amax = max(abs(int(a.min())), abs(int(a.max())))
    bmax = max(abs(int(b.min())), abs(int(b.max())))
    bound = amax * bmax * int(k)
    if bound > _INT64_MAX:
        raise OverflowError(
            f"fixed_point_matmul would overflow int64: max|a|·max|b|·K = {bound} > 2^63-1 (K={k}); "
            f"activation magnitude left the fixed-point envelope — a determinism bug, not a quiet wrap")


def fixed_point_matmul(a_fp: np.ndarray, b_fp: np.ndarray, frac_bits: int = 16) -> np.ndarray:
    """A_fp @ B_fp in fixed-point: (m,k)@(k,n) int64 -> (m,n) int64 at scale 2**frac_bits.

    acc = A_fp @ B_fp is an exact integer sum (scale 2**(2f)); >> frac returns to scale 2**f.
    Reduction-order invariant (integer sum) -> bit-exact across machines.
    """
    a = np.asarray(a_fp, dtype=np.int64)
    b = np.asarray(b_fp, dtype=np.int64)
    _assert_no_int64_overflow(a, b)          # fail loud rather than silently wrap (receipt integrity)
    acc = a @ b                              # exact integer accumulation, scale 2**(2*frac)
    return acc >> frac_bits                  # arithmetic shift (floor); deterministic


def fixed_point_matmul_ordered(a_fp: np.ndarray, b_fp: np.ndarray, perm: np.ndarray,
                               frac_bits: int = 16) -> np.ndarray:
    """Same as fixed_point_matmul but contracts in the explicit order `perm` (for the invariance test)."""
    a = np.asarray(a_fp, dtype=np.int64)[:, perm]
    b = np.asarray(b_fp, dtype=np.int64)[perm, :]
    _assert_no_int64_overflow(a, b)
    return (a @ b) >> frac_bits


def fixed_point_squared_relu(x_fp: np.ndarray, frac_bits: int = 16) -> np.ndarray:
    """squared-ReLU in fixed-point: relu(x)^2. Exactly representable (clamp then integer square)."""
    x = np.maximum(np.asarray(x_fp, dtype=np.int64), 0)
    # Fail loud rather than silently wrap the int64 square (the module's overflow contract): a wrapped x*x
    # is deterministic-but-wrong, the failure a receipt cannot detect. max(x) computed as a Python int.
    xm = int(x.max()) if x.size else 0
    if xm * xm > _INT64_MAX:
        raise OverflowError(
            f"fixed_point_squared_relu would overflow int64: max(x)^2 = {xm * xm} > 2^63-1 "
            f"(activation left the fixed-point envelope — a determinism bug, not a quiet wrap)")
    return (x * x) >> frac_bits              # scale 2**(2f) -> 2**f

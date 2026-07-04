"""Q4_K / Q6_K integer codec for the north-mini-code deterministic engine (Stage 2).

These are the two k-quant block formats north-mini-code-1.0 uses (Q4_K_M = Q4_K + Q6_K + F32). This module is
the integer-engine counterpart of llama.cpp's float dequant:

  * `dequant_q4k_float` / `dequant_q6k_float` — the CANONICAL float dequant (ports the llama.cpp formula). The
    fidelity reference: the integer path should approximate this.
  * `dequant_q4k_int`   / `dequant_q6k_int`   — the INTEGER (fixed-point) dequant the engine actually uses. The
    only float step is a ONE-TIME, deterministic conversion of each block's fp16 scale to fixed-point
    (`d_q = round(d * 2**frac)`) — exactly as the Bonsai importer turns Q1_0 fp scales into ints. Everything
    else is integer; the result is the weight in fixed-point at scale 2**frac.
  * `*_int_vec` — numpy-vectorized integer dequant, used as the byte-exact parity counterpart to the scalar
    reference (the determinism gate: integer ⇒ implementation/order-independent).
  * `matmul_fixed` — integer fixed-point GEMM (weights·activations >> frac), the inference primitive.

Block layouts (GGML, QK_K=256, little-endian, ggml_half = IEEE fp16):
  Q4_K (144B): d(fp16) dmin(fp16) scales[12]u8 qs[128]u8         — 8 sub-blocks of 32, 6-bit sub scale+min
  Q6_K (210B): ql[128]u8 qh[64]u8 scales[16]i8 d(fp16)           — 6-bit quants (q-32), per-16 int8 sub-scale
"""
from __future__ import annotations

import numpy as np

QK_K = 256


# --------------------------------------------------------------------------------------------------- Q4_K ----
def get_scale_min_k4(j: int, scales) -> tuple[int, int]:
    """Unpack the j-th (0..7) 6-bit sub-block scale `d` and min `m` from the packed 12-byte `scales`
    (verbatim port of llama.cpp `get_scale_min_k4`)."""
    if j < 4:
        return int(scales[j] & 63), int(scales[j + 4] & 63)
    d = (int(scales[j + 4]) & 0xF) | ((int(scales[j - 4]) >> 6) << 4)
    m = (int(scales[j + 4]) >> 4) | ((int(scales[j]) >> 6) << 4)
    return d, m


def dequant_q4k_float(d, dmin, scales, qs) -> np.ndarray:
    """Canonical float dequant of one Q4_K super-block -> float32[256] (ports `dequantize_row_q4_K`)."""
    d = np.float32(np.float16(d)); dmin = np.float32(np.float16(dmin))
    y = np.empty(QK_K, dtype=np.float32)
    out = is_ = qoff = 0
    for _ in range(0, QK_K, 64):
        sc, m = get_scale_min_k4(is_, scales);     d1 = d * np.float32(sc); m1 = dmin * np.float32(m)
        sc, m = get_scale_min_k4(is_ + 1, scales); d2 = d * np.float32(sc); m2 = dmin * np.float32(m)
        for l in range(32):
            y[out + l] = d1 * np.float32(int(qs[qoff + l]) & 0xF) - m1
        for l in range(32):
            y[out + 32 + l] = d2 * np.float32(int(qs[qoff + l]) >> 4) - m2
        out += 64; qoff += 32; is_ += 2
    return y


def dequant_q4k_int(d, dmin, scales, qs, frac: int) -> np.ndarray:
    """Integer fixed-point dequant of one Q4_K super-block -> int64[256] at scale 2**frac (scalar reference).
    The block fp16 scales become fixed-point once (`round(d*2**frac)`); the rest is exact integer affine."""
    d_q = int(round(float(np.float16(d)) * (1 << frac)))
    dmin_q = int(round(float(np.float16(dmin)) * (1 << frac)))
    y = np.empty(QK_K, dtype=np.int64)
    out = is_ = qoff = 0
    for _ in range(0, QK_K, 64):
        sc, m = get_scale_min_k4(is_, scales)
        dm = dmin_q * m
        for l in range(32):
            y[out + l] = d_q * sc * (int(qs[qoff + l]) & 0xF) - dm
        sc, m = get_scale_min_k4(is_ + 1, scales)
        dm = dmin_q * m
        for l in range(32):
            y[out + 32 + l] = d_q * sc * (int(qs[qoff + l]) >> 4) - dm
        out += 64; qoff += 32; is_ += 2
    return y


def dequant_q4k_int_vec(d, dmin, scales, qs, frac: int) -> np.ndarray:
    """Vectorized integer Q4_K dequant -> int64[256]. MUST be byte-identical to `dequant_q4k_int`."""
    d_q = int(round(float(np.float16(d)) * (1 << frac)))
    dmin_q = int(round(float(np.float16(dmin)) * (1 << frac)))
    qs = np.asarray(qs, dtype=np.int64)
    sc = np.array([get_scale_min_k4(j, scales)[0] for j in range(8)], dtype=np.int64)
    mn = np.array([get_scale_min_k4(j, scales)[1] for j in range(8)], dtype=np.int64)
    q = np.empty(QK_K, dtype=np.int64)          # per-position nibble
    sc_pos = np.empty(QK_K, dtype=np.int64)     # per-position sub-scale
    mn_pos = np.empty(QK_K, dtype=np.int64)     # per-position sub-min
    for g in range(4):                           # 4 groups of 64 = sub-blocks (2g, 2g+1)
        lo = qs[32 * g:32 * g + 32] & 0xF
        hi = qs[32 * g:32 * g + 32] >> 4
        q[64 * g:64 * g + 32] = lo;        q[64 * g + 32:64 * g + 64] = hi
        sc_pos[64 * g:64 * g + 32] = sc[2 * g];   sc_pos[64 * g + 32:64 * g + 64] = sc[2 * g + 1]
        mn_pos[64 * g:64 * g + 32] = mn[2 * g];   mn_pos[64 * g + 32:64 * g + 64] = mn[2 * g + 1]
    return d_q * sc_pos * q - dmin_q * mn_pos


# --------------------------------------------------------------------------------------------------- Q6_K ----
def dequant_q6k_float(d, ql, qh, scales) -> np.ndarray:
    """Canonical float dequant of one Q6_K super-block -> float32[256] (ports `dequantize_row_q6_K`)."""
    d = np.float32(np.float16(d))
    y = np.empty(QK_K, dtype=np.float32)
    yo = qlo = qho = sco = 0
    for _ in range(0, QK_K, 128):
        for l in range(32):
            is_ = l // 16
            q1 = ((int(ql[qlo + l]) & 0xF) | (((int(qh[qho + l]) >> 0) & 3) << 4)) - 32
            q2 = ((int(ql[qlo + l + 32]) & 0xF) | (((int(qh[qho + l]) >> 2) & 3) << 4)) - 32
            q3 = ((int(ql[qlo + l]) >> 4) | (((int(qh[qho + l]) >> 4) & 3) << 4)) - 32
            q4 = ((int(ql[qlo + l + 32]) >> 4) | (((int(qh[qho + l]) >> 6) & 3) << 4)) - 32
            y[yo + l] = d * np.float32(int(scales[sco + is_ + 0])) * np.float32(q1)
            y[yo + l + 32] = d * np.float32(int(scales[sco + is_ + 2])) * np.float32(q2)
            y[yo + l + 64] = d * np.float32(int(scales[sco + is_ + 4])) * np.float32(q3)
            y[yo + l + 96] = d * np.float32(int(scales[sco + is_ + 6])) * np.float32(q4)
        yo += 128; qlo += 64; qho += 32; sco += 8
    return y


def dequant_q6k_int(d, ql, qh, scales, frac: int) -> np.ndarray:
    """Integer fixed-point dequant of one Q6_K super-block -> int64[256] at scale 2**frac (scalar reference)."""
    d_q = int(round(float(np.float16(d)) * (1 << frac)))
    y = np.empty(QK_K, dtype=np.int64)
    yo = qlo = qho = sco = 0
    for _ in range(0, QK_K, 128):
        for l in range(32):
            is_ = l // 16
            q1 = ((int(ql[qlo + l]) & 0xF) | (((int(qh[qho + l]) >> 0) & 3) << 4)) - 32
            q2 = ((int(ql[qlo + l + 32]) & 0xF) | (((int(qh[qho + l]) >> 2) & 3) << 4)) - 32
            q3 = ((int(ql[qlo + l]) >> 4) | (((int(qh[qho + l]) >> 4) & 3) << 4)) - 32
            q4 = ((int(ql[qlo + l + 32]) >> 4) | (((int(qh[qho + l]) >> 6) & 3) << 4)) - 32
            y[yo + l] = d_q * int(scales[sco + is_ + 0]) * q1
            y[yo + l + 32] = d_q * int(scales[sco + is_ + 2]) * q2
            y[yo + l + 64] = d_q * int(scales[sco + is_ + 4]) * q3
            y[yo + l + 96] = d_q * int(scales[sco + is_ + 6]) * q4
        yo += 128; qlo += 64; qho += 32; sco += 8
    return y


def dequant_q6k_int_vec(d, ql, qh, scales, frac: int) -> np.ndarray:
    """Vectorized integer Q6_K dequant -> int64[256]. MUST be byte-identical to `dequant_q6k_int`."""
    d_q = int(round(float(np.float16(d)) * (1 << frac)))
    ql = np.asarray(ql, dtype=np.int64); qh = np.asarray(qh, dtype=np.int64)
    sca = np.asarray(scales, dtype=np.int64)
    y = np.empty(QK_K, dtype=np.int64)
    for half in range(2):
        qlo, qho, sco, yo = 64 * half, 32 * half, 8 * half, 128 * half
        l = np.arange(32)
        is_ = l // 16
        q1 = ((ql[qlo + l] & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) - 32
        q2 = ((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32
        q3 = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32
        q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32
        y[yo + l] = d_q * sca[sco + is_ + 0] * q1
        y[yo + l + 32] = d_q * sca[sco + is_ + 2] * q2
        y[yo + l + 64] = d_q * sca[sco + is_ + 4] * q3
        y[yo + l + 96] = d_q * sca[sco + is_ + 6] * q4
    return y


# ------------------------------------------------------------------------ vectorized tensor dequant ---------
# Whole-tensor numpy dequant (no per-block python loop) — needed for real-weight forwards (the 262144x2048
# head is ~2M blocks). Byte-identical to the per-block integer functions; float matches within float rounding.
def dequant_q4k_tensor(raw: bytes, n_elements: int, frac: int | None = None) -> np.ndarray:
    nb = n_elements // QK_K
    b = np.frombuffer(raw, np.uint8, nb * 144).reshape(nb, 144)
    d = b[:, 0:2].copy().view(np.float16).reshape(nb).astype(np.float64)
    dmin = b[:, 2:4].copy().view(np.float16).reshape(nb).astype(np.float64)
    scales = b[:, 4:16].astype(np.int64); qs = b[:, 16:144].astype(np.int64)
    sc = np.empty((nb, 8), np.int64); mn = np.empty((nb, 8), np.int64)
    for j in range(8):
        if j < 4:
            sc[:, j] = scales[:, j] & 63; mn[:, j] = scales[:, j + 4] & 63
        else:
            sc[:, j] = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
            mn[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    q = np.empty((nb, QK_K), np.int64); scp = np.empty((nb, QK_K), np.int64); mnp = np.empty((nb, QK_K), np.int64)
    for g in range(4):
        q[:, 64 * g:64 * g + 32] = qs[:, 32 * g:32 * g + 32] & 0xF
        q[:, 64 * g + 32:64 * g + 64] = qs[:, 32 * g:32 * g + 32] >> 4
        scp[:, 64 * g:64 * g + 32] = sc[:, 2 * g:2 * g + 1]; scp[:, 64 * g + 32:64 * g + 64] = sc[:, 2 * g + 1:2 * g + 2]
        mnp[:, 64 * g:64 * g + 32] = mn[:, 2 * g:2 * g + 1]; mnp[:, 64 * g + 32:64 * g + 64] = mn[:, 2 * g + 1:2 * g + 2]
    if frac is None:
        d32 = d.astype(np.float32)[:, None]; dm32 = dmin.astype(np.float32)[:, None]
        return (d32 * scp.astype(np.float32) * q.astype(np.float32) - dm32 * mnp.astype(np.float32)).reshape(-1)
    dq = np.round(d * (1 << frac)).astype(np.int64)[:, None]; dmq = np.round(dmin * (1 << frac)).astype(np.int64)[:, None]
    return (dq * scp * q - dmq * mnp).reshape(-1)


def dequant_q6k_tensor(raw: bytes, n_elements: int, frac: int | None = None) -> np.ndarray:
    nb = n_elements // QK_K
    b = np.frombuffer(raw, np.uint8, nb * 210).reshape(nb, 210)
    ql = b[:, 0:128].astype(np.int64); qh = b[:, 128:192].astype(np.int64)
    sca = b[:, 192:208].copy().view(np.int8).astype(np.int64)
    d = b[:, 208:210].copy().view(np.float16).reshape(nb).astype(np.float64)
    y = np.empty((nb, QK_K), np.float64 if frac is None else np.int64)
    for half in range(2):
        qlo, qho, sco, yo = 64 * half, 32 * half, 8 * half, 128 * half
        l = np.arange(32); is_ = l // 16
        Ql = ql[:, qlo:qlo + 32]; Ql2 = ql[:, qlo + 32:qlo + 64]; Qh = qh[:, qho:qho + 32]
        qq = [((Ql & 0xF) | (((Qh >> 0) & 3) << 4)) - 32, ((Ql2 & 0xF) | (((Qh >> 2) & 3) << 4)) - 32,
              ((Ql >> 4) | (((Qh >> 4) & 3) << 4)) - 32, ((Ql2 >> 4) | (((Qh >> 6) & 3) << 4)) - 32]
        ss = [sca[:, sco + is_ + k] for k in (0, 2, 4, 6)]
        if frac is None:
            d32 = d.astype(np.float32)[:, None]
            for k in range(4):
                y[:, yo + 32 * k:yo + 32 * k + 32] = d32 * ss[k].astype(np.float32) * qq[k].astype(np.float32)
        else:
            dq = np.round(d * (1 << frac)).astype(np.int64)[:, None]
            for k in range(4):
                y[:, yo + 32 * k:yo + 32 * k + 32] = dq * ss[k] * qq[k]
    return y.reshape(-1)


# ----------------------------------------------------------------------------------------------- matmul ------
def matmul_fixed(w_fixed: np.ndarray, x_fixed: np.ndarray, frac: int) -> np.ndarray:
    """Integer fixed-point GEMM: out[o] = (Σ_i w[o,i]·x[i]) >> frac, all int64. Both inputs at scale 2**frac;
    output at scale 2**frac. Integer addition is associative ⇒ order/implementation-independent (the
    determinism contract). Caller must keep the accumulator inside int64 (Σ|w·x| < 2**63)."""
    w = np.asarray(w_fixed, dtype=np.int64)
    x = np.asarray(x_fixed, dtype=np.int64)
    acc = w @ x                          # exact int64 dot (no overflow within range)
    return acc >> frac                   # arithmetic shift = floor toward -inf


# --------------------------------------------------------------------------------- synthetic test blocks -----
def _rng(seed): return np.random.default_rng(seed)


def random_q4k(seed, *, dmag=0.01):
    """A valid random Q4_K super-block (any bytes are a valid block; both dequants parse identically)."""
    r = _rng(seed)
    d = np.float16(r.uniform(0.2 * dmag, dmag))
    dmin = np.float16(r.uniform(0.0, 0.5 * dmag))
    scales = r.integers(0, 256, size=12, dtype=np.uint8)
    qs = r.integers(0, 256, size=128, dtype=np.uint8)
    return dict(d=d, dmin=dmin, scales=scales, qs=qs)


def random_q6k(seed, *, dmag=0.01):
    r = _rng(seed)
    d = np.float16(r.uniform(0.2 * dmag, dmag))
    ql = r.integers(0, 256, size=128, dtype=np.uint8)
    qh = r.integers(0, 256, size=64, dtype=np.uint8)
    scales = r.integers(-64, 64, size=16, dtype=np.int8)
    return dict(d=d, ql=ql, qh=qh, scales=scales)


# ------------------------------------------------------------------------------------------- self-test -------
def _selftest(frac: int = 16, n: int = 64) -> int:
    """Run the two gates over n random blocks: (1) byte-exact int parity scalar==vec; (2) float fidelity."""
    import sys
    max_rel4 = max_rel6 = 0.0
    for s in range(n):
        b4 = random_q4k(s)
        i4 = dequant_q4k_int(**b4, frac=frac); v4 = dequant_q4k_int_vec(**b4, frac=frac)
        assert np.array_equal(i4, v4), f"Q4_K parity FAIL @seed {s}"
        f4 = dequant_q4k_float(**b4)
        scale = max(np.max(np.abs(f4)), 1e-9)
        max_rel4 = max(max_rel4, np.max(np.abs(i4.astype(np.float64) / (1 << frac) - f4)) / scale)

        b6 = random_q6k(s)
        i6 = dequant_q6k_int(**b6, frac=frac); v6 = dequant_q6k_int_vec(**b6, frac=frac)
        assert np.array_equal(i6, v6), f"Q6_K parity FAIL @seed {s}"
        f6 = dequant_q6k_float(**b6)
        scale = max(np.max(np.abs(f6)), 1e-9)
        max_rel6 = max(max_rel6, np.max(np.abs(i6.astype(np.float64) / (1 << frac) - f6)) / scale)

    # matmul: dequant a small weight matrix (Q4_K rows) and check fixed-point GEMM vs float GEMM
    rows = 16
    W_int = np.stack([dequant_q4k_int(**random_q4k(1000 + r), frac=frac) for r in range(rows)])
    W_flt = np.stack([dequant_q4k_float(**random_q4k(1000 + r)) for r in range(rows)])
    xf = _rng(7).uniform(-1, 1, size=QK_K).astype(np.float64)
    x_int = np.round(xf * (1 << frac)).astype(np.int64)
    out_int = matmul_fixed(W_int, x_int, frac).astype(np.float64) / (1 << frac)
    out_flt = W_flt.astype(np.float64) @ xf
    mm_rel = np.max(np.abs(out_int - out_flt)) / max(np.max(np.abs(out_flt)), 1e-9)

    print(f"[qk_codec self-test] frac={frac} blocks={n}")
    print(f"  Q4_K  byte-exact int parity: PASS   float-fidelity max rel err: {max_rel4:.2e}")
    print(f"  Q6_K  byte-exact int parity: PASS   float-fidelity max rel err: {max_rel6:.2e}")
    print(f"  fixed-point GEMM vs float GEMM: max rel err: {mm_rel:.2e}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Q4_K/Q6_K integer codec self-test")
    ap.add_argument("--frac", type=int, default=16)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    raise SystemExit(_selftest(a.frac, a.n))

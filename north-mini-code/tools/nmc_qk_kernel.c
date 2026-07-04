// nmc_qk_kernel.c — CPU integer kernels for north-mini-code: fused Q4_K/Q6_K dequant + fixed-point matmul.
//
// Computes  out[t,o] = ( Σ_i W_fixed[o,i] * x[t,i] ) >> fw   in integer, BYTE-IDENTICAL to the numpy reference
// (qk_codec.dequant_q4k/q6k_int  +  cohere2.linear). Weights are consumed as RAW GGUF block bytes (dequant is
// fused inline). Accumulation is __int128 (the per-row dot can exceed int64) — exactly the big-int the numpy
// oracle uses, with no overflow for these magnitudes, so the result matches bit-for-bit. The fp16 block scale
// is converted to fixed-point as round-half-to-even(d * 2^fw), matching Python's round() on the same double.
//
// NOT a committed/portable artifact (per-host build, like the Bonsai kernel). The numpy oracle stays canonical.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// IEEE half -> double (exact). Subnormals and inf/nan handled; weights never use inf/nan.
static double half_to_double(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
    double v;
    if (exp == 0)        v = ldexp((double)mant, -24);                  // 2^-14 * (mant/1024)
    else if (exp == 31)  v = mant ? NAN : INFINITY;
    else                 v = ldexp((double)(mant | 0x400u), (int)exp - 25);
    return sign ? -v : v;
}

// round-half-to-even(d * 2^fw) — matches Python round(float(np.float16(h)) * (1<<fw)).
//
// IMPLEMENTATION-DEFINED RELIANCE (documented, per-host-build mitigation): llrint() rounds using the
// CURRENT (dynamic) floating-point rounding mode. We rely on the default FE_TONEAREST (round-half-to-EVEN),
// which equals Python's banker's-rounding round(); no code here calls fesetround(), so the mode is the
// process default. Unlike the sibling Bonsai kernel (tools/bonsai_q1_kernel.c) — whose Q1 scales arrive
// already converted to int64 fixed-point from Python, so it needs NO fp16->fixed step and thus has no
// helper to copy here — there is no integer-only helper that reproduces round-half-to-even on the exact
// double bit-for-bit without risking divergence from the NumPy oracle. This kernel is therefore NOT a
// committed/portable artifact: it is rebuilt per host and the NumPy oracle (qk_codec + cohere2.linear)
// stays canonical; tests pin this build to the oracle bit-for-bit. Do not assume portability of llrint's
// rounding across a process that has changed the FP rounding mode.
static inline int64_t fp16_fixed(uint16_t h, int fw) {
    return (int64_t)llrint(half_to_double(h) * (double)(1ULL << fw));
}

// llama.cpp get_scale_min_k4: unpack the j-th (0..7) 6-bit sub-scale d and min m from the packed 12 bytes.
static inline void get_scale_min_k4(int j, const uint8_t *q, int *d, int *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j]     >> 6) << 4);
    }
}

// Dequant one Q4_K super-block (raw 144B) -> wb[256] int64 at scale 2^fw (order matches dequant_q4k_int).
static inline void deq_q4k_block(const uint8_t *blk, int fw, int64_t *wb) {
    uint16_t dh, dmh; memcpy(&dh, blk, 2); memcpy(&dmh, blk + 2, 2);
    int64_t dq = fp16_fixed(dh, fw), dmq = fp16_fixed(dmh, fw);
    const uint8_t *scales = blk + 4, *qs = blk + 16;
    for (int g = 0; g < 4; g++) {
        int sc, m;
        get_scale_min_k4(2 * g, scales, &sc, &m);     int64_t dmm = dmq * (int64_t)m;
        for (int l = 0; l < 32; l++) wb[64 * g + l]      = dq * (int64_t)sc * (int64_t)(qs[32 * g + l] & 0xF) - dmm;
        get_scale_min_k4(2 * g + 1, scales, &sc, &m); dmm = dmq * (int64_t)m;
        for (int l = 0; l < 32; l++) wb[64 * g + 32 + l] = dq * (int64_t)sc * (int64_t)(qs[32 * g + l] >> 4)  - dmm;
    }
}

// Dequant one Q6_K super-block (raw 210B) -> wb[256] int64 at scale 2^fw (order matches dequant_q6k_int).
static inline void deq_q6k_block(const uint8_t *blk, int fw, int64_t *wb) {
    const uint8_t *ql = blk, *qh = blk + 128; const int8_t *sc = (const int8_t *)(blk + 192);
    uint16_t dh; memcpy(&dh, blk + 208, 2); int64_t dq = fp16_fixed(dh, fw);
    for (int half = 0; half < 2; half++) {
        int qlo = 64 * half, qho = 32 * half, sco = 8 * half, yo = 128 * half;
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int64_t q1 = ((ql[qlo + l]      & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) - 32;
            int64_t q2 = ((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32;
            int64_t q3 = ((ql[qlo + l]      >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32;
            int64_t q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32;
            wb[yo + l]      = dq * (int64_t)sc[sco + is + 0] * q1;
            wb[yo + l + 32] = dq * (int64_t)sc[sco + is + 2] * q2;
            wb[yo + l + 64] = dq * (int64_t)sc[sco + is + 4] * q3;
            wb[yo + l + 96] = dq * (int64_t)sc[sco + is + 6] * q4;
        }
    }
}

// out[t*out_f + o] = (Σ_i W_fixed[o,i] * x[t*in_f + i]) >> fw.  qtype: 0=Q4_K, 1=Q6_K.  W is [out_f][n_blocks*bs].
void qk_linear(const uint8_t *W, const int64_t *x, int64_t T, int64_t out_f, int64_t n_blocks,
               int fw, int qtype, int64_t *out) {
    const int64_t in_f = n_blocks * 256;
    const int64_t bs = qtype == 0 ? 144 : 210;
    #pragma omp parallel
    {
        int64_t *wb = (int64_t *)malloc((size_t)in_f * sizeof(int64_t));
        #pragma omp for schedule(static)
        for (int64_t o = 0; o < out_f; o++) {
            const uint8_t *row = W + (size_t)o * n_blocks * bs;
            for (int64_t b = 0; b < n_blocks; b++) {
                if (qtype == 0) deq_q4k_block(row + (size_t)b * 144, fw, wb + b * 256);
                else            deq_q6k_block(row + (size_t)b * 210, fw, wb + b * 256);
            }
            for (int64_t t = 0; t < T; t++) {
                const int64_t *xt = x + t * in_f;
                __int128 acc = 0;
                for (int64_t i = 0; i < in_f; i++) acc += (__int128)wb[i] * (__int128)xt[i];
                // IMPLEMENTATION-DEFINED RELIANCE (documented; per-host-build + NumPy-oracle mitigation):
                //   (a) `acc >> fw` on a NEGATIVE signed __int128 — the C standard leaves the result of a
                //       signed right-shift implementation-defined; every real toolchain used to build this
                //       does an arithmetic shift = floor toward -inf, matching Python's `>>` on big-ints in
                //       cohere2.linear (the NumPy oracle), so the value matches bit-for-bit on the build host.
                //   (b) the `(int64_t)` truncation of the 128-bit result — well-defined only because the
                //       contract guarantees the per-row dot fits int64 ("no overflow for these magnitudes");
                //       an out-of-range value would be implementation-defined.
                // The sibling Bonsai kernel avoids (a)/(b) with explicit portable helpers (floor_shift_i128 /
                // i128_to_i64 in tools/bonsai_q1_kernel.c). They are NOT ported here because this kernel is a
                // per-host build whose result is pinned bit-for-bit to the canonical NumPy oracle by tests —
                // that build+oracle gate, not source portability, is the guarantee. Never diverge from the oracle.
                out[t * out_f + o] = (int64_t)(acc >> fw);     // arithmetic >> = floor toward -inf (matches py)
            }
        }
        free(wb);
    }
}

int qk_kernel_available(void) { return 0; }   // probe symbol

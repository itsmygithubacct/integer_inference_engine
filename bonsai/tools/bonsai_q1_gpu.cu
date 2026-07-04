// bonsai_q1_gpu.cu — PER-HOST OPT-IN CUDA Q1_0 linear apply (NOT a committed/portable artifact).
//
// Byte-identical to the int64 CPU oracle (reference_bonsai.q1_linear_ref / tools/bonsai_q1_kernel.c
// DEFINE_Q1_ELEMENT). The GPU is a PRODUCER; the CPU oracle is the canonical VERIFIER — a GPU-produced
// receipt re-executes bit-for-bit on a CPU-only host. Build: tools/build_bonsai_q1_gpu.sh (nvcc -arch=sm_86).
// Plan + parity gate: research/bonsai-notary/IMPLEMENT-GPU-MODE.md. Math: Q1-BITMATMUL-REFORMULATION.md.
//
// MILESTONE M1 (this file) = the byte-exact BEACHHEAD: a direct-int64 masked-sum kernel, one warp per (t,o)
// output. It proves integer determinism on the GPU (GPU-FEASIBILITY §2) and validates the whole int64 tail —
// little-endian bit->±1 unpack, per-block signed_sum (warp-reduced; integer add is order-free), the per-block
// uint64 scale multiply (mod 2^64), the arshift_i64 floor-toward-−∞ port, per-block-floor-then-sum, and the
// cross-block accumulation. It deliberately does NOT yet use DP4A/IMMA or the base-256 limb decomposition or
// weight residency — those are the perf follow-ups (M1-perf / M2 IMMA), layered on against THIS same parity
// gate. Correctness first, exactly per GPU-FEASIBILITY phase 1 ("prove byte-exact, not speed").
//
// Invariants reproduced verbatim from the CPU kernel (see Q1-BITMATMUL-REFORMULATION.md §2/§7):
//   I3 per-block int64 scale  | I4 per-block arshift floor THEN sum (never floor-once; never trunc/CUDA >>)
//   I5 order-free integer reductions | mod-2^64 wrap at every stage (NO wider-than-64-bit accumulator)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Floor-toward-−∞ arithmetic shift, exact port of bonsai_q1_kernel.c:29-42 (arshift_i64).
// NOT a CUDA signed `>>` (implementation-defined) and NOT truncating division (rounds toward zero -> wrong
// for negatives). For v<0 returns -ceil(|v| / 2^shift) via the two's-complement magnitude.
__device__ __forceinline__ long long arshift_i64_floor(long long v, long long shift) {
    if (shift <= 0) return v;
    if (shift >= 63) return v < 0 ? -1LL : 0LL;          // off the apply path (frac=16), but matches the C edge
    if (v >= 0) return (long long)((unsigned long long)v >> shift);
    unsigned long long mag = (~(unsigned long long)v) + 1ULL;          // two's-complement magnitude of v
    unsigned long long q = (mag + ((1ULL << shift) - 1ULL)) >> shift;  // ceil(mag / 2^shift)
    return -(long long)q;
}

// One warp computes one output element (t, o). The 32 lanes split the 128-weight block (4 weights/lane),
// each lane forms its int64 partial (sign * activation), the warp sums them (order-free), and lane 0 does the
// per-block scale + floor + cross-block accumulate serially (ascending block index, matching the CPU loop).
__global__ void q1_linear_kernel(
        const long long* __restrict__ x,            // (tokens, n_blocks*128)  int64, C-contiguous
        const unsigned char* __restrict__ bits,     // (out_features, n_blocks, 16)  uint8, C-contiguous
        const long long* __restrict__ scale,        // (out_features, n_blocks)  int64, C-contiguous
        long long tokens, long long out_f, long long n_blocks, long long frac,
        long long* __restrict__ out)                // (tokens, out_features)  int64, C-contiguous
{
    const long long gtid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long warp_id = gtid >> 5;            // 32 lanes per warp
    const int lane = (int)(threadIdx.x & 31);
    const long long total_warps = tokens * out_f;
    if (warp_id >= total_warps) return;             // uniform across the warp (blockDim.x % 32 == 0)

    const long long t = warp_id / out_f;
    const long long o = warp_id % out_f;
    const long long* xrow = x + t * (n_blocks * 128);
    const unsigned char* brow = bits + o * (n_blocks * 16);
    const long long* srow = scale + o * n_blocks;

    unsigned long long total = 0ULL;                // accumulate mod 2^64 (lane 0 only)
    for (long long b = 0; b < n_blocks; ++b) {
        const long long* xb = xrow + b * 128;
        const unsigned char* bb = brow + b * 16;
        long long lane_partial = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = lane * 4 + j;             // weight index 0..127
            const int sbit = (bb[i >> 3] >> (i & 7)) & 1;   // little-endian within byte (matches np.unpackbits)
            const long long sgn = (long long)(2 * sbit - 1); // 1 -> +1, 0 -> -1
            lane_partial += sgn * xb[i];            // int64 (mod 2^64)
        }
        // Warp-reduce the 32 lane partials -> block signed_sum. Integer add is exactly associative, so this
        // tree order is bit-identical to the CPU's serial 128-element sum (I5).
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            lane_partial += __shfl_down_sync(0xffffffffu, lane_partial, off);
        if (lane == 0) {
            const long long signed_sum = lane_partial;
            const unsigned long long prod =
                (unsigned long long)signed_sum * (unsigned long long)srow[b];   // mod 2^64 (I3)
            total += (unsigned long long) arshift_i64_floor((long long)prod, frac); // per-block floor (I4)
        }
    }
    if (lane == 0)
        out[t * out_f + o] = (long long)total;      // u64 -> i64 two's-complement reinterpret
}

// ---- DP4A Q1 apply (compute lever for prefill: ncu showed the int64 masked-sum is ALU-bound, dram 0.2%) ----
// Reformulation (Q1-BITMATMUL-REFORMULATION.md §4): decompose each int64 activation into L signed base-256
// digits d_l∈[−128,127]; per block the 128-wide masked sum becomes L int8 dp4a reductions S_l=Σ w_i·d_l(x_i)
// (int32, |S_l|≤16384), recombined signed_sum=Σ_l 2^(8l)·S_l in int64. The hot 128-element summation runs in
// int32 dp4a (4 MACs/instr, NOT emulated) instead of emulated int64 add; int64 only at the per-block recombine
// + scale + floor (n_blocks×, not 128·n_blocks×). Byte-identical to q1_linear_kernel: same signed_sum mod 2^64
// (exact for L=8 by 2^64 closure; for L<8 when |x| is in the balanced range — host-gated), same scale/floor/sum.

// Precompute L signed base-256 digits per activation. d_limb laid out (L, tokens, K) int8 so the 4 digits for
// one dp4a group are contiguous (4-byte-aligned). Recurrence on the uint64 bit pattern (portable; §4.1 note).
__global__ void q1_digits_kernel(const long long* __restrict__ x, long long tokens, long long K, int L,
                                 signed char* __restrict__ d_limb) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= tokens * K) return;
    const long long t = gid / K, k = gid % K;
    unsigned long long r = (unsigned long long) x[gid];
    const long long TK = tokens * K;
    for (int l = 0; l < L; ++l) {
        int lb = (int)(r & 0xFFu);
        int d = lb >= 128 ? lb - 256 : lb;                       // balanced digit [−128,127]
        d_limb[(long long) l * TK + t * K + k] = (signed char) d;
        r = (r - (unsigned long long)(long long) d) >> 8;        // (r−d) divisible by 256 → exact logical shift
    }
}

// One thread per (t,o). dp4a over the L limbs; int64 only per block. d_limb is (L,tokens,K), bits/scale as usual.
__global__ void q1_dp4a_apply_kernel(const signed char* __restrict__ d_limb, const unsigned char* __restrict__ bits,
                                     const long long* __restrict__ scale, long long tokens, long long out_f,
                                     long long n_blocks, long long frac, int L, long long* __restrict__ out) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= tokens * out_f) return;
    const long long t = gid / out_f, o = gid % out_f;
    const long long K = n_blocks * 128, TK = tokens * K;
    const unsigned char* brow = bits + o * (n_blocks * 16);
    const long long* srow = scale + o * n_blocks;
    unsigned long long total = 0ULL;
    for (long long b = 0; b < n_blocks; ++b) {
        int S[8];
        #pragma unroll
        for (int l = 0; l < 8; ++l) S[l] = 0;
        for (int g = 0; g < 32; ++g) {                          // 32 groups of 4 within the 128-block
            const unsigned char wbyte = brow[b * 16 + (g >> 1)];
            const int bb = (g & 1) * 4;
            const int w0 = ((wbyte >> (bb + 0)) & 1) ? 1 : -1;
            const int w1 = ((wbyte >> (bb + 1)) & 1) ? 1 : -1;
            const int w2 = ((wbyte >> (bb + 2)) & 1) ? 1 : -1;
            const int w3 = ((wbyte >> (bb + 3)) & 1) ? 1 : -1;
            const int wpk = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
            const long long off = t * K + b * 128 + (long long) g * 4;
            for (int l = 0; l < L; ++l) {
                const int dpk = *reinterpret_cast<const int*>(d_limb + (long long) l * TK + off);
                S[l] = __dp4a(wpk, dpk, S[l]);
            }
        }
        long long signed_sum = 0;
        #pragma unroll
        for (int l = 0; l < 8; ++l) if (l < L) signed_sum += ((long long) S[l]) << (8 * l);
        const unsigned long long prod = (unsigned long long) signed_sum * (unsigned long long)(long long) srow[b];
        total += (unsigned long long) arshift_i64_floor((long long) prod, frac);
    }
    out[t * out_f + o] = (long long) total;
}

// L=4 envelope guard: if ANY activation falls outside the balanced base-256 L=4 range (|x| > 2139062143), the
// L=4 digit decomposition is not byte-exact → set the overflow flag so the monolith returns rc 4 → CPU
// fallback (no silent wrap). For the committed model (|x|~2^25 « 2.14e9) this never fires.
__global__ void range_guard_l4_kernel(const long long* __restrict__ x, long long n, int* __restrict__ overflow) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long v = x[i];
    unsigned long long a = v < 0 ? ((unsigned long long)(~(unsigned long long) v) + 1u) : (unsigned long long) v;
    if (a > 2139062143ULL) atomicOr(overflow, 1);
}

// Warp-per-(t,o) DP4A variant (recovers the 32-way parallelism the thread-per-output version lost). Each lane
// does L dp4a over its 4 elements, recombines its limb partials to an int64 lane partial, then ONE int64
// warp-reduce per block (same reduction count as the int64 kernel). Byte-identical.
__global__ void q1_dp4a_warp_kernel(const signed char* __restrict__ d_limb, const unsigned char* __restrict__ bits,
                                    const long long* __restrict__ scale, long long tokens, long long out_f,
                                    long long n_blocks, long long frac, int L, long long* __restrict__ out) {
    const long long gtid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long warp_id = gtid >> 5;
    const int lane = (int)(threadIdx.x & 31);
    if (warp_id >= tokens * out_f) return;
    const long long t = warp_id / out_f, o = warp_id % out_f;
    const long long K = n_blocks * 128, TK = tokens * K;
    const unsigned char* brow = bits + o * (n_blocks * 16);
    const long long* srow = scale + o * n_blocks;
    unsigned long long total = 0ULL;
    for (long long b = 0; b < n_blocks; ++b) {
        const unsigned char wbyte = brow[b * 16 + (lane >> 1)];
        const int bb = (lane & 1) * 4;
        const int w0 = ((wbyte >> (bb + 0)) & 1) ? 1 : -1, w1 = ((wbyte >> (bb + 1)) & 1) ? 1 : -1;
        const int w2 = ((wbyte >> (bb + 2)) & 1) ? 1 : -1, w3 = ((wbyte >> (bb + 3)) & 1) ? 1 : -1;
        const int wpk = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
        const long long off = t * K + b * 128 + (long long) lane * 4;
        long long lane_partial = 0;
        for (int l = 0; l < L; ++l) {
            const int dpk = *reinterpret_cast<const int*>(d_limb + (long long) l * TK + off);
            lane_partial += ((long long) __dp4a(wpk, dpk, 0)) << (8 * l);
        }
        #pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) lane_partial += __shfl_down_sync(0xffffffffu, lane_partial, o2);
        if (lane == 0) {
            const unsigned long long prod = (unsigned long long) lane_partial * (unsigned long long)(long long) srow[b];
            total += (unsigned long long) arshift_i64_floor((long long) prod, frac);
        }
    }
    if (lane == 0) out[t * out_f + o] = (long long) total;
}

// ---- M2: RMSNorm (byte-exact port of bonsai_rmsnorm_i64, tools/bonsai_q1_kernel.c:173-258) ---------------
// nvcc supports __int128 in device code (emulated), so this is a near-verbatim port: 128-bit sum-of-squares
// (the residual stream is unbounded across 36 layers → exceeds int64), bit-exact integer isqrt, floor-division
// (toward −∞, NOT truncation), and the rc=4 "refuse when 128 bits insufficient / gain leaves the envelope"
// fallback so the GPU declines EXACTLY when the CPU does (→ wrapper None → CPU big-int oracle). One thread per
// row; the per-row ssq is a serial integer sum identical to the CPU loop (order-free, so this is byte-exact).

__device__ __forceinline__ unsigned __int128 g_abs_i64_u128(long long v) {
    if (v >= 0) return (unsigned __int128) v;
    return (unsigned __int128)(-(v + 1)) + 1u;        // |INT64_MIN| safe (two's-complement)
}

// Returns 0 (and leaves *acc unchanged) if v² or the running sum would exceed 128 bits — the rc=4 trigger.
__device__ __forceinline__ int g_add_square_u128(unsigned __int128* acc, long long v) {
    const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
    unsigned __int128 a = g_abs_i64_u128(v);
    if (a != 0 && a > max_u128 / a) return 0;
    unsigned __int128 term = a * a;
    if (*acc > max_u128 - term) return 0;
    *acc += term;
    return 1;
}

__device__ __forceinline__ unsigned long long g_isqrt_u128(unsigned __int128 n) {
    unsigned __int128 res = 0;
    unsigned __int128 bit = (unsigned __int128) 1 << 126;
    while (bit > n) bit >>= 2;
    while (bit != 0) {
        if (n >= res + bit) { n -= res + bit; res = (res >> 1) + bit; }
        else res >>= 1;
        bit >>= 2;
    }
    return (unsigned long long) res;
}

__device__ __forceinline__ __int128 g_floor_div_i128_u64(__int128 n, unsigned long long d) {
    __int128 denom = (__int128) d;
    __int128 q = n / denom, r = n % denom;
    if (r != 0 && n < 0) q -= 1;                       // floor toward −∞ (C truncates toward 0)
    return q;
}

__device__ __forceinline__ __int128 g_floor_shift_i128(__int128 v, long long shift) {
    if (shift <= 0) return v;
    if (shift >= 126) return v < 0 ? -1 : 0;
    __int128 denom = (__int128) 1 << shift;
    __int128 q = v / denom, r = v % denom;
    if (r != 0 && v < 0) q -= 1;
    return q;
}

__device__ __forceinline__ int g_i128_to_i64(__int128 v, long long* out) {
    const __int128 lo = -((__int128) 0x7fffffffffffffffLL) - 1;
    const __int128 hi = (__int128) 0x7fffffffffffffffLL;
    if (v < lo || v > hi) return 0;
    *out = (long long) v;
    return 1;
}

// One thread per row. Writes per-row rc into row_rc[r] (0 ok, 4 overflow/refuse) if row_rc != null; and/or
// atomicOr's a shared `overflow` flag if non-null (the monolith uses one flag across all rmsnorm calls).
__global__ void rmsnorm_kernel(const long long* __restrict__ x, long long rows, long long cols,
                               long long frac, long long eps, const long long* __restrict__ gain,
                               long long* __restrict__ out, int* __restrict__ row_rc,
                               int* __restrict__ overflow) {
    const long long r = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const long long* row = x + r * cols;
    long long* dst = out + r * cols;
    const __int128 fp = (__int128) 1 << frac;
    int local_rc = 0;

    unsigned __int128 ssq = 0;
    for (long long c = 0; c < cols; ++c) {
        if (!g_add_square_u128(&ssq, row[c])) { local_rc = 4; break; }
    }
    if (local_rc == 0) {
        const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
        unsigned __int128 mean = ssq / (unsigned __int128) cols;
        if (mean > max_u128 - (unsigned __int128) eps) local_rc = 4;
        else mean += (unsigned __int128) eps;
        unsigned long long rms = local_rc == 0 ? g_isqrt_u128(mean) : 0;
        if (rms == 0) local_rc = 4;
        if (local_rc == 0 && gain) {
            // Same fail-loud envelope as the oracle: refuse if max|normalized|·max|gain| > INT64_MAX.
            unsigned __int128 max_norm = 0, max_gain = 0;
            for (long long c = 0; c < cols; ++c) {
                __int128 nrm = g_floor_div_i128_u64((__int128) row[c] * fp, rms);
                unsigned __int128 an = (unsigned __int128)(nrm < 0 ? -nrm : nrm);
                if (an > max_norm) max_norm = an;
                long long gc = gain[c];
                unsigned __int128 ag = (unsigned __int128)(gc < 0 ? -(__int128) gc : (__int128) gc);
                if (ag > max_gain) max_gain = ag;
            }
            const unsigned __int128 i64max = (unsigned __int128) 0x7fffffffffffffffLL;
            if (max_norm != 0 && max_gain > i64max / max_norm) local_rc = 4;
        }
        for (long long c = 0; local_rc == 0 && c < cols; ++c) {
            __int128 normalized = g_floor_div_i128_u64((__int128) row[c] * fp, rms);
            __int128 y = normalized;
            if (gain) y = g_floor_shift_i128(normalized * (__int128) gain[c], frac);
            if (!g_i128_to_i64(y, &dst[c])) { local_rc = 4; break; }
        }
    }
    if (row_rc) row_rc[r] = local_rc;
    if (overflow && local_rc != 0) atomicOr(overflow, 1);
}

// ---- M3: prefill attention (byte-exact port of bonsai_attention_prefill_i64, bonsai_q1_kernel.c:1060) ----
// Integer score → integer softmax (poly 2^-f, NO expf) → @V, causal M=N. inv_sqrt_fp / log2e / d_clip are
// computed HOST-side and passed in (never an on-device sqrtf). Fail-loud overflow (q@Kᵀ and probs@V bounds,
// division-form so the 128-bit test can't wrap) sets a flag → host returns rc 2 → wrapper None → CPU oracle.
// One thread per (h,m); per-thread scratch row of length L in global memory (the two-pass softmax needs the
// stored scores). Byte-identical to the CPU: same arshift floors, same poly, same floor-div normalize, same
// accumulation order over keys [0, Lv).

__device__ __forceinline__ long long g_exp2_neg_fixed(long long u, long long frac) {
    const long long C0 = 65536, C1 = 45426, C2 = 15743, C3 = 3638;   // matches _exp2_neg_fixed / bonsai_exp2_neg_fixed
    long long FP = (long long) 1 << frac, mask = FP - 1;
    long long k = u >> frac, f = u & mask;
    long long shift = 16 - frac, c0, c1, c2, c3;
    if (shift >= 0) { c0 = C0 >> shift; c1 = C1 >> shift; c2 = C2 >> shift; c3 = C3 >> shift; }
    else { long long s = -shift; c0 = C0 << s; c1 = C1 << s; c2 = C2 << s; c3 = C3 << s; }
    long long f2 = (f * f) >> frac, f3 = (f2 * f) >> frac;
    long long poly = c0 - ((c1 * f) >> frac) + ((c2 * f2) >> frac) - ((c3 * f3) >> frac);
    if (poly < 0) poly = 0;
    long long kk = k < 63 ? k : 63;
    return poly >> kk;
}

// One thread per (h, m). maxk/maxv are per-kv (Hkv) host-computed maxabs. scratch is H*M*L int64 (row per (h,m)).
__global__ void attention_prefill_kernel(
        const long long* __restrict__ q, const long long* __restrict__ k, const long long* __restrict__ v,
        long long H, long long Hkv, long long hd, long long M, long long L, long long start,
        long long frac, long long inv_sqrt_fp, long long log2e, long long d_clip,
        const unsigned long long* __restrict__ maxk, const unsigned long long* __restrict__ maxv,
        long long* __restrict__ out, long long* __restrict__ scratch, int* __restrict__ overflow) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= H * M) return;
    const long long h = gid / M, m = gid % M;
    if (*overflow) return;

    const long long rep = H / Hkv, kv = h / rep;
    const long long Lv = start + m + 1;                  // causal: query m sees keys [0, Lv)
    const long long* qh = q + (h * M + m) * hd;
    const long long* kh = k + kv * L * hd;
    const long long* vh = v + kv * L * hd;
    long long* sc = scratch + gid * L;
    const unsigned __int128 i64max = (unsigned __int128) 0x7fffffffffffffffLL;

    // q@Kᵀ fail-loud bound (contract hd), division-form so it can't wrap
    unsigned long long maxq = 0;
    for (long long d = 0; d < hd; ++d) {
        long long vv = qh[d];
        unsigned long long a = vv < 0 ? ((unsigned long long)(~(unsigned long long) vv) + 1u) : (unsigned long long) vv;
        if (a > maxq) maxq = a;
    }
    if ((unsigned __int128) maxq * (unsigned __int128) maxk[kv] > i64max / (unsigned __int128) hd) {
        *overflow = 1; return;
    }
    long long mx = (long long) 0x8000000000000000LL;     // INT64_MIN
    for (long long j = 0; j < Lv; ++j) {
        const long long* kj = kh + j * hd;
        long long dot = 0;
        for (long long d = 0; d < hd; ++d) dot += qh[d] * kj[d];
        long long s = arshift_i64_floor(dot, frac);
        s = arshift_i64_floor(s * inv_sqrt_fp, frac);
        sc[j] = s;
        if (s > mx) mx = s;
    }
    long long Z = 0;
    for (long long j = 0; j < Lv; ++j) {
        long long d = mx - sc[j];
        if (d > d_clip) d = d_clip;
        long long u = (d * log2e) >> frac;
        long long e = g_exp2_neg_fixed(u, frac);
        sc[j] = e;
        Z += e;
    }
    for (long long j = 0; j < Lv; ++j) sc[j] = Z ? ((sc[j] << frac) / Z) : 0;

    // probs@V fail-loud bound (contract Lv)
    unsigned long long maxp = 0;
    for (long long j = 0; j < Lv; ++j) {
        long long vv = sc[j];
        unsigned long long a = vv < 0 ? ((unsigned long long)(~(unsigned long long) vv) + 1u) : (unsigned long long) vv;
        if (a > maxp) maxp = a;
    }
    if ((unsigned __int128) maxp * (unsigned __int128) maxv[kv] > i64max / (unsigned __int128) Lv) {
        *overflow = 1; return;
    }
    long long* oh = out + (h * M + m) * hd;
    for (long long d = 0; d < hd; ++d) {
        long long acc = 0;
        for (long long j = 0; j < Lv; ++j) acc += sc[j] * vh[j * hd + d];
        oh[d] = arshift_i64_floor(acc, frac);
    }
}

// ---- M3 true-residency: elementwise device kernels so the residual stream never leaves the GPU ------------
// Each mirrors its CPU op bit-for-bit; all integer, mod-2⁶⁴ where the oracle wraps, arithmetic floor-shift.

__device__ __forceinline__ long long g_u64_to_i64(unsigned long long u) {
    if (u <= 0x7fffffffffffffffULL) return (long long) u;
    unsigned long long mag = (~u) + 1ULL;
    if (u == 0x8000000000000000ULL) return (long long) 0x8000000000000000LL;
    return -(long long) mag;
}

// RoPE NeoX rotate-half on (Hh, T, hd), cos/sin (T, half). out0=(x0*c-x1*s)>>frac, out1=(x0*s+x1*c)>>frac.
// One thread per (h,t); in-place safe (reads x0,x1 for each e before writing e and half+e). Matches
// apply_rope_fixed_neox (rope_v2.py). (No overflow guard: committed RoPE'd Q/K stay ~2^41 « int64.)
__global__ void rope_kernel(long long* __restrict__ x, const long long* __restrict__ cos,
                            const long long* __restrict__ sin, long long Hh, long long T, long long hd,
                            long long frac) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= Hh * T) return;
    const long long t = gid % T;
    long long* row = x + gid * hd;
    const long long half = hd / 2;
    const long long* c = cos + t * half;
    const long long* s = sin + t * half;
    for (long long e = 0; e < half; ++e) {
        long long x0 = row[e], x1 = row[half + e];
        row[e]        = arshift_i64_floor(x0 * c[e] - x1 * s[e], frac);
        row[half + e] = arshift_i64_floor(x0 * s[e] + x1 * c[e], frac);
    }
}

// (T, H*hd) -> (H, T, hd):  out[h,t,e] = in[t, h*hd+e].  One thread per (gid over H*T*hd).
__global__ void transpose_thd_to_htd(const long long* __restrict__ in, long long* __restrict__ out,
                                     long long T, long long H, long long hd) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= H * T * hd) return;
    const long long e = gid % hd, t = (gid / hd) % T, h = gid / (hd * T);
    out[gid] = in[t * (H * hd) + h * hd + e];
}

// (H, T, hd) -> (T, H*hd):  out[t, h*hd+e] = in[h,t,e].  One thread per (gid over T*H*hd).
__global__ void transpose_htd_to_thd(const long long* __restrict__ in, long long* __restrict__ out,
                                     long long H, long long T, long long hd) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= T * H * hd) return;
    const long long e = gid % hd, h = (gid / hd) % H, t = gid / (hd * H);
    out[t * (H * hd) + h * hd + e] = in[h * (T * hd) + t * hd + e];
}

// Fixed-point SiLU: out[i] = (x*sigmoid(x))>>frac. Verbatim port of bonsai_silu_i64. log2e/d_clip host-passed.
__global__ void silu_kernel(const long long* __restrict__ x, long long* __restrict__ out, long long n,
                            long long frac, long long log2e, long long d_clip) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long xi = x[i];
    long long m = xi > 0 ? xi : 0;
    long long d0 = m;
    long long d1 = g_u64_to_i64((unsigned long long) m - (unsigned long long) xi);
    if (d0 > d_clip) d0 = d_clip;
    if (d1 > d_clip) d1 = d_clip;
    long long e0 = g_exp2_neg_fixed((d0 * log2e) >> frac, frac);
    long long e1 = g_exp2_neg_fixed((d1 * log2e) >> frac, frac);
    long long z = e0 + e1;
    long long sig = z ? ((e1 << frac) / z) : 0;
    unsigned long long prod = (unsigned long long) xi * (unsigned long long) sig;
    out[i] = arshift_i64_floor(g_u64_to_i64(prod), frac);
}

// out[i] = ((a*b) mod 2^64) >> frac  (== (silu(gate)*up)>>frac).  numpy int64 * then arithmetic >>.
__global__ void mulshift_kernel(const long long* __restrict__ a, const long long* __restrict__ b,
                                long long* __restrict__ out, long long n, long long frac) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = arshift_i64_floor(g_u64_to_i64((unsigned long long) a[i] * (unsigned long long) b[i]), frac);
}

// out[i] = a[i] + b[i]  (int64 wrap; residual add).
__global__ void add_kernel(const long long* __restrict__ a, const long long* __restrict__ b,
                           long long* __restrict__ out, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = g_u64_to_i64((unsigned long long) a[i] + (unsigned long long) b[i]);
}

// maxabs over each kv's (L,hd) block of a (Hkv, L, hd) tensor -> maxout[kv]. One thread per kv.
__global__ void maxabs_per_kv_kernel(const long long* __restrict__ p, long long Hkv, long long L, long long hd,
                                     unsigned long long* __restrict__ maxout) {
    const long long kv = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (kv >= Hkv) return;
    const long long* base = p + kv * L * hd;
    unsigned long long m = 0;
    for (long long i = 0; i < L * hd; ++i) {
        long long vv = base[i];
        unsigned long long a = vv < 0 ? ((unsigned long long)(~(unsigned long long) vv) + 1u) : (unsigned long long) vv;
        if (a > m) m = a;
    }
    maxout[kv] = m;
}

// ---- M=B fully-resident batched decode (widen the batch-serving plateau: KV + RMSNorm/RoPE/attn on device) -
// Per-(b,kv) maxabs over the valid cache range [0, lengths[b]) of a padded (B,Hkv,cap,hd) cache.
__global__ void maxabs_bkv_kernel(const long long* __restrict__ c, const long long* __restrict__ lengths,
                                  long long B, long long Hkv, long long hd, long long cap,
                                  unsigned long long* __restrict__ out) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= B * Hkv) return;
    const long long b = gid / Hkv, kv = gid % Hkv, Lv = lengths[b];
    const long long* base = c + (b * Hkv + kv) * cap * hd;
    unsigned long long m = 0;
    for (long long i = 0; i < Lv * hd; ++i) {
        long long v = base[i];
        unsigned long long a = v < 0 ? ((unsigned long long)(~(unsigned long long) v) + 1u) : (unsigned long long) v;
        if (a > m) m = a;
    }
    out[gid] = m;
}

// B independent M=1 decode attentions, one thread per (b,h): query q[b,h] attends its sequence's cached keys
// [0, lengths[b]) (no causal mask — a single decode query sees all cached positions). Same integer
// score/softmax/@V math as the prefill kernel (m=0). Cache is padded (B,Hkv,cap,hd); only [0,Lv) is read.
// Byte-identical to attention_decode_batched_native (which is byte-identical to B× M=1 decode attention).
__global__ void attention_decode_batched_kernel(
        const long long* __restrict__ q, const long long* __restrict__ Kc, const long long* __restrict__ Vc,
        const long long* __restrict__ lengths, long long B, long long H, long long Hkv, long long hd, long long cap,
        long long frac, long long inv_sqrt_fp, long long log2e, long long d_clip,
        const unsigned long long* __restrict__ maxk, const unsigned long long* __restrict__ maxv,
        long long* __restrict__ out, long long* __restrict__ scratch, int* __restrict__ overflow) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= B * H) return;
    const long long b = gid / H, h = gid % H;
    if (*overflow) return;
    const long long rep = H / Hkv, kv = h / rep, Lv = lengths[b];
    if (Lv <= 0) { long long* o0 = out + gid * hd; for (long long d = 0; d < hd; ++d) o0[d] = 0; return; }
    const long long* qh = q + gid * hd;
    const long long* Kb = Kc + (b * Hkv + kv) * cap * hd;
    const long long* Vb = Vc + (b * Hkv + kv) * cap * hd;
    long long* sc = scratch + gid * cap;
    const unsigned __int128 i64max = (unsigned __int128) 0x7fffffffffffffffLL;

    unsigned long long maxq = 0;
    for (long long d = 0; d < hd; ++d) {
        long long vv = qh[d];
        unsigned long long a = vv < 0 ? ((unsigned long long)(~(unsigned long long) vv) + 1u) : (unsigned long long) vv;
        if (a > maxq) maxq = a;
    }
    if ((unsigned __int128) maxq * (unsigned __int128) maxk[b * Hkv + kv] > i64max / (unsigned __int128) hd) {
        *overflow = 1; return;
    }
    long long mx = (long long) 0x8000000000000000LL;
    for (long long j = 0; j < Lv; ++j) {
        const long long* kj = Kb + j * hd;
        long long dot = 0;
        for (long long d = 0; d < hd; ++d) dot += qh[d] * kj[d];
        long long s = arshift_i64_floor(dot, frac);
        s = arshift_i64_floor(s * inv_sqrt_fp, frac);
        sc[j] = s;
        if (s > mx) mx = s;
    }
    long long Z = 0;
    for (long long j = 0; j < Lv; ++j) {
        long long d = mx - sc[j];
        if (d > d_clip) d = d_clip;
        long long u = (d * log2e) >> frac;
        long long e = g_exp2_neg_fixed(u, frac);
        sc[j] = e; Z += e;
    }
    for (long long j = 0; j < Lv; ++j) sc[j] = Z ? ((sc[j] << frac) / Z) : 0;
    unsigned long long maxp = 0;
    for (long long j = 0; j < Lv; ++j) {
        long long vv = sc[j];
        unsigned long long a = vv < 0 ? ((unsigned long long)(~(unsigned long long) vv) + 1u) : (unsigned long long) vv;
        if (a > maxp) maxp = a;
    }
    if ((unsigned __int128) maxp * (unsigned __int128) maxv[b * Hkv + kv] > i64max / (unsigned __int128) Lv) {
        *overflow = 1; return;
    }
    long long* oh = out + gid * hd;
    for (long long d = 0; d < hd; ++d) {
        long long acc = 0;
        for (long long j = 0; j < Lv; ++j) acc += sc[j] * Vb[j * hd + d];
        oh[d] = arshift_i64_floor(acc, frac);
    }
}

// Per-sequence-position RoPE for decode: x (B, Hh, hd), each sequence b at absolute position pos[b].
__global__ void rope_decode_kernel(long long* __restrict__ x, const long long* __restrict__ pos,
                                   const long long* __restrict__ cos, const long long* __restrict__ sin,
                                   long long B, long long Hh, long long hd, long long frac) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= B * Hh) return;
    const long long b = gid / Hh;
    const long long half = hd / 2;
    const long long* c = cos + pos[b] * half;
    const long long* s = sin + pos[b] * half;
    long long* row = x + gid * hd;
    for (long long e = 0; e < half; ++e) {
        long long x0 = row[e], x1 = row[half + e];
        row[e]        = arshift_i64_floor(x0 * c[e] - x1 * s[e], frac);
        row[half + e] = arshift_i64_floor(x0 * s[e] + x1 * c[e], frac);
    }
}

// Append the new token's K/V to each sequence's cache slot: cache[b][kv][pos[b]] = src[b][kv]. src is
// (B,Hkv,hd); cache is (B,Hkv,cap,hd). One thread per (b,kv,e).
__global__ void kv_append_kernel(const long long* __restrict__ src, long long* __restrict__ cache,
                                 const long long* __restrict__ pos, long long B, long long Hkv, long long hd,
                                 long long cap) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= B * Hkv * hd) return;
    const long long e = gid % hd, kv = (gid / hd) % Hkv, b = gid / (hd * Hkv);
    cache[((b * Hkv + kv) * cap + pos[b]) * hd + e] = src[(b * Hkv + kv) * hd + e];
}

// Seed sequence b's prefilled KV into the padded decode cache: src (n_layers,Hkv,Lb,hd) -> cache
// (n_layers,B,Hkv,cap,hd) at [li,b,kv,0..Lb). One thread per (li,kv,j<Lb,e).
__global__ void kv_seed_kernel(const long long* __restrict__ src, long long* __restrict__ cache, long long b,
                               long long n_layers, long long B, long long Hkv, long long Lb, long long hd,
                               long long cap) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_layers * Hkv * Lb * hd) return;
    const long long e = gid % hd, j = (gid / hd) % Lb, kv = (gid / (hd * Lb)) % Hkv, li = gid / (hd * Lb * Hkv);
    cache[(((li * B + b) * Hkv + kv) * cap + j) * hd + e] = src[((li * Hkv + kv) * Lb + j) * hd + e];
}

// ---- Weight residency (Phase-2 lever: upload bits/scale ONCE, reuse across all tokens) -------------------
// The M1 per-call upload (bonsai_q1_linear_gpu) re-sends each projection's weight every call — the dominant
// decode cost. Registration uploads bits+scale once and returns an opaque handle; bonsai_q1_apply_resident
// then moves only the per-call activations/output. Byte-identical: the SAME q1_linear_kernel runs, only the
// weight source differs (resident device copy vs freshly-uploaded). Registration is keyed by an explicit
// handle (NOT the host pointer) so it is safe for transient/test weights too.
namespace {
struct ResidentWeight {
    unsigned char* dbits;
    long long* dscale;
    long long out_f;
    long long n_blocks;
};
std::vector<ResidentWeight> g_weights;   // handle = index into this registry (process-global)

// Resident int64 device buffers (gains, cos/sin tables) for the M3 monolith — uploaded once, reused.
struct ResidentBuf { long long* ptr; size_t n; };
std::vector<ResidentBuf> g_buffers;       // handle = index

// Stateful M=B decode context: device KV cache (persists across steps) + per-step scratch + the weight/gain/
// table handles, all stored at create so each step only moves (B,d) in and (B,vocab) logits out.
struct DecodeCtx {
    long long B, n_layers, H, Hkv, hd, d, dff, cap, vocab, frac, eps, inv_sqrt;
    long long *K, *V;                                          // (n_layers, B, Hkv, cap, hd)
    long long *x, *dn, *dq, *dk, *dv, *dah, *dg, *du, *dh, *dtmp, *dscr, *dlog, *dpos, *dlen;
    unsigned long long *dmk, *dmv;
    int *dov;
    // weight handles (n_layers each) + gain buffer handles (n_layers each) + scalar handles
    std::vector<long long> wq,wk,wv,wo,w1,wu,w2,n1g,n2g,qng,kng;
    long long finalg, out_head, cos_h, sin_h;
    bool alive;
};
std::vector<DecodeCtx> g_decode;
}  // namespace

extern "C" {

// Upload an int64 host array to the device once; return a handle (>=0) or -1. Used for gains/cos/sin in the
// resident prefill forward. Free via bonsai_q1_free_weights (which also frees these).
long long bonsai_buf_upload_i64(const long long* host, long long n) {
    if (n <= 0) return -1;
    long long* d = nullptr;
    if (cudaMalloc(&d, (size_t) n * sizeof(long long)) != cudaSuccess) return -1;
    if (cudaMemcpy(d, host, (size_t) n * sizeof(long long), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d); return -1;
    }
    g_buffers.push_back(ResidentBuf{d, (size_t) n});
    return (long long)(g_buffers.size() - 1);
}

// Returns 0 iff a usable CUDA device exists (so gpu_native._load_lib can degrade to the CPU path otherwise).
int bonsai_gpu_available(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess && n > 0) ? 0 : 1;     // 0 = available (matches the rc=0-is-good convention)
}

// Packed-Q1_0 linear x @ W.T. Returns 0 on success; nonzero on any CUDA failure so the Python wrapper
// (gpu_native.q1_apply_gpu) falls back to the CPU native/oracle path rather than aborting a notarized run.
// M1 uploads weights per call (no residency) — correctness milestone; residency/DP4A are perf follow-ups.
int bonsai_q1_linear_gpu(
        const long long* x, const unsigned char* bits, const long long* scale,
        long long tokens, long long out_f, long long n_blocks, long long frac,
        long long* out)
{
    if (tokens <= 0 || out_f <= 0 || n_blocks <= 0) return 1;
    const size_t xsz = (size_t)tokens * (size_t)n_blocks * 128 * sizeof(long long);
    const size_t bsz = (size_t)out_f * (size_t)n_blocks * 16 * sizeof(unsigned char);
    const size_t ssz = (size_t)out_f * (size_t)n_blocks * sizeof(long long);
    const size_t osz = (size_t)tokens * (size_t)out_f * sizeof(long long);

    long long *dx = nullptr, *ds = nullptr, *dout = nullptr;
    unsigned char *db = nullptr;
    bool ok = true;
    int rc = 1;

    ok = ok && (cudaMalloc(&dx, xsz) == cudaSuccess);
    ok = ok && (cudaMalloc(&db, bsz) == cudaSuccess);
    ok = ok && (cudaMalloc(&ds, ssz) == cudaSuccess);
    ok = ok && (cudaMalloc(&dout, osz) == cudaSuccess);
    ok = ok && (cudaMemcpy(dx, x, xsz, cudaMemcpyHostToDevice) == cudaSuccess);
    ok = ok && (cudaMemcpy(db, bits, bsz, cudaMemcpyHostToDevice) == cudaSuccess);
    ok = ok && (cudaMemcpy(ds, scale, ssz, cudaMemcpyHostToDevice) == cudaSuccess);

    if (ok) {
        const long long total_warps = tokens * out_f;
        const int threads = 128;                     // 4 warps/block; blockDim % 32 == 0 (uniform warp return)
        const unsigned long long nblk =
            ((unsigned long long)total_warps * 32ULL + (threads - 1)) / (unsigned long long)threads;
        q1_linear_kernel<<<(unsigned int)nblk, threads>>>(dx, db, ds, tokens, out_f, n_blocks, frac, dout);
        ok = ok && (cudaGetLastError() == cudaSuccess);
        ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
    }
    ok = ok && (cudaMemcpy(out, dout, osz, cudaMemcpyDeviceToHost) == cudaSuccess);
    if (ok) rc = 0;

    if (dx) cudaFree(dx);
    if (db) cudaFree(db);
    if (ds) cudaFree(ds);
    if (dout) cudaFree(dout);
    return rc;
}

// Standalone DP4A Q1 apply (per-call upload; for the parity gate + microbench). L is chosen by the caller
// (4 for the committed envelope, 8 unconditionally). Byte-identical to q1_linear_ref. rc 0 ok, 1 bad, 2 cuda.
int bonsai_q1_linear_dp4a_gpu(const long long* x, const unsigned char* bits, const long long* scale,
                              long long tokens, long long out_f, long long n_blocks, long long frac,
                              int L, long long* out) {
    const bool warp = L < 0; const int LL = warp ? -L : L;       // L<0 selects the warp-per-output variant
    if (tokens <= 0 || out_f <= 0 || n_blocks <= 0 || LL < 1 || LL > 8) return 1;
    const long long K = n_blocks * 128;
    const size_t xsz = (size_t) tokens * K * sizeof(long long);
    const size_t bsz = (size_t) out_f * n_blocks * 16;
    const size_t ssz = (size_t) out_f * n_blocks * sizeof(long long);
    const size_t osz = (size_t) tokens * out_f * sizeof(long long);
    const size_t dsz = (size_t) LL * tokens * K;                 // int8 digits
    long long *dx=nullptr,*ds=nullptr,*dout=nullptr; unsigned char* db=nullptr; signed char* dd=nullptr;
    bool ok=true; int rc=1;
    ok=ok&&(cudaMalloc(&dx,xsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&db,bsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&ds,ssz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dout,osz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dd,dsz)==cudaSuccess);
    ok=ok&&(cudaMemcpy(dx,x,xsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(db,bits,bsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(ds,scale,ssz,cudaMemcpyHostToDevice)==cudaSuccess);
    if (ok) {
        const int TPB=128;
        const bool prof = getenv("BONSAI_GPU_PROFILE") != nullptr;
        cudaEvent_t e0,e1,e2; if (prof){cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);cudaEventRecord(e0);}
        q1_digits_kernel<<<(unsigned)((tokens*K+TPB-1)/TPB),TPB>>>(dx,tokens,K,LL,dd);
        if (prof) cudaEventRecord(e1);
        if (warp)
            q1_dp4a_warp_kernel<<<(unsigned)((tokens*out_f*32+TPB-1)/TPB),TPB>>>(dd,db,ds,tokens,out_f,n_blocks,frac,LL,dout);
        else
            q1_dp4a_apply_kernel<<<(unsigned)((tokens*out_f+TPB-1)/TPB),TPB>>>(dd,db,ds,tokens,out_f,n_blocks,frac,LL,dout);
        ok=ok&&(cudaGetLastError()==cudaSuccess);
        if (prof){cudaEventRecord(e2);cudaEventSynchronize(e2);float md=0,ma=0;cudaEventElapsedTime(&md,e0,e1);cudaEventElapsedTime(&ma,e1,e2);
            fprintf(stderr,"[dp4a-prof] digits=%.2fms apply=%.2fms (L=%d warp=%d)\n",md,ma,LL,(int)warp);
            cudaEventDestroy(e0);cudaEventDestroy(e1);cudaEventDestroy(e2);}
        ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);
    }
    ok=ok&&(cudaMemcpy(out,dout,osz,cudaMemcpyDeviceToHost)==cudaSuccess);
    if (ok) rc=0; else rc=2;
    if (dx) cudaFree(dx); if (db) cudaFree(db); if (ds) cudaFree(ds);
    if (dout) cudaFree(dout); if (dd) cudaFree(dd);
    return rc;
}

// Upload one projection's bits+scale to the device ONCE; return a non-negative handle, or -1 on failure.
// out_f, n_blocks describe the weight (bits: out_f*n_blocks*16 uint8; scale: out_f*n_blocks int64).
long long bonsai_q1_register_weight(const unsigned char* bits, const long long* scale,
                                    long long out_f, long long n_blocks) {
    if (out_f <= 0 || n_blocks <= 0) return -1;
    const size_t bsz = (size_t)out_f * (size_t)n_blocks * 16;
    const size_t ssz = (size_t)out_f * (size_t)n_blocks * sizeof(long long);
    unsigned char* dbits = nullptr;
    long long* dscale = nullptr;
    if (cudaMalloc(&dbits, bsz) != cudaSuccess) return -1;
    if (cudaMalloc(&dscale, ssz) != cudaSuccess) { cudaFree(dbits); return -1; }
    if (cudaMemcpy(dbits, bits, bsz, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dscale, scale, ssz, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dbits); cudaFree(dscale); return -1;
    }
    g_weights.push_back(ResidentWeight{dbits, dscale, out_f, n_blocks});
    return (long long)(g_weights.size() - 1);
}

// Apply a registered weight to fresh activations: uploads x, runs the SAME kernel against the resident
// bits/scale, downloads out. Returns 0 on success; nonzero -> Python wrapper falls back to CPU.
int bonsai_q1_apply_resident(long long handle, const long long* x, long long tokens,
                             long long frac, long long* out) {
    if (handle < 0 || (size_t)handle >= g_weights.size() || tokens <= 0) return 1;
    const ResidentWeight w = g_weights[(size_t)handle];
    const long long out_f = w.out_f, n_blocks = w.n_blocks;
    const size_t xsz = (size_t)tokens * (size_t)n_blocks * 128 * sizeof(long long);
    const size_t osz = (size_t)tokens * (size_t)out_f * sizeof(long long);
    long long *dx = nullptr, *dout = nullptr;
    bool ok = true;
    int rc = 1;
    ok = ok && (cudaMalloc(&dx, xsz) == cudaSuccess);
    ok = ok && (cudaMalloc(&dout, osz) == cudaSuccess);
    ok = ok && (cudaMemcpy(dx, x, xsz, cudaMemcpyHostToDevice) == cudaSuccess);
    if (ok) {
        const long long total_warps = tokens * out_f;
        const int threads = 128;
        const unsigned long long nblk =
            ((unsigned long long)total_warps * 32ULL + (threads - 1)) / (unsigned long long)threads;
        q1_linear_kernel<<<(unsigned int)nblk, threads>>>(dx, w.dbits, w.dscale, tokens, out_f, n_blocks, frac, dout);
        ok = ok && (cudaGetLastError() == cudaSuccess);
        ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
    }
    ok = ok && (cudaMemcpy(out, dout, osz, cudaMemcpyDeviceToHost) == cudaSuccess);
    if (ok) rc = 0;
    if (dx) cudaFree(dx);
    if (dout) cudaFree(dout);
    return rc;
}

// M2: RMSNorm of fixed-point rows. Byte-identical to fixed_point_rmsnorm / bonsai_rmsnorm_i64.
// gain may be NULL (no gain). Returns 0 on success; 4 if any row overflowed 128 bits or left the gain
// envelope (→ wrapper None → CPU big-int oracle); 1 on bad args / CUDA failure.
int bonsai_rmsnorm_gpu(const long long* x, long long rows, long long cols, long long frac, long long eps,
                       const long long* gain, long long* out) {
    if (!x || !out || rows < 0 || cols <= 0 || frac < 0 || frac > 62 || eps < 0) return 1;
    if (rows == 0) return 0;
    const size_t xsz = (size_t)rows * (size_t)cols * sizeof(long long);
    const size_t gsz = (size_t)cols * sizeof(long long);
    const size_t rcsz = (size_t)rows * sizeof(int);
    long long *dx = nullptr, *dout = nullptr, *dgain = nullptr;
    int *drc = nullptr, *hrc = nullptr;
    bool ok = true;
    int rc = 1;
    ok = ok && (cudaMalloc(&dx, xsz) == cudaSuccess);
    ok = ok && (cudaMalloc(&dout, xsz) == cudaSuccess);
    ok = ok && (cudaMalloc(&drc, rcsz) == cudaSuccess);
    ok = ok && (cudaMemcpy(dx, x, xsz, cudaMemcpyHostToDevice) == cudaSuccess);
    if (ok && gain) {
        ok = ok && (cudaMalloc(&dgain, gsz) == cudaSuccess);
        ok = ok && (cudaMemcpy(dgain, gain, gsz, cudaMemcpyHostToDevice) == cudaSuccess);
    }
    if (ok) {
        const int threads = 64;
        const unsigned int nblk = (unsigned int)((rows + threads - 1) / threads);
        rmsnorm_kernel<<<nblk, threads>>>(dx, rows, cols, frac, eps, dgain, dout, drc, nullptr);
        ok = ok && (cudaGetLastError() == cudaSuccess);
        ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
    }
    if (ok) {
        hrc = (int*) malloc(rcsz);
        ok = ok && (hrc != nullptr);
        ok = ok && (cudaMemcpy(hrc, drc, rcsz, cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    int any = 0;
    if (ok) for (long long r = 0; r < rows; ++r) if (hrc[r] != 0) { any = hrc[r]; break; }
    if (ok && any == 0)
        ok = (cudaMemcpy(out, dout, xsz, cudaMemcpyDeviceToHost) == cudaSuccess);
    if (ok) rc = (any != 0) ? any : 0;                 // 4 → caller falls back to the CPU oracle

    if (hrc) free(hrc);
    if (dx) cudaFree(dx);
    if (dout) cudaFree(dout);
    if (dgain) cudaFree(dgain);
    if (drc) cudaFree(drc);
    return rc;
}

// M3: causal M=N prefill attention. q:(H,M,hd), k/v:(Hkv,L,hd), L==start+M. Byte-identical to
// bonsai_attention_prefill_i64 / the NumPy causal path. Returns 0 ok, 1 bad args, 2 overflow (→ wrapper None).
int bonsai_attention_prefill_gpu(
        const long long* q, const long long* k, const long long* v,
        long long H, long long Hkv, long long hd, long long M, long long L, long long start,
        long long frac, long long inv_sqrt_fp, long long* out) {
    if (!q || !k || !v || !out || H <= 0 || Hkv <= 0 || hd <= 0 || M <= 0 || L <= 0 || start < 0 ||
        frac < 1 || frac > 29 || H % Hkv != 0 || L != start + M) return 1;
    // host-side scalars (no on-device sqrt): log2e, d_clip — mirror bonsai_scaled_log2e / the d_clip formula
    const long long LOG2E_Q16 = 94548;
    long long shift = 16 - frac;
    long long log2e = shift >= 0 ? (LOG2E_Q16 >> shift) : (LOG2E_Q16 << (-shift));
    if (log2e <= 0) return 1;
    long long dca = ((frac + 2) << (2 * frac)) / log2e;
    long long dcb = ((long long) 1 << 62) / log2e;
    long long d_clip = dca < dcb ? dca : dcb;

    // host-side per-kv maxabs (order-free max) for the fail-loud bound
    unsigned long long *hmaxk = (unsigned long long*) malloc((size_t) Hkv * sizeof(unsigned long long));
    unsigned long long *hmaxv = (unsigned long long*) malloc((size_t) Hkv * sizeof(unsigned long long));
    if (!hmaxk || !hmaxv) { free(hmaxk); free(hmaxv); return 2; }
    for (long long kv = 0; kv < Hkv; ++kv) {
        unsigned long long mk = 0, mv = 0;
        const long long* kbase = k + (size_t) kv * L * hd;
        const long long* vbase = v + (size_t) kv * L * hd;
        for (size_t i = 0; i < (size_t) L * hd; ++i) {
            long long a = kbase[i]; unsigned long long ua = a < 0 ? ((unsigned long long)(~(unsigned long long)a)+1u) : (unsigned long long)a;
            if (ua > mk) mk = ua;
            long long b = vbase[i]; unsigned long long ub = b < 0 ? ((unsigned long long)(~(unsigned long long)b)+1u) : (unsigned long long)b;
            if (ub > mv) mv = ub;
        }
        hmaxk[kv] = mk; hmaxv[kv] = mv;
    }

    const size_t qsz = (size_t) H * M * hd * sizeof(long long);
    const size_t kvsz = (size_t) Hkv * L * hd * sizeof(long long);
    const size_t scrsz = (size_t) H * M * L * sizeof(long long);
    const size_t mxsz = (size_t) Hkv * sizeof(unsigned long long);
    long long *dq=nullptr,*dk=nullptr,*dv=nullptr,*dout=nullptr,*dscr=nullptr;
    unsigned long long *dmk=nullptr,*dmv=nullptr;
    int *dov=nullptr; int hov=0, rc=1; bool ok=true;
    ok = ok && (cudaMalloc(&dq,qsz)==cudaSuccess);
    ok = ok && (cudaMalloc(&dk,kvsz)==cudaSuccess);
    ok = ok && (cudaMalloc(&dv,kvsz)==cudaSuccess);
    ok = ok && (cudaMalloc(&dout,qsz)==cudaSuccess);
    ok = ok && (cudaMalloc(&dscr,scrsz)==cudaSuccess);
    ok = ok && (cudaMalloc(&dmk,mxsz)==cudaSuccess);
    ok = ok && (cudaMalloc(&dmv,mxsz)==cudaSuccess);
    ok = ok && (cudaMalloc(&dov,sizeof(int))==cudaSuccess);
    ok = ok && (cudaMemcpy(dq,q,qsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok = ok && (cudaMemcpy(dk,k,kvsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok = ok && (cudaMemcpy(dv,v,kvsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok = ok && (cudaMemcpy(dmk,hmaxk,mxsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok = ok && (cudaMemcpy(dmv,hmaxv,mxsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok = ok && (cudaMemset(dov,0,sizeof(int))==cudaSuccess);
    if (ok) {
        const int threads=64;
        const unsigned int nblk=(unsigned int)((H*M + threads-1)/threads);
        attention_prefill_kernel<<<nblk,threads>>>(dq,dk,dv,H,Hkv,hd,M,L,start,frac,inv_sqrt_fp,
                                                    log2e,d_clip,dmk,dmv,dout,dscr,dov);
        ok = ok && (cudaGetLastError()==cudaSuccess);
        ok = ok && (cudaDeviceSynchronize()==cudaSuccess);
    }
    ok = ok && (cudaMemcpy(&hov,dov,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
    if (ok && hov==0) ok = (cudaMemcpy(out,dout,qsz,cudaMemcpyDeviceToHost)==cudaSuccess);
    if (ok) rc = hov ? 2 : 0;

    free(hmaxk); free(hmaxv);
    if (dq) cudaFree(dq); if (dk) cudaFree(dk); if (dv) cudaFree(dv);
    if (dout) cudaFree(dout); if (dscr) cudaFree(dscr);
    if (dmk) cudaFree(dmk); if (dmv) cudaFree(dmv); if (dov) cudaFree(dov);
    return rc;
}

// Standalone batched M=1 decode attention (per-call upload; for the parity gate). q:(B,H,hd); Kc/Vc padded
// (B,Hkv,cap,hd); lengths:(B,) valid per-seq length. Byte-identical to attention_decode_batched_native.
// Returns 0 ok, 1 bad, 2 overflow.
int bonsai_attention_decode_batched_gpu(
        const long long* q, const long long* Kc, const long long* Vc, const long long* lengths,
        long long B, long long H, long long Hkv, long long hd, long long cap,
        long long frac, long long inv_sqrt_fp, long long* out) {
    if (!q || !Kc || !Vc || !out || B <= 0 || H <= 0 || Hkv <= 0 || hd <= 0 || cap <= 0 ||
        H % Hkv != 0 || frac < 1 || frac > 29) return 1;
    const long long LOG2E_Q16 = 94548; long long shift = 16 - frac;
    long long log2e = shift >= 0 ? (LOG2E_Q16 >> shift) : (LOG2E_Q16 << (-shift));
    if (log2e <= 0) return 1;
    long long dca = ((frac + 2) << (2 * frac)) / log2e, dcb = ((long long) 1 << 62) / log2e;
    long long d_clip = dca < dcb ? dca : dcb;
    const size_t qsz = (size_t) B * H * hd * 8, csz = (size_t) B * Hkv * cap * hd * 8;
    const size_t lsz = (size_t) B * 8, mxsz = (size_t) B * Hkv * 8, scrsz = (size_t) B * H * cap * 8;
    long long *dq=nullptr,*dK=nullptr,*dV=nullptr,*dlen=nullptr,*dout=nullptr,*dscr=nullptr;
    unsigned long long *dmk=nullptr,*dmv=nullptr; int *dov=nullptr, hov=0, rc=1; bool ok=true;
    ok=ok&&(cudaMalloc(&dq,qsz)==cudaSuccess); ok=ok&&(cudaMalloc(&dK,csz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dV,csz)==cudaSuccess); ok=ok&&(cudaMalloc(&dlen,lsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dout,qsz)==cudaSuccess); ok=ok&&(cudaMalloc(&dscr,scrsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dmk,mxsz)==cudaSuccess); ok=ok&&(cudaMalloc(&dmv,mxsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dov,sizeof(int))==cudaSuccess);
    ok=ok&&(cudaMemcpy(dq,q,qsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(dK,Kc,csz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(dV,Vc,csz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(dlen,lengths,lsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemset(dov,0,sizeof(int))==cudaSuccess);
    if (ok) {
        const int T=64;
        maxabs_bkv_kernel<<<(unsigned)((B*Hkv+T-1)/T),T>>>(dK,dlen,B,Hkv,hd,cap,dmk);
        maxabs_bkv_kernel<<<(unsigned)((B*Hkv+T-1)/T),T>>>(dV,dlen,B,Hkv,hd,cap,dmv);
        attention_decode_batched_kernel<<<(unsigned)((B*H+T-1)/T),T>>>(dq,dK,dV,dlen,B,H,Hkv,hd,cap,frac,
                                                inv_sqrt_fp,log2e,d_clip,dmk,dmv,dout,dscr,dov);
        ok=ok&&(cudaGetLastError()==cudaSuccess); ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);
    }
    ok=ok&&(cudaMemcpy(&hov,dov,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
    if (ok && hov==0) ok=(cudaMemcpy(out,dout,qsz,cudaMemcpyDeviceToHost)==cudaSuccess);
    if (ok) rc = hov ? 2 : 0;
    if (dq)cudaFree(dq); if(dK)cudaFree(dK); if(dV)cudaFree(dV); if(dlen)cudaFree(dlen);
    if(dout)cudaFree(dout); if(dscr)cudaFree(dscr); if(dmk)cudaFree(dmk); if(dmv)cudaFree(dmv); if(dov)cudaFree(dov);
    return rc;
}

// Grid helpers shared by the monolith + the batched-decode step (one-thread-per-item vs one-warp-per-output).
static const int MONO_TPB = 64;
static inline unsigned int mono_blocks(long long nthreads) { return (unsigned int)((nthreads + MONO_TPB - 1) / MONO_TPB); }
static inline unsigned int mono_wblocks(long long nwarps) { return (unsigned int)((nwarps * 32 + MONO_TPB - 1) / MONO_TPB); }

// ---- M=B fully-resident batched decode: stateful context (KV persists on device across steps) ------------
// Create the context: allocate the device KV cache + per-step scratch, and stash all weight/gain/table handles
// + scalars so each step only moves (B,d) in / (B,vocab) logits out. Returns a handle (>=0) or -1.
long long bonsai_decode_ctx_create(
        long long B, long long n_layers, long long H, long long Hkv, long long hd, long long d, long long dff,
        long long cap, long long vocab, long long frac, long long eps, long long inv_sqrt,
        const long long* wq, const long long* wk, const long long* wv, const long long* wo,
        const long long* w1, const long long* wu, const long long* w2,
        const long long* n1g, const long long* n2g, const long long* qng, const long long* kng,
        long long finalg, long long out_head, long long cos_h, long long sin_h) {
    if (B <= 0 || n_layers <= 0 || H <= 0 || Hkv <= 0 || hd <= 0 || cap <= 0 || H % Hkv != 0) return -1;
    DecodeCtx c{};
    c.B=B; c.n_layers=n_layers; c.H=H; c.Hkv=Hkv; c.hd=hd; c.d=d; c.dff=dff; c.cap=cap; c.vocab=vocab;
    c.frac=frac; c.eps=eps; c.inv_sqrt=inv_sqrt; c.alive=true;
    c.finalg=finalg; c.out_head=out_head; c.cos_h=cos_h; c.sin_h=sin_h;
    c.wq.assign(wq,wq+n_layers); c.wk.assign(wk,wk+n_layers); c.wv.assign(wv,wv+n_layers);
    c.wo.assign(wo,wo+n_layers); c.w1.assign(w1,w1+n_layers); c.wu.assign(wu,wu+n_layers);
    c.w2.assign(w2,w2+n_layers); c.n1g.assign(n1g,n1g+n_layers); c.n2g.assign(n2g,n2g+n_layers);
    c.qng.assign(qng,qng+n_layers); c.kng.assign(kng,kng+n_layers);
    const size_t kvsz = (size_t)n_layers*B*Hkv*cap*hd*8;
    bool ok = true;
    ok=ok&&(cudaMalloc(&c.K,kvsz)==cudaSuccess); ok=ok&&(cudaMalloc(&c.V,kvsz)==cudaSuccess);
    if (ok) { cudaMemset(c.K,0,kvsz); cudaMemset(c.V,0,kvsz); }
    ok=ok&&(cudaMalloc(&c.x,(size_t)B*d*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dn,(size_t)B*d*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dq,(size_t)B*H*hd*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dk,(size_t)B*Hkv*hd*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dv,(size_t)B*Hkv*hd*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dah,(size_t)B*H*hd*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dg,(size_t)B*dff*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.du,(size_t)B*dff*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dh,(size_t)B*dff*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dtmp,(size_t)B*d*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dscr,(size_t)B*H*cap*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dlog,(size_t)B*vocab*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dpos,(size_t)B*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dlen,(size_t)B*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dmk,(size_t)B*Hkv*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dmv,(size_t)B*Hkv*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&c.dov,sizeof(int))==cudaSuccess);
    if (!ok) {
        cudaFree(c.K);cudaFree(c.V);cudaFree(c.x);cudaFree(c.dn);cudaFree(c.dq);cudaFree(c.dk);cudaFree(c.dv);
        cudaFree(c.dah);cudaFree(c.dg);cudaFree(c.du);cudaFree(c.dh);cudaFree(c.dtmp);cudaFree(c.dscr);
        cudaFree(c.dlog);cudaFree(c.dpos);cudaFree(c.dlen);cudaFree(c.dmk);cudaFree(c.dmv);cudaFree(c.dov);
        return -1;
    }
    g_decode.push_back(c);
    return (long long)(g_decode.size()-1);
}

// Seed sequence b's prefilled KV (host (n_layers,Hkv,Lb,hd)) into the device cache. Returns 0/1.
int bonsai_decode_ctx_seed_seq(long long ctx_h, long long b, const long long* k_src, const long long* v_src,
                               long long Lb) {
    if (ctx_h < 0 || (size_t)ctx_h >= g_decode.size() || !g_decode[(size_t)ctx_h].alive) return 1;
    DecodeCtx& c = g_decode[(size_t)ctx_h];
    if (b < 0 || b >= c.B || Lb < 0 || Lb > c.cap) return 1;
    if (Lb == 0) return 0;
    const size_t n = (size_t)c.n_layers*c.Hkv*Lb*c.hd, sz = n*8;
    long long *tk=nullptr,*tv=nullptr; bool ok=true;
    ok=ok&&(cudaMalloc(&tk,sz)==cudaSuccess); ok=ok&&(cudaMalloc(&tv,sz)==cudaSuccess);
    ok=ok&&(cudaMemcpy(tk,k_src,sz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(tv,v_src,sz,cudaMemcpyHostToDevice)==cudaSuccess);
    if (ok) {
        const int T=128; const unsigned nb=(unsigned)((n+T-1)/T);
        kv_seed_kernel<<<nb,T>>>(tk,c.K,b,c.n_layers,c.B,c.Hkv,Lb,c.hd,c.cap);
        kv_seed_kernel<<<nb,T>>>(tv,c.V,b,c.n_layers,c.B,c.Hkv,Lb,c.hd,c.cap);
        ok=ok&&(cudaGetLastError()==cudaSuccess); ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);
    }
    if (tk) cudaFree(tk); if (tv) cudaFree(tv);
    return ok ? 0 : 1;
}

// One M=B decode step: x_in (B,d) new-token residual; pos (B,) each seq's absolute position (= cache len before
// append). Writes out_logits (B,vocab). Byte-identical to B× the CPU M=1 decode step. rc 0 ok, 1 bad, 4 overflow.
int bonsai_decode_step(long long ctx_h, const long long* x_in, const long long* pos, long long* out_logits) {
    if (ctx_h < 0 || (size_t)ctx_h >= g_decode.size() || !g_decode[(size_t)ctx_h].alive) return 1;
    DecodeCtx& c = g_decode[(size_t)ctx_h];
    const long long B=c.B,H=c.H,Hkv=c.Hkv,hd=c.hd,d=c.d,dff=c.dff,cap=c.cap,frac=c.frac,eps=c.eps;
    const long long LOG2E_Q16=94548; long long shift=16-frac;
    long long log2e = shift>=0 ? (LOG2E_Q16>>shift) : (LOG2E_Q16<<(-shift));
    if (log2e<=0) return 1;
    long long dca=((frac+2)<<(2*frac))/log2e, dcb=((long long)1<<62)/log2e;
    long long d_clip = dca<dcb?dca:dcb, d_clip_silu=((frac+2)<<(2*frac))/log2e;
    auto wbits=[](long long h){return g_weights[(size_t)h].dbits;};
    auto wscale=[](long long h){return g_weights[(size_t)h].dscale;};
    auto wof=[](long long h){return g_weights[(size_t)h].out_f;};
    auto wnb=[](long long h){return g_weights[(size_t)h].n_blocks;};
    auto buf=[](long long h){return g_buffers[(size_t)h].ptr;};
    // lengths = pos+1 (host), then upload pos + lengths
    std::vector<long long> hlen(B); for (long long b=0;b<B;++b) hlen[b]=pos[b]+1;
    bool ok=true;
    ok=ok&&(cudaMemcpy(c.x,x_in,(size_t)B*d*8,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(c.dpos,pos,(size_t)B*8,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(c.dlen,hlen.data(),(size_t)B*8,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemset(c.dov,0,sizeof(int))==cudaSuccess);
    for (long long li=0; ok && li<c.n_layers; ++li) {
        long long* Kli = c.K + (size_t)li*B*Hkv*cap*hd;
        long long* Vli = c.V + (size_t)li*B*Hkv*cap*hd;
        rmsnorm_kernel<<<mono_blocks(B),MONO_TPB>>>(c.x,B,d,frac,eps,buf(c.n1g[li]),c.dn,nullptr,c.dov);
        q1_linear_kernel<<<mono_wblocks(B*wof(c.wq[li])),MONO_TPB>>>(c.dn,wbits(c.wq[li]),wscale(c.wq[li]),B,wof(c.wq[li]),wnb(c.wq[li]),frac,c.dq);
        q1_linear_kernel<<<mono_wblocks(B*wof(c.wk[li])),MONO_TPB>>>(c.dn,wbits(c.wk[li]),wscale(c.wk[li]),B,wof(c.wk[li]),wnb(c.wk[li]),frac,c.dk);
        q1_linear_kernel<<<mono_wblocks(B*wof(c.wv[li])),MONO_TPB>>>(c.dn,wbits(c.wv[li]),wscale(c.wv[li]),B,wof(c.wv[li]),wnb(c.wv[li]),frac,c.dv);
        rmsnorm_kernel<<<mono_blocks(B*H),MONO_TPB>>>(c.dq,B*H,hd,frac,eps,buf(c.qng[li]),c.dq,nullptr,c.dov);
        rmsnorm_kernel<<<mono_blocks(B*Hkv),MONO_TPB>>>(c.dk,B*Hkv,hd,frac,eps,buf(c.kng[li]),c.dk,nullptr,c.dov);
        rope_decode_kernel<<<mono_blocks(B*H),MONO_TPB>>>(c.dq,c.dpos,buf(c.cos_h),buf(c.sin_h),B,H,hd,frac);
        rope_decode_kernel<<<mono_blocks(B*Hkv),MONO_TPB>>>(c.dk,c.dpos,buf(c.cos_h),buf(c.sin_h),B,Hkv,hd,frac);
        kv_append_kernel<<<mono_blocks(B*Hkv*hd),MONO_TPB>>>(c.dk,Kli,c.dpos,B,Hkv,hd,cap);
        kv_append_kernel<<<mono_blocks(B*Hkv*hd),MONO_TPB>>>(c.dv,Vli,c.dpos,B,Hkv,hd,cap);
        maxabs_bkv_kernel<<<mono_blocks(B*Hkv),MONO_TPB>>>(Kli,c.dlen,B,Hkv,hd,cap,c.dmk);
        maxabs_bkv_kernel<<<mono_blocks(B*Hkv),MONO_TPB>>>(Vli,c.dlen,B,Hkv,hd,cap,c.dmv);
        attention_decode_batched_kernel<<<mono_blocks(B*H),MONO_TPB>>>(c.dq,Kli,Vli,c.dlen,B,H,Hkv,hd,cap,frac,
                                                c.inv_sqrt,log2e,d_clip,c.dmk,c.dmv,c.dah,c.dscr,c.dov);
        q1_linear_kernel<<<mono_wblocks(B*wof(c.wo[li])),MONO_TPB>>>(c.dah,wbits(c.wo[li]),wscale(c.wo[li]),B,wof(c.wo[li]),wnb(c.wo[li]),frac,c.dtmp);
        add_kernel<<<mono_blocks(B*d),MONO_TPB>>>(c.x,c.dtmp,c.x,B*d);
        rmsnorm_kernel<<<mono_blocks(B),MONO_TPB>>>(c.x,B,d,frac,eps,buf(c.n2g[li]),c.dn,nullptr,c.dov);
        q1_linear_kernel<<<mono_wblocks(B*wof(c.w1[li])),MONO_TPB>>>(c.dn,wbits(c.w1[li]),wscale(c.w1[li]),B,wof(c.w1[li]),wnb(c.w1[li]),frac,c.dg);
        q1_linear_kernel<<<mono_wblocks(B*wof(c.wu[li])),MONO_TPB>>>(c.dn,wbits(c.wu[li]),wscale(c.wu[li]),B,wof(c.wu[li]),wnb(c.wu[li]),frac,c.du);
        silu_kernel<<<mono_blocks(B*dff),MONO_TPB>>>(c.dg,c.dg,B*dff,frac,log2e,d_clip_silu);
        mulshift_kernel<<<mono_blocks(B*dff),MONO_TPB>>>(c.dg,c.du,c.dh,B*dff,frac);
        q1_linear_kernel<<<mono_wblocks(B*wof(c.w2[li])),MONO_TPB>>>(c.dh,wbits(c.w2[li]),wscale(c.w2[li]),B,wof(c.w2[li]),wnb(c.w2[li]),frac,c.dtmp);
        add_kernel<<<mono_blocks(B*d),MONO_TPB>>>(c.x,c.dtmp,c.x,B*d);
        ok = ok && (cudaGetLastError()==cudaSuccess);
    }
    if (ok) {
        rmsnorm_kernel<<<mono_blocks(B),MONO_TPB>>>(c.x,B,d,frac,eps,buf(c.finalg),c.dn,nullptr,c.dov);
        q1_linear_kernel<<<mono_wblocks(B*wof(c.out_head)),MONO_TPB>>>(c.dn,wbits(c.out_head),wscale(c.out_head),B,wof(c.out_head),wnb(c.out_head),frac,c.dlog);
        ok=ok&&(cudaGetLastError()==cudaSuccess); ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);
    }
    int hov=0;
    ok=ok&&(cudaMemcpy(&hov,c.dov,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
    if (ok && hov==0) ok=(cudaMemcpy(out_logits,c.dlog,(size_t)B*c.vocab*8,cudaMemcpyDeviceToHost)==cudaSuccess);
    if (!ok) return 1;
    return hov ? 4 : 0;
}

void bonsai_decode_ctx_free(long long ctx_h) {
    if (ctx_h < 0 || (size_t)ctx_h >= g_decode.size() || !g_decode[(size_t)ctx_h].alive) return;
    DecodeCtx& c = g_decode[(size_t)ctx_h];
    cudaFree(c.K);cudaFree(c.V);cudaFree(c.x);cudaFree(c.dn);cudaFree(c.dq);cudaFree(c.dk);cudaFree(c.dv);
    cudaFree(c.dah);cudaFree(c.dg);cudaFree(c.du);cudaFree(c.dh);cudaFree(c.dtmp);cudaFree(c.dscr);
    cudaFree(c.dlog);cudaFree(c.dpos);cudaFree(c.dlen);cudaFree(c.dmk);cudaFree(c.dmv);cudaFree(c.dov);
    c.alive=false;
}

// Free all resident weights AND buffers (optional; process exit frees them anyway).
void bonsai_q1_free_weights(void) {
    for (size_t i = 0; i < g_weights.size(); ++i) {
        if (g_weights[i].dbits) cudaFree(g_weights[i].dbits);
        if (g_weights[i].dscale) cudaFree(g_weights[i].dscale);
    }
    g_weights.clear();
    for (size_t i = 0; i < g_buffers.size(); ++i)
        if (g_buffers[i].ptr) cudaFree(g_buffers[i].ptr);
    g_buffers.clear();
}

// ---- M3: monolithic on-device prefill forward (TRUE x-residency) ----------------------------------------
// Runs the full prefill forward on the GPU: the residual `x` lives in a device buffer for the whole pass, so
// NO per-op host↔device transfer happens between projections/norms/attention (the per-op-dispatch regression
// the TRINOTE_GPU_FULL path showed). Only the embedded x uploads once and the final-position logits download.
// Weights are passed by resident handle (g_weights); gains/cos/sin by resident buffer handle (g_buffers).
// Byte-identical to forward(..., last_only=True): every kernel is the same byte-exact op already gated above.
// Returns 0 ok, 1 bad-args, 2 attention overflow, 4 rmsnorm overflow (→ Python redoes the forward on CPU).
//
// Layout per layer mirrors reference_bonsai._forward_impl:
//   n1 = rmsnorm(x, n1g); qkv = q1(n1); reshape→(H/Hkv,T,hd); q/k headnorm; RoPE(q,k);
//   attn = prefill_attn(q,k,v)→(H,T,hd)→(T,H*hd); x += q1_wo(attn);
//   n2 = rmsnorm(x, n2g); h = (silu(q1_w1(n2)) * q1_wu(n2))>>frac; x += q1_w2(h)
// then final rmsnorm + output-head q1 on the LAST row only → logits (vocab,).

int bonsai_prefill_forward_gpu(
        const long long* x_embed, long long T, long long d, long long n_layers,
        long long H, long long Hkv, long long hd, long long dff,
        long long frac, long long eps, long long inv_sqrt_fp,
        const long long* wq, const long long* wk, const long long* wv, const long long* wo,
        const long long* w1, const long long* wu, const long long* w2,
        const long long* n1g, const long long* n2g, const long long* qng, const long long* kng,
        long long finalg, long long out_head, long long cos_h, long long sin_h,
        long long* out_logits, long long* out_k, long long* out_v) {   // out_k/out_v nullable: (n_layers,Hkv,T,hd) KV export
    if (!x_embed || !out_logits || T <= 0 || d <= 0 || n_layers <= 0 || H <= 0 || Hkv <= 0 || hd <= 0 ||
        dff <= 0 || H % Hkv != 0 || frac < 1 || frac > 29) return 1;
    const long long rep = H / Hkv;
    // host-side softmax scalars (no on-device sqrt)
    const long long LOG2E_Q16 = 94548;
    long long shift = 16 - frac;
    long long log2e = shift >= 0 ? (LOG2E_Q16 >> shift) : (LOG2E_Q16 << (-shift));
    if (log2e <= 0) return 1;
    long long dca = ((frac + 2) << (2 * frac)) / log2e;
    long long dcb = ((long long) 1 << 62) / log2e;
    long long d_clip_attn = dca < dcb ? dca : dcb;
    long long d_clip_silu = ((frac + 2) << (2 * frac)) / log2e;   // sigmoid form (no 1<<62 cap)

    auto wbits = [](long long h) { return g_weights[(size_t) h].dbits; };
    auto wscale = [](long long h) { return g_weights[(size_t) h].dscale; };
    auto wof = [](long long h) { return g_weights[(size_t) h].out_f; };
    auto wnb = [](long long h) { return g_weights[(size_t) h].n_blocks; };
    auto buf = [](long long h) { return g_buffers[(size_t) h].ptr; };

    // device scratch (allocate once, reuse across layers)
    long long *dx=nullptr,*dn=nullptr,*dq=nullptr,*dk=nullptr,*dv=nullptr;       // x, normed, qkv flat
    long long *dqh=nullptr,*dkh=nullptr,*dvh=nullptr,*dah=nullptr,*daf=nullptr;  // transposed heads, attn
    long long *dg=nullptr,*du=nullptr,*dh=nullptr,*dtmp=nullptr,*dscr=nullptr;   // ffn, residual tmp, attn scratch
    unsigned long long *dmk=nullptr,*dmv=nullptr; int *dov=nullptr;
    long long *dlog=nullptr;
    bool ok=true; int rc=1, hov=0;
    const size_t Sx=(size_t)T*d, Sq=(size_t)T*H*hd, Skv=(size_t)T*Hkv*hd, Sff=(size_t)T*dff;
    const size_t Shtd=(size_t)H*T*hd, Skhtd=(size_t)Hkv*T*hd, Sscr=(size_t)H*T*T;
    ok=ok&&(cudaMalloc(&dx,Sx*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dn,Sx*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dq,Sq*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dk,Skv*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dv,Skv*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dqh,Shtd*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dkh,Skhtd*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dvh,Skhtd*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dah,Shtd*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&daf,Sq*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dg,Sff*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&du,Sff*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dh,Sff*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dtmp,Sx*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dscr,Sscr*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dmk,(size_t)Hkv*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dmv,(size_t)Hkv*8)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dov,sizeof(int))==cudaSuccess);
    ok=ok&&(cudaMalloc(&dlog,(size_t)wof(out_head)*8)==cudaSuccess);
    ok=ok&&(cudaMemcpy(dx,x_embed,Sx*8,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemset(dov,0,sizeof(int))==cudaSuccess);

    // optional per-kernel-group timing (BONSAI_GPU_PROFILE=1 → cudaEvent timing to stderr). Off by default.
    const bool prof = getenv("BONSAI_GPU_PROFILE") != nullptr;
    cudaEvent_t pe0=nullptr, pe1=nullptr;
    double t_rms=0,t_q1=0,t_attn=0,t_trans=0,t_rope=0,t_elem=0,t_max=0;
    if (prof) { cudaEventCreate(&pe0); cudaEventCreate(&pe1); }
    #define PT(BUCKET, ...) do { if (prof) { cudaEventRecord(pe0); __VA_ARGS__; cudaEventRecord(pe1); \
        cudaEventSynchronize(pe1); float _ms=0; cudaEventElapsedTime(&_ms,pe0,pe1); BUCKET += _ms; } \
        else { __VA_ARGS__; } } while(0)

    for (long long li=0; ok && li<n_layers; ++li) {
        // --- attention block ---
        PT(t_rms,   rmsnorm_kernel<<<mono_blocks(T),MONO_TPB>>>(dx,T,d,frac,eps,buf(n1g[li]),dn,nullptr,dov));
        PT(t_q1,    q1_linear_kernel<<<mono_wblocks(T*wof(wq[li])),MONO_TPB>>>(dn,wbits(wq[li]),wscale(wq[li]),T,wof(wq[li]),wnb(wq[li]),frac,dq));
        PT(t_q1,    q1_linear_kernel<<<mono_wblocks(T*wof(wk[li])),MONO_TPB>>>(dn,wbits(wk[li]),wscale(wk[li]),T,wof(wk[li]),wnb(wk[li]),frac,dk));
        PT(t_q1,    q1_linear_kernel<<<mono_wblocks(T*wof(wv[li])),MONO_TPB>>>(dn,wbits(wv[li]),wscale(wv[li]),T,wof(wv[li]),wnb(wv[li]),frac,dv));
        PT(t_trans, transpose_thd_to_htd<<<mono_blocks(Shtd),MONO_TPB>>>(dq,dqh,T,H,hd));
        PT(t_trans, transpose_thd_to_htd<<<mono_blocks(Skhtd),MONO_TPB>>>(dk,dkh,T,Hkv,hd));
        PT(t_trans, transpose_thd_to_htd<<<mono_blocks(Skhtd),MONO_TPB>>>(dv,dvh,T,Hkv,hd));
        PT(t_rms,   rmsnorm_kernel<<<mono_blocks(H*T),MONO_TPB>>>(dqh,H*T,hd,frac,eps,buf(qng[li]),dqh,nullptr,dov));
        PT(t_rms,   rmsnorm_kernel<<<mono_blocks(Hkv*T),MONO_TPB>>>(dkh,Hkv*T,hd,frac,eps,buf(kng[li]),dkh,nullptr,dov));
        PT(t_rope,  rope_kernel<<<mono_blocks(H*T),MONO_TPB>>>(dqh,buf(cos_h),buf(sin_h),H,T,hd,frac));
        PT(t_rope,  rope_kernel<<<mono_blocks(Hkv*T),MONO_TPB>>>(dkh,buf(cos_h),buf(sin_h),Hkv,T,hd,frac));
        // KV-export: dkh (post head-norm + RoPE K) and dvh (raw V) ARE the per-layer cache.k[li]/cache.v[li]
        // (Hkv,T,hd). Copy each layer's slot to host so generative-decode prefill can seed the KV cache and
        // continue decode byte-identically. Done before attention (dkh/dvh are final and not modified by it).
        if (out_k) cudaMemcpy(out_k + (size_t) li * Skhtd, dkh, Skhtd * 8, cudaMemcpyDeviceToHost);
        if (out_v) cudaMemcpy(out_v + (size_t) li * Skhtd, dvh, Skhtd * 8, cudaMemcpyDeviceToHost);
        PT(t_max,   maxabs_per_kv_kernel<<<mono_blocks(Hkv),MONO_TPB>>>(dkh,Hkv,T,hd,dmk));
        PT(t_max,   maxabs_per_kv_kernel<<<mono_blocks(Hkv),MONO_TPB>>>(dvh,Hkv,T,hd,dmv));
        PT(t_attn,  attention_prefill_kernel<<<mono_blocks(H*T),MONO_TPB>>>(dqh,dkh,dvh,H,Hkv,hd,T,T,0,frac,inv_sqrt_fp,
                                                                log2e,d_clip_attn,dmk,dmv,dah,dscr,dov));
        PT(t_trans, transpose_htd_to_thd<<<mono_blocks(Sq),MONO_TPB>>>(dah,daf,H,T,hd));
        PT(t_q1,    q1_linear_kernel<<<mono_wblocks(T*wof(wo[li])),MONO_TPB>>>(daf,wbits(wo[li]),wscale(wo[li]),T,wof(wo[li]),wnb(wo[li]),frac,dtmp));
        PT(t_elem,  add_kernel<<<mono_blocks(Sx),MONO_TPB>>>(dx,dtmp,dx,Sx));
        // --- ffn block ---
        PT(t_rms,   rmsnorm_kernel<<<mono_blocks(T),MONO_TPB>>>(dx,T,d,frac,eps,buf(n2g[li]),dn,nullptr,dov));
        PT(t_q1,    q1_linear_kernel<<<mono_wblocks(T*wof(w1[li])),MONO_TPB>>>(dn,wbits(w1[li]),wscale(w1[li]),T,wof(w1[li]),wnb(w1[li]),frac,dg));
        PT(t_q1,    q1_linear_kernel<<<mono_wblocks(T*wof(wu[li])),MONO_TPB>>>(dn,wbits(wu[li]),wscale(wu[li]),T,wof(wu[li]),wnb(wu[li]),frac,du));
        PT(t_elem,  silu_kernel<<<mono_blocks(Sff),MONO_TPB>>>(dg,dg,Sff,frac,log2e,d_clip_silu));
        PT(t_elem,  mulshift_kernel<<<mono_blocks(Sff),MONO_TPB>>>(dg,du,dh,Sff,frac));
        PT(t_q1,    q1_linear_kernel<<<mono_wblocks(T*wof(w2[li])),MONO_TPB>>>(dh,wbits(w2[li]),wscale(w2[li]),T,wof(w2[li]),wnb(w2[li]),frac,dtmp));
        PT(t_elem,  add_kernel<<<mono_blocks(Sx),MONO_TPB>>>(dx,dtmp,dx,Sx));
        ok = ok && (cudaGetLastError()==cudaSuccess);
    }
    // final norm on the LAST row, then output head on that single row
    if (ok) {
        long long* dlast = dx + (size_t)(T-1)*d;
        PT(t_rms, rmsnorm_kernel<<<1,MONO_TPB>>>(dlast,1,d,frac,eps,buf(finalg),dn,nullptr,dov));
        PT(t_q1,  q1_linear_kernel<<<mono_wblocks(wof(out_head)),MONO_TPB>>>(dn,wbits(out_head),wscale(out_head),1,wof(out_head),wnb(out_head),frac,dlog));
        ok = ok && (cudaGetLastError()==cudaSuccess);
        ok = ok && (cudaDeviceSynchronize()==cudaSuccess);
    }
    if (prof) {
        fprintf(stderr, "[gpu-prof] rms=%.1fms q1=%.1fms attn=%.1fms trans=%.1fms rope=%.1fms elem=%.1fms maxabs=%.1fms\n",
                t_rms, t_q1, t_attn, t_trans, t_rope, t_elem, t_max);
        cudaEventDestroy(pe0); cudaEventDestroy(pe1);
    }
    #undef PT
    ok = ok && (cudaMemcpy(&hov,dov,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
    if (ok && hov==0) ok = (cudaMemcpy(out_logits,dlog,(size_t)wof(out_head)*8,cudaMemcpyDeviceToHost)==cudaSuccess);
    if (ok) rc = hov ? 4 : 0;       // any rmsnorm/attention overflow → 4 (Python redoes on CPU)

    cudaFree(dx);cudaFree(dn);cudaFree(dq);cudaFree(dk);cudaFree(dv);
    cudaFree(dqh);cudaFree(dkh);cudaFree(dvh);cudaFree(dah);cudaFree(daf);
    cudaFree(dg);cudaFree(du);cudaFree(dh);cudaFree(dtmp);cudaFree(dscr);
    cudaFree(dmk);cudaFree(dmv);cudaFree(dov);cudaFree(dlog);
    return rc;
}

}  // extern "C"

// nmc_qk_cuda.cu — CUDA port of the integer Q4_K/Q6_K fused dequant + fixed-point matmul (qk_linear).
//
// BYTE-IDENTICAL to the CPU kernel (nmc_qk_kernel.c) and the numpy oracle: integer addition is associative, so
// the per-thread __int128 accumulation matches the big-int reference regardless of launch geometry. The GPU is
// a PRODUCER; the CPU oracle stays the canonical VERIFIER (parity gate: tests/test_qk_cuda.py). One thread per
// (output row o, token t): stream the row's blocks, dequant inline, MAC into __int128, arithmetic-shift >> fw.
//
// Per-host, arch-specific (build with tools/build_nmc_cuda.sh, -arch=sm_<cc>). The fp16→fixed conversion
// (round-half-to-even of d*2^fw) must match the CPU llrint — gated byte-exact.
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

// Opt-in native telemetry.  The bridge owns a process-wide lock around this
// process-global runtime, so reset/snapshot cannot race a normal Engine call.
// Atomic increments additionally keep the legacy per-call entry safe when it
// is invoked directly by more than one client thread.
enum ProfileMetric {
    PM_REGISTER_CALLS, PM_REGISTER_NS,
    PM_H2D_CALLS, PM_H2D_BYTES, PM_H2D_NS,
    PM_D2H_CALLS, PM_D2H_BYTES, PM_D2H_NS,
    PM_ALLOC_CALLS, PM_ALLOC_BYTES, PM_ALLOC_NS, PM_FREE_CALLS,
    PM_NATIVE_CALLS, PM_RESIDENT_APPLY_CALLS, PM_GROUPED_APPLY_CALLS,
    PM_MOE_CALLS, PM_MOE_BATCH_CALLS, PM_MOE_BATCH_DP4A_CALLS,
    PM_Q_CALLS, PM_Q_NS, PM_K_CALLS, PM_K_NS, PM_V_CALLS, PM_V_NS,
    PM_O_CALLS, PM_O_NS, PM_OTHER_PROJ_CALLS, PM_OTHER_PROJ_NS,
    PM_COUNT
};
static unsigned long long g_profile[PM_COUNT];
static int g_profile_enabled = 0;

static unsigned long long monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;
}
static void profile_add(int metric, unsigned long long value) {
    if (g_profile_enabled && metric >= 0 && metric < PM_COUNT)
        __atomic_fetch_add(&g_profile[metric], value, __ATOMIC_RELAXED);
}
static void profile_native_call(int metric) {
    profile_add(PM_NATIVE_CALLS, 1);
    if (metric >= 0) profile_add(metric, 1);
}

static cudaError_t tracked_cuda_malloc(void **p, size_t bytes) {
    unsigned long long started = g_profile_enabled ? monotonic_ns() : 0;
    cudaError_t rc = cudaMalloc(p, bytes);
    if (g_profile_enabled) {
        profile_add(PM_ALLOC_CALLS, 1); profile_add(PM_ALLOC_BYTES, bytes);
        profile_add(PM_ALLOC_NS, monotonic_ns() - started);
    }
    return rc;
}
static cudaError_t tracked_cuda_free(void *p) {
    if (g_profile_enabled && p) profile_add(PM_FREE_CALLS, 1);
    return cudaFree(p);
}
static cudaError_t tracked_cuda_memcpy(void *dst, const void *src, size_t bytes, enum cudaMemcpyKind kind) {
    unsigned long long started = g_profile_enabled ? monotonic_ns() : 0;
    cudaError_t rc = cudaMemcpy(dst, src, bytes, kind);
    if (g_profile_enabled) {
        unsigned long long elapsed = monotonic_ns() - started;
        if (kind == cudaMemcpyHostToDevice) {
            profile_add(PM_H2D_CALLS, 1); profile_add(PM_H2D_BYTES, bytes); profile_add(PM_H2D_NS, elapsed);
        } else if (kind == cudaMemcpyDeviceToHost) {
            profile_add(PM_D2H_CALLS, 1); profile_add(PM_D2H_BYTES, bytes); profile_add(PM_D2H_NS, elapsed);
        }
    }
    return rc;
}

// Track every explicit runtime allocation/copy below without changing the
// arithmetic kernels.  CUDA events are deliberately not interposed.
#define cudaMalloc(p, bytes) tracked_cuda_malloc((void **)(p), (bytes))
#define cudaFree(p) tracked_cuda_free((void *)(p))
#define cudaMemcpy(dst, src, bytes, kind) tracked_cuda_memcpy((void *)(dst), (const void *)(src), (bytes), (kind))

extern "C" void qk_profile_reset(void) {
    for (int i = 0; i < PM_COUNT; i++) __atomic_store_n(&g_profile[i], 0ULL, __ATOMIC_RELAXED);
}
extern "C" void qk_profile_set_enabled(int enabled) { g_profile_enabled = enabled ? 1 : 0; }
extern "C" int qk_profile_snapshot(unsigned long long *out, int capacity) {
    if (!out || capacity < PM_COUNT) return PM_COUNT;
    for (int i = 0; i < PM_COUNT; i++) out[i] = __atomic_load_n(&g_profile[i], __ATOMIC_RELAXED);
    return PM_COUNT;
}

__device__ __forceinline__ double half_to_double(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
    double v;
    if (exp == 0)        v = ldexp((double)mant, -24);
    else if (exp == 31)  v = mant ? __longlong_as_double(0x7ff8000000000000LL) : __longlong_as_double(0x7ff0000000000000LL);
    else                 v = ldexp((double)(mant | 0x400u), (int)exp - 25);
    return sign ? -v : v;
}

__device__ __forceinline__ int64_t fp16_fixed(uint16_t h, int fw) {
    return (int64_t)llrint(half_to_double(h) * (double)(1ULL << fw));   // round-half-to-even, matches CPU
}

__device__ __forceinline__ uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }  // LE, align-safe

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t *q, int *d, int *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// One output: dot(dequant(row o), xt) >> fw. Shared by the per-call, resident, and batched-MoE paths.
__device__ __forceinline__ int64_t dot_q4k(const uint8_t *row, const int64_t *xt, long long n_blocks, int fw) {
    __int128 acc = 0;
    for (long long b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row + b * 144;
        int64_t dq = fp16_fixed(rd16(blk), fw), dmq = fp16_fixed(rd16(blk + 2), fw);
        const uint8_t *scales = blk + 4, *qs = blk + 16;
        const int64_t *xb = xt + b * 256;
        for (int g = 0; g < 4; g++) {
            int sc, m;
            get_scale_min_k4(2 * g, scales, &sc, &m);     int64_t dmm = dmq * (int64_t)m;
            for (int l = 0; l < 32; l++) acc += (__int128)(dq * (int64_t)sc * (int64_t)(qs[32 * g + l] & 0xF) - dmm) * xb[64 * g + l];
            get_scale_min_k4(2 * g + 1, scales, &sc, &m); dmm = dmq * (int64_t)m;
            for (int l = 0; l < 32; l++) acc += (__int128)(dq * (int64_t)sc * (int64_t)(qs[32 * g + l] >> 4) - dmm) * xb[64 * g + 32 + l];
        }
    }
    return (int64_t)(acc >> fw);
}

__device__ __forceinline__ int64_t dot_q6k(const uint8_t *row, const int64_t *xt, long long n_blocks, int fw) {
    __int128 acc = 0;
    for (long long b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row + b * 210, *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        int64_t dq = fp16_fixed(rd16(blk + 208), fw);
        const int64_t *xb = xt + b * 256;
        for (int half = 0; half < 2; half++) {
            int qlo = 64 * half, qho = 32 * half, sco = 8 * half, yo = 128 * half;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int64_t q1 = ((ql[qlo + l]      & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) - 32;
                int64_t q2 = ((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32;
                int64_t q3 = ((ql[qlo + l]      >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32;
                int64_t q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32;
                acc += (__int128)(dq * (int64_t)sc[sco + is + 0] * q1) * xb[yo + l];
                acc += (__int128)(dq * (int64_t)sc[sco + is + 2] * q2) * xb[yo + l + 32];
                acc += (__int128)(dq * (int64_t)sc[sco + is + 4] * q3) * xb[yo + l + 64];
                acc += (__int128)(dq * (int64_t)sc[sco + is + 6] * q4) * xb[yo + l + 96];
            }
        }
    }
    return (int64_t)(acc >> fw);
}

__device__ __forceinline__ int64_t dot_any(const uint8_t *W, long long o, const int64_t *x, long long n_blocks, int fw, int qtype) {
    return qtype == 0 ? dot_q4k(W + (size_t)o * n_blocks * 144, x, n_blocks, fw)
                      : dot_q6k(W + (size_t)o * n_blocks * 210, x, n_blocks, fw);
}

__global__ void qk_q4k(const uint8_t *W, const int64_t *x, long long T, long long out_f,
                       long long n_blocks, int fw, int64_t *out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_f * T) return;
    long long o = idx / T, t = idx % T, in_f = n_blocks * 256;
    out[t * out_f + o] = dot_q4k(W + (size_t)o * n_blocks * 144, x + t * in_f, n_blocks, fw);
}

__global__ void qk_q6k(const uint8_t *W, const int64_t *x, long long T, long long out_f,
                       long long n_blocks, int fw, int64_t *out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_f * T) return;
    long long o = idx / T, t = idx % T, in_f = n_blocks * 256;
    out[t * out_f + o] = dot_q6k(W + (size_t)o * n_blocks * 210, x + t * in_f, n_blocks, fw);
}

// ---- batched MoE-FFN on-GPU (levers: one launch per stage over all experts + SiLU/combine on the GPU) -------
// Byte-exact ports of the integer SiLU (fixed_point_sigmoid + x*sig>>fa) — same cubic 2^-f poly + clamp.
__constant__ long long C_LOG2E_Q16 = 94548;
__constant__ long long C_POLY[4] = {65536, 45426, 15743, 3638};

__device__ __forceinline__ long long exp2_neg_fixed_dev(long long u, int fa) {
    long long FP = 1LL << fa, k = u >> fa, f = u & (FP - 1);
    int shift = 16 - fa;
    long long c0, c1, c2, c3;
    if (shift >= 0) { c0 = C_POLY[0] >> shift; c1 = C_POLY[1] >> shift; c2 = C_POLY[2] >> shift; c3 = C_POLY[3] >> shift; }
    else { int s = -shift; c0 = C_POLY[0] << s; c1 = C_POLY[1] << s; c2 = C_POLY[2] << s; c3 = C_POLY[3] << s; }
    long long f2 = (f * f) >> fa, f3 = (f2 * f) >> fa;
    long long poly = c0 - ((c1 * f) >> fa) + ((c2 * f2) >> fa) - ((c3 * f3) >> fa);
    if (poly < 0) poly = 0;
    long long kk = k < 63 ? k : 63;
    return poly >> kk;
}

__device__ __forceinline__ long long sigmoid_dev(long long x, int fa) {
    int shift = 16 - fa;
    long long log2e = shift >= 0 ? (C_LOG2E_Q16 >> shift) : (C_LOG2E_Q16 << (-shift));
    long long d_clip = (((long long)(fa + 2) << (2 * fa))) / log2e;
    long long m = x > 0 ? x : 0;
    long long d0 = m;            if (d0 > d_clip) d0 = d_clip;
    long long d1 = m - x;        if (d1 > d_clip) d1 = d_clip;
    long long e0 = exp2_neg_fixed_dev((d0 * log2e) >> fa, fa);
    long long e1 = exp2_neg_fixed_dev((d1 * log2e) >> fa, fa);
    long long z = e0 + e1;
    return z == 0 ? 0 : (long long)(((__int128)e1 << fa) / z);
}

__global__ void silu_k(int64_t *x, long long n, int fa) {           // x <- (x * sigmoid(x)) >> fa
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = (int64_t)(((__int128)x[i] * sigmoid_dev(x[i], fa)) >> fa);
}

__global__ void mul_shift_k(int64_t *a, const int64_t *b, long long n, int fa) {   // a <- (a*b) >> fa
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = (int64_t)(((__int128)a[i] * b[i]) >> fa);
}

// batched matmul: yout[e, o] = dot(expert e's weight row o, x) >> fw, for all (e, o). x shared (gate/up) or
// per-expert (down, x_per_expert=1). qtypes[e] = each expert's quant type.
__global__ void matmul_multi_k(uint8_t **wptrs, const int *qtypes, const int64_t *xin, int x_per_expert,
                               long long out_f, long long n_blocks, int fw, int64_t *yout, int n_e) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n_e * out_f) return;
    int e = idx / out_f; long long o = idx % out_f;
    const int64_t *x = xin + (x_per_expert ? (size_t)e * (n_blocks * 256) : 0);
    yout[(size_t)e * out_f + o] = dot_any(wptrs[e], o, x, n_blocks, fw, qtypes[e]);
}

__global__ void combine_k(const int64_t *dd, int n_e, long long d_model, const int64_t *gates, int fa, int64_t *out) {
    long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= d_model) return;
    int64_t acc = 0;                                              // Σ_e (down_e[o] * gate_e) >> fa  (matches CPU)
    for (int e = 0; e < n_e; e++) acc += (int64_t)(((__int128)dd[(size_t)e * d_model + o] * gates[e]) >> fa);
    out[o] = acc;
}

// Resident attention kernels.  A bank owns one K/V cache per model layer and
// one shared committed RoPE table.  Q/K/V, RoPE, cache append, deterministic
// fixed-point softmax attention, and O projection stay on the device for the
// whole attention sublayer.
__device__ __forceinline__ void set_guard_error(int *error) { atomicExch(error, 1); }

__global__ void inherit_guard_error_k(const int *source, int *error) {
    if (blockIdx.x || threadIdx.x) return;
    if (source && *source) set_guard_error(error);
}

__global__ void maxabs_contiguous_k(const int64_t *x, long long n, unsigned long long *out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t v = x[i];
    unsigned long long a = v < 0 ? (~(unsigned long long)v + 1ULL) : (unsigned long long)v;
    atomicMax(out, a);
}

__global__ void maxabs_cache_k(const int64_t *x, int heads, long long length, long long max_length,
                               int head_dim, unsigned long long *out, const int *error) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)heads * length * head_dim;
    if (i >= n || (error && *error)) return;
    int lane = i % head_dim; long long z = i / head_dim, pos = z % length, head = z / length;
    int64_t v = x[((size_t)head * max_length + pos) * head_dim + lane];
    unsigned long long a = v < 0 ? (~(unsigned long long)v + 1ULL) : (unsigned long long)v;
    atomicMax(out, a);
}

__global__ void rope_envelope_guard_k(const unsigned long long *maxima, int *error) {
    if (blockIdx.x || threadIdx.x) return;
    __int128 cs = maxima[2];
    if ((__int128)2 * maxima[0] * cs > LLONG_MAX || (__int128)2 * maxima[1] * cs > LLONG_MAX)
        set_guard_error(error);
}

__global__ void attention_envelope_guard_k(const unsigned long long *maxima, int head_dim, int fa,
                                           int *error) {
    if (blockIdx.x || threadIdx.x) return;
    if ((__int128)maxima[0] * maxima[1] * head_dim > LLONG_MAX ||
        ((__int128)1 << fa) * maxima[2] > LLONG_MAX) set_guard_error(error);
}

__global__ void rope_interleaved_k(int64_t *x, long long rows, int heads, int head_dim,
                                   long long start, const int64_t *cos, const int64_t *sin,
                                   int fa, int *error) {
    long long half = head_dim / 2;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * heads * half) return;
    long long pair = idx % half, z = idx / half, head = z % heads, row = z / heads;
    size_t base = ((size_t)row * heads + head) * head_dim + (size_t)pair * 2;
    int64_t a = x[base], b = x[base + 1];
    int64_t c = cos[(size_t)(start + row) * half + pair];
    int64_t s = sin[(size_t)(start + row) * half + pair];
    __int128 p0 = (__int128)a * c - (__int128)b * s;
    __int128 p1 = (__int128)a * s + (__int128)b * c;
    if (p0 > LLONG_MAX || p0 < LLONG_MIN || p1 > LLONG_MAX || p1 < LLONG_MIN) {
        set_guard_error(error); return;
    }
    x[base] = (int64_t)(p0 >> fa); x[base + 1] = (int64_t)(p1 >> fa);
}

__global__ void append_kv_k(const int64_t *k, const int64_t *v, long long rows, int n_kv,
                            int head_dim, long long start, long long max_length,
                            int64_t *cache_k, int64_t *cache_v, const int *error) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = rows * n_kv * head_dim;
    if (idx >= n || (error && *error)) return;
    int lane = idx % head_dim;
    long long z = idx / head_dim, kv = z % n_kv, row = z / n_kv;
    size_t src = ((size_t)row * n_kv + kv) * head_dim + lane;
    size_t dst = ((size_t)kv * max_length + (start + row)) * head_dim + lane;
    cache_k[dst] = k[src]; cache_v[dst] = v[src];
}

__device__ __forceinline__ int64_t attention_score(const int64_t *q, const int64_t *k, int head_dim,
                                                    int fa, int64_t inv_sqrt, int *error) {
    __int128 acc = 0;
    for (int lane = 0; lane < head_dim; lane++) acc += (__int128)q[lane] * k[lane];
    __int128 shifted = acc >> fa;
    __int128 scaled = (shifted * inv_sqrt) >> fa;
    if (shifted > LLONG_MAX || shifted < LLONG_MIN || scaled > LLONG_MAX || scaled < LLONG_MIN) {
        set_guard_error(error); return 0;
    }
    return (int64_t)scaled;
}

__device__ __forceinline__ int64_t softmax_exp_delta(int64_t best, int64_t score, int fa) {
    int shift = 16 - fa;
    int64_t log2e = shift >= 0 ? (C_LOG2E_Q16 >> shift) : (C_LOG2E_Q16 << (-shift));
    int64_t d_clip = (((int64_t)(fa + 2) << (2 * fa))) / log2e;
    int64_t hard_clip = ((int64_t)1 << 62) / log2e;
    if (hard_clip < d_clip) d_clip = hard_clip;
    __int128 wide = (__int128)best - score;
    int64_t d = wide > d_clip ? d_clip : (wide < 0 ? 0 : (int64_t)wide);
    return exp2_neg_fixed_dev((d * log2e) >> fa, fa);
}

// One deterministic worker per (new query, query head).  Integer reductions
// are exact and order-independent; the sequential form also mirrors the CPU
// reference's floor points literally. The bank ABI is deliberately M=1:
// batched prefill keeps its separately gated DP4A path, while decode exposes
// one worker per query head.
__global__ void attention_cached_k(const int64_t *q, const int64_t *cache_k, const int64_t *cache_v,
                                   long long rows, long long start, long long max_length,
                                   int heads, int n_kv, int head_dim, int window, int fa,
                                   int64_t inv_sqrt, int64_t *probs, long long prob_stride,
                                   int64_t *out, int *error) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * heads || (error && *error)) return;
    int head = idx % heads; long long row = idx / heads;
    int rep = heads / n_kv, kv = head / rep;
    long long pos = start + row;
    long long first = (window > 0 && pos - window >= 0) ? pos - window + 1 : 0;
    const int64_t *qh = q + ((size_t)row * heads + head) * head_dim;
    int64_t best = LLONG_MIN;
    for (long long key = first; key <= pos; key++) {
        const int64_t *kh = cache_k + ((size_t)kv * max_length + key) * head_dim;
        int64_t score = attention_score(qh, kh, head_dim, fa, inv_sqrt, error);
        if (score > best) best = score;
    }
    int64_t Z = 0;
    for (long long key = first; key <= pos; key++) {
        const int64_t *kh = cache_k + ((size_t)kv * max_length + key) * head_dim;
        int64_t e = softmax_exp_delta(best, attention_score(qh, kh, head_dim, fa, inv_sqrt, error), fa);
        probs[(size_t)idx * prob_stride + key] = e; Z += e;
    }
    if (Z != 0) {
        for (long long key = first; key <= pos; key++)
            probs[(size_t)idx * prob_stride + key] =
                (int64_t)(((__int128)probs[(size_t)idx * prob_stride + key] << fa) / Z);
    }
    for (int lane = 0; lane < head_dim; lane++) {
        __int128 acc = 0;
        if (Z != 0) {
            for (long long key = first; key <= pos; key++) {
                const int64_t *vh = cache_v + ((size_t)kv * max_length + key) * head_dim;
                acc += (__int128)probs[(size_t)idx * prob_stride + key] * vh[lane];
            }
        }
        __int128 shifted = acc >> fa;
        if (shifted > LLONG_MAX || shifted < LLONG_MIN) { set_guard_error(error); shifted = 0; }
        out[((size_t)row * heads + head) * head_dim + lane] = (int64_t)shifted;
    }
}

__device__ __forceinline__ unsigned __int128 isqrt_u128(unsigned __int128 value) {
    unsigned __int128 result = 0;
    unsigned __int128 bit = (unsigned __int128)1 << 126;
    while (bit > value) bit >>= 2;
    while (bit != 0) {
        if (value >= result + bit) { value -= result + bit; result = (result >> 1) + bit; }
        else result >>= 1;
        bit >>= 2;
    }
    return result;
}

__device__ __forceinline__ __int128 floor_div_positive(__int128 numerator, unsigned long long denominator) {
    __int128 q = numerator / (__int128)denominator;
    __int128 r = numerator % (__int128)denominator;
    return (numerator < 0 && r != 0) ? q - 1 : q;
}

enum { PREPROCESS_TPB = 256 };

__device__ __forceinline__ unsigned long long uabs_i64(int64_t value) {
    return value < 0 ? (~(unsigned long long)value + 1ULL) : (unsigned long long)value;
}

__device__ __forceinline__ unsigned long long isqrt_u64_fast(unsigned long long value) {
    unsigned long long result = 0, bit = 1ULL << 62;
    while (bit > value) bit >>= 2;
    while (bit != 0) {
        if (value >= result + bit) { value -= result + bit; result = (result >> 1) + bit; }
        else result >>= 1;
        bit >>= 2;
    }
    return result;
}

// Python // rounds toward negative infinity, unlike C integer division.
// The fast RMS envelope proves the signed numerator fits before calling this.
__device__ __forceinline__ int64_t floor_div_i64_u64(int64_t numerator,
                                                      unsigned long long denominator) {
    if (numerator >= 0) return (int64_t)((unsigned long long)numerator / denominator);
    unsigned long long magnitude = ~(unsigned long long)numerator + 1ULL;
    unsigned long long quotient = magnitude / denominator + (magnitude % denominator != 0);
    return quotient == (1ULL << 63) ? LLONG_MIN : -(int64_t)quotient;
}

// Block-parallel RMSNorm for rows whose complete arithmetic envelope fits in
// uint64/int64. Unsupported rows set fallback[row] and are recomputed by the
// exact i128 kernel immediately afterward; this flag is not a model error.
__global__ void rmsnorm_fast_u64_k(const int64_t *x, long long rows, long long width,
                                   const int64_t *gain, int fa, unsigned long long eps,
                                   int64_t *out, int *fallback) {
    long long row = (long long)blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const int64_t *xr = x + (size_t)row * width;
    int64_t *yr = out + (size_t)row * width;
    __shared__ unsigned long long sums[PREPROCESS_TPB];
    __shared__ unsigned long long maxima[PREPROCESS_TPB];
    __shared__ unsigned long long gains[PREPROCESS_TPB];
    __shared__ unsigned long long rms;
    __shared__ int unsupported;

    unsigned long long max_x = 0, max_gain = 0;
    for (long long i = tid; i < width; i += blockDim.x) {
        unsigned long long ax = uabs_i64(xr[i]);
        unsigned long long ag = uabs_i64(gain[i]);
        if (ax > max_x) max_x = ax;
        if (ag > max_gain) max_gain = ag;
    }
    maxima[tid] = max_x;
    gains[tid] = max_gain;
    if (tid == 0) { unsupported = 0; fallback[row] = 0; }
    for (int offset = blockDim.x / 2; offset != 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) {
            if (maxima[tid + offset] > maxima[tid]) maxima[tid] = maxima[tid + offset];
            if (gains[tid + offset] > gains[tid]) gains[tid] = gains[tid + offset];
        }
    }
    __syncthreads();
    if (tid == 0) {
        unsigned long long mx = maxima[0], fp = 1ULL << fa;
        if (mx > (unsigned long long)LLONG_MAX / fp) unsupported = 1;
        if (mx != 0 && (mx > ULLONG_MAX / mx ||
            mx * mx > ULLONG_MAX / (unsigned long long)width)) unsupported = 1;
    }
    __syncthreads();
    if (unsupported) { if (tid == 0) fallback[row] = 1; return; }

    unsigned long long local_sum = 0;
    for (long long i = tid; i < width; i += blockDim.x) {
        unsigned long long ax = uabs_i64(xr[i]);
        local_sum += ax * ax;
    }
    sums[tid] = local_sum;
    for (int offset = blockDim.x / 2; offset != 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) sums[tid] += sums[tid + offset];
    }
    __syncthreads();
    if (tid == 0) {
        unsigned long long mean = sums[0] / (unsigned long long)width;
        if (mean > ULLONG_MAX - eps) unsupported = 1;
        else {
            rms = isqrt_u64_fast(mean + eps);
            if (rms == 0) unsupported = 1;
        }
    }
    __syncthreads();
    if (unsupported) { if (tid == 0) fallback[row] = 1; return; }

    unsigned long long max_normalized = 0;
    int64_t fp = (int64_t)1 << fa;
    for (long long i = tid; i < width; i += blockDim.x) {
        int64_t normalized = floor_div_i64_u64(xr[i] * fp, rms);
        yr[i] = normalized;
        unsigned long long an = uabs_i64(normalized);
        if (an > max_normalized) max_normalized = an;
    }
    maxima[tid] = max_normalized;
    for (int offset = blockDim.x / 2; offset != 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset && maxima[tid + offset] > maxima[tid]) maxima[tid] = maxima[tid + offset];
    }
    __syncthreads();
    if (tid == 0 && maxima[0] != 0 &&
        gains[0] > (unsigned long long)LLONG_MAX / maxima[0]) unsupported = 1;
    __syncthreads();
    if (unsupported) { if (tid == 0) fallback[row] = 1; return; }

    for (long long i = tid; i < width; i += blockDim.x)
        yr[i] = (yr[i] * gain[i]) >> fa;
}

// Exact RMSNorm over an i128-safe row. The Python bridge proves the sum of
// squares fits before dispatch; this kernel mirrors Python isqrt and negative
// floor division, then enforces the CPU gain-multiply envelope before writing.
__global__ void rmsnorm_exact_k(const int64_t *x, long long rows, long long width, const int64_t *gain,
                                int fa, unsigned long long eps, int64_t *out, int *error,
                                const int *fallback) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    if (fallback && !fallback[row]) return;
    const int64_t *xr = x + (size_t)row * width;
    int64_t *yr = out + (size_t)row * width;
    unsigned __int128 ssq = 0;
    const unsigned __int128 i128_max = ((unsigned __int128)1 << 127) - 1;
    for (long long i = 0; i < width; i++) {
        __int128 v = xr[i];
        unsigned __int128 square = (unsigned __int128)(v * v);
        if (square > i128_max - ssq) { set_guard_error(error); return; }
        ssq += square;
    }
    unsigned __int128 mean = ssq / (unsigned long long)width + eps;
    unsigned __int128 rms = isqrt_u128(mean);
    if (rms == 0 || rms > ULLONG_MAX) { set_guard_error(error); return; }
    unsigned long long denominator = (unsigned long long)rms;
    unsigned long long max_y = 0, max_g = 0;
    for (long long i = 0; i < width; i++) {
        __int128 numerator = (__int128)xr[i] * ((__int128)1 << fa);
        __int128 q = floor_div_positive(numerator, denominator);
        if (q > LLONG_MAX || q < LLONG_MIN) { set_guard_error(error); return; }
        yr[i] = (int64_t)q;
        unsigned long long ay = yr[i] < 0 ? (~(unsigned long long)yr[i] + 1ULL) : (unsigned long long)yr[i];
        unsigned long long ag = gain[i] < 0 ? (~(unsigned long long)gain[i] + 1ULL) : (unsigned long long)gain[i];
        if (ay > max_y) max_y = ay; if (ag > max_g) max_g = ag;
    }
    if ((unsigned __int128)max_y * max_g > LLONG_MAX) { set_guard_error(error); return; }
    for (long long i = 0; i < width; i++) yr[i] = (int64_t)(((__int128)yr[i] * gain[i]) >> fa);
}

// One block contracts a row/expert pair. The absolute-value bound proves all
// products, partial sums, and reduction sums fit signed int64. Pairs outside
// that conservative envelope retain the exact serial i128 implementation.
__global__ void router_fast_i64_k(const int64_t *h, long long rows, long long width,
                                  const int64_t *weights, int experts, int fw,
                                  int64_t *logits, int *fallback) {
    long long idx = (long long)blockIdx.x;
    if (idx >= rows * experts) return;
    int tid = threadIdx.x;
    long long row = idx / experts;
    int expert = (int)(idx % experts);
    const int64_t *hr = h + (size_t)row * width;
    const int64_t *wr = weights + (size_t)expert * width;
    __shared__ unsigned long long max_h[PREPROCESS_TPB];
    __shared__ unsigned long long max_w[PREPROCESS_TPB];
    __shared__ int64_t sums[PREPROCESS_TPB];
    __shared__ int unsupported;

    unsigned long long local_h = 0, local_w = 0;
    for (long long i = tid; i < width; i += blockDim.x) {
        unsigned long long ah = uabs_i64(hr[i]);
        unsigned long long aw = uabs_i64(wr[i]);
        if (ah > local_h) local_h = ah;
        if (aw > local_w) local_w = aw;
    }
    max_h[tid] = local_h;
    max_w[tid] = local_w;
    if (tid == 0) { unsupported = 0; fallback[idx] = 0; }
    for (int offset = blockDim.x / 2; offset != 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) {
            if (max_h[tid + offset] > max_h[tid]) max_h[tid] = max_h[tid + offset];
            if (max_w[tid + offset] > max_w[tid]) max_w[tid] = max_w[tid + offset];
        }
    }
    __syncthreads();
    if (tid == 0 && max_h[0] != 0 && max_w[0] != 0) {
        if (max_h[0] > ULLONG_MAX / max_w[0]) unsupported = 1;
        else {
            unsigned long long max_product = max_h[0] * max_w[0];
            if (max_product > (unsigned long long)LLONG_MAX / (unsigned long long)width)
                unsupported = 1;
        }
    }
    __syncthreads();
    if (unsupported) { if (tid == 0) fallback[idx] = 1; return; }

    int64_t local_sum = 0;
    for (long long i = tid; i < width; i += blockDim.x) local_sum += hr[i] * wr[i];
    sums[tid] = local_sum;
    for (int offset = blockDim.x / 2; offset != 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) sums[tid] += sums[tid + offset];
    }
    __syncthreads();
    if (tid == 0) logits[idx] = sums[0] >> fw;
}

__global__ void router_i64_k(const int64_t *h, long long rows, long long width, const int64_t *weights,
                             int experts, int fw, int64_t *logits, int *error,
                             const int *fallback) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * experts) return;
    if (fallback && !fallback[idx]) return;
    long long row = idx / experts; int expert = idx % experts;
    const int64_t *hr = h + (size_t)row * width;
    const int64_t *wr = weights + (size_t)expert * width;
    __int128 acc = 0;
    const __int128 i128_max = (__int128)(((unsigned __int128)1 << 127) - 1);
    const __int128 i128_min = -i128_max - 1;
    for (long long i = 0; i < width; i++) {
        __int128 term = (__int128)hr[i] * wr[i];
        if ((term > 0 && acc > i128_max - term) || (term < 0 && acc < i128_min - term)) {
            set_guard_error(error); logits[idx] = 0; return;
        }
        acc += term;
    }
    // NumPy's production router accumulates in int64 and is self-checked
    // against big-int. Reject the same out-of-envelope row before shifting.
    if (acc > LLONG_MAX || acc < LLONG_MIN) { set_guard_error(error); logits[idx] = 0; return; }
    logits[idx] = ((int64_t)acc) >> fw;
}

__global__ void topk_lowidx_k(const int64_t *logits, long long rows, int experts, int used, int fa,
                              int *ids, int64_t *gates, int *error) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int64_t *lr = logits + (size_t)row * experts;
    int *ir = ids + (size_t)row * used; int64_t *gr = gates + (size_t)row * used;
    for (int rank = 0; rank < used; rank++) {
        int best_id = -1; int64_t best = LLONG_MIN;
        for (int expert = 0; expert < experts; expert++) {
            bool already = false;
            for (int j = 0; j < rank; j++) if (ir[j] == expert) { already = true; break; }
            if (!already && (best_id < 0 || lr[expert] > best)) { best = lr[expert]; best_id = expert; }
        }
        ir[rank] = best_id;
        if (best == LLONG_MIN) { set_guard_error(error); gr[rank] = 0; }
        else gr[rank] = sigmoid_dev(best, fa);
    }
}

// Compact only routes whose expert triplet has not yet been bound to this
// request's layer bank.  One worker preserves selected-route order and removes
// duplicates without an atomic ordering dependency.  The full selected IDs
// and gates remain device-side; only this compact cold list is copied out.
__global__ void cold_routes_k(const int *ids, int used, const unsigned char *bound,
                              int experts, int *cold_ids, int *cold_count, int *error) {
    if (blockIdx.x || threadIdx.x) return;
    int count = 0;
    for (int rank = 0; rank < used; rank++) {
        int expert = ids[rank];
        if (expert < 0 || expert >= experts) { set_guard_error(error); continue; }
        if (bound[expert]) continue;
        bool duplicate = false;
        for (int prior = 0; prior < count; prior++) {
            if (cold_ids[prior] == expert) { duplicate = true; break; }
        }
        if (!duplicate) cold_ids[count++] = expert;
    }
    *cold_count = count;
}

// Gather the selected expert pointers and quantization kinds without exposing
// warm route IDs to the host.  Per-expert pointer tables contain only already
// registered slices; cold entries are rejected by the preceding compaction.
__global__ void gather_bound_experts_k(
        const int *ids, int used, int experts, const unsigned char *bound,
        uint8_t *const *gate_by_id, uint8_t *const *up_by_id, uint8_t *const *down_by_id,
        const int *gate_q_by_id, const int *up_q_by_id, const int *down_q_by_id,
        uint8_t **gate_selected, uint8_t **up_selected, uint8_t **down_selected,
        int *gate_q_selected, int *up_q_selected, int *down_q_selected, int *error) {
    int rank = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= used) return;
    int expert = ids[rank];
    if (expert < 0 || expert >= experts || !bound[expert] || !gate_by_id[expert] ||
        !up_by_id[expert] || !down_by_id[expert]) {
        set_guard_error(error); return;
    }
    gate_selected[rank] = gate_by_id[expert];
    up_selected[rank] = up_by_id[expert];
    down_selected[rank] = down_by_id[expert];
    gate_q_selected[rank] = gate_q_by_id[expert];
    up_q_selected[rank] = up_q_by_id[expert];
    down_q_selected[rank] = down_q_by_id[expert];
}

// NumPy int64 residual addition has committed two's-complement wrap semantics.
// Express the sum in uint64 so native signed-overflow optimization cannot alter
// that behavior, then retain the exact bits for the next resident layer.
__global__ void residual_add3_k(const int64_t *residual, const int64_t *attention,
                                const int64_t *moe, long long width, int64_t *out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    unsigned long long sum = (unsigned long long)residual[i] +
                             (unsigned long long)attention[i] +
                             (unsigned long long)moe[i];
    out[i] = (int64_t)sum;
}

#define CK(call) do { if ((call) != cudaSuccess) { return 1; } } while (0)

// qtype: 0=Q4_K, 1=Q6_K. Handles H2D/D2H internally (like the Bonsai gpu_native entries). Returns 0 ok.
extern "C" int qk_linear_cuda(const uint8_t *W, const int64_t *x, long long T, long long out_f,
                              long long n_blocks, int fw, int qtype, int64_t *out) {
    profile_native_call(-1);
    if (!W || !x || !out || T <= 0 || out_f <= 0 || n_blocks <= 0 ||
        (qtype != 0 && qtype != 1)) return 1;
    long long in_f = n_blocks * 256, bs = qtype == 0 ? 144 : 210;
    size_t wbytes = (size_t)out_f * n_blocks * bs;
    uint8_t *dW = nullptr; int64_t *dx = nullptr, *dout = nullptr;
    CK(cudaMalloc(&dW, wbytes)); CK(cudaMalloc(&dx, (size_t)T * in_f * 8)); CK(cudaMalloc(&dout, (size_t)T * out_f * 8));
    CK(cudaMemcpy(dW, W, wbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x, (size_t)T * in_f * 8, cudaMemcpyHostToDevice));
    long long n = out_f * T; int tpb = 128; long long blocks = (n + tpb - 1) / tpb;
    if (qtype == 0) qk_q4k<<<blocks, tpb>>>(dW, dx, T, out_f, n_blocks, fw, dout);
    else            qk_q6k<<<blocks, tpb>>>(dW, dx, T, out_f, n_blocks, fw, dout);
    int rc = (cudaGetLastError() != cudaSuccess) || (cudaDeviceSynchronize() != cudaSuccess);
    if (!rc) rc = (cudaMemcpy(out, dout, (size_t)T * out_f * 8, cudaMemcpyDeviceToHost) != cudaSuccess);
    cudaFree(dW); cudaFree(dx); cudaFree(dout);
    return rc;
}

// ----- resident-weight register API (REGISTER-API.md): upload the quantized weights to VRAM ONCE; apply
// reads the resident bytes (only activations cross PCIe). Keeps the COMPACT quantized form resident (18GB
// fits 24GB; dequant is still inline in the kernel — dequantized int64 would be ~240GB). -------------------
// Heap-grown plain-C registry (no std::vector → no libstdc++ dependency, so the .so stays a pure C/CUDA
// object that loads on freshly-built hosts). A fixed 16,384-entry table was smaller than the model's 18,432
// possible expert slices and eventually turned a valid registration into handle -1. Handles are indices, so
// growing/reallocating this host-side metadata does not invalidate either handles or resident device pointers.
struct DevWeight { uint8_t *p; long long out_f, n_blocks; int qtype; };
static struct DevWeight *g_reg = nullptr;
static size_t g_nreg = 0, g_reg_cap = 0;

// Bump whenever the required exported runtime contract changes.  The Python
// bridge rejects a stale per-host build instead of silently selecting the old
// fixed-size registry or allocator-heavy MoE implementation.
#define NMC_CUDA_ABI_VERSION 5
extern "C" int qk_cuda_abi_version(void) { return NMC_CUDA_ABI_VERSION; }

static int ensure_reg_capacity(size_t need) {
    if (need <= g_reg_cap) return 0;
    size_t cap = g_reg_cap ? g_reg_cap : 1024;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) return 1;
        cap *= 2;
    }
    if (cap > SIZE_MAX / sizeof(struct DevWeight)) return 1;
    void *p = realloc(g_reg, cap * sizeof(struct DevWeight));
    if (!p) return 1;
    g_reg = (struct DevWeight *)p;
    memset(g_reg + g_reg_cap, 0, (cap - g_reg_cap) * sizeof(struct DevWeight));
    g_reg_cap = cap;
    return 0;
}

static int valid_handle(long long h) {
    return h >= 0 && (unsigned long long)h < (unsigned long long)g_nreg && g_reg[h].p != nullptr;
}
static int valid_quant_handle(long long h) {
    return valid_handle(h) && (g_reg[h].qtype == 0 || g_reg[h].qtype == 1);
}

struct LayerExpertBinding {
    long long gate_h, up_h, down_h;
    int bound;
};

struct LayerKV {
    int64_t *k, *v;
    long long length;
    int poisoned;
    struct LayerExpertBinding *expert_bindings;
    int *pending_cold_ids;
    int pending_cold_count;
    uint8_t **gate_by_id, **up_by_id, **down_by_id;
    int *gate_q_by_id, *up_q_by_id, *down_q_by_id;
    unsigned char *bound_by_id;
    int expert_count, used;
    long long d_model, expert_ffn;
};

enum BankScratchSlot {
    BS_RESIDUAL_0, BS_RESIDUAL_1, BS_NORMALIZED, BS_LOGITS, BS_ROUTE_IDS, BS_ROUTE_GATES,
    BS_COLD_IDS, BS_COLD_COUNT, BS_ROUTE_ERROR, BS_PREPROCESS_FALLBACK, BS_ATTENTION,
    BS_GATE_SELECTED, BS_UP_SELECTED, BS_DOWN_SELECTED,
    BS_GATE_Q_SELECTED, BS_UP_Q_SELECTED, BS_DOWN_Q_SELECTED,
    BS_GATE_OUT, BS_UP_OUT, BS_DOWN_OUT, BS_MOE_OUT, BS_COUNT
};
struct BankScratch { void *p; size_t cap; };
struct AttentionBank {
    struct LayerKV *layers;
    int64_t *cos, *sin;
    struct BankScratch scratch[BS_COUNT];
    int n_layers, heads, n_kv, head_dim, fa;
    long long max_length, d_model;
    int residual_slot, has_residual, retained_layer;
    int pending, pending_layer;
    int live;
};
static struct AttentionBank *g_attn_banks = nullptr;
static size_t g_nattn_banks = 0, g_attn_bank_cap = 0;

static int ensure_attn_bank_capacity(size_t need) {
    if (need <= g_attn_bank_cap) return 0;
    size_t cap = g_attn_bank_cap ? g_attn_bank_cap : 4;
    while (cap < need) { if (cap > SIZE_MAX / 2) return 1; cap *= 2; }
    if (cap > SIZE_MAX / sizeof(struct AttentionBank)) return 1;
    void *p = realloc(g_attn_banks, cap * sizeof(struct AttentionBank));
    if (!p) return 1;
    g_attn_banks = (struct AttentionBank *)p;
    memset(g_attn_banks + g_attn_bank_cap, 0, (cap - g_attn_bank_cap) * sizeof(struct AttentionBank));
    g_attn_bank_cap = cap;
    return 0;
}
static int valid_attn_bank(long long h) {
    return h >= 0 && (unsigned long long)h < (unsigned long long)g_nattn_banks && g_attn_banks[h].live;
}
static void *bank_scratch(struct AttentionBank &bank, int slot, size_t need) {
    if (slot < 0 || slot >= BS_COUNT || need == 0) return nullptr;
    struct BankScratch &scratch = bank.scratch[slot];
    if (scratch.cap >= need) return scratch.p;
    void *next = nullptr;
    if (cudaMalloc(&next, need) != cudaSuccess) return nullptr;
    if (scratch.p) cudaFree(scratch.p);
    scratch.p = next; scratch.cap = need;
    return next;
}
static void destroy_layer_experts(struct LayerKV &layer) {
    free(layer.expert_bindings); layer.expert_bindings = nullptr;
    free(layer.pending_cold_ids); layer.pending_cold_ids = nullptr;
    if (layer.gate_by_id) cudaFree(layer.gate_by_id);
    if (layer.up_by_id) cudaFree(layer.up_by_id);
    if (layer.down_by_id) cudaFree(layer.down_by_id);
    if (layer.gate_q_by_id) cudaFree(layer.gate_q_by_id);
    if (layer.up_q_by_id) cudaFree(layer.up_q_by_id);
    if (layer.down_q_by_id) cudaFree(layer.down_q_by_id);
    if (layer.bound_by_id) cudaFree(layer.bound_by_id);
    layer.gate_by_id = layer.up_by_id = layer.down_by_id = nullptr;
    layer.gate_q_by_id = layer.up_q_by_id = layer.down_q_by_id = nullptr;
    layer.bound_by_id = nullptr;
    layer.pending_cold_count = 0;
    layer.expert_count = layer.used = 0; layer.d_model = layer.expert_ffn = 0;
}
static void destroy_attn_bank(struct AttentionBank *bank) {
    if (!bank || !bank->live) return;
    if (bank->layers) {
        for (int i = 0; i < bank->n_layers; i++) {
            if (bank->layers[i].k) cudaFree(bank->layers[i].k);
            if (bank->layers[i].v) cudaFree(bank->layers[i].v);
            destroy_layer_experts(bank->layers[i]);
        }
        free(bank->layers);
    }
    for (int i = 0; i < BS_COUNT; i++) if (bank->scratch[i].p) cudaFree(bank->scratch[i].p);
    if (bank->cos) cudaFree(bank->cos);
    if (bank->sin) cudaFree(bank->sin);
    memset(bank, 0, sizeof(*bank));
}

extern "C" long long qk_attention_bank_create(int n_layers, long long max_length, long long d_model,
                                               int heads, int n_kv, int head_dim, int fa,
                                               const int64_t *cos, const int64_t *sin) {
    profile_native_call(-1);
    if (!cos || !sin || n_layers <= 0 || max_length <= 0 || d_model <= 0 || d_model % 256 != 0 ||
        heads <= 0 || n_kv <= 0 ||
        heads % n_kv != 0 || head_dim <= 0 || head_dim % 2 != 0 || fa < 1 || fa > 29 ||
        (size_t)n_layers > SIZE_MAX / sizeof(struct LayerKV) ||
        (unsigned long long)heads > (unsigned long long)LLONG_MAX / (unsigned int)head_dim ||
        (unsigned long long)n_kv > (unsigned long long)LLONG_MAX / (unsigned int)head_dim ||
        ((long long)heads * head_dim) % 256 != 0 ||
        (unsigned long long)d_model > SIZE_MAX / sizeof(int64_t)) return -1;
    size_t half = (size_t)head_dim / 2;
    size_t slot = g_nattn_banks;
    for (size_t i = 0; i < g_nattn_banks; i++) if (!g_attn_banks[i].live) { slot = i; break; }
    if ((unsigned long long)max_length > SIZE_MAX / half ||
        (size_t)max_length * half > SIZE_MAX / sizeof(int64_t) ||
        (size_t)heads > SIZE_MAX / (size_t)max_length ||
        (size_t)heads * (size_t)max_length > SIZE_MAX / sizeof(int64_t) ||
        ensure_attn_bank_capacity(slot + 1) != 0) return -1;
    struct AttentionBank bank = {};
    bank.layers = (struct LayerKV *)calloc((size_t)n_layers, sizeof(struct LayerKV));
    size_t table_bytes = (size_t)max_length * half * 8;
    if (!bank.layers || cudaMalloc(&bank.cos, table_bytes) != cudaSuccess ||
        cudaMalloc(&bank.sin, table_bytes) != cudaSuccess ||
        cudaMemcpy(bank.cos, cos, table_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(bank.sin, sin, table_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (bank.cos) cudaFree(bank.cos); if (bank.sin) cudaFree(bank.sin); free(bank.layers); return -1;
    }
    bank.n_layers = n_layers; bank.max_length = max_length; bank.d_model = d_model;
    bank.heads = heads; bank.n_kv = n_kv;
    bank.head_dim = head_dim; bank.fa = fa; bank.retained_layer = -1; bank.pending_layer = -1; bank.live = 1;
    g_attn_banks[slot] = bank;
    if (slot == g_nattn_banks) g_nattn_banks++;
    return (long long)slot;
}

extern "C" int qk_attention_bank_reset(long long h) {
    profile_native_call(-1);
    if (!valid_attn_bank(h)) return 1;
    struct AttentionBank &bank = g_attn_banks[h];
    for (int i = 0; i < bank.n_layers; i++) {
        bank.layers[i].length = 0;
        bank.layers[i].poisoned = 0;
        bank.layers[i].pending_cold_count = 0;
    }
    bank.residual_slot = 0; bank.has_residual = 0; bank.retained_layer = -1;
    bank.pending = 0; bank.pending_layer = -1;
    return 0;
}
extern "C" void qk_attention_bank_destroy(long long h) {
    profile_native_call(-1);
    if (valid_attn_bank(h)) destroy_attn_bank(&g_attn_banks[h]);
}

// Configure only compact expert-pointer metadata for one request layer.  This
// allocates no expert weight storage: actual Q4_K/Q6_K slices are still
// registered lazily and individually, then bound below after route discovery.
extern "C" int qk_attention_bank_moe_configure(long long h, int layer, int experts, int used,
                                                long long d_model, long long expert_ffn) {
    profile_native_call(-1);
    if (!valid_attn_bank(h) || layer < 0 || experts <= 0 || used <= 0 || used > experts ||
        d_model <= 0 || expert_ffn <= 0 || d_model % 256 != 0 || expert_ffn % 256 != 0) return 1;
    struct AttentionBank &bank = g_attn_banks[h];
    if (layer >= bank.n_layers || d_model != bank.d_model ||
        bank.pending || bank.layers[layer].poisoned) return 1;
    struct LayerKV &state = bank.layers[layer];
    if (state.expert_count) {
        return (state.expert_count == experts && state.used == used && state.d_model == d_model &&
                state.expert_ffn == expert_ffn) ? 0 : 1;
    }
    size_t n = (size_t)experts;
    if (n > SIZE_MAX / sizeof(struct LayerExpertBinding) || n > SIZE_MAX / sizeof(uint8_t *) ||
        n > SIZE_MAX / sizeof(int)) return 1;
    struct LayerExpertBinding *bindings =
        (struct LayerExpertBinding *)calloc(n, sizeof(struct LayerExpertBinding));
    int *pending_cold_ids = (int *)calloc((size_t)used, sizeof(int));
    uint8_t **gate = nullptr, **up = nullptr, **down = nullptr;
    int *gate_q = nullptr, *up_q = nullptr, *down_q = nullptr;
    unsigned char *bound = nullptr;
    if (!bindings || !pending_cold_ids || cudaMalloc(&gate, n * sizeof(uint8_t *)) != cudaSuccess ||
        cudaMalloc(&up, n * sizeof(uint8_t *)) != cudaSuccess ||
        cudaMalloc(&down, n * sizeof(uint8_t *)) != cudaSuccess ||
        cudaMalloc(&gate_q, n * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&up_q, n * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&down_q, n * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&bound, n) != cudaSuccess ||
        cudaMemset(gate, 0, n * sizeof(uint8_t *)) != cudaSuccess ||
        cudaMemset(up, 0, n * sizeof(uint8_t *)) != cudaSuccess ||
        cudaMemset(down, 0, n * sizeof(uint8_t *)) != cudaSuccess ||
        cudaMemset(bound, 0, n) != cudaSuccess) {
        free(bindings); free(pending_cold_ids);
        if (gate) cudaFree(gate); if (up) cudaFree(up); if (down) cudaFree(down);
        if (gate_q) cudaFree(gate_q); if (up_q) cudaFree(up_q); if (down_q) cudaFree(down_q);
        if (bound) cudaFree(bound);
        return 1;
    }
    state.expert_bindings = bindings;
    state.pending_cold_ids = pending_cold_ids;
    state.pending_cold_count = 0;
    state.gate_by_id = gate; state.up_by_id = up; state.down_by_id = down;
    state.gate_q_by_id = gate_q; state.up_q_by_id = up_q; state.down_q_by_id = down_q;
    state.bound_by_id = bound; state.expert_count = experts; state.used = used;
    state.d_model = d_model; state.expert_ffn = expert_ffn;
    return 0;
}

extern "C" int qk_attention_bank_moe_bind(long long h, int layer, int expert,
                                            long long gate_h, long long up_h, long long down_h) {
    profile_native_call(-1);
    if (!valid_attn_bank(h) || layer < 0) return 1;
    struct AttentionBank &bank = g_attn_banks[h];
    if (layer >= bank.n_layers || bank.layers[layer].poisoned) return 1;
    struct LayerKV &state = bank.layers[layer];
    if (expert < 0 || expert >= state.expert_count || !valid_quant_handle(gate_h) ||
        !valid_quant_handle(up_h) || !valid_quant_handle(down_h)) return 1;
    struct DevWeight &gate = g_reg[gate_h], &up = g_reg[up_h], &down = g_reg[down_h];
    long long nb_in = state.d_model / 256, nb_down = state.expert_ffn / 256;
    if (gate.out_f != state.expert_ffn || gate.n_blocks != nb_in ||
        up.out_f != state.expert_ffn || up.n_blocks != nb_in ||
        down.out_f != state.d_model || down.n_blocks != nb_down) return 1;
    struct LayerExpertBinding &binding = state.expert_bindings[expert];
    if (binding.bound) {
        if (binding.gate_h != gate_h || binding.up_h != up_h || binding.down_h != down_h) return 1;
        if (bank.pending && bank.pending_layer == layer) {
            for (int i = 0; i < state.pending_cold_count; i++) {
                if (state.pending_cold_ids[i] == expert) {
                    state.pending_cold_ids[i] = state.pending_cold_ids[--state.pending_cold_count];
                    break;
                }
            }
        }
        return 0;
    }
    uint8_t *gate_p = gate.p, *up_p = up.p, *down_p = down.p;
    int gate_q = gate.qtype, up_q = up.qtype, down_q = down.qtype;
    unsigned char one = 1;
    // Publish the bound byte last.  A failed intermediate copy therefore
    // cannot make a partially initialized triplet selectable by continuation.
    if (cudaMemcpy(state.gate_by_id + expert, &gate_p, sizeof(gate_p), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(state.up_by_id + expert, &up_p, sizeof(up_p), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(state.down_by_id + expert, &down_p, sizeof(down_p), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(state.gate_q_by_id + expert, &gate_q, sizeof(gate_q), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(state.up_q_by_id + expert, &up_q, sizeof(up_q), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(state.down_q_by_id + expert, &down_q, sizeof(down_q), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(state.bound_by_id + expert, &one, sizeof(one), cudaMemcpyHostToDevice) != cudaSuccess) {
        state.poisoned = 1; return 1;
    }
    binding.gate_h = gate_h; binding.up_h = up_h; binding.down_h = down_h; binding.bound = 1;
    if (bank.pending && bank.pending_layer == layer) {
        for (int i = 0; i < state.pending_cold_count; i++) {
            if (state.pending_cold_ids[i] == expert) {
                state.pending_cold_ids[i] = state.pending_cold_ids[--state.pending_cold_count];
                break;
            }
        }
    }
    return 0;
}

// Reserve host registry metadata without allocating VRAM.  Besides avoiding
// growth jitter during model upload, this makes the former 16,384-entry limit
// directly regression-testable without issuing thousands of cudaMalloc calls.
extern "C" int qk_resident_reserve(long long count) {
    if (count < 0) return 1;
    return ensure_reg_capacity((size_t)count);
}

// Upload one weight tensor's raw Q4_K/Q6_K bytes; return a handle (index) or -1.
extern "C" long long qk_register_weight(const uint8_t *W, long long out_f, long long n_blocks, int qtype) {
    profile_native_call(PM_REGISTER_CALLS);
    unsigned long long started = g_profile_enabled ? monotonic_ns() : 0;
    #define REG_RETURN(value) do { if (g_profile_enabled) profile_add(PM_REGISTER_NS, monotonic_ns() - started); return (value); } while (0)
    if (!W || out_f <= 0 || n_blocks <= 0 || (qtype != 0 && qtype != 1)) REG_RETURN(-1);
    size_t bs = qtype == 0 ? 144 : 210;
    if ((size_t)out_f > SIZE_MAX / (size_t)n_blocks ||
        (size_t)out_f * (size_t)n_blocks > SIZE_MAX / bs ||
        ensure_reg_capacity(g_nreg + 1) != 0) REG_RETURN(-1);
    size_t bytes = (size_t)out_f * (size_t)n_blocks * bs;
    uint8_t *d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) REG_RETURN(-1);
    if (cudaMemcpy(d, W, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); REG_RETURN(-1); }
    g_reg[g_nreg].p = d; g_reg[g_nreg].out_f = out_f; g_reg[g_nreg].n_blocks = n_blocks; g_reg[g_nreg].qtype = qtype;
    { long long handle = (long long)g_nreg++; REG_RETURN(handle); }
    #undef REG_RETURN
}

// Upload a dense fixed-int64 matrix.  qtype=2 is deliberately a distinct
// registry kind: quantized projection entries reject it, while the bounded
// resident RMSNorm/router executor below requires it.  ``rows``/``cols`` are
// stored in the existing metadata fields to preserve stable integer handles.
extern "C" long long qk_register_i64(const int64_t *W, long long rows, long long cols) {
    profile_native_call(PM_REGISTER_CALLS);
    unsigned long long started = g_profile_enabled ? monotonic_ns() : 0;
    #define REG_I64_RETURN(value) do { if (g_profile_enabled) profile_add(PM_REGISTER_NS, monotonic_ns() - started); return (value); } while (0)
    if (!W || rows <= 0 || cols <= 0 || (size_t)rows > SIZE_MAX / (size_t)cols ||
        (size_t)rows * (size_t)cols > SIZE_MAX / sizeof(int64_t) ||
        ensure_reg_capacity(g_nreg + 1) != 0) REG_I64_RETURN(-1);
    size_t bytes = (size_t)rows * (size_t)cols * sizeof(int64_t);
    int64_t *d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) REG_I64_RETURN(-1);
    if (cudaMemcpy(d, W, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d); REG_I64_RETURN(-1);
    }
    g_reg[g_nreg].p = (uint8_t *)d;
    g_reg[g_nreg].out_f = rows;
    g_reg[g_nreg].n_blocks = cols;
    g_reg[g_nreg].qtype = 2;
    { long long handle = (long long)g_nreg++; REG_I64_RETURN(handle); }
    #undef REG_I64_RETURN
}

// Persistent activation scratch (dx/dout), grown on demand and REUSED across applies — decode does ~1350
// applies/token, and a cudaMalloc+cudaFree pair per call (each a device sync) dominated the m=1 hot path.
static int64_t *g_dx = nullptr, *g_dout = nullptr;
static int64_t *g_attn_work = nullptr, *g_attn_probs = nullptr, *g_attn_error = nullptr, *g_attn_max = nullptr;
static int64_t *g_pre_x = nullptr, *g_pre_h = nullptr, *g_pre_logits = nullptr;
static int64_t *g_pre_ids = nullptr, *g_pre_gates = nullptr, *g_pre_error = nullptr, *g_pre_fallback = nullptr;
static int8_t *g_xl = nullptr; static size_t g_xl_cap = 0;       // DP4A activation-limb scratch
static int64_t *g_xs = nullptr; static size_t g_xs_cap = 0;      // Q4_K DP4A per-subgroup Σx scratch
static size_t g_dx_cap = 0, g_dout_cap = 0, g_attn_work_cap = 0, g_attn_probs_cap = 0;
static size_t g_attn_error_cap = 0, g_attn_max_cap = 0;
static size_t g_pre_x_cap = 0, g_pre_h_cap = 0, g_pre_logits_cap = 0;
static size_t g_pre_ids_cap = 0, g_pre_gates_cap = 0, g_pre_error_cap = 0, g_pre_fallback_cap = 0;
static int64_t *scratch(int64_t **p, size_t *cap, size_t need) {
    if (*cap < need) {
        if (*p) cudaFree(*p);
        if (cudaMalloc(p, need) != cudaSuccess) { *p = nullptr; *cap = 0; return nullptr; }
        *cap = need;
    }
    return *p;
}

// Exact bounded preprocessing for a MoE layer.  Dense fixed-int64 gain and
// router weights stay resident; x crosses once, while normalized h plus only
// the compact selected IDs/gates return.  This intentionally does not upload
// every expert slice: the Python engine registers just the selected experts
// after this route-ID boundary.  Return 3 is a deterministic arithmetic
// envelope rejection; return 1 is a shape/runtime failure.
extern "C" int qk_rmsnorm_router(long long gain_h, long long router_h, const int64_t *x,
                                  long long rows, int fa, int fw, unsigned long long eps, int used,
                                  long long h_capacity, long long route_capacity,
                                  int64_t *h_out, int *ids_out, int64_t *gates_out) {
    profile_native_call(-1);
    if (!valid_handle(gain_h) || !valid_handle(router_h) || !x || !h_out || !ids_out || !gates_out ||
        rows <= 0 || used <= 0 || h_capacity < 0 || route_capacity < 0 ||
        fa < 1 || fa > 29 || fw < 0 || fw > 62) return 1;
    struct DevWeight &gain = g_reg[gain_h], &router = g_reg[router_h];
    if (gain.qtype != 2 || router.qtype != 2 || gain.out_f != 1 || gain.n_blocks <= 0 ||
        router.n_blocks != gain.n_blocks || router.out_f <= 0 || router.out_f > INT_MAX ||
        used > router.out_f) return 1;
    size_t nr = (size_t)rows, width = (size_t)gain.n_blocks;
    size_t experts = (size_t)router.out_f, k = (size_t)used;
    if (nr > SIZE_MAX / width || nr * width > SIZE_MAX / sizeof(int64_t) ||
        nr > SIZE_MAX / experts || nr * experts > SIZE_MAX / sizeof(int64_t) ||
        nr > SIZE_MAX / k || nr * k > SIZE_MAX / sizeof(int64_t) ||
        nr * k > SIZE_MAX / sizeof(int) ||
        nr * width > (size_t)h_capacity || nr * k > (size_t)route_capacity ||
        nr > (size_t)INT_MAX || nr * experts > (size_t)INT_MAX) return 1;
    size_t h_elems = nr * width, logit_elems = nr * experts, route_elems = nr * k;
    size_t h_bytes = h_elems * sizeof(int64_t), logit_bytes = logit_elems * sizeof(int64_t);
    size_t id_bytes = route_elems * sizeof(int), gate_bytes = route_elems * sizeof(int64_t);
    int64_t *dx = scratch(&g_pre_x, &g_pre_x_cap, h_bytes);
    int64_t *dh = scratch(&g_pre_h, &g_pre_h_cap, h_bytes);
    int64_t *dlogits = scratch(&g_pre_logits, &g_pre_logits_cap, logit_bytes);
    int *dids = (int *)scratch(&g_pre_ids, &g_pre_ids_cap, id_bytes);
    int64_t *dgates = scratch(&g_pre_gates, &g_pre_gates_cap, gate_bytes);
    int *derror = (int *)scratch(&g_pre_error, &g_pre_error_cap, sizeof(int));
    int *dfallback = (int *)scratch(&g_pre_fallback, &g_pre_fallback_cap,
                                    logit_elems * sizeof(int));
    if (!dx || !dh || !dlogits || !dids || !dgates || !derror || !dfallback) return 1;
    if (cudaMemcpy(dx, x, h_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(dh, 0, h_bytes) != cudaSuccess ||
        cudaMemset(derror, 0, sizeof(int)) != cudaSuccess) return 1;
    int tpb = 128;
    rmsnorm_fast_u64_k<<<rows, PREPROCESS_TPB>>>(
        dx, rows, gain.n_blocks, (const int64_t *)gain.p, fa, eps, dh, dfallback);
    rmsnorm_exact_k<<<(rows + tpb - 1) / tpb, tpb>>>(
        dx, rows, gain.n_blocks, (const int64_t *)gain.p, fa, eps, dh, derror, dfallback);
    long long nlogits = rows * router.out_f;
    router_fast_i64_k<<<nlogits, PREPROCESS_TPB>>>(
        dh, rows, gain.n_blocks, (const int64_t *)router.p, (int)router.out_f,
        fw, dlogits, dfallback);
    router_i64_k<<<(nlogits + tpb - 1) / tpb, tpb>>>(
        dh, rows, gain.n_blocks, (const int64_t *)router.p, (int)router.out_f,
        fw, dlogits, derror, dfallback);
    topk_lowidx_k<<<(rows + tpb - 1) / tpb, tpb>>>(
        dlogits, rows, (int)router.out_f, used, fa, dids, dgates, derror);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 1;
    int guard_error = 0;
    if (cudaMemcpy(&guard_error, derror, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (guard_error) return 3;
    if (cudaMemcpy(h_out, dh, h_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(ids_out, dids, id_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(gates_out, dgates, gate_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    return 0;
}

// qk_moe_ffn used to allocate/free 12 device buffers per layer call (and the batched path allocated ten).
// cudaMalloc/cudaFree are synchronizing operations, so that allocator churn dominated the many small decode
// calls. Retain one process-local workspace per buffer role, growing a role only when a later shape needs it.
// The resident API already uses process-global activation scratch and is intentionally serialized by its
// Python caller; these buffers have the same lifetime/concurrency contract and are released by qk_free_all.
enum MoeScratchSlot {
    MS_PG, MS_PU, MS_PD, MS_QG, MS_QU, MS_QD, MS_H, MS_GATES,
    MS_GATE_OUT, MS_UP_OUT, MS_DOWN_OUT, MS_OUT, MS_TOK,
    MS_H_LIMBS, MS_H_XSUM, MS_G_LIMBS, MS_G_XSUM, MS_MAX, MS_COUNT
};
struct DeviceScratch { void *p; size_t cap; };
static struct DeviceScratch g_moe_scratch[MS_COUNT];
static unsigned long long g_moe_scratch_allocations = 0;

static void *moe_scratch(int slot, size_t need) {
    if (slot < 0 || slot >= MS_COUNT || need == 0) return nullptr;
    struct DeviceScratch *s = &g_moe_scratch[slot];
    if (s->cap >= need) return s->p;
    void *p = nullptr;
    if (cudaMalloc(&p, need) != cudaSuccess) return nullptr;
    if (s->p) cudaFree(s->p);
    s->p = p; s->cap = need;
    g_moe_scratch_allocations++;
    return p;
}

static void free_moe_scratch(void) {
    for (int i = 0; i < MS_COUNT; i++) {
        if (g_moe_scratch[i].p) cudaFree(g_moe_scratch[i].p);
        g_moe_scratch[i].p = nullptr; g_moe_scratch[i].cap = 0;
    }
    g_moe_scratch_allocations = 0;
}

static size_t moe_scratch_bytes(void) {
    size_t total = 0;
    for (int i = 0; i < MS_COUNT; i++) {
        if (g_moe_scratch[i].cap > SIZE_MAX - total) return SIZE_MAX;
        total += g_moe_scratch[i].cap;
    }
    return total;
}

static unsigned char *g_moe_host = nullptr;
static size_t g_moe_host_cap = 0;

// Drop prefill-sized activation buffers without invalidating resident weight
// handles.  Decode will lazily allocate only its much smaller one-token shape.
extern "C" void qk_moe_workspace_release(void) {
    free_moe_scratch();
    free(g_moe_host); g_moe_host = nullptr; g_moe_host_cap = 0;
}

// Batched MoE also rebuilt four host metadata arrays per layer. Keep them in one realloc-grown C allocation;
// offsets use the allocation capacity (not the current P), so the four arrays never overlap after reuse.
static int moe_host_scratch(size_t need, uint8_t ***pg, uint8_t ***pu, uint8_t ***pd, int **tok) {
    if (need == 0 || need > SIZE_MAX / (3 * sizeof(uint8_t *) + sizeof(int))) return 1;
    if (g_moe_host_cap < need) {
        size_t cap = g_moe_host_cap ? g_moe_host_cap : 64;
        while (cap < need) {
            if (cap > SIZE_MAX / 2) return 1;
            cap *= 2;
        }
        size_t bytes = cap * (3 * sizeof(uint8_t *) + sizeof(int));
        void *p = realloc(g_moe_host, bytes);
        if (!p) return 1;
        g_moe_host = (unsigned char *)p; g_moe_host_cap = cap;
    }
    *pg = (uint8_t **)g_moe_host;
    *pu = *pg + g_moe_host_cap;
    *pd = *pu + g_moe_host_cap;
    *tok = (int *)(*pd + g_moe_host_cap);
    return 0;
}

// Apply a resident weight: y[t,o] = (Σ_i W_fixed[o,i]·x[t,i]) >> fw. Only x is H2D'd, y D2H'd — no weight
// transfer. Byte-identical to qk_linear (same kernel, resident pointer). Returns 0 ok.
extern "C" int qk_apply_resident(long long h, const int64_t *x, long long T, int fw, int64_t *out) {
    profile_native_call(PM_RESIDENT_APPLY_CALLS);
    if (!valid_quant_handle(h)) return 1;
    struct DevWeight &w = g_reg[h];
    long long in_f = w.n_blocks * 256;
    int64_t *dx = scratch(&g_dx, &g_dx_cap, (size_t)T * in_f * 8);
    int64_t *dout = scratch(&g_dout, &g_dout_cap, (size_t)T * w.out_f * 8);
    if (!dx || !dout) return 1;
    if (cudaMemcpy(dx, x, (size_t)T * in_f * 8, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    long long n = w.out_f * T; int tpb = 128; long long blocks = (n + tpb - 1) / tpb;
    if (w.qtype == 0) qk_q4k<<<blocks, tpb>>>(w.p, dx, T, w.out_f, w.n_blocks, fw, dout);
    else              qk_q6k<<<blocks, tpb>>>(w.p, dx, T, w.out_f, w.n_blocks, fw, dout);
    int rc = (cudaGetLastError() != cudaSuccess) || (cudaDeviceSynchronize() != cudaSuccess);
    if (!rc) rc = (cudaMemcpy(out, dout, (size_t)T * w.out_f * 8, cudaMemcpyDeviceToHost) != cudaSuccess);
    return rc;
}

// Apply several resident weights that share one activation.  The activation
// crosses PCIe once, every output remains device-side until all projections
// complete, and a single synchronized D2H copy returns their row-major arrays
// concatenated in handle order.  Arithmetic inside every output row is the
// unchanged dot_q4k/dot_q6k path, so grouping cannot alter rounding or sums.
// phase_ids: 1=Q, 2=K, 3=V, 4=O, anything else=other.
static void profile_projection(int phase, unsigned long long elapsed_ns) {
    int calls = PM_OTHER_PROJ_CALLS, ns = PM_OTHER_PROJ_NS;
    if (phase == 1) { calls = PM_Q_CALLS; ns = PM_Q_NS; }
    else if (phase == 2) { calls = PM_K_CALLS; ns = PM_K_NS; }
    else if (phase == 3) { calls = PM_V_CALLS; ns = PM_V_NS; }
    else if (phase == 4) { calls = PM_O_CALLS; ns = PM_O_NS; }
    profile_add(calls, 1); profile_add(ns, elapsed_ns);
}

extern "C" int qk_apply_resident_grouped(const long long *handles, const int *phase_ids, int count,
                                           const int64_t *x, long long T, int fw,
                                           long long output_capacity, int64_t *out) {
    profile_native_call(PM_GROUPED_APPLY_CALLS);
    if (!handles || !x || !out || count <= 0 || count > 16 || T <= 0 || output_capacity <= 0) return 1;
    long long in_f = 0;
    size_t output_elems = 0;
    for (int i = 0; i < count; i++) {
        if (!valid_quant_handle(handles[i])) return 1;
        struct DevWeight &w = g_reg[handles[i]];
        long long candidate = w.n_blocks * 256;
        if (i == 0) in_f = candidate;
        else if (candidate != in_f) return 1;
        if ((size_t)w.out_f > SIZE_MAX / (size_t)T ||
            (size_t)w.out_f * (size_t)T > SIZE_MAX - output_elems) return 1;
        output_elems += (size_t)w.out_f * (size_t)T;
    }
    if ((size_t)in_f > SIZE_MAX / (size_t)T || (size_t)T * (size_t)in_f > SIZE_MAX / 8 ||
        output_elems > SIZE_MAX / 8 || output_elems > (size_t)output_capacity) return 1;
    int64_t *dx = scratch(&g_dx, &g_dx_cap, (size_t)T * (size_t)in_f * 8);
    int64_t *dout = scratch(&g_dout, &g_dout_cap, output_elems * 8);
    if (!dx || !dout) return 1;
    if (cudaMemcpy(dx, x, (size_t)T * (size_t)in_f * 8, cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    cudaEvent_t starts[16] = {}, stops[16] = {};
    bool timed = g_profile_enabled != 0;
    if (timed) {
        for (int i = 0; i < count; i++) {
            if (cudaEventCreate(&starts[i]) != cudaSuccess || cudaEventCreate(&stops[i]) != cudaSuccess) {
                timed = false;
                break;
            }
        }
    }
    size_t offset = 0;
    int rc = 0, tpb = 128;
    for (int i = 0; i < count; i++) {
        struct DevWeight &w = g_reg[handles[i]];
        long long n = w.out_f * T, blocks = (n + tpb - 1) / tpb;
        if (timed && cudaEventRecord(starts[i]) != cudaSuccess) { rc = 1; break; }
        if (w.qtype == 0) qk_q4k<<<blocks, tpb>>>(w.p, dx, T, w.out_f, w.n_blocks, fw, dout + offset);
        else              qk_q6k<<<blocks, tpb>>>(w.p, dx, T, w.out_f, w.n_blocks, fw, dout + offset);
        if (timed && cudaEventRecord(stops[i]) != cudaSuccess) { rc = 1; break; }
        offset += (size_t)T * (size_t)w.out_f;
    }
    if (!rc) rc = (cudaGetLastError() != cudaSuccess) || (cudaDeviceSynchronize() != cudaSuccess);
    if (!rc && timed) {
        for (int i = 0; i < count; i++) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, starts[i], stops[i]) != cudaSuccess) { rc = 1; break; }
            profile_projection(phase_ids ? phase_ids[i] : 0, (unsigned long long)(ms * 1000000.0f));
        }
    } else if (!rc) {
        for (int i = 0; i < count; i++) profile_projection(phase_ids ? phase_ids[i] : 0, 0);
    }
    if (!rc) rc = cudaMemcpy(out, dout, output_elems * 8, cudaMemcpyDeviceToHost) != cudaSuccess;
    for (int i = 0; i < count; i++) {
        if (starts[i]) cudaEventDestroy(starts[i]);
        if (stops[i]) cudaEventDestroy(stops[i]);
    }
    return rc;
}

static int ensure_layer_kv(struct AttentionBank &bank, int layer) {
    struct LayerKV &kv = bank.layers[layer];
    if (kv.k && kv.v) return 0;
    size_t hkv = (size_t)bank.n_kv, ml = (size_t)bank.max_length, hd = (size_t)bank.head_dim;
    if (hkv > SIZE_MAX / ml || hkv * ml > SIZE_MAX / hd || hkv * ml * hd > SIZE_MAX / 8) return 1;
    size_t bytes = hkv * ml * hd * 8;
    int64_t *k = nullptr, *v = nullptr;
    if (cudaMalloc(&k, bytes) != cudaSuccess || cudaMalloc(&v, bytes) != cudaSuccess) {
        if (k) cudaFree(k); if (v) cudaFree(v); return 1;
    }
    kv.k = k; kv.v = v;
    return 0;
}

// Import a host prefill cache once, after the existing batched/DP4A prefill.
// Subsequent M=1 decode attention remains entirely inside the bank.
extern "C" int qk_attention_bank_import(long long h, int layer, const int64_t *k, const int64_t *v,
                                         long long length) {
    profile_native_call(-1);
    if (!valid_attn_bank(h)) return 1;
    struct AttentionBank &bank = g_attn_banks[h];
    if (!k || !v || layer < 0 || layer >= bank.n_layers || length < 0 || length > bank.max_length ||
        bank.layers[layer].poisoned || ensure_layer_kv(bank, layer) != 0) return 1;
    struct LayerKV &kv = bank.layers[layer];
    size_t per_head = (size_t)length * (size_t)bank.head_dim;
    if (per_head > SIZE_MAX / 8) return 1;
    for (int head = 0; head < bank.n_kv; head++) {
        size_t src = (size_t)head * per_head;
        size_t dst = (size_t)head * (size_t)bank.max_length * (size_t)bank.head_dim;
        if (per_head && (cudaMemcpy(kv.k + dst, k + src, per_head * 8, cudaMemcpyHostToDevice) != cudaSuccess ||
                         cudaMemcpy(kv.v + dst, v + src, per_head * 8, cudaMemcpyHostToDevice) != cudaSuccess)) {
            kv.poisoned = 1; return 1;
        }
    }
    kv.length = length;
    return 0;
}

// Validate every handle/shape/bound and reserve all attention storage before
// a retained MoE begin queues preprocessing kernels.  The later device helper
// can then run without an early host return that would strand asynchronous
// normalization/router work in the default stream.
static int attention_bank_preflight(struct AttentionBank &bank, int layer, long long qh, long long kh,
                                    long long vh, long long oh, int window) {
    if (layer < 0 || layer >= bank.n_layers || !valid_quant_handle(qh) ||
        !valid_quant_handle(kh) || !valid_quant_handle(vh) || !valid_quant_handle(oh) ||
        window == 0 || window < -1) return 1;
    struct LayerKV &cache = bank.layers[layer];
    if (cache.poisoned || cache.length >= bank.max_length || ensure_layer_kv(bank, layer) != 0) return 1;
    long long d_model = bank.d_model;
    long long q_width = (long long)bank.heads * bank.head_dim;
    long long kv_width = (long long)bank.n_kv * bank.head_dim;
    long long input_nb = d_model / 256, output_nb = q_width / 256;
    struct DevWeight &wq = g_reg[qh], &wk = g_reg[kh], &wv = g_reg[vh], &wo = g_reg[oh];
    if (d_model <= 0 || d_model % 256 != 0 || q_width <= 0 || q_width % 256 != 0 ||
        wq.n_blocks != input_nb || wk.n_blocks != input_nb || wv.n_blocks != input_nb ||
        wq.out_f != q_width || wk.out_f != kv_width || wv.out_f != kv_width ||
        wo.n_blocks != output_nb || wo.out_f != d_model) return 1;
    if ((unsigned long long)d_model > SIZE_MAX / 8 ||
        (unsigned long long)q_width > SIZE_MAX / 8 ||
        (unsigned long long)kv_width > SIZE_MAX / 16 ||
        (size_t)q_width > SIZE_MAX - 2 * (size_t)kv_width ||
        (size_t)bank.heads > SIZE_MAX / (size_t)(cache.length + 1) ||
        (size_t)bank.heads * (size_t)(cache.length + 1) > SIZE_MAX / 8) return 1;
    size_t qkv_elems = (size_t)q_width + 2 * (size_t)kv_width;
    if (qkv_elems > SIZE_MAX / sizeof(int64_t)) return 1;
    size_t projection_elems = qkv_elems > (size_t)d_model ? qkv_elems : (size_t)d_model;
    if ((size_t)bank.heads > SIZE_MAX / (size_t)bank.max_length) return 1;
    size_t prob_elems = (size_t)bank.heads * (size_t)bank.max_length;
    return (!scratch(&g_dout, &g_dout_cap, projection_elems * 8) ||
            !scratch(&g_attn_work, &g_attn_work_cap, (size_t)q_width * 8) ||
            !scratch(&g_attn_probs, &g_attn_probs_cap, prob_elems * 8) ||
            !scratch(&g_attn_error, &g_attn_error_cap, 8) ||
            !scratch(&g_attn_max, &g_attn_max_cap, 3 * 8)) ? 1 : 0;
}

// Any host-side failure after work has been queued must drain the default
// stream before publishing failure.  Otherwise a later reset/destruction can
// race kernels that still reference request scratch or K/V storage.
static int poison_after_queued_work(struct LayerKV &state) {
    (void)cudaDeviceSynchronize();
    state.poisoned = 1;
    return 1;
}

// Device-to-device core shared by the legacy attention call and the resident
// layer continuation.  It commits the K/V append only after all exact guards
// pass and returns the projected attention row in process-global serialized
// scratch; the layer executor immediately retains a copy in its bank scratch.
static int attention_bank_apply_device(struct AttentionBank &bank, int layer, long long qh, long long kh,
                                       long long vh, long long oh, const int64_t *dx, int fw,
                                       int window, int rope, int64_t inv_sqrt,
                                       const int *pre_error, int preflighted, int64_t **device_out) {
    if (!dx || !device_out) return 1;
    if (!preflighted && attention_bank_preflight(bank, layer, qh, kh, vh, oh, window) != 0) return 1;
    struct LayerKV &cache = bank.layers[layer];
    long long d_model = bank.d_model;
    long long q_width = (long long)bank.heads * bank.head_dim;
    long long kv_width = (long long)bank.n_kv * bank.head_dim;
    struct DevWeight &wq = g_reg[qh], &wk = g_reg[kh], &wv = g_reg[vh], &wo = g_reg[oh];
    size_t qkv_elems = (size_t)q_width + 2 * (size_t)kv_width;
    size_t projection_elems = qkv_elems > (size_t)d_model ? qkv_elems : (size_t)d_model;
    int64_t *dqkv = scratch(&g_dout, &g_dout_cap, projection_elems * 8);
    int64_t *dattn = scratch(&g_attn_work, &g_attn_work_cap, (size_t)q_width * 8);
    // Reserve the request's committed maximum once. Growing this buffer by
    // one position on every decode token caused a synchronizing free/malloc
    // pair in an otherwise warm resident path.
    size_t prob_elems = (size_t)bank.heads * (size_t)bank.max_length;
    int64_t *dprobs = scratch(&g_attn_probs, &g_attn_probs_cap, prob_elems * 8);
    int64_t *derror64 = scratch(&g_attn_error, &g_attn_error_cap, 8);
    int64_t *dmax64 = scratch(&g_attn_max, &g_attn_max_cap, 3 * 8);
    if (!dqkv || !dattn || !dprobs || !derror64 || !dmax64)
        return poison_after_queued_work(cache);
    int *derror = (int *)derror64;
    unsigned long long *dmax = (unsigned long long *)dmax64;
    if (cudaMemset(derror, 0, sizeof(int)) != cudaSuccess ||
        cudaMemset(dattn, 0, (size_t)q_width * sizeof(int64_t)) != cudaSuccess)
        return poison_after_queued_work(cache);
    if (pre_error) inherit_guard_error_k<<<1, 1>>>(pre_error, derror);

    int tpb = 128;
    if (wq.qtype == 0) qk_q4k<<<(wq.out_f + tpb - 1) / tpb, tpb>>>(wq.p, dx, 1, wq.out_f, wq.n_blocks, fw, dqkv);
    else               qk_q6k<<<(wq.out_f + tpb - 1) / tpb, tpb>>>(wq.p, dx, 1, wq.out_f, wq.n_blocks, fw, dqkv);
    int64_t *dk = dqkv + q_width, *dv = dk + kv_width;
    if (wk.qtype == 0) qk_q4k<<<(wk.out_f + tpb - 1) / tpb, tpb>>>(wk.p, dx, 1, wk.out_f, wk.n_blocks, fw, dk);
    else               qk_q6k<<<(wk.out_f + tpb - 1) / tpb, tpb>>>(wk.p, dx, 1, wk.out_f, wk.n_blocks, fw, dk);
    if (wv.qtype == 0) qk_q4k<<<(wv.out_f + tpb - 1) / tpb, tpb>>>(wv.p, dx, 1, wv.out_f, wv.n_blocks, fw, dv);
    else               qk_q6k<<<(wv.out_f + tpb - 1) / tpb, tpb>>>(wv.p, dx, 1, wv.out_f, wv.n_blocks, fw, dv);
    if (rope) {
        long long nq = (long long)bank.heads * (bank.head_dim / 2);
        long long nk = (long long)bank.n_kv * (bank.head_dim / 2);
        if (cudaMemset(dmax, 0, 3 * sizeof(unsigned long long)) != cudaSuccess)
            return poison_after_queued_work(cache);
        maxabs_contiguous_k<<<(q_width + tpb - 1) / tpb, tpb>>>(dqkv, q_width, dmax);
        maxabs_contiguous_k<<<(kv_width + tpb - 1) / tpb, tpb>>>(dk, kv_width, dmax + 1);
        maxabs_contiguous_k<<<(bank.head_dim / 2 + tpb - 1) / tpb, tpb>>>(
            bank.cos + (size_t)cache.length * (bank.head_dim / 2), bank.head_dim / 2, dmax + 2);
        maxabs_contiguous_k<<<(bank.head_dim / 2 + tpb - 1) / tpb, tpb>>>(
            bank.sin + (size_t)cache.length * (bank.head_dim / 2), bank.head_dim / 2, dmax + 2);
        rope_envelope_guard_k<<<1, 1>>>(dmax, derror);
        rope_interleaved_k<<<(nq + tpb - 1) / tpb, tpb>>>(dqkv, 1, bank.heads, bank.head_dim,
                                                          cache.length, bank.cos, bank.sin, bank.fa, derror);
        rope_interleaved_k<<<(nk + tpb - 1) / tpb, tpb>>>(dk, 1, bank.n_kv, bank.head_dim,
                                                          cache.length, bank.cos, bank.sin, bank.fa, derror);
    }
    append_kv_k<<<(kv_width + tpb - 1) / tpb, tpb>>>(
        dk, dv, 1, bank.n_kv, bank.head_dim, cache.length, bank.max_length,
        cache.k, cache.v, derror);
    if (cudaMemset(dmax, 0, 3 * sizeof(unsigned long long)) != cudaSuccess)
        return poison_after_queued_work(cache);
    maxabs_contiguous_k<<<(q_width + tpb - 1) / tpb, tpb>>>(dqkv, q_width, dmax);
    long long logical_cache = (cache.length + 1) * bank.n_kv * bank.head_dim;
    maxabs_cache_k<<<(logical_cache + tpb - 1) / tpb, tpb>>>(
        cache.k, bank.n_kv, cache.length + 1, bank.max_length, bank.head_dim, dmax + 1, derror);
    maxabs_cache_k<<<(logical_cache + tpb - 1) / tpb, tpb>>>(
        cache.v, bank.n_kv, cache.length + 1, bank.max_length, bank.head_dim, dmax + 2, derror);
    attention_envelope_guard_k<<<1, 1>>>(dmax, bank.head_dim, bank.fa, derror);
    attention_cached_k<<<(bank.heads + tpb - 1) / tpb, tpb>>>(
        dqkv, cache.k, cache.v, 1, cache.length, bank.max_length, bank.heads, bank.n_kv,
        bank.head_dim, window, bank.fa, inv_sqrt, dprobs, cache.length + 1, dattn, derror);
    if (wo.qtype == 0) qk_q4k<<<(wo.out_f + tpb - 1) / tpb, tpb>>>(wo.p, dattn, 1, wo.out_f, wo.n_blocks, fw, dqkv);
    else               qk_q6k<<<(wo.out_f + tpb - 1) / tpb, tpb>>>(wo.p, dattn, 1, wo.out_f, wo.n_blocks, fw, dqkv);
    cudaError_t launch_status = cudaGetLastError();
    cudaError_t sync_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess || sync_status != cudaSuccess) {
        cache.poisoned = 1; return 1;
    }
    int guard_error = 0;
    if (cudaMemcpy(&guard_error, derror, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cache.poisoned = 1; return 1;
    }
    if (guard_error) { cache.poisoned = 1; return 3; }
    cache.length++;
    *device_out = dqkv;
    profile_projection(1, 0); profile_projection(2, 0); profile_projection(3, 0); profile_projection(4, 0);
    return 0;
}

// Exact resident decode attention: normalized hidden -> Q/K/V -> optional
// interleaved RoPE -> transactional K/V append -> fixed-point GQA -> O.
// The only large host/device payloads are one hidden row in and one projected
// attention row out.  Return 3 means an exact overflow guard fired and poisons
// this bank layer; callers must fail loud and destroy the request context.
extern "C" int qk_attention_bank_apply(long long h, int layer, long long qh, long long kh,
                                        long long vh, long long oh, const int64_t *x, int fw,
                                        int window, int rope, int64_t inv_sqrt,
                                        long long output_capacity, int64_t *out) {
    profile_native_call(-1);
    if (!valid_attn_bank(h) || !x || !out || output_capacity <= 0) return 1;
    struct AttentionBank &bank = g_attn_banks[h];
    long long d_model = bank.d_model;
    if (output_capacity < d_model || (size_t)d_model > SIZE_MAX / sizeof(int64_t)) return 1;
    int64_t *dx = scratch(&g_dx, &g_dx_cap, (size_t)d_model * sizeof(int64_t));
    if (!dx || cudaMemcpy(dx, x, (size_t)d_model * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess)
        return 1;
    int64_t *device_out = nullptr;
    int rc = attention_bank_apply_device(bank, layer, qh, kh, vh, oh, dx, fw,
                                         window, rope, inv_sqrt, nullptr, 0, &device_out);
    if (rc != 0) return rc;
    if (!device_out || cudaMemcpy(out, device_out, (size_t)d_model * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost) != cudaSuccess) {
        bank.layers[layer].poisoned = 1; return 1;
    }
    return 0;
}

// Begin one exact M=1 MoE layer while retaining all large intermediates in the
// request bank.  Host input is accepted only at a chain boundary; subsequent
// layers pass use_retained=1 and consume the prior continuation's device
// residual.  Only compact cold expert IDs are copied to the caller.
extern "C" int qk_attention_bank_moe_begin(
        long long h, int layer, long long gain_h, long long router_h,
        long long qh, long long kh, long long vh, long long oh,
        const int64_t *x, int use_retained, int fw, unsigned long long eps,
        int window, int rope, int64_t inv_sqrt, int cold_capacity,
        int *cold_count_out, int *cold_ids_out) {
    profile_native_call(-1);
    if (!valid_attn_bank(h) || layer < 0 || !cold_count_out || !cold_ids_out ||
        use_retained < 0 || use_retained > 1 || fw < 0 || fw > 62 ||
        window == 0 || window < -1) return 1;
    struct AttentionBank &bank = g_attn_banks[h];
    if (layer >= bank.n_layers || bank.pending || !valid_handle(gain_h) || !valid_handle(router_h)) return 1;
    struct LayerKV &state = bank.layers[layer];
    struct DevWeight &gain = g_reg[gain_h], &router = g_reg[router_h];
    if (state.poisoned || state.expert_count <= 0 || state.used <= 0 ||
        state.pending_cold_count != 0 || !state.pending_cold_ids ||
        cold_capacity < state.used || gain.qtype != 2 || router.qtype != 2 ||
        gain.out_f != 1 || gain.n_blocks != state.d_model ||
        router.out_f != state.expert_count || router.n_blocks != state.d_model ||
        state.d_model != bank.d_model) return 1;
    if ((!use_retained && !x) || (use_retained && (!bank.has_residual ||
        bank.retained_layer + 1 != layer))) return 1;
    if (attention_bank_preflight(bank, layer, qh, kh, vh, oh, window) != 0) return 1;

    size_t dm = (size_t)state.d_model, experts = (size_t)state.expert_count, used = (size_t)state.used;
    if (dm > SIZE_MAX / sizeof(int64_t) || experts > SIZE_MAX / sizeof(int64_t) ||
        used > SIZE_MAX / sizeof(int64_t) || used > SIZE_MAX / sizeof(int)) return 1;
    int64_t *residual0 = (int64_t *)bank_scratch(bank, BS_RESIDUAL_0, dm * sizeof(int64_t));
    int64_t *residual1 = (int64_t *)bank_scratch(bank, BS_RESIDUAL_1, dm * sizeof(int64_t));
    int64_t *normalized = (int64_t *)bank_scratch(bank, BS_NORMALIZED, dm * sizeof(int64_t));
    int64_t *logits = (int64_t *)bank_scratch(bank, BS_LOGITS, experts * sizeof(int64_t));
    int *route_ids = (int *)bank_scratch(bank, BS_ROUTE_IDS, used * sizeof(int));
    int64_t *route_gates = (int64_t *)bank_scratch(bank, BS_ROUTE_GATES, used * sizeof(int64_t));
    int *cold_ids = (int *)bank_scratch(bank, BS_COLD_IDS, used * sizeof(int));
    int *cold_count = (int *)bank_scratch(bank, BS_COLD_COUNT, sizeof(int));
    int *route_error = (int *)bank_scratch(bank, BS_ROUTE_ERROR, sizeof(int));
    int *preprocess_fallback = (int *)bank_scratch(
        bank, BS_PREPROCESS_FALLBACK, experts * sizeof(int));
    int64_t *attention = (int64_t *)bank_scratch(bank, BS_ATTENTION, dm * sizeof(int64_t));
    if (!residual0 || !residual1 || !normalized || !logits || !route_ids || !route_gates ||
        !cold_ids || !cold_count || !route_error || !preprocess_fallback || !attention) return 1;
    int64_t *residual = use_retained ?
        (bank.residual_slot ? residual1 : residual0) : residual0;
    if (!use_retained) {
        if (cudaMemcpy(residual, x, dm * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
        bank.residual_slot = 0;
    }
    if (cudaMemset(normalized, 0, dm * sizeof(int64_t)) != cudaSuccess ||
        cudaMemset(route_error, 0, sizeof(int)) != cudaSuccess ||
        cudaMemset(cold_count, 0, sizeof(int)) != cudaSuccess) return 1;

    int tpb = 128;
    rmsnorm_fast_u64_k<<<1, PREPROCESS_TPB>>>(
        residual, 1, state.d_model, (const int64_t *)gain.p, bank.fa, eps,
        normalized, preprocess_fallback);
    rmsnorm_exact_k<<<1, 1>>>(
        residual, 1, state.d_model, (const int64_t *)gain.p, bank.fa, eps,
        normalized, route_error, preprocess_fallback);
    router_fast_i64_k<<<state.expert_count, PREPROCESS_TPB>>>(
        normalized, 1, state.d_model, (const int64_t *)router.p,
        state.expert_count, fw, logits, preprocess_fallback);
    router_i64_k<<<(state.expert_count + tpb - 1) / tpb, tpb>>>(
        normalized, 1, state.d_model, (const int64_t *)router.p,
        state.expert_count, fw, logits, route_error, preprocess_fallback);
    topk_lowidx_k<<<1, 1>>>(logits, 1, state.expert_count, state.used,
                            bank.fa, route_ids, route_gates, route_error);
    // Route discovery does not depend on attention. Compact cold IDs before
    // attention in the same default-stream sequence. The attention core
    // inherits route_error on-device; its guarded append performs no K/V
    // mutation on a preprocessing envelope failure, and its existing final
    // guard synchronization validates the entire sequence once.
    cold_routes_k<<<1, 1>>>(route_ids, state.used, state.bound_by_id, state.expert_count,
                             cold_ids, cold_count, route_error);
    int64_t *device_attention = nullptr;
    int rc = attention_bank_apply_device(bank, layer, qh, kh, vh, oh, normalized, fw,
                                         window, rope, inv_sqrt, route_error, 1, &device_attention);
    if (rc != 0) return rc;
    int host_count = 0;
    if (cudaMemcpy(&host_count, cold_count, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
        host_count < 0 || host_count > state.used) {
        state.poisoned = 1; return 1;
    }
    if (host_count && cudaMemcpy(cold_ids_out, cold_ids, (size_t)host_count * sizeof(int),
                                 cudaMemcpyDeviceToHost) != cudaSuccess) {
        state.poisoned = 1; return 1;
    }
    if (!device_attention || cudaMemcpy(attention, device_attention, dm * sizeof(int64_t),
                                        cudaMemcpyDeviceToDevice) != cudaSuccess) {
        state.poisoned = 1; return 1;
    }
    if (host_count) memcpy(state.pending_cold_ids, cold_ids_out, (size_t)host_count * sizeof(int));
    state.pending_cold_count = host_count;
    *cold_count_out = host_count;
    bank.has_residual = 1; bank.pending = 1; bank.pending_layer = layer;
    return 0;
}

// Continue a prepared layer after the caller has registered and bound only the
// reported cold expert slices.  Normalized h, selected IDs/gates, attention,
// MoE intermediates, and residual addition never leave the device.  publish=0
// returns no activation and makes the retained residual available to layer+1.
extern "C" int qk_attention_bank_moe_continue(long long h, int layer, int fw, int publish,
                                               long long output_capacity, int64_t *out) {
    profile_native_call(PM_MOE_CALLS);
    if (!valid_attn_bank(h) || layer < 0 || fw < 0 || fw > 62 ||
        publish < 0 || publish > 1 || (publish && (!out || output_capacity <= 0))) return 1;
    struct AttentionBank &bank = g_attn_banks[h];
    if (layer >= bank.n_layers || !bank.pending || bank.pending_layer != layer) return 1;
    struct LayerKV &state = bank.layers[layer];
    if (state.poisoned || state.expert_count <= 0 || state.used <= 0 ||
        (publish && output_capacity < state.d_model)) return 1;
    size_t dm = (size_t)state.d_model, ef = (size_t)state.expert_ffn, used = (size_t)state.used;
    if (dm > SIZE_MAX / sizeof(int64_t) || ef > SIZE_MAX / used || used * ef > SIZE_MAX / sizeof(int64_t) ||
        dm > SIZE_MAX / used || used * dm > SIZE_MAX / sizeof(int64_t)) return 1;

    int *route_ids = (int *)bank.scratch[BS_ROUTE_IDS].p;
    int64_t *route_gates = (int64_t *)bank.scratch[BS_ROUTE_GATES].p;
    int *route_error = (int *)bank.scratch[BS_ROUTE_ERROR].p;
    int64_t *normalized = (int64_t *)bank.scratch[BS_NORMALIZED].p;
    int64_t *attention = (int64_t *)bank.scratch[BS_ATTENTION].p;
    int64_t *residual0 = (int64_t *)bank.scratch[BS_RESIDUAL_0].p;
    int64_t *residual1 = (int64_t *)bank.scratch[BS_RESIDUAL_1].p;
    if (!route_ids || !route_gates || !route_error ||
        !normalized || !attention || !residual0 || !residual1) return 1;
    // begin recorded the exact compact cold set in host metadata.  Successful
    // binds retire those entries synchronously, so no second cold-route kernel
    // or D2H synchronization is needed here.  This check still fails closed
    // before any MoE kernel when the caller omitted a required binding.
    if (state.pending_cold_count != 0) return 4;
    int guard_error = 0;

    uint8_t **gate_selected = (uint8_t **)bank_scratch(
        bank, BS_GATE_SELECTED, used * sizeof(uint8_t *));
    uint8_t **up_selected = (uint8_t **)bank_scratch(
        bank, BS_UP_SELECTED, used * sizeof(uint8_t *));
    uint8_t **down_selected = (uint8_t **)bank_scratch(
        bank, BS_DOWN_SELECTED, used * sizeof(uint8_t *));
    int *gate_q_selected = (int *)bank_scratch(bank, BS_GATE_Q_SELECTED, used * sizeof(int));
    int *up_q_selected = (int *)bank_scratch(bank, BS_UP_Q_SELECTED, used * sizeof(int));
    int *down_q_selected = (int *)bank_scratch(bank, BS_DOWN_Q_SELECTED, used * sizeof(int));
    int64_t *gate_out = (int64_t *)bank_scratch(bank, BS_GATE_OUT, used * ef * sizeof(int64_t));
    int64_t *up_out = (int64_t *)bank_scratch(bank, BS_UP_OUT, used * ef * sizeof(int64_t));
    int64_t *down_out = (int64_t *)bank_scratch(bank, BS_DOWN_OUT, used * dm * sizeof(int64_t));
    int64_t *moe_out = (int64_t *)bank_scratch(bank, BS_MOE_OUT, dm * sizeof(int64_t));
    if (!gate_selected || !up_selected || !down_selected || !gate_q_selected || !up_q_selected ||
        !down_q_selected || !gate_out || !up_out || !down_out || !moe_out) return 1;
    if (cudaMemset(route_error, 0, sizeof(int)) != cudaSuccess) return 1;
    int tpb = 128;
    gather_bound_experts_k<<<(state.used + tpb - 1) / tpb, tpb>>>(
        route_ids, state.used, state.expert_count, state.bound_by_id,
        state.gate_by_id, state.up_by_id, state.down_by_id,
        state.gate_q_by_id, state.up_q_by_id, state.down_q_by_id,
        gate_selected, up_selected, down_selected,
        gate_q_selected, up_q_selected, down_q_selected, route_error);
    long long n_gate = (long long)state.used * state.expert_ffn;
    long long n_down = (long long)state.used * state.d_model;
    matmul_multi_k<<<(n_gate + tpb - 1) / tpb, tpb>>>(
        gate_selected, gate_q_selected, normalized, 0, state.expert_ffn,
        state.d_model / 256, fw, gate_out, state.used);
    silu_k<<<(n_gate + tpb - 1) / tpb, tpb>>>(gate_out, n_gate, bank.fa);
    matmul_multi_k<<<(n_gate + tpb - 1) / tpb, tpb>>>(
        up_selected, up_q_selected, normalized, 0, state.expert_ffn,
        state.d_model / 256, fw, up_out, state.used);
    mul_shift_k<<<(n_gate + tpb - 1) / tpb, tpb>>>(gate_out, up_out, n_gate, bank.fa);
    matmul_multi_k<<<(n_down + tpb - 1) / tpb, tpb>>>(
        down_selected, down_q_selected, gate_out, 1, state.d_model,
        state.expert_ffn / 256, fw, down_out, state.used);
    combine_k<<<(state.d_model + tpb - 1) / tpb, tpb>>>(
        down_out, state.used, state.d_model, route_gates, bank.fa, moe_out);
    int current = bank.residual_slot, next = current ? 0 : 1;
    int64_t *residual = current ? residual1 : residual0;
    int64_t *next_residual = next ? residual1 : residual0;
    residual_add3_k<<<(state.d_model + tpb - 1) / tpb, tpb>>>(
        residual, attention, moe_out, state.d_model, next_residual);
    cudaError_t launch_status = cudaGetLastError();
    cudaError_t sync_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess || sync_status != cudaSuccess ||
        cudaMemcpy(&guard_error, route_error, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        state.poisoned = 1; return 1;
    }
    if (guard_error) { state.poisoned = 1; return 3; }
    if (publish && cudaMemcpy(out, next_residual, dm * sizeof(int64_t),
                              cudaMemcpyDeviceToHost) != cudaSuccess) {
        state.poisoned = 1; return 1;
    }
    bank.residual_slot = next; bank.has_residual = 1; bank.retained_layer = layer;
    state.pending_cold_count = 0;
    bank.pending = 0; bank.pending_layer = -1;
    return 0;
}

extern "C" int qk_attention_bank_moe_export(long long h, long long output_capacity, int64_t *out) {
    profile_native_call(-1);
    if (!valid_attn_bank(h) || !out || output_capacity <= 0) return 1;
    struct AttentionBank &bank = g_attn_banks[h];
    if (bank.pending || !bank.has_residual || bank.retained_layer < 0) return 1;
    struct LayerKV &state = bank.layers[bank.retained_layer];
    if (output_capacity < state.d_model) return 1;
    int slot = bank.residual_slot ? BS_RESIDUAL_1 : BS_RESIDUAL_0;
    int64_t *residual = (int64_t *)bank.scratch[slot].p;
    if (!residual || cudaMemcpy(out, residual, (size_t)state.d_model * sizeof(int64_t),
                                cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    return 0;
}

// Device bytes owned exclusively by this request bank. Process-global
// serialized projection/attention scratch and registry weights are excluded;
// they are already exposed through allocation telemetry and resident counts.
extern "C" unsigned long long qk_attention_bank_workspace_bytes(long long h) {
    if (!valid_attn_bank(h)) return ULLONG_MAX;
    struct AttentionBank &bank = g_attn_banks[h];
    unsigned __int128 total = 0;
    total += (unsigned __int128)2 * (size_t)bank.max_length * (bank.head_dim / 2) * sizeof(int64_t);
    for (int slot = 0; slot < BS_COUNT; slot++) total += bank.scratch[slot].cap;
    for (int layer = 0; layer < bank.n_layers; layer++) {
        struct LayerKV &state = bank.layers[layer];
        if (state.k) total += (unsigned __int128)bank.n_kv * bank.max_length * bank.head_dim * sizeof(int64_t);
        if (state.v) total += (unsigned __int128)bank.n_kv * bank.max_length * bank.head_dim * sizeof(int64_t);
        if (state.expert_count > 0) {
            total += (unsigned __int128)state.expert_count *
                     (3 * sizeof(uint8_t *) + 3 * sizeof(int) + sizeof(unsigned char));
        }
    }
    return total > ULLONG_MAX ? ULLONG_MAX : (unsigned long long)total;
}

extern "C" void qk_free_all(void) {
    for (size_t i = 0; i < g_nreg; i++) if (g_reg[i].p) cudaFree(g_reg[i].p);
    free(g_reg); g_reg = nullptr; g_nreg = 0; g_reg_cap = 0;
    if (g_dx) { cudaFree(g_dx); g_dx = nullptr; g_dx_cap = 0; }
    if (g_dout) { cudaFree(g_dout); g_dout = nullptr; g_dout_cap = 0; }
    if (g_attn_work) { cudaFree(g_attn_work); g_attn_work = nullptr; g_attn_work_cap = 0; }
    if (g_attn_probs) { cudaFree(g_attn_probs); g_attn_probs = nullptr; g_attn_probs_cap = 0; }
    if (g_attn_error) { cudaFree(g_attn_error); g_attn_error = nullptr; g_attn_error_cap = 0; }
    if (g_attn_max) { cudaFree(g_attn_max); g_attn_max = nullptr; g_attn_max_cap = 0; }
    if (g_pre_x) { cudaFree(g_pre_x); g_pre_x = nullptr; g_pre_x_cap = 0; }
    if (g_pre_h) { cudaFree(g_pre_h); g_pre_h = nullptr; g_pre_h_cap = 0; }
    if (g_pre_logits) { cudaFree(g_pre_logits); g_pre_logits = nullptr; g_pre_logits_cap = 0; }
    if (g_pre_ids) { cudaFree(g_pre_ids); g_pre_ids = nullptr; g_pre_ids_cap = 0; }
    if (g_pre_gates) { cudaFree(g_pre_gates); g_pre_gates = nullptr; g_pre_gates_cap = 0; }
    if (g_pre_error) { cudaFree(g_pre_error); g_pre_error = nullptr; g_pre_error_cap = 0; }
    if (g_pre_fallback) { cudaFree(g_pre_fallback); g_pre_fallback = nullptr; g_pre_fallback_cap = 0; }
    if (g_xl) { cudaFree(g_xl); g_xl = nullptr; g_xl_cap = 0; }
    if (g_xs) { cudaFree(g_xs); g_xs = nullptr; g_xs_cap = 0; }
    qk_moe_workspace_release();
    for (size_t i = 0; i < g_nattn_banks; i++) destroy_attn_bank(&g_attn_banks[i]);
    free(g_attn_banks); g_attn_banks = nullptr; g_nattn_banks = 0; g_attn_bank_cap = 0;
}
extern "C" long long qk_resident_count(void) { return (long long)g_nreg; }
extern "C" long long qk_resident_capacity(void) { return (long long)g_reg_cap; }
extern "C" long long qk_moe_workspace_allocations(void) { return (long long)g_moe_scratch_allocations; }
extern "C" unsigned long long qk_moe_workspace_bytes(void) {
    return (unsigned long long)moe_scratch_bytes();
}

// Fused batched MoE expert-FFN on the GPU: out[d_model] = Σ_e gate_e · down_e( silu(gate_e·h) * up_e·h ).
// gate_h/up_h/down_h are resident handles for the n_e selected experts; gates[e] = sigmoid(router logit).
// ONE call per MoE layer (vs n_e·3 apply_resident + host silu/combine) with no intermediate H2D/D2H. Byte-
// identical to the per-expert CPU path (same dequant matmul, same integer SiLU, same shift-then-sum combine).
extern "C" int qk_moe_ffn(const long long *gate_h, const long long *up_h, const long long *down_h, int n_e,
                          const int64_t *h, const int64_t *gates, long long d_model, long long e_ffn,
                          int fa, int fw, int64_t *out) {
    profile_native_call(PM_MOE_CALLS);
    if (!gate_h || !up_h || !down_h || !h || !gates || !out || n_e <= 0 || n_e > 256 ||
        d_model <= 0 || e_ffn <= 0 || d_model % 256 != 0 || e_ffn % 256 != 0) return 1;
    long long nb_in = d_model / 256, nb_dn = e_ffn / 256;
    uint8_t *pg[256], *pu[256], *pd[256]; int qg[256], qu[256], qd[256];
    for (int e = 0; e < n_e; e++) {
        long long g = gate_h[e], u = up_h[e], d = down_h[e];
        if (!valid_quant_handle(g) || !valid_quant_handle(u) || !valid_quant_handle(d) ||
            g_reg[g].out_f != e_ffn || g_reg[g].n_blocks != nb_in ||
            g_reg[u].out_f != e_ffn || g_reg[u].n_blocks != nb_in ||
            g_reg[d].out_f != d_model || g_reg[d].n_blocks != nb_dn) return 1;
        pg[e] = g_reg[g].p; pu[e] = g_reg[u].p; pd[e] = g_reg[d].p;
        qg[e] = g_reg[g].qtype; qu[e] = g_reg[u].qtype; qd[e] = g_reg[d].qtype;
    }
    size_t ne = (size_t)n_e, dm = (size_t)d_model, ef = (size_t)e_ffn;
    if (dm > SIZE_MAX / 8 || ef > SIZE_MAX / ne || ne * ef > SIZE_MAX / 8 ||
        dm > SIZE_MAX / ne || ne * dm > SIZE_MAX / 8) return 1;
    uint8_t **dpg = (uint8_t **)moe_scratch(MS_PG, ne * sizeof(uint8_t *));
    uint8_t **dpu = (uint8_t **)moe_scratch(MS_PU, ne * sizeof(uint8_t *));
    uint8_t **dpd = (uint8_t **)moe_scratch(MS_PD, ne * sizeof(uint8_t *));
    int *dqg = (int *)moe_scratch(MS_QG, ne * sizeof(int));
    int *dqu = (int *)moe_scratch(MS_QU, ne * sizeof(int));
    int *dqd = (int *)moe_scratch(MS_QD, ne * sizeof(int));
    int64_t *dh = (int64_t *)moe_scratch(MS_H, dm * 8);
    int64_t *dgs = (int64_t *)moe_scratch(MS_GATES, ne * 8);
    int64_t *dg = (int64_t *)moe_scratch(MS_GATE_OUT, ne * ef * 8);
    int64_t *du = (int64_t *)moe_scratch(MS_UP_OUT, ne * ef * 8);
    int64_t *dd = (int64_t *)moe_scratch(MS_DOWN_OUT, ne * dm * 8);
    int64_t *dout = (int64_t *)moe_scratch(MS_OUT, dm * 8);
    if (!dpg || !dpu || !dpd || !dqg || !dqu || !dqd || !dh || !dgs || !dg || !du || !dd || !dout) return 1;
    #define H2D(d, s, sz) if (cudaMemcpy((d), (s), (sz), cudaMemcpyHostToDevice) != cudaSuccess) return 1
    H2D(dpg, pg, n_e * sizeof(uint8_t *)); H2D(dpu, pu, n_e * sizeof(uint8_t *)); H2D(dpd, pd, n_e * sizeof(uint8_t *));
    H2D(dqg, qg, n_e * sizeof(int)); H2D(dqu, qu, n_e * sizeof(int)); H2D(dqd, qd, n_e * sizeof(int));
    H2D(dh, h, d_model * 8); H2D(dgs, gates, (long long)n_e * 8);
    #undef H2D
    {
        int tpb = 128;
        long long nge = (long long)n_e * e_ffn, ngd = (long long)n_e * d_model;
        matmul_multi_k<<<(nge + tpb - 1) / tpb, tpb>>>(dpg, dqg, dh, 0, e_ffn, nb_in, fw, dg, n_e);  // gate
        silu_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, nge, fa);                                          // silu(gate)
        matmul_multi_k<<<(nge + tpb - 1) / tpb, tpb>>>(dpu, dqu, dh, 0, e_ffn, nb_in, fw, du, n_e);  // up
        mul_shift_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, du, nge, fa);                                 // gu = silu(g)*u
        matmul_multi_k<<<(ngd + tpb - 1) / tpb, tpb>>>(dpd, dqd, dg, 1, d_model, nb_dn, fw, dd, n_e); // down(gu)
        combine_k<<<(d_model + tpb - 1) / tpb, tpb>>>(dd, n_e, d_model, dgs, fa, dout);               // Σ gate·down
    }
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 1;
    return cudaMemcpy(out, dout, d_model * 8, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}

// ---- DP4A path (CUDA-KERNEL.md §7): Q6_K resident apply via base-256 activation limbs + __dp4a, byte-exact
// to qk_q6k. The weight (q-32)∈[-32,31] is int8-clean; the int64 activation is decomposed into Ln balanced
// base-256 digits (int8) so int8×int8 DP4A (int32 accum) recombines to the exact int128 dot. -----------------
__device__ __forceinline__ int pk4(const int8_t *p) {
    return (p[0] & 0xFF) | ((p[1] & 0xFF) << 8) | ((p[2] & 0xFF) << 16) | ((p[3] & 0xFF) << 24);
}

// xlimb[l*T*in_f + t*in_f + i] = l-th balanced base-256 digit of x[t*in_f+i] (signed int8). The greedy
// decomposition keeps Ln signed digits and discards the final carry, so it reconstructs x exactly over ℤ
// ONLY when |x| <= 127*(256^Ln - 1)/255 (NOT the looser 256^Ln/2 bound — that over-claims a high band and
// wraps by 256^Ln, e.g. Ln=2 fails on [32640,32767]). The caller (_ln_for) picks Ln for that exact bound.
__global__ void make_limbs(const int64_t *x, long long n, long long stride, int Ln, int8_t *xlimb) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int64_t r = x[idx];
    for (int l = 0; l < Ln; l++) {
        int lb = (int)((uint64_t)r & 0xFF);
        int d = (lb >= 128) ? lb - 256 : lb;
        xlimb[(size_t)l * stride + idx] = (int8_t)d;
        r = (r - d) >> 8;                       // exact: (r-d) divisible by 256
    }
}

__global__ void qk_q6k_dp4a(const uint8_t *W, const int8_t *xlimb, long long T, long long out_f,
                            long long n_blocks, int fw, int Ln, int64_t *out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_f * T) return;
    long long o = idx / T, t = idx % T, in_f = n_blocks * 256;
    const uint8_t *row = W + (size_t)o * n_blocks * 210;
    __int128 acc = 0;
    for (long long b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row + b * 210, *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        int64_t dq = fp16_fixed(rd16(blk + 208), fw);
        int64_t blockacc = 0;
        for (int half = 0; half < 2; half++) {
            int qlo = 64 * half, qho = 32 * half, sco = 8 * half;
            long long xbase = b * 256 + 128 * half;          // x index base for this half
            int8_t qbuf[128];                                 // (q-32) in x-index order for the 128-wide half
            for (int l = 0; l < 32; l++) {                    // q1..q4 each cover 32 consecutive x in the half
                qbuf[l]      = (int8_t)(((ql[qlo + l]      & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) - 32);
                qbuf[32 + l] = (int8_t)(((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32);
                qbuf[64 + l] = (int8_t)(((ql[qlo + l]      >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32);
                qbuf[96 + l] = (int8_t)(((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32);
            }
            for (int g = 0; g < 8; g++) {                     // 8 sub-scale groups of 16; group g uses sc[sco+g]
                const int8_t *qg = qbuf + g * 16;
                int64_t gdot = 0;
                for (int l = 0; l < Ln; l++) {
                    const int8_t *xg = xlimb + (size_t)l * T * in_f + t * in_f + xbase + g * 16;
                    int s = 0;
                    for (int j = 0; j < 16; j += 4) s = __dp4a(pk4(qg + j), pk4(xg + j), s);
                    gdot += (int64_t)s << (8 * l);            // recombine 256^l
                }
                blockacc += (int64_t)sc[sco + g] * gdot;
            }
        }
        acc += (__int128)dq * blockacc;
    }
    out[(size_t)t * out_f + o] = (int64_t)(acc >> fw);
}

// Apply a resident Q6_K weight via DP4A. x int64 [T,in_f] -> out [T,out_f]. Ln = activation limbs (caller picks
// from max|x|). Byte-identical to qk_apply_resident for the same handle when Ln covers x. Returns 0 ok, 2 = not Q6_K.
extern "C" int qk_apply_resident_q6k_dp4a(long long h, const int64_t *x, long long T, int fw, int Ln, int64_t *out) {
    if (!valid_handle(h)) return 1;
    if (g_reg[h].qtype != 1) return 2;                        // Q6_K only (qtype 1)
    struct DevWeight &w = g_reg[h];
    long long in_f = w.n_blocks * 256, nx = T * in_f;
    int64_t *dx = scratch(&g_dx, &g_dx_cap, (size_t)nx * 8);
    int64_t *dout = scratch(&g_dout, &g_dout_cap, (size_t)T * w.out_f * 8);
    if (g_xl_cap < (size_t)Ln * nx) {
        if (g_xl) cudaFree(g_xl);
        if (cudaMalloc(&g_xl, (size_t)Ln * nx) != cudaSuccess) { g_xl = nullptr; g_xl_cap = 0; return 1; }
        g_xl_cap = (size_t)Ln * nx;
    }
    if (!dx || !dout) return 1;
    if (cudaMemcpy(dx, x, (size_t)nx * 8, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    int tpb = 128;
    make_limbs<<<(nx + tpb - 1) / tpb, tpb>>>(dx, nx, nx, Ln, g_xl);
    long long n = w.out_f * T;
    qk_q6k_dp4a<<<(n + tpb - 1) / tpb, tpb>>>(w.p, g_xl, T, w.out_f, w.n_blocks, fw, Ln, dout);
    int rc = (cudaGetLastError() != cudaSuccess) || (cudaDeviceSynchronize() != cudaSuccess);
    if (!rc) rc = (cudaMemcpy(out, dout, (size_t)T * w.out_f * 8, cudaMemcpyDeviceToHost) != cudaSuccess);
    return rc;
}

// Q4_K DP4A: w = dq·sc·q - dmq·m (q∈[0,15], 4-bit). Σ w·x = dq·sc·(Σ q·x) - dmq·m·(Σ x) per 32-wide subgroup.
// Σ q·x via DP4A over the activation limbs (q int8); Σ x precomputed per subgroup (same for all output rows).
// xsum[t*(n_blocks*8) + b*8 + s] = Σ_{j<32} x[t*in_f + b*256 + s*32 + j]  (subgroup s covers x[s*32 .. s*32+31]).
__global__ void make_xsum(const int64_t *x, long long T, long long n_blocks, int64_t *xsum) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * n_blocks * 8) return;
    long long in_f = n_blocks * 256, t = idx / (n_blocks * 8), rem = idx % (n_blocks * 8), b = rem / 8, s = rem % 8;
    const int64_t *xb = x + t * in_f + b * 256 + s * 32;
    int64_t sum = 0;
    for (int j = 0; j < 32; j++) sum += xb[j];
    xsum[idx] = sum;
}

__global__ void qk_q4k_dp4a(const uint8_t *W, const int8_t *xlimb, const int64_t *xsum, long long T,
                            long long out_f, long long n_blocks, int fw, int Ln, int64_t *out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_f * T) return;
    long long o = idx / T, t = idx % T, in_f = n_blocks * 256;
    const uint8_t *row = W + (size_t)o * n_blocks * 144;
    __int128 acc = 0;
    for (long long b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row + b * 144;
        int64_t dq = fp16_fixed(rd16(blk), fw), dmq = fp16_fixed(rd16(blk + 2), fw);
        const uint8_t *scales = blk + 4, *qs = blk + 16;
        int8_t qbuf[256];                                  // q nibbles in x-index order (subgroup s -> x[s*32..])
        for (int s = 0; s < 8; s++) {
            int g = s / 2, sub = s % 2;
            for (int l = 0; l < 32; l++)
                qbuf[s * 32 + l] = (int8_t)(sub == 0 ? (qs[32 * g + l] & 0xF) : (qs[32 * g + l] >> 4));
        }
        for (int s = 0; s < 8; s++) {
            int sc, m;
            get_scale_min_k4(s, scales, &sc, &m);
            const int8_t *qg = qbuf + s * 32;
            long long xbase = b * 256 + s * 32;
            int64_t qdot = 0;
            for (int l = 0; l < Ln; l++) {
                const int8_t *xg = xlimb + (size_t)l * T * in_f + t * in_f + xbase;
                int ss = 0;
                for (int j = 0; j < 32; j += 4) ss = __dp4a(pk4(qg + j), pk4(xg + j), ss);
                qdot += (int64_t)ss << (8 * l);
            }
            acc += (__int128)(dq * (int64_t)sc) * qdot - (__int128)(dmq * (int64_t)m) * xsum[t * (n_blocks * 8) + b * 8 + s];
        }
    }
    out[(size_t)t * out_f + o] = (int64_t)(acc >> fw);
}

extern "C" int qk_apply_resident_q4k_dp4a(long long h, const int64_t *x, long long T, int fw, int Ln, int64_t *out) {
    if (!valid_handle(h)) return 1;
    if (g_reg[h].qtype != 0) return 2;                     // Q4_K only (qtype 0)
    struct DevWeight &w = g_reg[h];
    long long in_f = w.n_blocks * 256, nx = T * in_f, nxs = T * w.n_blocks * 8;
    int64_t *dx = scratch(&g_dx, &g_dx_cap, (size_t)nx * 8);
    int64_t *dout = scratch(&g_dout, &g_dout_cap, (size_t)T * w.out_f * 8);
    if (g_xl_cap < (size_t)Ln * nx) { if (g_xl) cudaFree(g_xl);
        if (cudaMalloc(&g_xl, (size_t)Ln * nx) != cudaSuccess) { g_xl = nullptr; g_xl_cap = 0; return 1; } g_xl_cap = (size_t)Ln * nx; }
    if (g_xs_cap < (size_t)nxs * 8) { if (g_xs) cudaFree(g_xs);
        if (cudaMalloc(&g_xs, (size_t)nxs * 8) != cudaSuccess) { g_xs = nullptr; g_xs_cap = 0; return 1; } g_xs_cap = (size_t)nxs * 8; }
    if (!dx || !dout) return 1;
    if (cudaMemcpy(dx, x, (size_t)nx * 8, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    int tpb = 128;
    make_limbs<<<(nx + tpb - 1) / tpb, tpb>>>(dx, nx, nx, Ln, g_xl);
    make_xsum<<<(nxs + tpb - 1) / tpb, tpb>>>(dx, T, w.n_blocks, g_xs);
    long long n = w.out_f * T;
    qk_q4k_dp4a<<<(n + tpb - 1) / tpb, tpb>>>(w.p, g_xl, g_xs, T, w.out_f, w.n_blocks, fw, Ln, dout);
    int rc = (cudaGetLastError() != cudaSuccess) || (cudaDeviceSynchronize() != cudaSuccess);
    if (!rc) rc = (cudaMemcpy(out, dout, (size_t)T * w.out_f * 8, cudaMemcpyDeviceToHost) != cudaSuccess);
    return rc;
}

// ---- fused-MoE DP4A: the experts (the per-token decode bulk). matmul_multi_dp4a batches the n_e selected
// experts and dispatches Q4_K (affine, needs Σx) or Q6_K (no min) per stage; qk_moe_ffn_dp4a runs the whole
// gate+silu+up+gu+down+combine on-GPU with DP4A. Byte-identical to qk_moe_ffn. -----------------------------
__global__ void maxabs_k(const int64_t *x, long long n, unsigned long long *out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t v = x[i];
    unsigned long long a = v < 0 ? (~(unsigned long long)v + 1ULL) : (unsigned long long)v;   // |v|, INT64_MIN-safe
    atomicMax(out, a);
}

// per (expert e, output row o): DP4A dot of expert e's weight row o against its input limbs (+ Q4_K affine Σx).
// xlimb base for this expert's input is xlimb + (per_expert? e*in_f : 0); limbs are `ls` apart.
__global__ void matmul_multi_dp4a(uint8_t **wptrs, int qt, const int8_t *xlimb, const int64_t *xsum,
    int x_per_expert, const int *tok_of, long long input_count, long long out_f, long long n_blocks,
    int fw, int Ln, int n_e, int64_t *yout) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n_e * out_f) return;
    int e = idx / out_f; long long o = idx % out_f, in_f = n_blocks * 256;
    const uint8_t *W = wptrs[e];
    long long input = x_per_expert ? e : (tok_of ? tok_of[e] : 0);
    long long ls = input_count * in_f;
    const int8_t *xlb = xlimb + (size_t)input * in_f;
    __int128 acc = 0;
    if (qt == 0) {                                              // Q4_K (affine)
        const int64_t *xsb = xsum + input * n_blocks * 8;
        const uint8_t *Wrow = W + (size_t)o * n_blocks * 144;   // this output row's blocks
        for (long long b = 0; b < n_blocks; b++) {
            const uint8_t *blk = Wrow + b * 144;
            int64_t dq = fp16_fixed(rd16(blk), fw), dmq = fp16_fixed(rd16(blk + 2), fw);
            const uint8_t *scales = blk + 4, *qs = blk + 16;
            int8_t qbuf[256];
            for (int s = 0; s < 8; s++) { int g = s / 2, sub = s % 2;
                for (int l = 0; l < 32; l++) qbuf[s * 32 + l] = (int8_t)(sub == 0 ? (qs[32 * g + l] & 0xF) : (qs[32 * g + l] >> 4)); }
            for (int s = 0; s < 8; s++) {
                int sc, m; get_scale_min_k4(s, scales, &sc, &m);
                const int8_t *qg = qbuf + s * 32; long long xbase = b * 256 + s * 32;
                int64_t qdot = 0;
                for (int l = 0; l < Ln; l++) { const int8_t *xg = xlb + (size_t)l * ls + xbase;
                    int ss = 0; for (int j = 0; j < 32; j += 4) ss = __dp4a(pk4(qg + j), pk4(xg + j), ss);
                    qdot += (int64_t)ss << (8 * l); }
                acc += (__int128)(dq * (int64_t)sc) * qdot - (__int128)(dmq * (int64_t)m) * xsb[b * 8 + s];
            }
        }
    } else {                                                   // Q6_K
        const uint8_t *Wrow = W + (size_t)o * n_blocks * 210;  // this output row's blocks
        for (long long b = 0; b < n_blocks; b++) {
            const uint8_t *blk = Wrow + b * 210, *ql = blk, *qh = blk + 128;
            const int8_t *sc = (const int8_t *)(blk + 192);
            int64_t dq = fp16_fixed(rd16(blk + 208), fw), blockacc = 0;
            for (int half = 0; half < 2; half++) {
                int qlo = 64 * half, qho = 32 * half, sco = 8 * half; long long xbase = b * 256 + 128 * half;
                int8_t qbuf[128];
                for (int l = 0; l < 32; l++) {
                    qbuf[l]      = (int8_t)(((ql[qlo + l]      & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) - 32);
                    qbuf[32 + l] = (int8_t)(((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32);
                    qbuf[64 + l] = (int8_t)(((ql[qlo + l]      >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32);
                    qbuf[96 + l] = (int8_t)(((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32);
                }
                for (int g = 0; g < 8; g++) {
                    const int8_t *qg = qbuf + g * 16; int64_t gdot = 0;
                    for (int l = 0; l < Ln; l++) { const int8_t *xg = xlb + (size_t)l * ls + xbase + g * 16;
                        int ss = 0; for (int j = 0; j < 16; j += 4) ss = __dp4a(pk4(qg + j), pk4(xg + j), ss);
                        gdot += (int64_t)ss << (8 * l); }
                    blockacc += (int64_t)sc[sco + g] * gdot;
                }
            }
            acc += (__int128)dq * blockacc;
        }
    }
    yout[(size_t)e * out_f + o] = (int64_t)(acc >> fw);
}

static int _ln_for(unsigned long long maxabs) {                // safe DP4A limbs, or 0 for exact-kernel fallback
    // L balanced digits (each in [-128,127]) reconstruct x exactly only when |x| <= 127*(256^L-1)/255, NOT
    // the naive 2^(8L-1). Build that capacity with the overflow-safe recurrence cap(L)=cap(L-1)*256+127.
    // The decomposition supports more limbs mathematically, but these kernels recombine each weighted limb in
    // int64. Four limbs are the proven-safe Q4_K/Q6_K envelope; above it the caller must use the int128 path.
    int L = 1;
    unsigned long long cap = 127ULL;                           // capacity for L=1
    while (L < 4 && maxabs > cap) { cap = cap * 256ULL + 127ULL; L++; }
    return maxabs <= cap ? L : 0;
}

extern "C" int qk_moe_ffn_dp4a(const long long *gate_h, const long long *up_h, const long long *down_h, int n_e,
                               const int64_t *h, const int64_t *gates, long long d_model, long long e_ffn,
                               int fa, int fw, int64_t *out) {
    profile_native_call(PM_MOE_CALLS);
    if (n_e <= 0 || n_e > 256) return 1;
    uint8_t *pg[256], *pu[256], *pd[256];
    int qg = -1, qu = -1, qd = -1;
    for (int e = 0; e < n_e; e++) {
        long long g = gate_h[e], u = up_h[e], d = down_h[e];
        if (!valid_quant_handle(g) || !valid_quant_handle(u) || !valid_quant_handle(d)) return 1;
        pg[e] = g_reg[g].p; pu[e] = g_reg[u].p; pd[e] = g_reg[d].p;
        int eg = g_reg[g].qtype, eu = g_reg[u].qtype, ed = g_reg[d].qtype;
        if (e == 0) { qg = eg; qu = eu; qd = ed; }
        else if (qg != eg || qu != eu || qd != ed) return 1;              // one qtype per tensor/stage
    }
    long long nb_in = d_model / 256, nb_dn = e_ffn / 256;
    // host: Ln for the shared input h
    unsigned long long mxh = 0;
    for (long long i = 0; i < d_model; i++) { int64_t v = h[i]; unsigned long long a = v < 0 ? (~(unsigned long long)v + 1ULL) : (unsigned long long)v; if (a > mxh) mxh = a; }
    int Lh = _ln_for(mxh);
    if (Lh == 0) return 1;                                      // outside safe DP4A envelope; use int128 path
    uint8_t **dpg = 0, **dpu = 0, **dpd = 0;
    int64_t *dh = 0, *dg = 0, *du = 0, *dd = 0, *dgs = 0, *dout = 0, *dhxs = 0, *dgxs = 0;
    int8_t *dhl = 0, *dgl = 0; unsigned long long *dmax = 0;
    int rc = 1;
    #define A(p, sz) if (cudaMalloc(&(p), (sz)) != cudaSuccess) goto done
    A(dpg, n_e * sizeof(uint8_t *)); A(dpu, n_e * sizeof(uint8_t *)); A(dpd, n_e * sizeof(uint8_t *));
    A(dh, d_model * 8); A(dgs, (long long)n_e * 8); A(dout, d_model * 8);
    A(dg, (long long)n_e * e_ffn * 8); A(du, (long long)n_e * e_ffn * 8); A(dd, (long long)n_e * d_model * 8);
    A(dhl, (size_t)Lh * d_model); A(dhxs, (long long)nb_in * 8 * 8);
    A(dgxs, (long long)n_e * nb_dn * 8 * 8); A(dmax, 8);
    #undef A
    #define H2D(d, s, sz) if (cudaMemcpy((d), (s), (sz), cudaMemcpyHostToDevice) != cudaSuccess) goto done
    H2D(dpg, pg, n_e * sizeof(uint8_t *)); H2D(dpu, pu, n_e * sizeof(uint8_t *)); H2D(dpd, pd, n_e * sizeof(uint8_t *));
    H2D(dh, h, d_model * 8); H2D(dgs, gates, (long long)n_e * 8);
    #undef H2D
    {
        int tpb = 128;
        long long nge = (long long)n_e * e_ffn, ngd = (long long)n_e * d_model;
        make_limbs<<<(d_model + tpb - 1) / tpb, tpb>>>(dh, d_model, d_model, Lh, dhl);
        make_xsum<<<(nb_in * 8 + tpb - 1) / tpb, tpb>>>(dh, 1, nb_in, dhxs);
        matmul_multi_dp4a<<<(nge + tpb - 1) / tpb, tpb>>>(dpg, qg, dhl, dhxs, 0, nullptr, 1,
                                                          e_ffn, nb_in, fw, Lh, n_e, dg);
        silu_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, nge, fa);
        matmul_multi_dp4a<<<(nge + tpb - 1) / tpb, tpb>>>(dpu, qu, dhl, dhxs, 0, nullptr, 1,
                                                          e_ffn, nb_in, fw, Lh, n_e, du);
        mul_shift_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, du, nge, fa);     // gu = silu(gate)*up  (in dg)
        // gu is device-computed -> reduce its max|.| to pick the limb count for the down matmul
        cudaMemset(dmax, 0, 8);
        maxabs_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, nge, dmax);
        unsigned long long mxg = 0;
        if (cudaMemcpy(&mxg, dmax, 8, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
        int Lg = _ln_for(mxg);
        if (Lg == 0) goto done;                                 // outside safe DP4A envelope; use int128 path
        if (cudaMalloc(&dgl, (size_t)Lg * nge) != cudaSuccess) goto done;    // gu limbs (Lg known only now)
        make_limbs<<<(nge + tpb - 1) / tpb, tpb>>>(dg, nge, nge, Lg, dgl);   // [Lg, n_e*e_ffn]
        make_xsum<<<((long long)n_e * nb_dn * 8 + tpb - 1) / tpb, tpb>>>(dg, n_e, nb_dn, dgxs);
        matmul_multi_dp4a<<<(ngd + tpb - 1) / tpb, tpb>>>(dpd, qd, dgl, dgxs, 1, nullptr, n_e,
                                                          d_model, nb_dn, fw, Lg, n_e, dd);
        combine_k<<<(d_model + tpb - 1) / tpb, tpb>>>(dd, n_e, d_model, dgs, fa, dout);
    }
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) goto done;
    if (cudaMemcpy(out, dout, d_model * 8, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    cudaFree(dpg); cudaFree(dpu); cudaFree(dpd); cudaFree(dh); cudaFree(dgs); cudaFree(dout);
    cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dhl); cudaFree(dhxs); cudaFree(dgl); cudaFree(dgxs); cudaFree(dmax);
    return rc;
}

// ---- batched-over-(token,expert) MoE: collapse the prefill's m per-token qk_moe_ffn calls into ONE set of
// kernels over all P = m·k (token, selected-expert) pairs. The gate/up input is the pair's TOKEN h[t]; the
// down input is the pair's gu. int128 (the experts are small; the win is launch-count + GPU occupancy, not
// DP4A). Byte-identical to calling qk_moe_ffn per token. ---------------------------------------------------
__global__ void matmul_multi_tok(uint8_t **wptrs, int qt, const int64_t *xin, int x_per_pair,
                                 const int *tok_of, long long out_f, long long n_blocks, int fw, int P, int64_t *yout) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)P * out_f) return;
    int p = idx / out_f; long long o = idx % out_f, in_f = n_blocks * 256;
    const int64_t *x = xin + (size_t)(x_per_pair ? p : tok_of[p]) * in_f;
    yout[(size_t)p * out_f + o] = dot_any(wptrs[p], o, x, n_blocks, fw, qt);
}

__global__ void combine_tok(const int64_t *dd, int m, int k, long long d_model, const int64_t *gates, int fa, int64_t *out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)m * d_model) return;
    int t = idx / d_model; long long o = idx % d_model;
    int64_t acc = 0;
    for (int j = 0; j < k; j++)
        acc += (int64_t)(((__int128)dd[((size_t)(t * k + j)) * d_model + o] * gates[t * k + j]) >> fa);
    out[(size_t)t * d_model + o] = acc;
}

extern "C" int qk_moe_ffn_batched(const long long *gate_h, const long long *up_h, const long long *down_h,
                                  int m, int k, const int64_t *h, const int64_t *gates, long long d_model,
                                  long long e_ffn, int fa, int fw, int64_t *out) {
    profile_native_call(PM_MOE_BATCH_CALLS);
    long long P = (long long)m * k;
    if (!gate_h || !up_h || !down_h || !h || !gates || !out || m <= 0 || k <= 0 || P > INT_MAX ||
        d_model <= 0 || e_ffn <= 0 || d_model % 256 != 0 || e_ffn % 256 != 0) return 1;
    size_t np = (size_t)P, nm = (size_t)m, dm = (size_t)d_model, ef = (size_t)e_ffn;
    if (dm > SIZE_MAX / nm || nm * dm > SIZE_MAX / 8 || ef > SIZE_MAX / np || np * ef > SIZE_MAX / 8 ||
        dm > SIZE_MAX / np || np * dm > SIZE_MAX / 8) return 1;
    uint8_t **pg = nullptr, **pu = nullptr, **pd = nullptr;
    int *tok = nullptr, qg = -1, qu = -1, qd = -1;
    if (moe_host_scratch(np, &pg, &pu, &pd, &tok) != 0) return 1;
    long long nb_in = d_model / 256, nb_dn = e_ffn / 256;
    for (long long p = 0; p < P; p++) {
        long long g = gate_h[p], u = up_h[p], d = down_h[p];
        if (!valid_quant_handle(g) || !valid_quant_handle(u) || !valid_quant_handle(d) ||
            g_reg[g].out_f != e_ffn || g_reg[g].n_blocks != nb_in ||
            g_reg[u].out_f != e_ffn || g_reg[u].n_blocks != nb_in ||
            g_reg[d].out_f != d_model || g_reg[d].n_blocks != nb_dn) return 1;
        pg[p] = g_reg[g].p; pu[p] = g_reg[u].p; pd[p] = g_reg[d].p; tok[p] = (int)(p / k);
        int eg = g_reg[g].qtype, eu = g_reg[u].qtype, ed = g_reg[d].qtype;
        if (p == 0) { qg = eg; qu = eu; qd = ed; }
        else if (qg != eg || qu != eu || qd != ed) return 1; // matmul_multi_tok takes one qtype per stage
    }
    uint8_t **dpg = (uint8_t **)moe_scratch(MS_PG, np * sizeof(uint8_t *));
    uint8_t **dpu = (uint8_t **)moe_scratch(MS_PU, np * sizeof(uint8_t *));
    uint8_t **dpd = (uint8_t **)moe_scratch(MS_PD, np * sizeof(uint8_t *));
    int *dtok = (int *)moe_scratch(MS_TOK, np * sizeof(int));
    int64_t *dh = (int64_t *)moe_scratch(MS_H, nm * dm * 8);
    int64_t *dgs = (int64_t *)moe_scratch(MS_GATES, np * 8);
    int64_t *dout = (int64_t *)moe_scratch(MS_OUT, nm * dm * 8);
    int64_t *dg = (int64_t *)moe_scratch(MS_GATE_OUT, np * ef * 8);
    int64_t *du = (int64_t *)moe_scratch(MS_UP_OUT, np * ef * 8);
    int64_t *dd = (int64_t *)moe_scratch(MS_DOWN_OUT, np * dm * 8);
    if (!dpg || !dpu || !dpd || !dtok || !dh || !dgs || !dout || !dg || !du || !dd) return 1;
    #define H2D(d, s, sz) if (cudaMemcpy((d), (s), (sz), cudaMemcpyHostToDevice) != cudaSuccess) return 1
    H2D(dpg, pg, P * sizeof(uint8_t *)); H2D(dpu, pu, P * sizeof(uint8_t *)); H2D(dpd, pd, P * sizeof(uint8_t *));
    H2D(dtok, tok, P * sizeof(int)); H2D(dh, h, (long long)m * d_model * 8); H2D(dgs, gates, P * 8);
    #undef H2D
    {
        int tpb = 128; long long nge = P * e_ffn, ngd = P * d_model;
        matmul_multi_tok<<<(nge + tpb - 1) / tpb, tpb>>>(dpg, qg, dh, 0, dtok, e_ffn, nb_in, fw, (int)P, dg);
        silu_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, nge, fa);
        matmul_multi_tok<<<(nge + tpb - 1) / tpb, tpb>>>(dpu, qu, dh, 0, dtok, e_ffn, nb_in, fw, (int)P, du);
        mul_shift_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, du, nge, fa);
        matmul_multi_tok<<<(ngd + tpb - 1) / tpb, tpb>>>(dpd, qd, dg, 1, dtok, d_model, nb_dn, fw, (int)P, dd);
        combine_tok<<<((long long)m * d_model + tpb - 1) / tpb, tpb>>>(dd, m, k, d_model, dgs, fa, dout);
    }
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 1;
    return cudaMemcpy(out, dout, (long long)m * d_model * 8, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}

// Batched expert-prefill DP4A.  Unlike the M=1 path, each selected expert pair
// maps to one of m shared token inputs through tok_of, so h limbs/x-sums are
// generated once per token rather than duplicated k times.  Return code 2 is a
// clean exact-envelope miss; the Python bridge immediately dispatches the
// unchanged int128 batched entry.  All other non-zero codes are runtime errors.
extern "C" int qk_moe_ffn_batched_dp4a(const long long *gate_h, const long long *up_h,
                                        const long long *down_h, int m, int k, const int64_t *h,
                                        const int64_t *gates, long long d_model, long long e_ffn,
                                        int fa, int fw, int64_t *out) {
    profile_native_call(PM_MOE_BATCH_DP4A_CALLS);
    long long P = (long long)m * k;
    if (!gate_h || !up_h || !down_h || !h || !gates || !out || m <= 0 || k <= 0 || P > INT_MAX ||
        d_model <= 0 || e_ffn <= 0 || d_model % 256 != 0 || e_ffn % 256 != 0) return 1;
    size_t np = (size_t)P, nm = (size_t)m, dm = (size_t)d_model, ef = (size_t)e_ffn;
    if (dm > SIZE_MAX / nm || nm * dm > SIZE_MAX / 8 || ef > SIZE_MAX / np || np * ef > SIZE_MAX / 8 ||
        dm > SIZE_MAX / np || np * dm > SIZE_MAX / 8) return 1;

    unsigned long long mxh = 0;
    for (size_t i = 0; i < nm * dm; i++) {
        int64_t v = h[i];
        unsigned long long a = v < 0 ? (~(unsigned long long)v + 1ULL) : (unsigned long long)v;
        if (a > mxh) mxh = a;
    }
    int Lh = _ln_for(mxh);
    if (Lh == 0 || (size_t)Lh > SIZE_MAX / (nm * dm)) return 2;

    uint8_t **pg = nullptr, **pu = nullptr, **pd = nullptr;
    int *tok = nullptr, qg = -1, qu = -1, qd = -1;
    if (moe_host_scratch(np, &pg, &pu, &pd, &tok) != 0) return 1;
    long long nb_in = d_model / 256, nb_dn = e_ffn / 256;
    for (long long p = 0; p < P; p++) {
        long long g = gate_h[p], u = up_h[p], d = down_h[p];
        if (!valid_quant_handle(g) || !valid_quant_handle(u) || !valid_quant_handle(d) ||
            g_reg[g].out_f != e_ffn || g_reg[g].n_blocks != nb_in ||
            g_reg[u].out_f != e_ffn || g_reg[u].n_blocks != nb_in ||
            g_reg[d].out_f != d_model || g_reg[d].n_blocks != nb_dn) return 1;
        pg[p] = g_reg[g].p; pu[p] = g_reg[u].p; pd[p] = g_reg[d].p; tok[p] = (int)(p / k);
        int eg = g_reg[g].qtype, eu = g_reg[u].qtype, ed = g_reg[d].qtype;
        if (p == 0) { qg = eg; qu = eu; qd = ed; }
        else if (qg != eg || qu != eu || qd != ed) return 1;
    }

    uint8_t **dpg = (uint8_t **)moe_scratch(MS_PG, np * sizeof(uint8_t *));
    uint8_t **dpu = (uint8_t **)moe_scratch(MS_PU, np * sizeof(uint8_t *));
    uint8_t **dpd = (uint8_t **)moe_scratch(MS_PD, np * sizeof(uint8_t *));
    int *dtok = (int *)moe_scratch(MS_TOK, np * sizeof(int));
    int64_t *dh = (int64_t *)moe_scratch(MS_H, nm * dm * 8);
    int64_t *dgs = (int64_t *)moe_scratch(MS_GATES, np * 8);
    int64_t *dout = (int64_t *)moe_scratch(MS_OUT, nm * dm * 8);
    int64_t *dg = (int64_t *)moe_scratch(MS_GATE_OUT, np * ef * 8);
    int64_t *du = (int64_t *)moe_scratch(MS_UP_OUT, np * ef * 8);
    int64_t *dd = (int64_t *)moe_scratch(MS_DOWN_OUT, np * dm * 8);
    int8_t *dhl = (int8_t *)moe_scratch(MS_H_LIMBS, (size_t)Lh * nm * dm);
    int64_t *dhxs = (int64_t *)moe_scratch(MS_H_XSUM, nm * (size_t)nb_in * 8 * 8);
    unsigned long long *dmax = (unsigned long long *)moe_scratch(MS_MAX, sizeof(unsigned long long));
    if (!dpg || !dpu || !dpd || !dtok || !dh || !dgs || !dout || !dg || !du || !dd ||
        !dhl || !dhxs || !dmax) return 1;
    #define H2D_DP4A(d, s, sz) if (cudaMemcpy((d), (s), (sz), cudaMemcpyHostToDevice) != cudaSuccess) return 1
    H2D_DP4A(dpg, pg, np * sizeof(uint8_t *)); H2D_DP4A(dpu, pu, np * sizeof(uint8_t *));
    H2D_DP4A(dpd, pd, np * sizeof(uint8_t *)); H2D_DP4A(dtok, tok, np * sizeof(int));
    H2D_DP4A(dh, h, nm * dm * 8); H2D_DP4A(dgs, gates, np * 8);
    #undef H2D_DP4A

    int tpb = 128;
    long long nh = (long long)m * d_model, nge = P * e_ffn, ngd = P * d_model;
    make_limbs<<<(nh + tpb - 1) / tpb, tpb>>>(dh, nh, nh, Lh, dhl);
    make_xsum<<<((long long)m * nb_in * 8 + tpb - 1) / tpb, tpb>>>(dh, m, nb_in, dhxs);
    matmul_multi_dp4a<<<(nge + tpb - 1) / tpb, tpb>>>(dpg, qg, dhl, dhxs, 0, dtok, m,
                                                       e_ffn, nb_in, fw, Lh, (int)P, dg);
    silu_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, nge, fa);
    matmul_multi_dp4a<<<(nge + tpb - 1) / tpb, tpb>>>(dpu, qu, dhl, dhxs, 0, dtok, m,
                                                       e_ffn, nb_in, fw, Lh, (int)P, du);
    mul_shift_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, du, nge, fa);

    if (cudaMemset(dmax, 0, sizeof(unsigned long long)) != cudaSuccess) return 1;
    maxabs_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, nge, dmax);
    unsigned long long mxg = 0;
    if (cudaMemcpy(&mxg, dmax, sizeof(mxg), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    int Lg = _ln_for(mxg);
    if (Lg == 0 || (size_t)Lg > SIZE_MAX / (np * ef)) return 2;
    int8_t *dgl = (int8_t *)moe_scratch(MS_G_LIMBS, (size_t)Lg * np * ef);
    int64_t *dgxs = (int64_t *)moe_scratch(MS_G_XSUM, np * (size_t)nb_dn * 8 * 8);
    if (!dgl || !dgxs) return 1;
    make_limbs<<<(nge + tpb - 1) / tpb, tpb>>>(dg, nge, nge, Lg, dgl);
    make_xsum<<<(P * nb_dn * 8 + tpb - 1) / tpb, tpb>>>(dg, P, nb_dn, dgxs);
    matmul_multi_dp4a<<<(ngd + tpb - 1) / tpb, tpb>>>(dpd, qd, dgl, dgxs, 1, nullptr, P,
                                                       d_model, nb_dn, fw, Lg, (int)P, dd);
    combine_tok<<<((long long)m * d_model + tpb - 1) / tpb, tpb>>>(dd, m, k, d_model, dgs, fa, dout);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 1;
    return cudaMemcpy(out, dout, nm * dm * 8, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}

extern "C" int qk_cuda_available(void) {     // 0 = a usable GPU is present
    int n = 0;
    return (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) ? 1 : 0;
}

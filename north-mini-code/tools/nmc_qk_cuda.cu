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
#include <cuda_runtime.h>

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

#define CK(call) do { if ((call) != cudaSuccess) { return 1; } } while (0)

// qtype: 0=Q4_K, 1=Q6_K. Handles H2D/D2H internally (like the Bonsai gpu_native entries). Returns 0 ok.
extern "C" int qk_linear_cuda(const uint8_t *W, const int64_t *x, long long T, long long out_f,
                              long long n_blocks, int fw, int qtype, int64_t *out) {
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
// Plain C array registry (no std::vector → no libstdc++ dependency, so the .so stays a pure C/CUDA object that
// loads on any arch — pulling in libstdc++ broke loading on freshly-built sm_89). Sized for the whole model
// (442 dense/attn/head + touched experts).
#define NMC_MAX_REG 16384
struct DevWeight { uint8_t *p; long long out_f, n_blocks; int qtype; };
static struct DevWeight g_reg[NMC_MAX_REG];
static int g_nreg = 0;

// Upload one weight tensor's raw Q4_K/Q6_K bytes; return a handle (index) or -1.
extern "C" long long qk_register_weight(const uint8_t *W, long long out_f, long long n_blocks, int qtype) {
    if (g_nreg >= NMC_MAX_REG) return -1;
    long long bs = qtype == 0 ? 144 : 210;
    size_t bytes = (size_t)out_f * n_blocks * bs;
    uint8_t *d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) return -1;
    if (cudaMemcpy(d, W, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return -1; }
    g_reg[g_nreg].p = d; g_reg[g_nreg].out_f = out_f; g_reg[g_nreg].n_blocks = n_blocks; g_reg[g_nreg].qtype = qtype;
    return (long long)(g_nreg++);
}

// Persistent activation scratch (dx/dout), grown on demand and REUSED across applies — decode does ~1350
// applies/token, and a cudaMalloc+cudaFree pair per call (each a device sync) dominated the m=1 hot path.
static int64_t *g_dx = nullptr, *g_dout = nullptr;
static int8_t *g_xl = nullptr; static size_t g_xl_cap = 0;       // DP4A activation-limb scratch
static int64_t *g_xs = nullptr; static size_t g_xs_cap = 0;      // Q4_K DP4A per-subgroup Σx scratch
static size_t g_dx_cap = 0, g_dout_cap = 0;
static int64_t *scratch(int64_t **p, size_t *cap, size_t need) {
    if (*cap < need) {
        if (*p) cudaFree(*p);
        if (cudaMalloc(p, need) != cudaSuccess) { *p = nullptr; *cap = 0; return nullptr; }
        *cap = need;
    }
    return *p;
}

// Apply a resident weight: y[t,o] = (Σ_i W_fixed[o,i]·x[t,i]) >> fw. Only x is H2D'd, y D2H'd — no weight
// transfer. Byte-identical to qk_linear (same kernel, resident pointer). Returns 0 ok.
extern "C" int qk_apply_resident(long long h, const int64_t *x, long long T, int fw, int64_t *out) {
    if (h < 0 || h >= g_nreg || g_reg[h].p == nullptr) return 1;
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

extern "C" void qk_free_all(void) {
    for (int i = 0; i < g_nreg; i++) if (g_reg[i].p) cudaFree(g_reg[i].p);
    g_nreg = 0;
    if (g_dx) { cudaFree(g_dx); g_dx = nullptr; g_dx_cap = 0; }
    if (g_dout) { cudaFree(g_dout); g_dout = nullptr; g_dout_cap = 0; }
    if (g_xl) { cudaFree(g_xl); g_xl = nullptr; g_xl_cap = 0; }
    if (g_xs) { cudaFree(g_xs); g_xs = nullptr; g_xs_cap = 0; }
}
extern "C" long long qk_resident_count(void) { return (long long)g_nreg; }

// Fused batched MoE expert-FFN on the GPU: out[d_model] = Σ_e gate_e · down_e( silu(gate_e·h) * up_e·h ).
// gate_h/up_h/down_h are resident handles for the n_e selected experts; gates[e] = sigmoid(router logit).
// ONE call per MoE layer (vs n_e·3 apply_resident + host silu/combine) with no intermediate H2D/D2H. Byte-
// identical to the per-expert CPU path (same dequant matmul, same integer SiLU, same shift-then-sum combine).
extern "C" int qk_moe_ffn(const long long *gate_h, const long long *up_h, const long long *down_h, int n_e,
                          const int64_t *h, const int64_t *gates, long long d_model, long long e_ffn,
                          int fa, int fw, int64_t *out) {
    if (n_e <= 0 || n_e > 256) return 1;
    uint8_t *pg[256], *pu[256], *pd[256]; int qg[256], qu[256], qd[256];
    for (int e = 0; e < n_e; e++) {
        long long g = gate_h[e], u = up_h[e], d = down_h[e];
        if (g < 0 || g >= g_nreg || u < 0 || u >= g_nreg || d < 0 || d >= g_nreg) return 1;
        pg[e] = g_reg[g].p; pu[e] = g_reg[u].p; pd[e] = g_reg[d].p;
        qg[e] = g_reg[g].qtype; qu[e] = g_reg[u].qtype; qd[e] = g_reg[d].qtype;
    }
    long long nb_in = d_model / 256, nb_dn = e_ffn / 256;
    uint8_t **dpg = 0, **dpu = 0, **dpd = 0; int *dqg = 0, *dqu = 0, *dqd = 0;
    int64_t *dh = 0, *dg = 0, *du = 0, *dd = 0, *dgs = 0, *dout = 0;
    int rc = 1;
    #define A(p, sz) if (cudaMalloc(&(p), (sz)) != cudaSuccess) goto done
    A(dpg, n_e * sizeof(uint8_t *)); A(dpu, n_e * sizeof(uint8_t *)); A(dpd, n_e * sizeof(uint8_t *));
    A(dqg, n_e * sizeof(int)); A(dqu, n_e * sizeof(int)); A(dqd, n_e * sizeof(int));
    A(dh, d_model * 8); A(dgs, (long long)n_e * 8);
    A(dg, (long long)n_e * e_ffn * 8); A(du, (long long)n_e * e_ffn * 8);
    A(dd, (long long)n_e * d_model * 8); A(dout, d_model * 8);
    #undef A
    #define H2D(d, s, sz) if (cudaMemcpy((d), (s), (sz), cudaMemcpyHostToDevice) != cudaSuccess) goto done
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
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) goto done;
    if (cudaMemcpy(out, dout, d_model * 8, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    cudaFree(dpg); cudaFree(dpu); cudaFree(dpd); cudaFree(dqg); cudaFree(dqu); cudaFree(dqd);
    cudaFree(dh); cudaFree(dgs); cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dout);
    return rc;
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
    if (h < 0 || h >= g_nreg || g_reg[h].p == nullptr) return 1;
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
    if (h < 0 || h >= g_nreg || g_reg[h].p == nullptr) return 1;
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
    int x_per_expert, long long out_f, long long n_blocks, int fw, int Ln, int n_e, int64_t *yout) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n_e * out_f) return;
    int e = idx / out_f; long long o = idx % out_f, in_f = n_blocks * 256;
    const uint8_t *W = wptrs[e];
    long long ls = x_per_expert ? (long long)n_e * in_f : in_f;
    const int8_t *xlb = xlimb + (x_per_expert ? (size_t)e * in_f : 0);
    __int128 acc = 0;
    if (qt == 0) {                                              // Q4_K (affine)
        const int64_t *xsb = xsum + (x_per_expert ? (long long)e * n_blocks * 8 : 0);
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
    if (n_e <= 0 || n_e > 256) return 1;
    uint8_t *pg[256], *pu[256], *pd[256];
    int qg = -1, qu = -1, qd = -1;
    for (int e = 0; e < n_e; e++) {
        long long g = gate_h[e], u = up_h[e], d = down_h[e];
        if (g < 0 || g >= g_nreg || u < 0 || u >= g_nreg || d < 0 || d >= g_nreg) return 1;
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
        matmul_multi_dp4a<<<(nge + tpb - 1) / tpb, tpb>>>(dpg, qg, dhl, dhxs, 0, e_ffn, nb_in, fw, Lh, n_e, dg);
        silu_k<<<(nge + tpb - 1) / tpb, tpb>>>(dg, nge, fa);
        matmul_multi_dp4a<<<(nge + tpb - 1) / tpb, tpb>>>(dpu, qu, dhl, dhxs, 0, e_ffn, nb_in, fw, Lh, n_e, du);
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
        matmul_multi_dp4a<<<(ngd + tpb - 1) / tpb, tpb>>>(dpd, qd, dgl, dgxs, 1, d_model, nb_dn, fw, Lg, n_e, dd);
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
    long long P = (long long)m * k;
    if (m <= 0 || k <= 0) return 1;
    uint8_t **pg = (uint8_t **)malloc(P * sizeof(uint8_t *)), **pu = (uint8_t **)malloc(P * sizeof(uint8_t *)),
            **pd = (uint8_t **)malloc(P * sizeof(uint8_t *));
    int *tok = (int *)malloc(P * sizeof(int)), qg = -1, qu = -1, qd = -1, ok = 1;
    if (!pg || !pu || !pd || !tok) ok = 0;
    for (long long p = 0; ok && p < P; p++) {
        long long g = gate_h[p], u = up_h[p], d = down_h[p];
        if (g < 0 || g >= g_nreg || u < 0 || u >= g_nreg || d < 0 || d >= g_nreg) { ok = 0; break; }
        pg[p] = g_reg[g].p; pu[p] = g_reg[u].p; pd[p] = g_reg[d].p; tok[p] = (int)(p / k);
        qg = g_reg[g].qtype; qu = g_reg[u].qtype; qd = g_reg[d].qtype;
    }
    long long nb_in = d_model / 256, nb_dn = e_ffn / 256;
    uint8_t **dpg = 0, **dpu = 0, **dpd = 0; int *dtok = 0;
    int64_t *dh = 0, *dg = 0, *du = 0, *dd = 0, *dgs = 0, *dout = 0;
    int rc = 1;
    if (!ok) goto done;
    #define A(p, sz) if (cudaMalloc(&(p), (sz)) != cudaSuccess) goto done
    A(dpg, P * sizeof(uint8_t *)); A(dpu, P * sizeof(uint8_t *)); A(dpd, P * sizeof(uint8_t *)); A(dtok, P * sizeof(int));
    A(dh, (long long)m * d_model * 8); A(dgs, P * 8); A(dout, (long long)m * d_model * 8);
    A(dg, P * e_ffn * 8); A(du, P * e_ffn * 8); A(dd, P * d_model * 8);
    #undef A
    #define H2D(d, s, sz) if (cudaMemcpy((d), (s), (sz), cudaMemcpyHostToDevice) != cudaSuccess) goto done
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
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) goto done;
    if (cudaMemcpy(out, dout, (long long)m * d_model * 8, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    free(pg); free(pu); free(pd); free(tok);
    cudaFree(dpg); cudaFree(dpu); cudaFree(dpd); cudaFree(dtok);
    cudaFree(dh); cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dgs); cudaFree(dout);
    return rc;
}

extern "C" int qk_cuda_available(void) {     // 0 = a usable GPU is present
    int n = 0;
    return (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) ? 1 : 0;
}

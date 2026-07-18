#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define BONSAI_CAN_BUILD_AVX2 1
#else
#define BONSAI_CAN_BUILD_AVX2 0
#endif

#ifdef _OPENMP
#include <omp.h>
#else
// Keep the portable single-thread build self-contained.  Unknown OpenMP
// pragmas are ignored by C compilers, but the runtime query calls still need
// definitions when the file is compiled and linked without -fopenmp.
static int omp_get_max_threads(void) { return 1; }
static int omp_get_num_threads(void) { return 1; }
static int omp_get_thread_num(void) { return 0; }
#endif

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int checked_add_size(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static int checked_mul3_size(size_t a, size_t b, size_t c, size_t *out) {
    size_t ab = 0;
    return checked_mul_size(a, b, &ab) && checked_mul_size(ab, c, out);
}

static int checked_i64_to_size(int64_t value, size_t *out) {
    if (value < 0 || (uint64_t) value > (uint64_t) SIZE_MAX) {
        return 0;
    }
    *out = (size_t) value;
    return 1;
}

static int64_t u64_to_i64(uint64_t u) {
    if (u <= (uint64_t) INT64_MAX) {
        return (int64_t) u;
    }
    uint64_t magnitude = (~u) + 1u;
    if (magnitude == (1ull << 63)) {
        return INT64_MIN;
    }
    return -(int64_t) magnitude;
}

static int64_t arshift_i64(int64_t v, int64_t shift) {
    if (shift <= 0) {
        return v;
    }
    if (shift >= 63) {
        return v < 0 ? -1 : 0;
    }
    if (v >= 0) {
        return v >> shift;
    }
    uint64_t magnitude = (uint64_t) (-(v + 1)) + 1u;
    uint64_t q = (magnitude + ((1ull << shift) - 1u)) >> shift;
    return -(int64_t) q;
}

// Shared per-output-element Q1_0 accumulation. Defined once and instantiated for int64 and int32 scale
// arrays so the optional narrow-scale-cache path (see *_scale32 kernels below) is byte-identical to the
// committed int64 path BY CONSTRUCTION: `(uint64_t)(int64_t) scale` is the same 64-bit multiply operand
// whether the scale was read from an int64 array or a lossless int32 cache of the same values. The
// staged-doubling LUT, the signed_sum = 2*pos_sum - block_total reduction, the per-block arithmetic
// shift, and the two's-complement wrap are all unchanged. `token_base` = t * n_blocks indexes this
// token's slice of the per-block totals/LUT workspace.
#define DEFINE_Q1_ELEMENT(NAME, STYPE)                                                  \
static inline int64_t NAME(const uint8_t *bits_row, const STYPE *scale_row,              \
                           int64_t n_blocks, int64_t frac, size_t token_base,            \
                           const uint64_t *totals, const uint64_t *lut) {                \
    uint64_t total = 0;                                                                  \
    for (int64_t b = 0; b < n_blocks; ++b) {                                             \
        const uint8_t *bb = bits_row + (size_t) b * 16u;                                 \
        const uint64_t block_total = totals[token_base + (size_t) b];                    \
        uint64_t pos_sum = 0;                                                            \
        const uint64_t *block_lut = lut + ((token_base + (size_t) b) * 16u) * 256u;      \
        for (int byte_i = 0; byte_i < 16; ++byte_i) {                                    \
            pos_sum += block_lut[(size_t) byte_i * 256u + (size_t) bb[byte_i]];          \
        }                                                                               \
        uint64_t signed_sum = 2u * pos_sum - block_total;                               \
        uint64_t prod = signed_sum * (uint64_t) (int64_t) scale_row[b];                 \
        total += (uint64_t) arshift_i64(u64_to_i64(prod), frac);                        \
    }                                                                                   \
    return u64_to_i64(total);                                                           \
}

DEFINE_Q1_ELEMENT(q1_element_s64, int64_t)
DEFINE_Q1_ELEMENT(q1_element_s32, int32_t)

static unsigned __int128 abs_i64_u128(int64_t v) {
    if (v >= 0) {
        return (unsigned __int128) v;
    }
    return (unsigned __int128) (-(v + 1)) + 1u;
}

static int add_square_u128(unsigned __int128 *acc, int64_t v) {
    const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
    unsigned __int128 a = abs_i64_u128(v);
    if (a != 0 && a > max_u128 / a) {
        return 0;
    }
    unsigned __int128 term = a * a;
    if (*acc > max_u128 - term) {
        return 0;
    }
    *acc += term;
    return 1;
}

static uint64_t isqrt_u128(unsigned __int128 n) {
    unsigned __int128 res = 0;
    unsigned __int128 bit = (unsigned __int128) 1 << 126;
    while (bit > n) {
        bit >>= 2;
    }
    while (bit != 0) {
        if (n >= res + bit) {
            n -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return (uint64_t) res;
}

static __int128 floor_div_i128_u64(__int128 n, uint64_t d) {
    __int128 denom = (__int128) d;
    __int128 q = n / denom;
    __int128 r = n % denom;
    if (r != 0 && n < 0) {
        q -= 1;
    }
    return q;
}

// Exact fast path for floor((value * 2^shift) / divisor) when the scaled
// magnitude fits uint64 and the quotient fits int64.  The unsigned magnitude
// formulation handles INT64_MIN without signed overflow and explicitly adds
// one to a negative non-integral quotient to reproduce floor rather than C's
// truncation toward zero.  Callers retain the i128 path for the uncommon
// out-of-envelope case.
static int floor_scaled_i64_u64(int64_t value, int64_t shift,
                                uint64_t divisor, int64_t *out) {
    if (!out || !divisor || shift < 0 || shift >= 64) return 0;
    uint64_t magnitude = value < 0
        ? (~(uint64_t) value) + 1u
        : (uint64_t) value;
    if (magnitude > (UINT64_MAX >> shift)) return 0;
    magnitude <<= shift;
    uint64_t quotient = magnitude / divisor;
    const uint64_t remainder = magnitude % divisor;
    if (value < 0) {
        if (remainder) {
            if (quotient == UINT64_MAX) return 0;
            quotient++;
        }
        if (quotient > (UINT64_C(1) << 63)) return 0;
        *out = quotient == (UINT64_C(1) << 63)
            ? INT64_MIN : -(int64_t) quotient;
        return 1;
    }
    if (quotient > (uint64_t) INT64_MAX) return 0;
    *out = (int64_t) quotient;
    return 1;
}

static __int128 floor_shift_i128(__int128 v, int64_t shift) {
    if (shift <= 0) {
        return v;
    }
    if (shift >= 126) {
        return v < 0 ? -1 : 0;
    }
    __int128 denom = (__int128) 1 << shift;
    __int128 q = v / denom;
    __int128 r = v % denom;
    if (r != 0 && v < 0) {
        q -= 1;
    }
    return q;
}

static int i128_to_i64(__int128 v, int64_t *out) {
    const __int128 lo = -((__int128) INT64_MAX) - 1;
    const __int128 hi = (__int128) INT64_MAX;
    if (v < lo || v > hi) {
        return 0;
    }
    *out = (int64_t) v;
    return 1;
}

static void prepare_q1_block_lut(const int64_t *xb, uint64_t *total_out, uint64_t *lut_block) {
    uint64_t total = 0;
    for (int i = 0; i < 128; ++i) {
        total += (uint64_t) xb[i];
    }
    *total_out = total;

    for (int byte_i = 0; byte_i < 16; ++byte_i) {
        const int64_t *xp = xb + byte_i * 8;
        uint64_t *table = lut_block + (size_t) byte_i * 256u;
        table[0] = 0;
        for (int bit = 0; bit < 8; ++bit) {
            const size_t base = (size_t) 1u << bit;
            const uint64_t add = (uint64_t) xp[bit];
            for (size_t mask = 0; mask < base; ++mask) {
                table[base + mask] = table[mask] + add;
            }
        }
    }
}

// Native RMSNorm fast path for in-envelope rows. It preserves the Python reference's integer semantics:
// exact sum-of-squares, integer sqrt, floor division for negative values, and floor power-of-two shifts.
// Return 4 means a row exceeded the 128-bit fast envelope; callers should fall back to the big-int oracle.
int bonsai_rmsnorm_i64(const int64_t *x,
                       int64_t rows,
                       int64_t cols,
                       int64_t frac,
                       int64_t eps,
                       const int64_t *gain,
                       int64_t *out) {
    if (!x || !out || rows < 0 || cols <= 0 || frac < 0 || frac > 62 || eps < 0) {
        return 1;
    }
    size_t srows = (size_t) rows;
    size_t scols = (size_t) cols;
    size_t total = 0;
    if (!checked_mul_size(srows, scols, &total)) {
        return 1;
    }

    int rc = 0;
    const __int128 fp = (__int128) 1 << frac;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        int local_rc = 0;
        const int64_t *row = x + (size_t) r * scols;
        int64_t *dst = out + (size_t) r * scols;
        unsigned __int128 ssq = 0;
        for (int64_t c = 0; c < cols; ++c) {
            if (!add_square_u128(&ssq, row[c])) {
                local_rc = 4;
                break;
            }
        }
        if (local_rc == 0) {
            const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
            unsigned __int128 mean = ssq / (unsigned __int128) cols;
            if (mean > max_u128 - (unsigned __int128) eps) {
                local_rc = 4;
            } else {
                mean += (unsigned __int128) eps;
            }
            uint64_t rms = local_rc == 0 ? isqrt_u128(mean) : 0;
            if (rms == 0) {
                local_rc = 4;
            }
            if (local_rc == 0 && gain) {
                // Mirror the oracle's coarse fail-loud envelope (determinism/fixedpoint.py::fixed_point_rmsnorm):
                // it does out*g in int64 and REFUSES when max|out|*max|gain| > INT64_MAX (before the >>frac).
                // We do the multiply in __int128 (no wrap), but must refuse under the SAME condition so native
                // falls back (rc 4 -> the oracle, which raises) instead of returning a value the oracle rejects
                // — keeping the two paths in lockstep on WHEN they refuse, not only on the value they compute.
                // Division form avoids any 128-bit wrap when |normalized| itself is huge.
                unsigned __int128 max_norm = 0, max_gain = 0;
                for (int64_t c = 0; c < cols; ++c) {
                    __int128 nrm = floor_div_i128_u64((__int128) row[c] * fp, rms);
                    unsigned __int128 an = (unsigned __int128) (nrm < 0 ? -nrm : nrm);
                    if (an > max_norm) max_norm = an;
                    int64_t gc = gain[c];
                    unsigned __int128 ag = (unsigned __int128) (gc < 0 ? -(__int128) gc : (__int128) gc);
                    if (ag > max_gain) max_gain = ag;
                }
                if (max_norm != 0 && max_gain > (unsigned __int128) INT64_MAX / max_norm) {
                    local_rc = 4;
                }
            }
            for (int64_t c = 0; local_rc == 0 && c < cols; ++c) {
                __int128 normalized = floor_div_i128_u64((__int128) row[c] * fp, rms);
                __int128 y = normalized;
                if (gain) {
                    y = floor_shift_i128(normalized * (__int128) gain[c], frac);
                }
                if (!i128_to_i64(y, &dst[c])) {
                    local_rc = 4;
                    break;
                }
            }
        }
        if (local_rc != 0) {
            #pragma omp critical
            {
                if (rc == 0) {
                    rc = local_rc;
                }
            }
        }
    }
    return rc;
}

// BUFFER-EXTENT CONTRACT (L16): these kernels take only the logical dims (tokens, out_features,
// n_blocks, frac); they DO NOT receive the byte length of x/bits/scale/out. The caller MUST size every
// buffer to match those dims exactly — x: tokens*n_blocks*128, bits: out_features*n_blocks*16,
// scale: out_features*n_blocks, out: tokens*out_features (and totals/lut for the *_workspace/*_prepared
// variants per checked_mul_size below). The Python ctypes wrapper derives all of these from the array
// shapes, so it is always consistent; a hand-written caller passing mismatched extents would read/write
// out of bounds. Only the explicit totals_count/lut_count are bounds-checked here (return 3 on short
// workspace); the x/bits/scale/out extents are NOT validated and are the caller's responsibility.
//
// Packed GGUF Q1_0 linear:
// x:     (tokens, n_blocks * 128) int64 fixed-point
// bits:  (out_features, n_blocks, 16) uint8, little-endian bits, 0 -> -1, 1 -> +1
// scale: (out_features, n_blocks) int64 fixed-point
// out:   (tokens, out_features) int64 fixed-point
int bonsai_q1_linear_i64(const int64_t *x,
                         const uint8_t *bits,
                         const int64_t *scale,
                         int64_t tokens,
                         int64_t out_features,
                         int64_t n_blocks,
                         int64_t frac,
                         int64_t *out) {
    if (!x || !bits || !scale || !out || tokens < 0 || out_features < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }

    size_t stokens = (size_t) tokens;
    size_t sout_features = (size_t) out_features;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t x_width = 0;
    size_t bits_row_stride = 0;
    size_t out_count = 0;
    size_t lut_count = 0;
    size_t totals_bytes = 0;
    size_t lut_bytes = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(stokens, sout_features, &out_count) ||
        !checked_mul_size(tb, 16u, &lut_count) ||
        !checked_mul_size(lut_count, 256u, &lut_count) ||
        !checked_mul_size(tb, sizeof(uint64_t), &totals_bytes) ||
        !checked_mul_size(lut_count, sizeof(uint64_t), &lut_bytes)) {
        return 1;
    }

    uint64_t *totals = (uint64_t *) malloc(totals_bytes);
    uint64_t *lut = (uint64_t *) malloc(lut_bytes);
    if (!totals || !lut) {
        free(totals);
        free(lut);
        return 2;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            prepare_q1_block_lut(xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t o = 0; o < out_features; ++o) {
            const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
            const int64_t *scale_row = scale + (size_t) o * sn_blocks;
            out[(size_t) t * sout_features + (size_t) o] =
                q1_element_s64(bits_row, scale_row, n_blocks, frac,
                               (size_t) t * (size_t) n_blocks, totals, lut);
        }
    }
    free(totals);
    free(lut);
    return 0;
}

int bonsai_q1_linear_i64_workspace(const int64_t *x,
                                   const uint8_t *bits,
                                   const int64_t *scale,
                                   int64_t tokens,
                                   int64_t out_features,
                                   int64_t n_blocks,
                                   int64_t frac,
                                   int64_t *out,
                                   uint64_t *totals,
                                   size_t totals_count,
                                   uint64_t *lut,
                                   size_t lut_count) {
    if (!x || !bits || !scale || !out || !totals || !lut ||
        tokens < 0 || out_features < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }

    size_t stokens = (size_t) tokens;
    size_t sout_features = (size_t) out_features;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t x_width = 0;
    size_t bits_row_stride = 0;
    size_t out_count = 0;
    size_t need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(stokens, sout_features, &out_count) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            prepare_q1_block_lut(xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t o = 0; o < out_features; ++o) {
            const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
            const int64_t *scale_row = scale + (size_t) o * sn_blocks;
            out[(size_t) t * sout_features + (size_t) o] =
                q1_element_s64(bits_row, scale_row, n_blocks, frac,
                               (size_t) t * (size_t) n_blocks, totals, lut);
        }
    }
    return 0;
}

// out_features==0 INCONSISTENCY (L17): the linear variants above accept out_features == 0 (out_features
// < 0 is the only reject) and simply produce an empty (tokens x 0) output, whereas this argmax variant
// REJECTS out_features == 0 (out_features <= 0 below) because argmax over zero candidates has no defined
// result. This asymmetry is intentional, not a bug — argmax needs at least one column to pick a winner.
// The Python wrapper never calls argmax with a 0-width head, so the divergence is unreachable in practice.
int bonsai_q1_argmax_i64_workspace(const int64_t *x,
                                   const uint8_t *bits,
                                   const int64_t *scale,
                                   int64_t tokens,
                                   int64_t out_features,
                                   int64_t n_blocks,
                                   int64_t frac,
                                   int64_t *argmax_out,
                                   int64_t *max_out,
                                   uint64_t *totals,
                                   size_t totals_count,
                                   uint64_t *lut,
                                   size_t lut_count) {
    if (!x || !bits || !scale || !argmax_out || !max_out || !totals || !lut ||
        tokens < 0 || out_features <= 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }

    size_t stokens = (size_t) tokens;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t x_width = 0;
    size_t bits_row_stride = 0;
    size_t need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            prepare_q1_block_lut(xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }

    for (int64_t t = 0; t < tokens; ++t) {
        int64_t best_idx = 0;
        int64_t best_val = INT64_MIN;
        #pragma omp parallel
        {
            int64_t local_idx = -1;
            int64_t local_val = INT64_MIN;
            #pragma omp for schedule(static)
            for (int64_t o = 0; o < out_features; ++o) {
                const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                const int64_t *scale_row = scale + (size_t) o * sn_blocks;
                int64_t value = q1_element_s64(bits_row, scale_row, n_blocks, frac,
                                               (size_t) t * (size_t) n_blocks, totals, lut);
                if (local_idx < 0 || value > local_val) {
                    local_val = value;
                    local_idx = o;
                }
            }
            #pragma omp critical
            {
                if (local_idx >= 0 && (local_val > best_val ||
                    (local_val == best_val && local_idx < best_idx))) {
                    best_val = local_val;
                    best_idx = local_idx;
                }
            }
        }
        argmax_out[t] = best_idx;
        max_out[t] = best_val;
    }
    return 0;
}

int bonsai_q1_prepare_i64(const int64_t *x,
                          int64_t tokens,
                          int64_t n_blocks,
                          uint64_t *totals,
                          size_t totals_count,
                          uint64_t *lut,
                          size_t lut_count) {
    if (!x || !totals || !lut || tokens < 0 || n_blocks < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t x_width = 0;
    size_t need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            prepare_q1_block_lut(xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }
    return 0;
}

int bonsai_q1_linear_i64_prepared(const uint8_t *bits,
                                  const int64_t *scale,
                                  int64_t tokens,
                                  int64_t out_features,
                                  int64_t n_blocks,
                                  int64_t frac,
                                  int64_t *out,
                                  const uint64_t *totals,
                                  size_t totals_count,
                                  const uint64_t *lut,
                                  size_t lut_count) {
    if (!bits || !scale || !out || !totals || !lut ||
        tokens < 0 || out_features < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens;
    size_t sout_features = (size_t) out_features;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t bits_row_stride = 0;
    size_t out_count = 0;
    size_t need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(stokens, sout_features, &out_count) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t o = 0; o < out_features; ++o) {
            const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
            const int64_t *scale_row = scale + (size_t) o * sn_blocks;
            out[(size_t) t * sout_features + (size_t) o] =
                q1_element_s64(bits_row, scale_row, n_blocks, frac,
                               (size_t) t * (size_t) n_blocks, totals, lut);
        }
    }
    return 0;
}

int bonsai_q1_linear_i64_prepared_multi(const uint8_t *const *bits_list,
                                        const int64_t *const *scale_list,
                                        const int64_t *out_features_list,
                                        int64_t n_items,
                                        int64_t tokens,
                                        int64_t n_blocks,
                                        int64_t frac,
                                        int64_t **out_list,
                                        const uint64_t *totals,
                                        size_t totals_count,
                                        const uint64_t *lut,
                                        size_t lut_count) {
    if (!bits_list || !scale_list || !out_features_list || !out_list || !totals || !lut ||
        n_items < 0 || tokens < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t bits_row_stride = 0;
    size_t need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    for (int64_t item = 0; item < n_items; ++item) {
        if (!bits_list[item] || !scale_list[item] || !out_list[item] || out_features_list[item] < 0) {
            return 1;
        }
    }

    #pragma omp parallel
    {
        for (int64_t item = 0; item < n_items; ++item) {
            const uint8_t *bits = bits_list[item];
            const int64_t *scale = scale_list[item];
            int64_t *out = out_list[item];
            int64_t out_features = out_features_list[item];
            size_t sout_features = (size_t) out_features;
            #pragma omp for collapse(2) schedule(static)
            for (int64_t t = 0; t < tokens; ++t) {
                for (int64_t o = 0; o < out_features; ++o) {
                    const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                    const int64_t *scale_row = scale + (size_t) o * sn_blocks;
                    out[(size_t) t * sout_features + (size_t) o] =
                        q1_element_s64(bits_row, scale_row, n_blocks, frac,
                                       (size_t) t * (size_t) n_blocks, totals, lut);
                }
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Narrow int32 scale-cache variants (optional reproducer path; PERFORMANCE-DETERMINISM-REVIEW.md
// Recommendation 7). These are byte-identical to the int64 kernels above for any scale that fits int32
// losslessly — the committed artifact's Q1 scales are small positive values — because they share the
// q1_element_s32 helper, which promotes the int32 scale to the same (uint64_t)(int64_t) operand. The
// committed artifact is unchanged; the int32 array is a runtime-only cache built by the Python loader,
// and the int64 oracle remains the canonical path. Only the `scale` argument type differs.
// ---------------------------------------------------------------------------

int bonsai_q1_linear_i64_workspace_scale32(const int64_t *x,
                                           const uint8_t *bits,
                                           const int32_t *scale,
                                           int64_t tokens,
                                           int64_t out_features,
                                           int64_t n_blocks,
                                           int64_t frac,
                                           int64_t *out,
                                           uint64_t *totals,
                                           size_t totals_count,
                                           uint64_t *lut,
                                           size_t lut_count) {
    if (!x || !bits || !scale || !out || !totals || !lut ||
        tokens < 0 || out_features < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }

    size_t stokens = (size_t) tokens;
    size_t sout_features = (size_t) out_features;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t x_width = 0;
    size_t bits_row_stride = 0;
    size_t out_count = 0;
    size_t need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(stokens, sout_features, &out_count) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            prepare_q1_block_lut(xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t o = 0; o < out_features; ++o) {
            const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
            const int32_t *scale_row = scale + (size_t) o * sn_blocks;
            out[(size_t) t * sout_features + (size_t) o] =
                q1_element_s32(bits_row, scale_row, n_blocks, frac,
                               (size_t) t * (size_t) n_blocks, totals, lut);
        }
    }
    return 0;
}

int bonsai_q1_argmax_i64_workspace_scale32(const int64_t *x,
                                           const uint8_t *bits,
                                           const int32_t *scale,
                                           int64_t tokens,
                                           int64_t out_features,
                                           int64_t n_blocks,
                                           int64_t frac,
                                           int64_t *argmax_out,
                                           int64_t *max_out,
                                           uint64_t *totals,
                                           size_t totals_count,
                                           uint64_t *lut,
                                           size_t lut_count) {
    if (!x || !bits || !scale || !argmax_out || !max_out || !totals || !lut ||
        tokens < 0 || out_features <= 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }

    size_t stokens = (size_t) tokens;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t x_width = 0;
    size_t bits_row_stride = 0;
    size_t need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            prepare_q1_block_lut(xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }

    for (int64_t t = 0; t < tokens; ++t) {
        int64_t best_idx = 0;
        int64_t best_val = INT64_MIN;
        #pragma omp parallel
        {
            int64_t local_idx = -1;
            int64_t local_val = INT64_MIN;
            #pragma omp for schedule(static)
            for (int64_t o = 0; o < out_features; ++o) {
                const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                const int32_t *scale_row = scale + (size_t) o * sn_blocks;
                int64_t value = q1_element_s32(bits_row, scale_row, n_blocks, frac,
                                               (size_t) t * (size_t) n_blocks, totals, lut);
                if (local_idx < 0 || value > local_val) {
                    local_val = value;
                    local_idx = o;
                }
            }
            #pragma omp critical
            {
                if (local_idx >= 0 && (local_val > best_val ||
                    (local_val == best_val && local_idx < best_idx))) {
                    best_val = local_val;
                    best_idx = local_idx;
                }
            }
        }
        argmax_out[t] = best_idx;
        max_out[t] = best_val;
    }
    return 0;
}

int bonsai_q1_linear_i64_prepared_scale32(const uint8_t *bits,
                                          const int32_t *scale,
                                          int64_t tokens,
                                          int64_t out_features,
                                          int64_t n_blocks,
                                          int64_t frac,
                                          int64_t *out,
                                          const uint64_t *totals,
                                          size_t totals_count,
                                          const uint64_t *lut,
                                          size_t lut_count) {
    if (!bits || !scale || !out || !totals || !lut ||
        tokens < 0 || out_features < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens;
    size_t sout_features = (size_t) out_features;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t bits_row_stride = 0;
    size_t out_count = 0;
    size_t need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(stokens, sout_features, &out_count) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t o = 0; o < out_features; ++o) {
            const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
            const int32_t *scale_row = scale + (size_t) o * sn_blocks;
            out[(size_t) t * sout_features + (size_t) o] =
                q1_element_s32(bits_row, scale_row, n_blocks, frac,
                               (size_t) t * (size_t) n_blocks, totals, lut);
        }
    }
    return 0;
}

int bonsai_q1_linear_i64_prepared_multi_scale32(const uint8_t *const *bits_list,
                                                const int32_t *const *scale_list,
                                                const int64_t *out_features_list,
                                                int64_t n_items,
                                                int64_t tokens,
                                                int64_t n_blocks,
                                                int64_t frac,
                                                int64_t **out_list,
                                                const uint64_t *totals,
                                                size_t totals_count,
                                                const uint64_t *lut,
                                                size_t lut_count) {
    if (!bits_list || !scale_list || !out_features_list || !out_list || !totals || !lut ||
        n_items < 0 || tokens < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens;
    size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0;
    size_t bits_row_stride = 0;
    size_t need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    for (int64_t item = 0; item < n_items; ++item) {
        if (!bits_list[item] || !scale_list[item] || !out_list[item] || out_features_list[item] < 0) {
            return 1;
        }
    }

    #pragma omp parallel
    {
        for (int64_t item = 0; item < n_items; ++item) {
            const uint8_t *bits = bits_list[item];
            const int32_t *scale = scale_list[item];
            int64_t *out = out_list[item];
            int64_t out_features = out_features_list[item];
            size_t sout_features = (size_t) out_features;
            #pragma omp for collapse(2) schedule(static)
            for (int64_t t = 0; t < tokens; ++t) {
                for (int64_t o = 0; o < out_features; ++o) {
                    const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                    const int32_t *scale_row = scale + (size_t) o * sn_blocks;
                    out[(size_t) t * sout_features + (size_t) o] =
                        q1_element_s32(bits_row, scale_row, n_blocks, frac,
                                       (size_t) t * (size_t) n_blocks, totals, lut);
                }
            }
        }
    }
    return 0;
}

// Prepare one shared activation LUT and apply every same-input projection in a
// single OpenMP region.  Besides avoiding a second Python/native transition,
// this keeps one worker team alive across LUT construction and all projection
// items.  The public wrappers differ only in the lossless scale storage width;
// both execute the exact same Q1 arithmetic and preserve the per-block shift.
static int q1_prepare_apply_multi_u64_core(const int64_t *x,
                                           const uint8_t *const *bits_list,
                                           const void *const *scale_list,
                                           const int64_t *out_features_list,
                                           int64_t n_items,
                                           int64_t tokens,
                                           int64_t n_blocks,
                                           int64_t frac,
                                           int64_t **out_list,
                                           uint64_t *totals,
                                           size_t totals_count,
                                           uint64_t *lut,
                                           size_t lut_count,
                                           int scale32) {
    if (!x || !bits_list || !scale_list || !out_features_list || !out_list ||
        !totals || !lut || n_items < 0 || tokens < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    const size_t stokens = (size_t) tokens;
    const size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0, x_width = 0, bits_row_stride = 0, need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    for (int64_t item = 0; item < n_items; ++item) {
        if (!bits_list[item] || !scale_list[item] || !out_list[item] ||
            out_features_list[item] < 0) {
            return 1;
        }
    }

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static)
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t b = 0; b < n_blocks; ++b) {
                const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
                const size_t idx = (size_t) t * sn_blocks + (size_t) b;
                prepare_q1_block_lut(xb, &totals[idx], lut + idx * 16u * 256u);
            }
        }
        // The implicit barrier above makes the complete LUT visible before
        // any output element is evaluated.  All workers encounter the same
        // item/scale-width sequence, as required by OpenMP work sharing.
        for (int64_t item = 0; item < n_items; ++item) {
            const uint8_t *bits = bits_list[item];
            int64_t *out = out_list[item];
            const int64_t out_features = out_features_list[item];
            const size_t sout_features = (size_t) out_features;
            if (scale32) {
                const int32_t *scale = (const int32_t *) scale_list[item];
                #pragma omp for collapse(2) schedule(static)
                for (int64_t t = 0; t < tokens; ++t) {
                    for (int64_t o = 0; o < out_features; ++o) {
                        const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                        const int32_t *scale_row = scale + (size_t) o * sn_blocks;
                        out[(size_t) t * sout_features + (size_t) o] =
                            q1_element_s32(bits_row, scale_row, n_blocks, frac,
                                           (size_t) t * sn_blocks, totals, lut);
                    }
                }
            } else {
                const int64_t *scale = (const int64_t *) scale_list[item];
                #pragma omp for collapse(2) schedule(static)
                for (int64_t t = 0; t < tokens; ++t) {
                    for (int64_t o = 0; o < out_features; ++o) {
                        const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                        const int64_t *scale_row = scale + (size_t) o * sn_blocks;
                        out[(size_t) t * sout_features + (size_t) o] =
                            q1_element_s64(bits_row, scale_row, n_blocks, frac,
                                           (size_t) t * sn_blocks, totals, lut);
                    }
                }
            }
        }
    }
    return 0;
}

int bonsai_q1_prepare_apply_multi_i64(const int64_t *x,
                                      const uint8_t *const *bits_list,
                                      const int64_t *const *scale_list,
                                      const int64_t *out_features_list,
                                      int64_t n_items,
                                      int64_t tokens,
                                      int64_t n_blocks,
                                      int64_t frac,
                                      int64_t **out_list,
                                      uint64_t *totals,
                                      size_t totals_count,
                                      uint64_t *lut,
                                      size_t lut_count) {
    return q1_prepare_apply_multi_u64_core(
        x, bits_list, (const void *const *) scale_list, out_features_list,
        n_items, tokens, n_blocks, frac, out_list, totals, totals_count, lut,
        lut_count, 0);
}

int bonsai_q1_prepare_apply_multi_i64_scale32(const int64_t *x,
                                              const uint8_t *const *bits_list,
                                              const int32_t *const *scale_list,
                                              const int64_t *out_features_list,
                                              int64_t n_items,
                                              int64_t tokens,
                                              int64_t n_blocks,
                                              int64_t frac,
                                              int64_t **out_list,
                                              uint64_t *totals,
                                              size_t totals_count,
                                              uint64_t *lut,
                                              size_t lut_count) {
    return q1_prepare_apply_multi_u64_core(
        x, bits_list, (const void *const *) scale_list, out_features_list,
        n_items, tokens, n_blocks, frac, out_list, totals, totals_count, lut,
        lut_count, 1);
}

// Exact Qwen3.5 Gated DeltaNet M=1 state update.  NumPy's int64 einsum and
// elementwise operations use two's-complement wrap; all products/additions
// below therefore operate through uint64 and convert back explicitly.  Heads
// are independent, so one native call parallelizes the 48 fixed 128x128
// states without BLAS dispatch or temporary einsum allocations.  `out` is
// used as per-head delta scratch, then overwritten with the final Q(frac)
// recurrent output.
static int bonsai_gdn_run_i64(int64_t *state,
                              const int64_t *q,
                              const int64_t *k,
                              const int64_t *v,
                              const int64_t *beta,
                              const int64_t *decay,
                              int64_t tokens,
                              int64_t heads,
                              int64_t dim,
                              int64_t frac,
                              int64_t state_frac,
                              int64_t outer_shift,
                              int64_t inv_sqrt_fp,
                              int64_t *out) {
    if (!state || !q || !k || !v || !beta || !decay || !out ||
        tokens < 0 || heads < 0 || dim <= 0 || frac < 0 || frac > 62 ||
        state_frac < 0 || state_frac > 62 ||
        outer_shift < 0 || outer_shift > 62) {
        return 1;
    }
    size_t hd = 0, hdd = 0, thd = 0;
    if (!checked_mul_size((size_t) heads, (size_t) dim, &hd) ||
        !checked_mul_size(hd, (size_t) dim, &hdd) ||
        !checked_mul_size((size_t) tokens, hd, &thd)) {
        return 1;
    }
    (void) hdd; (void) thd;

    #pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < heads; ++h) {
        int64_t *sh = state + (size_t) h * (size_t) dim * (size_t) dim;
        for (int64_t t = 0; t < tokens; ++t) {
            const size_t row_base = ((size_t) t * (size_t) heads + (size_t) h) * (size_t) dim;
            const size_t scalar = (size_t) t * (size_t) heads + (size_t) h;
            const int64_t *qh = q + row_base;
            const int64_t *kh = k + row_base;
            const int64_t *vh = v + row_base;
            int64_t *oh = out + row_base;
            const int64_t dh = decay[scalar];
            const int64_t bh = beta[scalar];

            for (int64_t i = 0; i < dim; ++i) {
                for (int64_t j = 0; j < dim; ++j) {
                    const size_t ij = (size_t) i * (size_t) dim + (size_t) j;
                    const uint64_t prod = (uint64_t) sh[ij] * (uint64_t) dh;
                    sh[ij] = arshift_i64(u64_to_i64(prod), frac);
                }
            }
            for (int64_t j = 0; j < dim; ++j) {
                oh[j] = 0;
            }
            // Traverse state rows contiguously while accumulating all output
            // columns.  The per-column i-order is unchanged from einsum.
            for (int64_t i = 0; i < dim; ++i) {
                const int64_t ki = kh[i];
                const int64_t *row = sh + (size_t) i * (size_t) dim;
                for (int64_t j = 0; j < dim; ++j) {
                    oh[j] = u64_to_i64(
                        (uint64_t) oh[j] + (uint64_t) row[j] * (uint64_t) ki);
                }
            }
            for (int64_t j = 0; j < dim; ++j) {
                const int64_t pred = arshift_i64(oh[j], state_frac);
                const int64_t diff = u64_to_i64((uint64_t) vh[j] - (uint64_t) pred);
                const uint64_t prod = (uint64_t) diff * (uint64_t) bh;
                oh[j] = arshift_i64(u64_to_i64(prod), frac);
            }
            for (int64_t i = 0; i < dim; ++i) {
                for (int64_t j = 0; j < dim; ++j) {
                    const size_t ij = (size_t) i * (size_t) dim + (size_t) j;
                    const uint64_t prod = (uint64_t) kh[i] * (uint64_t) oh[j];
                    const int64_t update = arshift_i64(u64_to_i64(prod), outer_shift);
                    sh[ij] = u64_to_i64((uint64_t) sh[ij] + (uint64_t) update);
                }
            }
            for (int64_t j = 0; j < dim; ++j) {
                oh[j] = 0;
            }
            for (int64_t i = 0; i < dim; ++i) {
                const int64_t qi = qh[i];
                const int64_t *row = sh + (size_t) i * (size_t) dim;
                for (int64_t j = 0; j < dim; ++j) {
                    oh[j] = u64_to_i64(
                        (uint64_t) oh[j] + (uint64_t) row[j] * (uint64_t) qi);
                }
            }
            for (int64_t j = 0; j < dim; ++j) {
                const int64_t score = arshift_i64(oh[j], frac);
                const uint64_t prod = (uint64_t) score * (uint64_t) inv_sqrt_fp;
                oh[j] = arshift_i64(u64_to_i64(prod), frac);
            }
        }
    }
    return 0;
}

int bonsai_gdn_decode_i64(int64_t *state,
                          const int64_t *q, const int64_t *k, const int64_t *v,
                          const int64_t *beta, const int64_t *decay,
                          int64_t heads, int64_t dim, int64_t frac,
                          int64_t state_frac, int64_t outer_shift,
                          int64_t inv_sqrt_fp, int64_t *out) {
    return bonsai_gdn_run_i64(
        state, q, k, v, beta, decay, 1, heads, dim, frac, state_frac,
        outer_shift, inv_sqrt_fp, out);
}

int bonsai_gdn_prefill_i64(int64_t *state,
                           const int64_t *q, const int64_t *k, const int64_t *v,
                           const int64_t *beta, const int64_t *decay,
                           int64_t tokens, int64_t heads, int64_t dim,
                           int64_t frac, int64_t state_frac,
                           int64_t outer_shift, int64_t inv_sqrt_fp,
                           int64_t *out) {
    return bonsai_gdn_run_i64(
        state, q, k, v, beta, decay, tokens, heads, dim, frac, state_frac,
        outer_shift, inv_sqrt_fp, out);
}

// ---------------------------------------------------------------------------
// Native M=1 cached-decode attention (PERFORMANCE-DETERMINISM-REVIEW.md Recommendation 4).
//
// Byte-identical to the NumPy fixed-point path in reference_bonsai._attention_with_cache_bonsai for the
// M=1 case (a single new query attending to all L cached positions — the causal mask is all-false, so it
// is omitted here; the caller MUST only invoke this when M==1). It ports the integer softmax from
// determinism/fixedpoint.py exactly (cubic 2^-f poly + power-of-two shift, integer normalize, floor div).
//
// It preserves attention's "fail loud on overflow" contract (UNLIKE the Q1 wrap policy): per head it
// checks the SAME int64 bound as fixed_point_matmul's _assert_no_int64_overflow (max|a|*max|b|*K), using a
// 128-bit product so the bound test itself cannot wrap. If any head would overflow it returns 2 and the
// caller falls back to the NumPy path, which raises — it never silently wraps the attention matmuls.
//
// q:       (H, hd)      int64 fixed-point  (the single decode query, post q-norm + RoPE)
// k, v:    (Hkv, L, hd) int64              (cached RoPE'd K and raw V; GQA: H % Hkv == 0)
// out:     (H, hd)      int64
// scratch: >= H*L int64                    (per-head scores/probs workspace; caller-sized)
static const int64_t BONSAI_LOG2E_Q16 = 94548;

static int64_t bonsai_scaled_log2e(int64_t frac) {
    int64_t shift = 16 - frac;
    return shift >= 0 ? (BONSAI_LOG2E_Q16 >> shift) : (BONSAI_LOG2E_Q16 << (-shift));
}

// 2^(-u_real) in fixed-point, u fixed-point >= 0 — mirrors fixedpoint.py::_exp2_neg_fixed exactly.
static int64_t bonsai_exp2_neg_fixed(int64_t u, int64_t frac) {
    const int64_t C0 = 65536, C1 = 45426, C2 = 15743, C3 = 3638;
    int64_t FP = (int64_t) 1 << frac, mask = FP - 1;
    int64_t k = u >> frac, f = u & mask;
    int64_t shift = 16 - frac, c0, c1, c2, c3;
    if (shift >= 0) { c0 = C0 >> shift; c1 = C1 >> shift; c2 = C2 >> shift; c3 = C3 >> shift; }
    else { int64_t s = -shift; c0 = C0 << s; c1 = C1 << s; c2 = C2 << s; c3 = C3 << s; }
    int64_t f2 = (f * f) >> frac, f3 = (f2 * f) >> frac;
    int64_t poly = c0 - ((c1 * f) >> frac) + ((c2 * f2) >> frac) - ((c3 * f3) >> frac);
    if (poly < 0) poly = 0;
    int64_t kk = k < 63 ? k : 63;
    return poly >> kk;
}

static uint64_t bonsai_maxabs_u64(const int64_t *p, size_t n) {
    uint64_t m = 0;
    for (size_t i = 0; i < n; ++i) {
        int64_t v = p[i];
        uint64_t a = v < 0 ? ((uint64_t)(~(uint64_t) v) + 1u) : (uint64_t) v;
        if (a > m) m = a;
    }
    return m;
}

// k_kv_stride / v_kv_stride: element offset between consecutive kv-heads in k/v. For a contiguous
// (Hkv, L, hd) tensor this is L*hd; passing the underlying KV-cache buffer's stride (cap*hd) lets the
// caller avoid copying the non-contiguous valid slice every decode step. Within a head the (L, hd) block
// must be contiguous (row stride hd).
int bonsai_attention_decode_i64(const int64_t *q, const int64_t *k, const int64_t *v,
                                int64_t H, int64_t Hkv, int64_t hd, int64_t L,
                                int64_t k_kv_stride, int64_t v_kv_stride,
                                int64_t frac, int64_t inv_sqrt_fp,
                                int64_t *out, int64_t *scratch, size_t scratch_count) {
    if (!q || !k || !v || !out || !scratch ||
        H <= 0 || Hkv <= 0 || hd <= 0 || L <= 0 || frac < 0 || frac > 29 || H % Hkv != 0) {
        return 1;
    }
    size_t sH = 0, sHkv = 0, shd = 0, sL = 0;
    size_t sk_stride = 0, sv_stride = 0;
    size_t per_head = 0, q_count = 0, need = 0;
    size_t k_last = 0, v_last = 0, k_span = 0, v_span = 0;
    size_t max_bytes = 0, extent_bytes = 0;
    if (!checked_i64_to_size(H, &sH) || !checked_i64_to_size(Hkv, &sHkv) ||
        !checked_i64_to_size(hd, &shd) || !checked_i64_to_size(L, &sL) ||
        !checked_i64_to_size(k_kv_stride, &sk_stride) ||
        !checked_i64_to_size(v_kv_stride, &sv_stride) ||
        !checked_mul_size(sL, shd, &per_head) || per_head > (size_t) INT64_MAX ||
        k_kv_stride < (int64_t) per_head || v_kv_stride < (int64_t) per_head ||
        !checked_mul_size(sH, shd, &q_count) ||
        !checked_mul_size(sH, sL, &need) ||
        !checked_mul_size(sHkv - 1u, sk_stride, &k_last) ||
        !checked_add_size(k_last, per_head, &k_span) ||
        !checked_mul_size(sHkv - 1u, sv_stride, &v_last) ||
        !checked_add_size(v_last, per_head, &v_span) ||
        !checked_mul_size(sHkv, sizeof(uint64_t), &max_bytes) ||
        !checked_mul_size(q_count, sizeof(int64_t), &extent_bytes) ||
        !checked_mul_size(need, sizeof(int64_t), &extent_bytes) ||
        !checked_mul_size(k_span, sizeof(int64_t), &extent_bytes) ||
        !checked_mul_size(v_span, sizeof(int64_t), &extent_bytes)) {
        return 1;
    }
    if (scratch_count < need) {
        return 3;
    }
    int64_t rep = H / Hkv;
    int64_t log2e = bonsai_scaled_log2e(frac);
    if (log2e <= 0) {
        return 1;
    }
    int64_t dca = ((frac + 2) << (2 * frac)) / log2e;
    int64_t dcb = ((int64_t) 1 << 62) / log2e;
    int64_t d_clip = dca < dcb ? dca : dcb;
    const unsigned __int128 i64max = (unsigned __int128) INT64_MAX;

    // per-kv max|k|, max|v| for the fail-loud bound (shared across each kv group's rep heads)
    uint64_t *maxk = (uint64_t *) malloc(max_bytes);
    uint64_t *maxv = (uint64_t *) malloc(max_bytes);
    if (!maxk || !maxv) {
        free(maxk);
        free(maxv);
        return 2;
    }
    for (int64_t kv = 0; kv < Hkv; ++kv) {
        maxk[kv] = bonsai_maxabs_u64(k + (size_t) kv * sk_stride, per_head);
        maxv[kv] = bonsai_maxabs_u64(v + (size_t) kv * sv_stride, per_head);
    }

    // `overflow` is a shared early-exit flag across the per-head parallel-for. Every write stores the SAME
    // value (1) and an aligned int cannot tear to a third value, so the race is benign: each head runs its
    // OWN input-deterministic 128-bit bound check (independent of the others), so whether ANY head exceeded
    // its bound — and thus the rc==2 verdict — is thread-count invariant. On rc==2 the wrapper discards `out`
    // and falls back to the raising NumPy oracle, so output is invariant too. The atomic annotations below
    // keep TSan/UBSan quiet; the `if (overflow)` read is a best-effort skip, not a correctness gate.
    int overflow = 0;
    #pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < H; ++h) {
        int ov_seen;
        #pragma omp atomic read
        ov_seen = overflow;
        if (ov_seen) {
            continue;
        }
        int64_t kv = h / rep;
        const int64_t *qh = q + (size_t) h * shd;
        const int64_t *kh = k + (size_t) kv * sk_stride;
        const int64_t *vh = v + (size_t) kv * sv_stride;
        int64_t *sc = scratch + (size_t) h * sL;

        // bound: q @ K^T contracts over hd. max|q|*max|k| <= 2^126 fits __int128; the *hd is checked via
        // division so the bound test itself cannot overflow (max|a|,max|b| can each be 2^63, and the naive
        // triple product would wrap mod 2^128 and silently defeat this fail-loud guard). Equivalent to the
        // big-int bound in fixedpoint.py::_assert_no_int64_overflow (a*b*K > INT64_MAX).
        uint64_t maxq = bonsai_maxabs_u64(qh, (size_t) hd);
        unsigned __int128 qk = (unsigned __int128) maxq * (unsigned __int128) maxk[kv];
        if (qk > i64max / (unsigned __int128) hd) {
            #pragma omp atomic write
            overflow = 1;
            continue;
        }
        int64_t m = INT64_MIN;
        for (int64_t j = 0; j < L; ++j) {
            const int64_t *kj = kh + (size_t) j * shd;
            int64_t dot = 0;
            for (int64_t d = 0; d < hd; ++d) {
                dot += qh[d] * kj[d];
            }
            int64_t s = arshift_i64(dot, frac);
            s = arshift_i64(s * inv_sqrt_fp, frac);
            sc[j] = s;
            if (s > m) {
                m = s;
            }
        }
        int64_t Z = 0;
        for (int64_t j = 0; j < L; ++j) {
            int64_t d = m - sc[j];
            if (d > d_clip) {
                d = d_clip;
            }
            int64_t u = (d * log2e) >> frac;
            int64_t e = bonsai_exp2_neg_fixed(u, frac);
            sc[j] = e;
            Z += e;
        }
        for (int64_t j = 0; j < L; ++j) {
            sc[j] = Z ? (int64_t) (((__int128) sc[j] *
                ((__int128) 1 << frac)) / Z) : 0;
        }
        // bound: probs @ V contracts over L (division-checked like the q@K^T bound above so the test
        // itself cannot wrap mod 2^128)
        uint64_t maxp = bonsai_maxabs_u64(sc, (size_t) L);
        unsigned __int128 pv = (unsigned __int128) maxp * (unsigned __int128) maxv[kv];
        if (pv > i64max / (unsigned __int128) L) {
            #pragma omp atomic write
            overflow = 1;
            continue;
        }
        int64_t *oh = out + (size_t) h * shd;
        for (int64_t d = 0; d < hd; ++d) {
            int64_t acc = 0;
            for (int64_t j = 0; j < L; ++j) {
                acc += sc[j] * vh[(size_t) j * shd + (size_t) d];
            }
            oh[d] = arshift_i64(acc, frac);
        }
    }
    free(maxk);
    free(maxv);
    return overflow ? 2 : 0;
}

// Native M=N PREFILL attention (deep-dive lever L5).
//
// Byte-identical to the NumPy causal path in reference_bonsai._attention_with_cache_bonsai for M>1: query m
// (absolute position start+m) attends to the causal key range [0, start+m+1). In this integer softmax the
// masked positions (key j > start+m) contribute EXACTLY 0 (their score is -inf-like, so max-score is clamped
// to d_clip and exp2_neg >> k underflows to 0), so iterating only the valid keys is bit-exact to the
// full-L-with-(-inf)-mask NumPy path — and ~2x cheaper (avg L/2 keys). Reuses the decode kernel's proven
// integer score/softmax/@V math verbatim. Same fail-loud overflow contract: returns 2 and the caller falls
// back to the raising NumPy oracle (never silently wraps).
//
// q:   (H, M, hd)   int64 fixed-point (post q-norm + RoPE), C-contiguous
// k,v: (Hkv, L, hd) int64 (RoPE'd K, raw V), C-contiguous, with L == start + M
// out: (H, M, hd)   int64, C-contiguous
int bonsai_attention_prefill_i64(const int64_t *q, const int64_t *k, const int64_t *v,
                                 int64_t H, int64_t Hkv, int64_t hd, int64_t M, int64_t L, int64_t start,
                                 int64_t frac, int64_t inv_sqrt_fp, int64_t *out) {
    if (!q || !k || !v || !out ||
        H <= 0 || Hkv <= 0 || hd <= 0 || M <= 0 || L <= 0 || start < 0 ||
        frac < 0 || frac > 29 || H % Hkv != 0) {
        return 1;
    }
    size_t sH = 0, sHkv = 0, shd = 0, sM = 0, sL = 0, sstart = 0;
    size_t end = 0, per_head = 0, kv_count = 0, q_count = 0;
    size_t max_bytes = 0, extent_bytes = 0;
    if (!checked_i64_to_size(H, &sH) || !checked_i64_to_size(Hkv, &sHkv) ||
        !checked_i64_to_size(hd, &shd) || !checked_i64_to_size(M, &sM) ||
        !checked_i64_to_size(L, &sL) || !checked_i64_to_size(start, &sstart) ||
        !checked_add_size(sstart, sM, &end) || end != sL ||
        !checked_mul_size(sL, shd, &per_head) ||
        !checked_mul_size(sHkv, per_head, &kv_count) ||
        !checked_mul3_size(sH, sM, shd, &q_count) ||
        !checked_mul_size(sHkv, sizeof(uint64_t), &max_bytes) ||
        !checked_mul_size(kv_count, sizeof(int64_t), &extent_bytes) ||
        !checked_mul_size(q_count, sizeof(int64_t), &extent_bytes)) {
        return 1;
    }
    int64_t rep = H / Hkv;
    int64_t log2e = bonsai_scaled_log2e(frac);
    if (log2e <= 0) {
        return 1;
    }
    int64_t dca = ((frac + 2) << (2 * frac)) / log2e;
    int64_t dcb = ((int64_t) 1 << 62) / log2e;
    int64_t d_clip = dca < dcb ? dca : dcb;
    const unsigned __int128 i64max = (unsigned __int128) INT64_MAX;

    uint64_t *maxk = (uint64_t *) malloc(max_bytes);
    uint64_t *maxv = (uint64_t *) malloc(max_bytes);
    if (!maxk || !maxv) {
        free(maxk);
        free(maxv);
        return 2;
    }
    for (int64_t kv = 0; kv < Hkv; ++kv) {
        maxk[kv] = bonsai_maxabs_u64(k + (size_t) kv * per_head, per_head);
        maxv[kv] = bonsai_maxabs_u64(v + (size_t) kv * per_head, per_head);
    }
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) {
        nthreads = 1;
    }
    size_t scr_count = 0, scratch_bytes = 0;
    if (!checked_mul_size((size_t) nthreads, sL, &scr_count) ||
        !checked_mul_size(scr_count, sizeof(int64_t), &scratch_bytes)) {
        free(maxk);
        free(maxv);
        return 1;
    }
    int64_t *scratch = (int64_t *) malloc(scratch_bytes);   // per-thread (L) score/prob buffer
    if (!scratch) {
        free(maxk);
        free(maxv);
        return 2;
    }

    int overflow = 0;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t m = 0; m < M; ++m) {
            int ov_seen;
            #pragma omp atomic read
            ov_seen = overflow;
            if (ov_seen) {
                continue;
            }
            int tid = omp_get_thread_num();
            int64_t *sc = scratch + (size_t) tid * sL;
            int64_t kv = h / rep;
            int64_t Lv = start + m + 1;                       // causal: query m sees keys [0, Lv)
            const int64_t *qh = q + ((size_t) h * sM + (size_t) m) * shd;
            const int64_t *kh = k + (size_t) kv * per_head;
            const int64_t *vh = v + (size_t) kv * per_head;

            uint64_t maxq = bonsai_maxabs_u64(qh, (size_t) hd);   // q@K bound (contract hd), 128-bit, no wrap
            unsigned __int128 qk = (unsigned __int128) maxq * (unsigned __int128) maxk[kv];
            if (qk > i64max / (unsigned __int128) hd) {
                #pragma omp atomic write
                overflow = 1;
                continue;
            }
            int64_t mx = INT64_MIN;
            for (int64_t j = 0; j < Lv; ++j) {
                const int64_t *kj = kh + (size_t) j * shd;
                int64_t dot = 0;
                for (int64_t d = 0; d < hd; ++d) {
                    dot += qh[d] * kj[d];
                }
                int64_t s = arshift_i64(dot, frac);
                s = arshift_i64(s * inv_sqrt_fp, frac);
                sc[j] = s;
                if (s > mx) {
                    mx = s;
                }
            }
            int64_t Z = 0;
            for (int64_t j = 0; j < Lv; ++j) {
                int64_t d = mx - sc[j];
                if (d > d_clip) {
                    d = d_clip;
                }
                int64_t u = (d * log2e) >> frac;
                int64_t e = bonsai_exp2_neg_fixed(u, frac);
                sc[j] = e;
                Z += e;
            }
            for (int64_t j = 0; j < Lv; ++j) {
                sc[j] = Z ? (int64_t) (((__int128) sc[j] *
                    ((__int128) 1 << frac)) / Z) : 0;
            }
            uint64_t maxp = bonsai_maxabs_u64(sc, (size_t) Lv);   // probs@V bound (contract Lv)
            unsigned __int128 pv = (unsigned __int128) maxp * (unsigned __int128) maxv[kv];
            if (pv > i64max / (unsigned __int128) Lv) {
                #pragma omp atomic write
                overflow = 1;
                continue;
            }
            int64_t *oh = out + ((size_t) h * sM + (size_t) m) * shd;
            for (int64_t d = 0; d < hd; ++d) {
                int64_t acc = 0;
                for (int64_t j = 0; j < Lv; ++j) {
                    acc += sc[j] * vh[(size_t) j * shd + (size_t) d];
                }
                oh[d] = arshift_i64(acc, frac);
            }
        }
    }
    free(scratch);
    free(maxk);
    free(maxv);
    return overflow ? 2 : 0;
}

// Native BATCHED M=1 decode attention (unblocks request-batching L11): B independent decode attentions — each
// query attending its OWN ragged KV cache — in ONE call, OpenMP over (b, h). Per (b,h) it runs the EXACT same
// integer score/softmax/@V math as bonsai_attention_decode_i64 over lengths[b] keys (a single decode query
// attends all cached positions, so no causal mask), so out[b] is bit-identical to the M=1 kernel for sequence
// b. Caches are passed as per-sequence pointers + inter-kv-head strides (the buffer's cap*hd) so the growing
// cache is never copied. Same fail-loud overflow contract (returns 2 -> caller uses the NumPy/M=1 path).
//
// q:            (B, H, hd)      int64, C-contiguous (post q-norm + RoPE, one query per sequence)
// k_ptrs/v_ptrs: const int64_t*[B]  per-sequence pointer to that cache's (Hkv, L_b, hd) block (kv-head 0)
// lengths:      int64[B]        L_b = valid cached length for sequence b (>= 1)
// k/v_kv_strides:int64[B]        element offset between consecutive kv-heads in cache b (>= L_b*hd)
// out:          (B, H, hd)      int64, C-contiguous
int bonsai_attention_decode_batched_i64(const int64_t *q, const int64_t *const *k_ptrs,
                                        const int64_t *const *v_ptrs, const int64_t *lengths,
                                        const int64_t *k_kv_strides, const int64_t *v_kv_strides,
                                        int64_t B, int64_t H, int64_t Hkv, int64_t hd,
                                        int64_t frac, int64_t inv_sqrt_fp, int64_t *out) {
    if (!q || !k_ptrs || !v_ptrs || !lengths || !k_kv_strides || !v_kv_strides || !out ||
        B <= 0 || H <= 0 || Hkv <= 0 || hd <= 0 || frac < 0 || frac > 29 || H % Hkv != 0) {
        return 1;
    }
    size_t sB = 0, sH = 0, sHkv = 0, shd = 0;
    size_t q_count = 0, nbk = 0, max_bytes = 0, extent_bytes = 0;
    if (!checked_i64_to_size(B, &sB) || !checked_i64_to_size(H, &sH) ||
        !checked_i64_to_size(Hkv, &sHkv) || !checked_i64_to_size(hd, &shd) ||
        !checked_mul3_size(sB, sH, shd, &q_count) ||
        !checked_mul_size(sB, sHkv, &nbk) ||
        !checked_mul_size(nbk, sizeof(uint64_t), &max_bytes) ||
        !checked_mul_size(q_count, sizeof(int64_t), &extent_bytes) ||
        !checked_mul_size(sB, sizeof(int64_t *), &extent_bytes) ||
        !checked_mul_size(sB, sizeof(int64_t), &extent_bytes)) {
        return 1;
    }
    int64_t rep = H / Hkv;
    int64_t log2e = bonsai_scaled_log2e(frac);
    if (log2e <= 0) {
        return 1;
    }
    int64_t dca = ((frac + 2) << (2 * frac)) / log2e;
    int64_t dcb = ((int64_t) 1 << 62) / log2e;
    int64_t d_clip = dca < dcb ? dca : dcb;
    const unsigned __int128 i64max = (unsigned __int128) INT64_MAX;

    size_t sLmax = 0;
    for (int64_t b = 0; b < B; ++b) {
        if (!k_ptrs[b] || !v_ptrs[b] || lengths[b] <= 0) {
            return 1;
        }
        size_t sL = 0, sk_stride = 0, sv_stride = 0, per_head = 0;
        size_t k_last = 0, v_last = 0, k_span = 0, v_span = 0;
        if (!checked_i64_to_size(lengths[b], &sL) ||
            !checked_i64_to_size(k_kv_strides[b], &sk_stride) ||
            !checked_i64_to_size(v_kv_strides[b], &sv_stride) ||
            !checked_mul_size(sL, shd, &per_head) || per_head > (size_t) INT64_MAX ||
            k_kv_strides[b] < (int64_t) per_head || v_kv_strides[b] < (int64_t) per_head ||
            !checked_mul_size(sHkv - 1u, sk_stride, &k_last) ||
            !checked_add_size(k_last, per_head, &k_span) ||
            !checked_mul_size(sHkv - 1u, sv_stride, &v_last) ||
            !checked_add_size(v_last, per_head, &v_span) ||
            !checked_mul_size(k_span, sizeof(int64_t), &extent_bytes) ||
            !checked_mul_size(v_span, sizeof(int64_t), &extent_bytes)) {
            return 1;
        }
        if (sL > sLmax) {
            sLmax = sL;
        }
    }
    uint64_t *maxk = (uint64_t *) malloc(max_bytes);   // per (b, kv) max|k|, max|v| for the bound
    uint64_t *maxv = (uint64_t *) malloc(max_bytes);
    if (!maxk || !maxv) {
        free(maxk);
        free(maxv);
        return 2;
    }
    for (int64_t b = 0; b < B; ++b) {
        const size_t per_head = (size_t) lengths[b] * shd;
        const size_t sk_stride = (size_t) k_kv_strides[b];
        const size_t sv_stride = (size_t) v_kv_strides[b];
        for (int64_t kv = 0; kv < Hkv; ++kv) {
            const size_t index = (size_t) b * sHkv + (size_t) kv;
            maxk[index] = bonsai_maxabs_u64(k_ptrs[b] + (size_t) kv * sk_stride, per_head);
            maxv[index] = bonsai_maxabs_u64(v_ptrs[b] + (size_t) kv * sv_stride, per_head);
        }
    }
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) {
        nthreads = 1;
    }
    size_t scr_count = 0, scratch_bytes = 0;
    if (!checked_mul_size((size_t) nthreads, sLmax, &scr_count) ||
        !checked_mul_size(scr_count, sizeof(int64_t), &scratch_bytes)) {
        free(maxk);
        free(maxv);
        return 1;
    }
    int64_t *scratch = (int64_t *) malloc(scratch_bytes);
    if (!scratch) {
        free(maxk);
        free(maxv);
        return 2;
    }

    int overflow = 0;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int ov_seen;
            #pragma omp atomic read
            ov_seen = overflow;
            if (ov_seen) {
                continue;
            }
            int tid = omp_get_thread_num();
            int64_t *sc = scratch + (size_t) tid * sLmax;
            int64_t kv = h / rep;
            int64_t L = lengths[b];
            const int64_t *qh = q + ((size_t) b * sH + (size_t) h) * shd;
            const int64_t *kh = k_ptrs[b] + (size_t) kv * k_kv_strides[b];
            const int64_t *vh = v_ptrs[b] + (size_t) kv * v_kv_strides[b];
            const size_t bound_index = (size_t) b * sHkv + (size_t) kv;

            uint64_t maxq = bonsai_maxabs_u64(qh, (size_t) hd);
            unsigned __int128 qk = (unsigned __int128) maxq * (unsigned __int128) maxk[bound_index];
            if (qk > i64max / (unsigned __int128) hd) {
                #pragma omp atomic write
                overflow = 1;
                continue;
            }
            int64_t mx = INT64_MIN;
            for (int64_t j = 0; j < L; ++j) {
                const int64_t *kj = kh + (size_t) j * shd;
                int64_t dot = 0;
                for (int64_t d = 0; d < hd; ++d) {
                    dot += qh[d] * kj[d];
                }
                int64_t s = arshift_i64(dot, frac);
                s = arshift_i64(s * inv_sqrt_fp, frac);
                sc[j] = s;
                if (s > mx) {
                    mx = s;
                }
            }
            int64_t Z = 0;
            for (int64_t j = 0; j < L; ++j) {
                int64_t d = mx - sc[j];
                if (d > d_clip) {
                    d = d_clip;
                }
                int64_t u = (d * log2e) >> frac;
                int64_t e = bonsai_exp2_neg_fixed(u, frac);
                sc[j] = e;
                Z += e;
            }
            for (int64_t j = 0; j < L; ++j) {
                sc[j] = Z ? (int64_t) (((__int128) sc[j] *
                    ((__int128) 1 << frac)) / Z) : 0;
            }
            uint64_t maxp = bonsai_maxabs_u64(sc, (size_t) L);
            unsigned __int128 pv = (unsigned __int128) maxp * (unsigned __int128) maxv[bound_index];
            if (pv > i64max / (unsigned __int128) L) {
                #pragma omp atomic write
                overflow = 1;
                continue;
            }
            int64_t *oh = out + ((size_t) b * sH + (size_t) h) * shd;
            for (int64_t d = 0; d < hd; ++d) {
                int64_t acc = 0;
                for (int64_t j = 0; j < L; ++j) {
                    acc += sc[j] * vh[(size_t) j * shd + (size_t) d];
                }
                oh[d] = arshift_i64(acc, frac);
            }
        }
    }
    free(scratch);
    free(maxk);
    free(maxv);
    return overflow ? 2 : 0;
}

// Element-wise fixed-point SiLU: out[i] = (x[i] * sigmoid(x[i])) >> frac, byte-identical to
// reference_bonsai.fixed_point_silu = (x * fixed_point_sigmoid(x, frac)) >> frac. Reuses the integer
// softmax helpers (bonsai_scaled_log2e, bonsai_exp2_neg_fixed) added for native attention. The sigmoid
// uses the fixedpoint.py::fixed_point_sigmoid d_clip form — min(((frac+2)<<(2*frac))/log2e,
// (1<<62)//log2e). The (1<<62)//log2e HARD cap mirrors the oracle EXACTLY (fixedpoint.py::fixed_point_sigmoid
// now applies it too); it is a no-op at the committed frac=16 (the first term dominates) and only binds near
// frac=29, keeping native SiLU byte-identical to the NumPy oracle across the whole [1,29] envelope. Without
// it the two diverge at frac=29. The final (x*sig)>>frac WRAPS mod 2^64 like the NumPy
// int64 multiply (NOT fail-loud — numpy does not assert here), via uint64 multiply + u64_to_i64 + floor
// arshift, so it is byte-exact even at overflow. OpenMP partitions independent output elements.
int bonsai_silu_i64(const int64_t *x, int64_t n, int64_t frac, int64_t *out) {
    if (!x || !out || n < 0 || frac < 1 || frac > 29) {
        return 1;
    }
    const int64_t log2e = bonsai_scaled_log2e(frac);
    if (log2e <= 0) {
        return 1;
    }
    const int64_t d_clip_first = ((frac + 2) << (2 * frac)) / log2e;
    const int64_t d_clip_cap = ((int64_t) 1 << 62) / log2e;      // HARD cap, mirrors the NumPy oracle
    const int64_t d_clip = d_clip_first < d_clip_cap ? d_clip_first : d_clip_cap;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        int64_t xi = x[i];
        int64_t m = xi > 0 ? xi : 0;                 // max(x, 0)
        int64_t d0 = m;                              // m - 0
        // d1 = m - xi. For xi == INT64_MIN this is 0 - INT64_MIN, which is signed-overflow UB in C but a
        // DEFINED two's-complement wrap in the NumPy oracle (fixedpoint.py::fixed_point_sigmoid). Compute it
        // as an unsigned subtraction so C reproduces NumPy's wrap bit-for-bit regardless of compiler/flags
        // (the kernel-wide guarantee; see also -fwrapv in build_bonsai_q1_kernel.sh). Normal inputs: >= 0.
        int64_t d1 = u64_to_i64((uint64_t) m - (uint64_t) xi);
        if (d0 > d_clip) d0 = d_clip;
        if (d1 > d_clip) d1 = d_clip;
        int64_t e0 = bonsai_exp2_neg_fixed((d0 * log2e) >> frac, frac);
        int64_t e1 = bonsai_exp2_neg_fixed((d1 * log2e) >> frac, frac);
        int64_t z = e0 + e1;
        // exp2_neg returns a non-negative Q(frac) polynomial bounded by
        // c0+c2.  Across the admitted frac<=29 envelope, e1*2^frac is
        // therefore below 2^59 and the positive quotient is exactly a
        // uint64 division.  Avoiding a signed 128-bit divide matters here:
        // the resident 27B graph evaluates roughly two million sigmoid/SiLU
        // elements per decode token.
        const uint64_t sig_num = (uint64_t) e1 * (UINT64_C(1) << frac);
        int64_t sig = z ? (int64_t) (sig_num / (uint64_t) z) : 0;
        uint64_t prod = (uint64_t) xi * (uint64_t) sig;   // defined wrap mod 2^64
        out[i] = arshift_i64(u64_to_i64(prod), frac);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// int32 activation-LUT-entry variants (optimization-scopes/INT32-LUT-ENTRY.md). The activation LUT entries
// are stored as int32 instead of uint64 (halving the gather data) for blocks whose subset sums fit int32.
// Byte-identical to the uint64-LUT path for in-envelope blocks BY CONSTRUCTION: pos_sum accumulates the
// sign-extended int32 entries in int64, and 2*(uint64)pos_sum - block_total equals the uint64-LUT
// signed_sum mod 2^64 (the int32 entries are the true subset sums, no wrap at the entry level). A build-time
// per-lane range guard (sum of |8 activations| <= INT32_MAX) makes every entry fit int32; if a block fails
// the guard the builder signals fallback (rc 5) and the caller uses the uint64-LUT path. totals stays
// uint64. Opt-in (TRINOTE_Q1_LUT32); the int64 path remains the canonical default + fallback.
#define DEFINE_Q1_ELEMENT_LUT32(NAME, STYPE)                                            \
static inline int64_t NAME(const uint8_t *bits_row, const STYPE *scale_row,              \
                           int64_t n_blocks, int64_t frac, size_t token_base,            \
                           const uint64_t *totals, const int32_t *lut) {                 \
    uint64_t total = 0;                                                                  \
    for (int64_t b = 0; b < n_blocks; ++b) {                                             \
        const uint8_t *bb = bits_row + (size_t) b * 16u;                                 \
        const uint64_t block_total = totals[token_base + (size_t) b];                    \
        int64_t pos_sum = 0;                                                             \
        const int32_t *block_lut = lut + ((token_base + (size_t) b) * 16u) * 256u;       \
        for (int byte_i = 0; byte_i < 16; ++byte_i) {                                    \
            pos_sum += (int64_t) block_lut[(size_t) byte_i * 256u + (size_t) bb[byte_i]];\
        }                                                                               \
        uint64_t signed_sum = 2u * (uint64_t) pos_sum - block_total;                     \
        uint64_t prod = signed_sum * (uint64_t) (int64_t) scale_row[b];                  \
        total += (uint64_t) arshift_i64(u64_to_i64(prod), frac);                         \
    }                                                                                   \
    return u64_to_i64(total);                                                            \
}

DEFINE_Q1_ELEMENT_LUT32(q1_element_lut32_s64, int64_t)
DEFINE_Q1_ELEMENT_LUT32(q1_element_lut32_s32, int32_t)

// Shared by both resident argmax paths and their model-free boundary test.
// Considering candidates in any worker/order still selects the lowest index
// on an exact tie.
static inline void q1_argmax_consider(int64_t index, int64_t value,
                                      int64_t *best_index,
                                      int64_t *best_value) {
    if (*best_index < 0 || value > *best_value ||
        (value == *best_value && index < *best_index)) {
        *best_index = index;
        *best_value = value;
    }
}

#if BONSAI_CAN_BUILD_AVX2
static int q1_hardware_has_avx2(void) {
    static int cached = -1;
    if (cached < 0) {
        __builtin_cpu_init();
        cached = __builtin_cpu_supports("avx2") ? 1 : 0;
    }
    return cached;
}

// Gather the sixteen lane-specific int32 subset sums with two AVX2 gathers.
// Each gathered value is widened before addition, so this is exactly the
// scalar int64 pos_sum even when four or sixteen int32 entries would overflow
// an int32 lane.  Scale multiply, per-block floor shift, and wrapping reduction
// remain shared with the portable implementation.
#define DEFINE_Q1_ELEMENT_LUT32_AVX2(NAME, STYPE)                                      \
__attribute__((target("avx2"), noinline))                                               \
static int64_t NAME(const uint8_t *bits_row, const STYPE *scale_row,                    \
                    int64_t n_blocks, int64_t frac, size_t token_base,                 \
                    const uint64_t *totals, const int32_t *lut) {                      \
    uint64_t total = 0;                                                                 \
    const __m256i off0 = _mm256_setr_epi32(                                             \
        0*256, 1*256, 2*256, 3*256, 4*256, 5*256, 6*256, 7*256);                       \
    const __m256i off1 = _mm256_setr_epi32(                                             \
        8*256, 9*256, 10*256, 11*256, 12*256, 13*256, 14*256, 15*256);                 \
    for (int64_t b = 0; b < n_blocks; ++b) {                                            \
        const uint8_t *bb = bits_row + (size_t) b * 16u;                                \
        const uint64_t block_total = totals[token_base + (size_t) b];                   \
        const int32_t *block_lut = lut + ((token_base + (size_t) b) * 16u) * 256u;      \
        const __m128i bytes = _mm_loadu_si128((const __m128i *) bb);                    \
        const __m256i idx0 = _mm256_add_epi32(_mm256_cvtepu8_epi32(bytes), off0);       \
        const __m128i bytes_hi = _mm_srli_si128(bytes, 8);                              \
        const __m256i idx1 = _mm256_add_epi32(_mm256_cvtepu8_epi32(bytes_hi), off1);    \
        const __m256i g0 = _mm256_i32gather_epi32(block_lut, idx0, 4);                  \
        const __m256i g1 = _mm256_i32gather_epi32(block_lut, idx1, 4);                  \
        __m256i sums = _mm256_add_epi64(                                                \
            _mm256_cvtepi32_epi64(_mm256_castsi256_si128(g0)),                         \
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(g0, 1)));                   \
        sums = _mm256_add_epi64(sums, _mm256_cvtepi32_epi64(                           \
            _mm256_castsi256_si128(g1)));                                               \
        sums = _mm256_add_epi64(sums, _mm256_cvtepi32_epi64(                           \
            _mm256_extracti128_si256(g1, 1)));                                         \
        int64_t lanes[4];                                                               \
        _mm256_storeu_si256((__m256i *) lanes, sums);                                   \
        const int64_t pos_sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];              \
        const uint64_t signed_sum = 2u * (uint64_t) pos_sum - block_total;              \
        const uint64_t prod = signed_sum * (uint64_t) (int64_t) scale_row[b];           \
        total += (uint64_t) arshift_i64(u64_to_i64(prod), frac);                        \
    }                                                                                   \
    return u64_to_i64(total);                                                           \
}

DEFINE_Q1_ELEMENT_LUT32_AVX2(q1_element_lut32_s64_avx2, int64_t)
DEFINE_Q1_ELEMENT_LUT32_AVX2(q1_element_lut32_s32_avx2, int32_t)
#else
static int q1_hardware_has_avx2(void) { return 0; }
#endif

int bonsai_q1_runtime_has_avx2(void) {
    return q1_hardware_has_avx2();
}

// -1 means read the process environment lazily; 0 auto, 1 portable, 2 AVX2.
static int q1_isa_mode = -1;

static int q1_current_isa_mode(void) {
    if (q1_isa_mode < 0) {
        const char *raw = getenv("TRINOTE_Q1_ISA");
        if (raw && (!strcmp(raw, "portable") || !strcmp(raw, "scalar"))) q1_isa_mode = 1;
        else if (raw && !strcmp(raw, "avx2")) q1_isa_mode = 2;
        else q1_isa_mode = 0;
    }
    return q1_isa_mode;
}

static int q1_runtime_use_avx2(void) {
    const int mode = q1_current_isa_mode();
    if (mode == 1) return 0;
    return q1_hardware_has_avx2();
}

int bonsai_q1_set_isa_mode(int mode) {
    if (mode < 0 || mode > 2) return 1;
    if (mode == 2 && !q1_hardware_has_avx2()) return 5;
    q1_isa_mode = mode;
    return 0;
}

int bonsai_q1_get_isa_mode(void) {
    const int mode = q1_current_isa_mode();
    return mode == 0 ? (q1_hardware_has_avx2() ? 2 : 1) : mode;
}

// Build the int32 activation LUT for one 128-wide block. Returns 0 on success, 1 if any lane's subset sums
// could exceed int32 (caller falls back to the uint64 LUT). Each lane's max |subset sum| <= sum of |8
// values|, so a per-lane abs-sum <= INT32_MAX guarantees every entry fits int32 and the staged-doubling add
// never overflows.
static int prepare_q1_block_lut32(const int64_t *xb, uint64_t *total_out, int32_t *lut_block) {
    uint64_t total = 0;
    for (int i = 0; i < 128; ++i) {
        total += (uint64_t) xb[i];
    }
    *total_out = total;
    for (int byte_i = 0; byte_i < 16; ++byte_i) {
        const int64_t *xp = xb + byte_i * 8;
        uint64_t lane_abs = 0;
        for (int k = 0; k < 8; ++k) {
            int64_t v = xp[k];
            uint64_t a = v < 0 ? ((uint64_t)(~(uint64_t) v) + 1u) : (uint64_t) v;
            if (a > (uint64_t) INT32_MAX) {
                return 1;
            }
            lane_abs += a;
        }
        if (lane_abs > (uint64_t) INT32_MAX) {
            return 1;
        }
        int32_t *table = lut_block + (size_t) byte_i * 256u;
        table[0] = 0;
        for (int bit = 0; bit < 8; ++bit) {
            const size_t base = (size_t) 1u << bit;
            const int32_t add = (int32_t) xp[bit];   // |xp[bit]| <= lane_abs <= INT32_MAX
            for (size_t mask = 0; mask < base; ++mask) {
                table[base + mask] = table[mask] + add;   // every entry is a subset sum, |.| <= lane_abs
            }
        }
    }
    return 0;
}

#if BONSAI_CAN_BUILD_AVX2
// Direct int32 dot-product signs.  Eight expanded signs occupy one aligned
// cache-line half and are loaded as a single YMM vector.  The complete table
// is 8 KiB and remains private-L1 resident while the 4 GiB packed weights are
// streamed.
#define Q1_I32_SIGN(V, B) (((V) & (1u << (B))) ? INT32_C(1) : -INT32_C(1))
#define Q1_I32_ROW(V) { \
    Q1_I32_SIGN((V), 0), Q1_I32_SIGN((V), 1), \
    Q1_I32_SIGN((V), 2), Q1_I32_SIGN((V), 3), \
    Q1_I32_SIGN((V), 4), Q1_I32_SIGN((V), 5), \
    Q1_I32_SIGN((V), 6), Q1_I32_SIGN((V), 7) }
#define Q1_I32_ROWS16(B) \
    Q1_I32_ROW((B) + 0), Q1_I32_ROW((B) + 1), \
    Q1_I32_ROW((B) + 2), Q1_I32_ROW((B) + 3), \
    Q1_I32_ROW((B) + 4), Q1_I32_ROW((B) + 5), \
    Q1_I32_ROW((B) + 6), Q1_I32_ROW((B) + 7), \
    Q1_I32_ROW((B) + 8), Q1_I32_ROW((B) + 9), \
    Q1_I32_ROW((B) + 10), Q1_I32_ROW((B) + 11), \
    Q1_I32_ROW((B) + 12), Q1_I32_ROW((B) + 13), \
    Q1_I32_ROW((B) + 14), Q1_I32_ROW((B) + 15)
static const int32_t q1_sign_i32[256][8] __attribute__((aligned(32))) = {
    Q1_I32_ROWS16(0), Q1_I32_ROWS16(16), Q1_I32_ROWS16(32),
    Q1_I32_ROWS16(48), Q1_I32_ROWS16(64), Q1_I32_ROWS16(80),
    Q1_I32_ROWS16(96), Q1_I32_ROWS16(112), Q1_I32_ROWS16(128),
    Q1_I32_ROWS16(144), Q1_I32_ROWS16(160), Q1_I32_ROWS16(176),
    Q1_I32_ROWS16(192), Q1_I32_ROWS16(208), Q1_I32_ROWS16(224),
    Q1_I32_ROWS16(240),
};
#undef Q1_I32_ROWS16
#undef Q1_I32_ROW
#undef Q1_I32_SIGN

// Compact resident-decode activation representation.  Each AVX2 accumulator
// lane receives the 16 values at indices lane + 8*k.  Guarding the sum of
// magnitudes for each lane by INT32_MAX makes every sequence of +/- additions
// exact in signed int32; INT32_MIN is rejected by the same envelope.  A reject
// occurs before any projection output and selects the established uint64 LUT.
static int prepare_q1_block_direct32(const int64_t *xb, int32_t *direct) {
    uint64_t lane_abs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 128; ++i) {
        const int64_t value = xb[i];
        if (value < -(int64_t) INT32_MAX || value > (int64_t) INT32_MAX) {
            return 1;
        }
        const uint64_t magnitude = value < 0
            ? (uint64_t) (-value) : (uint64_t) value;
        const int lane = i & 7;
        lane_abs[lane] += magnitude;
        if (lane_abs[lane] > (uint64_t) INT32_MAX) return 1;
        direct[i] = (int32_t) value;
    }
    return 0;
}

__attribute__((target("avx2"), always_inline))
static inline int64_t q1_hsum_epi64_avx2(__m256i value) {
    __m128i sum = _mm_add_epi64(
        _mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_add_epi64(sum, _mm_srli_si128(sum, 8));
    return (int64_t) _mm_cvtsi128_si64(sum);
}

// Direct two-row resident dot product.  The builder's per-lane envelope makes
// `_mm256_sign_epi32` plus `_mm256_add_epi32` exact (including every partial
// sum), after which the eight lanes widen to int64 for the final reduction.
// This removes the byte-plane reconstruction instructions while retaining the
// identical per-block scale/shift/wrap sequence.
static inline int64_t q1_arshift_q16_x86(uint64_t product) {
    // Spell the sign extension entirely in uint64 so exactness does not rely
    // on implementation-defined signed right shift.  The shifted result is in
    // [-2^47, 2^47-1], so u64_to_i64's INT64_MIN case is unreachable.  GCC and
    // Clang recognize this unsigned identity as a constant arithmetic shift.
    const uint64_t sign_fill = UINT64_C(0) - (product >> 63);
    const uint64_t shifted = (product >> 16) | (sign_fill << 48);
    return u64_to_i64(shifted);
}

__attribute__((target("avx2"), always_inline))
static inline void q1_elements2_direct32_s32_avx2_core(
        const uint8_t *bits0, const int32_t *scale0,
        const uint8_t *bits1, const int32_t *scale1, int have_second,
        int64_t n_blocks, int64_t frac, int q16_fast_shift,
        size_t token_base,
        const int32_t *direct, int64_t out[2]) {
    uint64_t total0 = 0, total1 = 0;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t *bb0 = bits0 + (size_t) b * 16u;
        const uint8_t *bb1 = have_second
            ? bits1 + (size_t) b * 16u : bb0;
        const int32_t *xb = direct +
            (token_base + (size_t) b) * 128u;
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        for (int byte_i = 0; byte_i < 16; ++byte_i) {
            const __m256i x = _mm256_loadu_si256(
                (const __m256i *) (xb + (size_t) byte_i * 8u));
            const __m256i signs0 = _mm256_load_si256(
                (const __m256i *) q1_sign_i32[bb0[byte_i]]);
            sum0 = _mm256_add_epi32(
                sum0, _mm256_sign_epi32(x, signs0));
            if (have_second) {
                const __m256i signs1 = _mm256_load_si256(
                    (const __m256i *) q1_sign_i32[bb1[byte_i]]);
                sum1 = _mm256_add_epi32(
                    sum1, _mm256_sign_epi32(x, signs1));
            }
        }
        const __m256i wide0 = _mm256_add_epi64(
            _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum0)),
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum0, 1)));
        const int64_t dot0 = q1_hsum_epi64_avx2(wide0);
        const uint64_t product0 =
            (uint64_t) dot0 * (uint64_t) (int64_t) scale0[b];
        total0 += (uint64_t) (q16_fast_shift
            ? q1_arshift_q16_x86(product0)
            : arshift_i64(u64_to_i64(product0), frac));
        if (have_second) {
            const __m256i wide1 = _mm256_add_epi64(
                _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum1)),
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum1, 1)));
            const int64_t dot1 = q1_hsum_epi64_avx2(wide1);
            const uint64_t product1 =
                (uint64_t) dot1 * (uint64_t) (int64_t) scale1[b];
            total1 += (uint64_t) (q16_fast_shift
                ? q1_arshift_q16_x86(product1)
                : arshift_i64(u64_to_i64(product1), frac));
        }
    }
    out[0] = u64_to_i64(total0);
    if (have_second) out[1] = u64_to_i64(total1);
}

// Keep the dynamic-frac entry for the model-free boundary test and any future
// non-resident reuse.  The admitted resident Bonsai-27B descriptor is fixed at
// frac=16, so its dedicated entry lets the compiler remove variable-shift
// branches from every streamed Q1 block without changing shared/legacy APIs.
__attribute__((target("avx2"), noinline))
static void q1_elements2_direct32_s32_avx2(
        const uint8_t *bits0, const int32_t *scale0,
        const uint8_t *bits1, const int32_t *scale1, int have_second,
        int64_t n_blocks, int64_t frac, size_t token_base,
        const int32_t *direct, int64_t out[2]) {
    q1_elements2_direct32_s32_avx2_core(
        bits0, scale0, bits1, scale1, have_second, n_blocks, frac, 0,
        token_base, direct, out);
}

__attribute__((target("avx2"), noinline))
static void q1_elements2_direct32_s32_avx2_q16(
        const uint8_t *bits0, const int32_t *scale0,
        const uint8_t *bits1, const int32_t *scale1, int have_second,
        int64_t n_blocks, size_t token_base,
        const int32_t *direct, int64_t out[2]) {
    q1_elements2_direct32_s32_avx2_core(
        bits0, scale0, bits1, scale1, have_second, n_blocks, 16, 1,
        token_base, direct, out);
}

// Independent scalar oracle for the direct32 self-test.  It intentionally
// does not use either activation-LUT implementation: signs are decoded one
// bit at a time, then the resident per-block multiply/shift/wrapping reduction
// is reproduced through unsigned arithmetic.
static int64_t q1_direct32_scalar_reference_s32(
        const uint8_t *bits, const int32_t *scale, const int64_t *x,
        int64_t n_blocks, int64_t frac) {
    uint64_t total = 0;
    for (int64_t b = 0; b < n_blocks; ++b) {
        uint64_t dot = 0;
        const uint8_t *bb = bits + (size_t) b * 16u;
        const int64_t *xb = x + (size_t) b * 128u;
        for (int byte_i = 0; byte_i < 16; ++byte_i) {
            for (int lane = 0; lane < 8; ++lane) {
                const uint64_t value = (uint64_t) xb[byte_i * 8 + lane];
                if (bb[byte_i] & (uint8_t) (1u << lane)) dot += value;
                else dot -= value;
            }
        }
        const uint64_t product = dot * (uint64_t) (int64_t) scale[b];
        total += (uint64_t) arshift_i64(u64_to_i64(product), frac);
    }
    return u64_to_i64(total);
}

// Focused ABI self-test for the guarded resident kernel's adversarial edges.
// This is intentionally model-free so CI can exercise the exact AVX2 helper
// without loading the 4 GiB release artifact. Return 5 when AVX2 is absent.
int bonsai_q1_direct32_boundary_selftest(void) {
    if (!q1_hardware_has_avx2()) return 5;
    const uint64_t product_edges[] = {
        UINT64_C(0), UINT64_C(1), UINT64_C(0xffff), UINT64_C(0x10000),
        UINT64_C(0x7fffffffffff0000), UINT64_C(0x7fffffffffffffff),
        UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000001),
        UINT64_C(0xffffffffffff0000), UINT64_C(0xffffffffffffffff),
    };
    for (size_t i = 0; i < sizeof(product_edges) / sizeof(product_edges[0]); ++i) {
        const int64_t generic = arshift_i64(
            u64_to_i64(product_edges[i]), 16);
        if (q1_arshift_q16_x86(product_edges[i]) != generic) return 10;
    }
    enum { blocks = 2, rows = 3 };
    int64_t x[blocks * 128] = {0};
    int32_t direct[blocks * 128] = {0};
    uint8_t bits[rows][blocks * 16];
    int32_t scale[rows][blocks] = {
        {INT32_MAX, INT32_MIN},
        {INT32_MAX, INT32_MIN},
        {-INT32_C(12345), INT32_MAX},
    };
    memset(bits, 0, sizeof(bits));
    for (int lane = 0; lane < 8; ++lane) {
        x[lane] = INT32_MAX;
        x[128 + lane] = INT32_MAX - INT32_C(17);
    }
    for (int byte_i = 0; byte_i < 16; ++byte_i) {
        bits[0][16 + byte_i] = UINT8_MAX;
        bits[2][byte_i] = (uint8_t) (0xa5u ^ (uint8_t) byte_i);
        bits[2][16 + byte_i] = (uint8_t) (0x3cu + (uint8_t) byte_i);
    }
    memcpy(bits[1], bits[0], sizeof(bits[0]));
    for (int b = 0; b < blocks; ++b) {
        if (prepare_q1_block_direct32(
                x + (size_t) b * 128u,
                direct + (size_t) b * 128u)) return 1;
    }

    // Two blocks are required here: both mathematical dot*scale products
    // exceed int64, and each must wrap and floor-shift independently before
    // the wrapping block reduction.  Rows 0 and 1 intentionally tie.
    int64_t actual[2] = {INT64_MIN, INT64_MIN};
    q1_elements2_direct32_s32_avx2(
        bits[0], scale[0], bits[1], scale[1], 1,
        blocks, 16, 0, direct, actual);
    int64_t q16_actual[2] = {INT64_MIN, INT64_MIN};
    q1_elements2_direct32_s32_avx2_q16(
        bits[0], scale[0], bits[1], scale[1], 1,
        blocks, 0, direct, q16_actual);
    if (q16_actual[0] != actual[0] || q16_actual[1] != actual[1]) return 11;
    uint64_t totals[blocks];
    uint64_t lut[blocks * 16u * 256u];
    for (int b = 0; b < blocks; ++b) {
        prepare_q1_block_lut(x + (size_t) b * 128u, &totals[b],
                             lut + (size_t) b * 16u * 256u);
    }
    for (int row = 0; row < 2; ++row) {
        const int64_t lut_expected = q1_element_s32(
            bits[row], scale[row], blocks, 16, 0, totals, lut);
        const int64_t scalar_expected = q1_direct32_scalar_reference_s32(
            bits[row], scale[row], x, blocks, 16);
        if (actual[row] != lut_expected || actual[row] != scalar_expected)
            return 2;
    }

    // Exercise the helper's odd final row.  The unused output lane must not
    // be touched, which is required by odd-width projections and vocabularies.
    const int64_t odd_sentinel = INT64_C(0x123456789abcdef);
    int64_t odd[2] = {INT64_MIN, odd_sentinel};
    q1_elements2_direct32_s32_avx2(
        bits[2], scale[2], bits[2], scale[2], 0,
        blocks, 16, 0, direct, odd);
    int64_t q16_odd[2] = {INT64_MIN, odd_sentinel};
    q1_elements2_direct32_s32_avx2_q16(
        bits[2], scale[2], bits[2], scale[2], 0,
        blocks, 0, direct, q16_odd);
    if (odd[1] != odd_sentinel ||
        odd[0] != q1_direct32_scalar_reference_s32(
            bits[2], scale[2], x, blocks, 16)) return 3;
    if (q16_odd[0] != odd[0] || q16_odd[1] != odd_sentinel) return 12;

    // Feed tied rows in descending index order through the exact helper used
    // by the resident parallel reduction. Lowest-index selection must win.
    if (actual[0] != actual[1]) return 4;
    int64_t best_index = -1, best_value = INT64_MIN;
    q1_argmax_consider(1, actual[1], &best_index, &best_value);
    q1_argmax_consider(0, actual[0], &best_index, &best_value);
    if (best_index != 0 || best_value != actual[0]) return 6;

    // Natural (data-driven) rejection in the second block must happen before
    // any projection output. Replay every block through the canonical uint64
    // LUT, including the already-prepared first block, and compare with the
    // independent bitwise scalar oracle.
    int64_t reject_x[blocks * 128];
    int32_t reject_direct[blocks * 128] = {0};
    memcpy(reject_x, x, sizeof(reject_x));
    reject_x[128] = INT32_MIN;
    int range_bad = 0;
    int64_t replay = odd_sentinel;
    for (int b = 0; b < blocks; ++b) {
        if (prepare_q1_block_direct32(
                reject_x + (size_t) b * 128u,
                reject_direct + (size_t) b * 128u)) range_bad = 1;
    }
    if (!range_bad || replay != odd_sentinel) return 7;
    for (int b = 0; b < blocks; ++b) {
        prepare_q1_block_lut(reject_x + (size_t) b * 128u, &totals[b],
                             lut + (size_t) b * 16u * 256u);
    }
    replay = q1_element_s32(bits[2], scale[2], blocks, 16, 0, totals, lut);
    if (replay != q1_direct32_scalar_reference_s32(
            bits[2], scale[2], reject_x, blocks, 16)) return 8;

    // A prompt row occupies a nonzero block offset in the compact activation
    // arena.  Exercise that address calculation through the same Q16 entry the
    // resident one-row prefill path uses; the preceding row deliberately holds
    // different values so an accidental token_base=0 is observable.
    int64_t prompt_x[2 * blocks * 128] = {0};
    int32_t prompt_direct[2 * blocks * 128] = {0};
    for (int i = 0; i < blocks * 128; ++i) {
        prompt_x[i] = (i & 1) ? INT64_C(37) : -INT64_C(19);
        prompt_x[blocks * 128 + i] = x[i];
    }
    for (int prompt_row = 0; prompt_row < 2; ++prompt_row) {
        for (int b = 0; b < blocks; ++b) {
            const size_t block_index =
                (size_t) prompt_row * (size_t) blocks + (size_t) b;
            if (prepare_q1_block_direct32(
                    prompt_x + block_index * 128u,
                    prompt_direct + block_index * 128u)) return 13;
        }
    }
    int64_t prompt_actual[2] = {INT64_MIN, odd_sentinel};
    q1_elements2_direct32_s32_avx2_q16(
        bits[2], scale[2], bits[2], scale[2], 0,
        blocks, blocks, prompt_direct, prompt_actual);
    if (prompt_actual[0] != q1_direct32_scalar_reference_s32(
            bits[2], scale[2], prompt_x + blocks * 128, blocks, 16) ||
        prompt_actual[1] != odd_sentinel) return 14;

    // Model the prompt dispatcher guard for a second row whose first block is
    // admissible but whose second block naturally rejects INT32_MIN. No output
    // may be published before every block is guarded; the replay must rebuild
    // both blocks through the canonical uint64 LUT, not mix a direct prefix
    // with a fallback suffix.
    int64_t reject_prompt_x[2 * blocks * 128];
    int32_t reject_prompt_direct[2 * blocks * 128] = {0};
    memcpy(reject_prompt_x, prompt_x, sizeof(reject_prompt_x));
    reject_prompt_x[(size_t) 3 * 128u] = INT32_MIN;
    int reject_prompt_bad = 0;
    int64_t reject_prompt_out = odd_sentinel;
    for (int b = 0; b < blocks; ++b) {
        const size_t block_index = (size_t) blocks + (size_t) b;
        if (prepare_q1_block_direct32(
                reject_prompt_x + block_index * 128u,
                reject_prompt_direct + block_index * 128u)) {
            reject_prompt_bad = 1;
        }
    }
    if (!reject_prompt_bad || reject_prompt_out != odd_sentinel) return 15;
    for (int b = 0; b < blocks; ++b) {
        const size_t block_index = (size_t) blocks + (size_t) b;
        prepare_q1_block_lut(
            reject_prompt_x + block_index * 128u, &totals[b],
            lut + (size_t) b * 16u * 256u);
    }
    reject_prompt_out = q1_element_s32(
        bits[2], scale[2], blocks, 16, 0, totals, lut);
    if (reject_prompt_out != q1_direct32_scalar_reference_s32(
            bits[2], scale[2],
            reject_prompt_x + (size_t) blocks * 128u, blocks, 16)) return 16;

    // Retain the distinct aggregate-lane guard as well as the INT32_MIN
    // element rejects exercised by both natural replay cases above.
    int64_t lane_over[128] = {0};
    lane_over[0] = INT32_MAX;
    lane_over[8] = 1;
    if (!prepare_q1_block_direct32(lane_over, direct)) return 9;
    return 0;
}

#endif

int bonsai_q1_prepare_i64_lut32(const int64_t *x,
                                int64_t tokens,
                                int64_t n_blocks,
                                uint64_t *totals,
                                size_t totals_count,
                                int32_t *lut,
                                size_t lut_count) {
    if (!x || !totals || !lut || tokens < 0 || n_blocks < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens, sn_blocks = (size_t) n_blocks;
    size_t tb = 0, x_width = 0, need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    int range_bad = 0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(|:range_bad)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            range_bad |= prepare_q1_block_lut32(
                xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }
    return range_bad ? 5 : 0;
}

int bonsai_q1_linear_i64_workspace_lut32(const int64_t *x,
                                         const uint8_t *bits,
                                         const int64_t *scale,
                                         int64_t tokens,
                                         int64_t out_features,
                                         int64_t n_blocks,
                                         int64_t frac,
                                         int64_t *out,
                                         uint64_t *totals,
                                         size_t totals_count,
                                         int32_t *lut,
                                         size_t lut_count) {
    if (!x || !bits || !scale || !out || !totals || !lut ||
        tokens < 0 || out_features < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens, sout_features = (size_t) out_features, sn_blocks = (size_t) n_blocks;
    size_t tb = 0, x_width = 0, bits_row_stride = 0, out_count = 0, need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(stokens, sout_features, &out_count) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    int range_bad = 0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(|:range_bad)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            range_bad |= prepare_q1_block_lut32(
                xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }
    if (range_bad) {
        return 5;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t o = 0; o < out_features; ++o) {
            const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
            const int64_t *scale_row = scale + (size_t) o * sn_blocks;
            out[(size_t) t * sout_features + (size_t) o] =
                q1_element_lut32_s64(bits_row, scale_row, n_blocks, frac,
                                     (size_t) t * (size_t) n_blocks, totals, lut);
        }
    }
    return 0;
}

int bonsai_q1_linear_i64_prepared_lut32(const uint8_t *bits,
                                        const int64_t *scale,
                                        int64_t tokens,
                                        int64_t out_features,
                                        int64_t n_blocks,
                                        int64_t frac,
                                        int64_t *out,
                                        const uint64_t *totals,
                                        size_t totals_count,
                                        const int32_t *lut,
                                        size_t lut_count) {
    if (!bits || !scale || !out || !totals || !lut ||
        tokens < 0 || out_features < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens, sout_features = (size_t) out_features, sn_blocks = (size_t) n_blocks;
    size_t tb = 0, bits_row_stride = 0, out_count = 0, need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(stokens, sout_features, &out_count) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t o = 0; o < out_features; ++o) {
            const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
            const int64_t *scale_row = scale + (size_t) o * sn_blocks;
            out[(size_t) t * sout_features + (size_t) o] =
                q1_element_lut32_s64(bits_row, scale_row, n_blocks, frac,
                                     (size_t) t * (size_t) n_blocks, totals, lut);
        }
    }
    return 0;
}

int bonsai_q1_linear_i64_prepared_multi_lut32(const uint8_t *const *bits_list,
                                              const int64_t *const *scale_list,
                                              const int64_t *out_features_list,
                                              int64_t n_items,
                                              int64_t tokens,
                                              int64_t n_blocks,
                                              int64_t frac,
                                              int64_t **out_list,
                                              const uint64_t *totals,
                                              size_t totals_count,
                                              const int32_t *lut,
                                              size_t lut_count) {
    if (!bits_list || !scale_list || !out_features_list || !out_list || !totals || !lut ||
        n_items < 0 || tokens < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens, sn_blocks = (size_t) n_blocks;
    size_t tb = 0, bits_row_stride = 0, need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    for (int64_t item = 0; item < n_items; ++item) {
        if (!bits_list[item] || !scale_list[item] || !out_list[item] || out_features_list[item] < 0) {
            return 1;
        }
    }
    #pragma omp parallel
    {
        for (int64_t item = 0; item < n_items; ++item) {
            const uint8_t *bits = bits_list[item];
            const int64_t *scale = scale_list[item];
            int64_t *out = out_list[item];
            int64_t out_features = out_features_list[item];
            size_t sout_features = (size_t) out_features;
            #pragma omp for collapse(2) schedule(static)
            for (int64_t t = 0; t < tokens; ++t) {
                for (int64_t o = 0; o < out_features; ++o) {
                    const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                    const int64_t *scale_row = scale + (size_t) o * sn_blocks;
                    out[(size_t) t * sout_features + (size_t) o] =
                        q1_element_lut32_s64(bits_row, scale_row, n_blocks, frac,
                                             (size_t) t * (size_t) n_blocks, totals, lut);
                }
            }
        }
    }
    return 0;
}

// int32-LUT counterpart of q1_prepare_apply_multi_u64_core.  The range guard
// completes before outputs are touched: if any block cannot be represented by
// the narrow LUT, every worker skips the projection loops and the caller gets
// rc=5, allowing an exact uint64 retry before any externally visible mutation.
static int q1_prepare_apply_multi_lut32_core(const int64_t *x,
                                             const uint8_t *const *bits_list,
                                             const void *const *scale_list,
                                             const int64_t *out_features_list,
                                             int64_t n_items,
                                             int64_t tokens,
                                             int64_t n_blocks,
                                             int64_t frac,
                                             int64_t **out_list,
                                             uint64_t *totals,
                                             size_t totals_count,
                                             int32_t *lut,
                                             size_t lut_count,
                                             int scale32) {
    if (!x || !bits_list || !scale_list || !out_features_list || !out_list ||
        !totals || !lut || n_items < 0 || tokens < 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    const size_t stokens = (size_t) tokens;
    const size_t sn_blocks = (size_t) n_blocks;
    size_t tb = 0, x_width = 0, bits_row_stride = 0, need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    for (int64_t item = 0; item < n_items; ++item) {
        if (!bits_list[item] || !scale_list[item] || !out_list[item] ||
            out_features_list[item] < 0) {
            return 1;
        }
    }

    int range_bad = 0;
    // AVX2 gather pays off once the per-row LUT spans the 17,408-wide
    // FFN-down shape on the target i7; at 40/48 blocks the portable scalar
    // loads are measurably cheaper than gather setup/sign extension.
    const int use_avx2 = q1_runtime_use_avx2() && n_blocks >= 128;
    #pragma omp parallel shared(range_bad)
    {
        #pragma omp for collapse(2) schedule(static)
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t b = 0; b < n_blocks; ++b) {
                const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
                const size_t idx = (size_t) t * sn_blocks + (size_t) b;
                if (prepare_q1_block_lut32(
                        xb, &totals[idx], lut + idx * 16u * 256u)) {
                    #pragma omp atomic write
                    range_bad = 1;
                }
            }
        }
        // The omp-for barrier flushes range_bad.  No output is written on a
        // failed guard, which makes the retry transactional from the caller's
        // perspective.
        if (!range_bad) {
            for (int64_t item = 0; item < n_items; ++item) {
                const uint8_t *bits = bits_list[item];
                int64_t *out = out_list[item];
                const int64_t out_features = out_features_list[item];
                const size_t sout_features = (size_t) out_features;
                if (scale32) {
                    const int32_t *scale = (const int32_t *) scale_list[item];
                    #pragma omp for collapse(2) schedule(static)
                    for (int64_t t = 0; t < tokens; ++t) {
                        for (int64_t o = 0; o < out_features; ++o) {
                            const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                            const int32_t *scale_row = scale + (size_t) o * sn_blocks;
                            #if BONSAI_CAN_BUILD_AVX2
                            out[(size_t) t * sout_features + (size_t) o] = use_avx2
                                ? q1_element_lut32_s32_avx2(
                                    bits_row, scale_row, n_blocks, frac,
                                    (size_t) t * sn_blocks, totals, lut)
                                : q1_element_lut32_s32(
                                    bits_row, scale_row, n_blocks, frac,
                                    (size_t) t * sn_blocks, totals, lut);
                            #else
                            out[(size_t) t * sout_features + (size_t) o] =
                                q1_element_lut32_s32(bits_row, scale_row, n_blocks, frac,
                                                     (size_t) t * sn_blocks, totals, lut);
                            #endif
                        }
                    }
                } else {
                    const int64_t *scale = (const int64_t *) scale_list[item];
                    #pragma omp for collapse(2) schedule(static)
                    for (int64_t t = 0; t < tokens; ++t) {
                        for (int64_t o = 0; o < out_features; ++o) {
                            const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                            const int64_t *scale_row = scale + (size_t) o * sn_blocks;
                            #if BONSAI_CAN_BUILD_AVX2
                            out[(size_t) t * sout_features + (size_t) o] = use_avx2
                                ? q1_element_lut32_s64_avx2(
                                    bits_row, scale_row, n_blocks, frac,
                                    (size_t) t * sn_blocks, totals, lut)
                                : q1_element_lut32_s64(
                                    bits_row, scale_row, n_blocks, frac,
                                    (size_t) t * sn_blocks, totals, lut);
                            #else
                            out[(size_t) t * sout_features + (size_t) o] =
                                q1_element_lut32_s64(bits_row, scale_row, n_blocks, frac,
                                                     (size_t) t * sn_blocks, totals, lut);
                            #endif
                        }
                    }
                }
            }
        }
    }
    return range_bad ? 5 : 0;
}

int bonsai_q1_prepare_apply_multi_i64_lut32(const int64_t *x,
                                            const uint8_t *const *bits_list,
                                            const int64_t *const *scale_list,
                                            const int64_t *out_features_list,
                                            int64_t n_items,
                                            int64_t tokens,
                                            int64_t n_blocks,
                                            int64_t frac,
                                            int64_t **out_list,
                                            uint64_t *totals,
                                            size_t totals_count,
                                            int32_t *lut,
                                            size_t lut_count) {
    return q1_prepare_apply_multi_lut32_core(
        x, bits_list, (const void *const *) scale_list, out_features_list,
        n_items, tokens, n_blocks, frac, out_list, totals, totals_count, lut,
        lut_count, 0);
}

int bonsai_q1_prepare_apply_multi_i64_lut32_scale32(
                                            const int64_t *x,
                                            const uint8_t *const *bits_list,
                                            const int32_t *const *scale_list,
                                            const int64_t *out_features_list,
                                            int64_t n_items,
                                            int64_t tokens,
                                            int64_t n_blocks,
                                            int64_t frac,
                                            int64_t **out_list,
                                            uint64_t *totals,
                                            size_t totals_count,
                                            int32_t *lut,
                                            size_t lut_count) {
    return q1_prepare_apply_multi_lut32_core(
        x, bits_list, (const void *const *) scale_list, out_features_list,
        n_items, tokens, n_blocks, frac, out_list, totals, totals_count, lut,
        lut_count, 1);
}

int bonsai_q1_argmax_i64_workspace_lut32(const int64_t *x,
                                         const uint8_t *bits,
                                         const int64_t *scale,
                                         int64_t tokens,
                                         int64_t out_features,
                                         int64_t n_blocks,
                                         int64_t frac,
                                         int64_t *argmax_out,
                                         int64_t *max_out,
                                         uint64_t *totals,
                                         size_t totals_count,
                                         int32_t *lut,
                                         size_t lut_count) {
    if (!x || !bits || !scale || !argmax_out || !max_out || !totals || !lut ||
        tokens < 0 || out_features <= 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    size_t stokens = (size_t) tokens, sn_blocks = (size_t) n_blocks;
    size_t tb = 0, x_width = 0, bits_row_stride = 0, need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    int range_bad = 0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(|:range_bad)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            range_bad |= prepare_q1_block_lut32(
                xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }
    if (range_bad) {
        return 5;
    }
    for (int64_t t = 0; t < tokens; ++t) {
        int64_t best_idx = 0;
        int64_t best_val = INT64_MIN;
        #pragma omp parallel
        {
            int64_t local_idx = -1;
            int64_t local_val = INT64_MIN;
            #pragma omp for schedule(static)
            for (int64_t o = 0; o < out_features; ++o) {
                const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                const int64_t *scale_row = scale + (size_t) o * sn_blocks;
                int64_t value = q1_element_lut32_s64(bits_row, scale_row, n_blocks, frac,
                                                     (size_t) t * (size_t) n_blocks, totals, lut);
                if (local_idx < 0 || value > local_val) {
                    local_val = value;
                    local_idx = o;
                }
            }
            #pragma omp critical
            {
                if (local_idx >= 0 && (local_val > best_val ||
                    (local_val == best_val && local_idx < best_idx))) {
                    best_val = local_val;
                    best_idx = local_idx;
                }
            }
        }
        argmax_out[t] = best_idx;
        max_out[t] = best_val;
    }
    return 0;
}

int bonsai_q1_argmax_i64_workspace_lut32_scale32(const int64_t *x,
                                                 const uint8_t *bits,
                                                 const int32_t *scale,
                                                 int64_t tokens,
                                                 int64_t out_features,
                                                 int64_t n_blocks,
                                                 int64_t frac,
                                                 int64_t *argmax_out,
                                                 int64_t *max_out,
                                                 uint64_t *totals,
                                                 size_t totals_count,
                                                 int32_t *lut,
                                                 size_t lut_count) {
    if (!x || !bits || !scale || !argmax_out || !max_out || !totals || !lut ||
        tokens < 0 || out_features <= 0 || n_blocks < 0 || frac < 0) {
        return 1;
    }
    const size_t stokens = (size_t) tokens, sn_blocks = (size_t) n_blocks;
    size_t tb = 0, x_width = 0, bits_row_stride = 0, need_lut_count = 0;
    if (!checked_mul_size(stokens, sn_blocks, &tb) ||
        !checked_mul_size(sn_blocks, 128u, &x_width) ||
        !checked_mul_size(sn_blocks, 16u, &bits_row_stride) ||
        !checked_mul_size(tb, 16u, &need_lut_count) ||
        !checked_mul_size(need_lut_count, 256u, &need_lut_count)) {
        return 1;
    }
    if (totals_count < tb || lut_count < need_lut_count) {
        return 3;
    }
    int range_bad = 0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(|:range_bad)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * sn_blocks + (size_t) b;
            range_bad |= prepare_q1_block_lut32(
                xb, &totals[idx], lut + idx * 16u * 256u);
        }
    }
    if (range_bad) {
        return 5;
    }
    for (int64_t t = 0; t < tokens; ++t) {
        int64_t best_idx = 0;
        int64_t best_val = INT64_MIN;
        #pragma omp parallel
        {
            int64_t local_idx = -1;
            int64_t local_val = INT64_MIN;
            #pragma omp for schedule(static)
            for (int64_t o = 0; o < out_features; ++o) {
                const uint8_t *bits_row = bits + (size_t) o * bits_row_stride;
                const int32_t *scale_row = scale + (size_t) o * sn_blocks;
                const int64_t value = q1_element_lut32_s32(
                    bits_row, scale_row, n_blocks, frac,
                    (size_t) t * sn_blocks, totals, lut);
                if (local_idx < 0 || value > local_val) {
                    local_val = value;
                    local_idx = o;
                }
            }
            #pragma omp critical
            {
                if (local_idx >= 0 && (local_val > best_val ||
                    (local_val == best_val && local_idx < best_idx))) {
                    best_val = local_val;
                    best_idx = local_idx;
                }
            }
        }
        argmax_out[t] = best_idx;
        max_out[t] = best_val;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Qwen3.5/Bonsai-27B resident M=1 executor.
//
// The correctness-first Python graph remains the canonical oracle.  This
// handle copies only immutable pointer descriptors; artifact arrays stay
// owned by Python.  Runtime caches/workspaces are disposable and excluded
// from the artifact digest.  A decode/prefill call enters exactly one OpenMP
// region and every primitive below is an `omp for` encountered by that same
// persistent team.

typedef struct {
    const uint8_t *bits;
    const int32_t *scale;
    int64_t out_features;
    int64_t n_blocks;
} bonsai35_q1_desc;

typedef struct {
    int64_t kind; // 0 recurrent, 1 full attention
    const int64_t *n1_gain;
    const int64_t *n2_gain;
    bonsai35_q1_desc w1, wu, w2;
    const int64_t *q_norm_gain;
    const int64_t *k_norm_gain;
    bonsai35_q1_desc wqg, wk, wv, wo;
    bonsai35_q1_desc wqkv, wz, walpha, wbeta, wout;
    const int64_t *conv_weight;
    const int64_t *dt_bias;
    const int64_t *ssm_a;
    const int64_t *ssm_norm_gain;
} bonsai35_layer_desc;

typedef struct {
    int64_t n_layers, context_len, frac, d_model, d_ff, vocab;
    int64_t n_heads, n_heads_kv, head_dim, rope_rot_dim;
    int64_t ssm_state_size, ssm_group_count, ssm_inner_size;
    int64_t ssm_value_heads, ssm_conv_kernel, ssm_state_frac;
    int64_t rms_eps, ssm_rms_eps, attention_scale, gdn_scale;
    int64_t lut_step, softplus_min, softplus_max, exp_min;
    int64_t softplus_count, exp_count, isa_mode;
    bonsai35_q1_desc embed, output;
    const int64_t *final_norm_gain;
    const int64_t *cos;
    const int64_t *sin;
    const int64_t *softplus_lut;
    const int64_t *exp_lut;
    const bonsai35_layer_desc *layers;
} bonsai35_model_desc;

typedef struct {
    uint64_t decode_calls, prefill_calls, team_entries, q1_groups;
    uint64_t lut32_hits, lut32_fallbacks, lut64_groups;
    uint64_t layer_major_prefills, layer_major_rows;
    uint64_t prefill_tiles_40, prefill_tiles_48, prefill_tiles_136;
    uint64_t fused_residual_rms_calls, parallel_rms_calls;
    int64_t last_team_size, selected_isa, selected_lut_bits, cache_width_bits;
    int64_t prefill_tile_40, prefill_tile_48, prefill_tile_136;
} bonsai35_exec_stats;

typedef struct bonsai35_model {
    bonsai35_model_desc d;
    bonsai35_layer_desc *layers;
    int64_t pos;
    int error;
    int range_bad;
    int use_lut32;
    int use_avx2;
    int force_lut_fallback;
    int debug_error_mode;
    int64_t argmax_index, argmax_value;
    bonsai35_exec_stats stats;

    // Decode retains the compact M=1 buffers below. Prompt execution uses a
    // lazily-grown residual arena plus bounded, reusable phase buffers so the
    // 4 GiB weights are traversed layer-major without retaining every prompt
    // intermediate.
    int64_t prefill_tile_rows;
    int64_t *prefill_x;
    size_t prefill_capacity_rows;
    int64_t *tile_n1, *tile_n2, *tile_branch;
    int64_t *tile_qkv, *tile_z, *tile_alpha, *tile_beta;
    int64_t *tile_kproj, *tile_vproj, *tile_conv_out;
    int64_t *tile_qmap, *tile_kmap, *tile_gdn;
    int64_t *tile_ff_gate, *tile_ff_up, *tile_ff_hidden;
    int64_t *tile_scores;
    int *tile_range_bad;
    uint64_t *tile_totals, *tile_lut64;
    int32_t *tile_lut32;
    size_t tile_lut_blocks_capacity;

    // Fixed-shape RMS reduction scratch. Partial sums are merged in thread-ID
    // order, preserving the exact u128 result across OpenMP schedules.
    unsigned __int128 *rms_partials;
    unsigned char *rms_partial_ok;
    int64_t rms_partial_capacity;
    uint64_t rms_value;
    int rms_ok;

    // Production-disabled selected-layer trace. Six complete int64 boundary
    // tensors are retained only after an explicit debug selection.
    int64_t trace_layer, trace_rows;
    int64_t trace_target_pos;
    size_t trace_capacity_rows;
    int64_t *trace[6]; // n1, branch, residual, n2, ffn, output
    int64_t *trace_internal[16];
    size_t trace_internal_capacity[16];
    int64_t trace_internal_count[16];

    int64_t **state, **conv, **kcache, **vcache;
    int64_t *x, *n1, *n2, *branch;
    int64_t *qkv, *z, *alpha, *beta;
    int64_t *kproj, *vproj, *conv_out;
    int64_t *qmap, *kmap, *gdn;
    int64_t *ff_gate, *ff_up, *ff_hidden;
    int64_t *scores;
    uint64_t *totals, *lut64;
    int32_t *lut32;
    size_t lut_count;
    // Every resident allocation/export extent is checked once at descriptor
    // admission and reused thereafter; no unchecked dimension products remain
    // on reset, diagnostics, or export paths.
    size_t state_n, conv_dim, conv_n, kv_per_head, kv_n;
    size_t qg_width, kv_width, attention_inner, scores_n;
} bonsai35_model;

#define BONSAI35_MODEL_ABI_VERSION UINT64_C(2)

uint64_t bonsai35_model_abi_version(void) {
    return BONSAI35_MODEL_ABI_VERSION;
}

// kind: 0=model descriptor, 1=layer descriptor, 2=Q1 descriptor,
// 3=executor stats.  Returning zero for an unknown kind fails closed.
size_t bonsai35_model_abi_sizeof(int kind) {
    if (kind == 0) return sizeof(bonsai35_model_desc);
    if (kind == 1) return sizeof(bonsai35_layer_desc);
    if (kind == 2) return sizeof(bonsai35_q1_desc);
    if (kind == 3) return sizeof(bonsai35_exec_stats);
    return 0;
}

static void *bonsai_calloc_array(size_t n, size_t width) {
    if (n != 0 && width > SIZE_MAX / n) return NULL;
    return calloc(n, width);
}

static int64_t bonsai35_env_tile_override(void) {
    const char *value = getenv("TRINOTE_BONSAI35_PREFILL_TILE");
    if (!value || !*value) value = getenv("TRINOTE_BONSAI35_Q1_CHUNK");
    if (!value || !*value) return 0;
    char *end = NULL;
    const long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > 32) return 0;
    return (int64_t) parsed;
}

static int64_t bonsai35_prefill_tile_for_blocks(const bonsai35_model *m,
                                                 int64_t n_blocks) {
    const int64_t override = bonsai35_env_tile_override();
    if (override) return override;
    // i7-10700F real-artifact sweeps found every M>1 activation-LUT tile
    // slower at all three committed widths for both storage modes. Keep the
    // width/LUT keyed policy surface and counters, but default each measured
    // cell to the conservative exact M=1 tile; explicit sweeps use override.
    (void) m;
    (void) n_blocks;
    return 1;
}

static void bonsai35_count_prefill_tile(bonsai35_model *m, int64_t n_blocks) {
    #pragma omp single
    {
        if (n_blocks == 40) m->stats.prefill_tiles_40++;
        else if (n_blocks == 48) m->stats.prefill_tiles_48++;
        else if (n_blocks == 136) m->stats.prefill_tiles_136++;
    }
}

static int bonsai35_q1_valid(const bonsai35_q1_desc *w) {
    return w && w->bits && w->scale && w->out_features > 0 && w->n_blocks > 0;
}

static int bonsai35_q1_shape(const bonsai35_q1_desc *w,
                             int64_t out_features, int64_t in_features) {
    return bonsai35_q1_valid(w) && in_features > 0 && in_features % 128 == 0 &&
           w->out_features == out_features && w->n_blocks == in_features / 128;
}

static void bonsai35_set_error(bonsai35_model *m, int code) {
    #pragma omp critical(bonsai35_error)
    { if (m->error == 0) m->error = code; }
}

static int64_t bonsai35_sigmoid_one(int64_t x, int64_t frac) {
    int64_t log2e, d_clip;
    if (frac == 16) {
        // bonsai35_model_create admits only the committed Q16 release graph.
        // Fold its two invariant clipping divisions once at compile time
        // instead of repeating them for roughly two million elementwise
        // sigmoid/SiLU evaluations per token.  Keep the generic branch for
        // direct helper reuse and future descriptor versions.
        log2e = BONSAI_LOG2E_Q16;
        d_clip = (INT64_C(18) << 32) / BONSAI_LOG2E_Q16;
    } else {
        log2e = bonsai_scaled_log2e(frac);
        const int64_t a = ((frac + 2) << (2 * frac)) / log2e;
        const int64_t b = ((int64_t) 1 << 62) / log2e;
        d_clip = a < b ? a : b;
    }
    const int64_t mx = x > 0 ? x : 0;
    int64_t d0 = mx;
    int64_t d1 = u64_to_i64((uint64_t) mx - (uint64_t) x);
    if (d0 > d_clip) d0 = d_clip;
    if (d1 > d_clip) d1 = d_clip;
    const int64_t e0 = bonsai_exp2_neg_fixed((d0 * log2e) >> frac, frac);
    const int64_t e1 = bonsai_exp2_neg_fixed((d1 * log2e) >> frac, frac);
    const int64_t den = e0 + e1;
    // Same exact bounded positive quotient as bonsai_silu_i64 above.  The
    // release descriptor fixes frac=16, but retain the full admitted helper
    // envelope without paying for a compiler-runtime 128-bit division.
    const uint64_t numerator = (uint64_t) e1 * (UINT64_C(1) << frac);
    return den ? (int64_t) (numerator / (uint64_t) den) : 0;
}

static int64_t bonsai35_silu_one(int64_t x, int64_t frac) {
    const int64_t sig = bonsai35_sigmoid_one(x, frac);
    return arshift_i64(u64_to_i64((uint64_t) x * (uint64_t) sig), frac);
}

static int64_t bonsai35_lut_interp(int64_t x, const int64_t *table,
                                   int64_t count, int64_t minimum, int64_t step) {
    const int64_t maximum = minimum + step * (count - 1);
    int64_t xc = x < minimum ? minimum : (x > maximum ? maximum : x);
    const int64_t pos = xc - minimum;
    int64_t idx = pos / step;
    if (idx > count - 2) idx = count - 2;
    const int64_t rem = pos - idx * step;
    const __int128 num = (__int128) (table[idx + 1] - table[idx]) * rem;
    return table[idx] + (int64_t) floor_div_i128_u64(num, (uint64_t) step);
}

static int64_t bonsai35_softplus(const bonsai35_model *m, int64_t x) {
    if (x <= m->d.softplus_min) return 0;
    if (x >= m->d.softplus_max) return x;
    return bonsai35_lut_interp(x, m->d.softplus_lut, m->d.softplus_count,
                               m->d.softplus_min, m->d.lut_step);
}

static int64_t bonsai35_exp_negative(const bonsai35_model *m, int64_t x) {
    if (x <= m->d.exp_min) return 0;
    if (x >= 0) return (int64_t) 1 << m->d.frac;
    return bonsai35_lut_interp(x, m->d.exp_lut, m->d.exp_count,
                               m->d.exp_min, m->d.lut_step);
}

// Called by every member of the resident team.
static void bonsai35_q1_group(bonsai35_model *m, const int64_t *x,
                              const bonsai35_q1_desc *const *weights,
                              int64_t *const *outputs, int n_items) {
    const int64_t n_blocks = weights[0]->n_blocks;
    #pragma omp single
    {
        m->range_bad = m->use_lut32 ? 0 : 2;
        m->stats.q1_groups++;
    }
    if (m->use_lut32) {
        #pragma omp for schedule(static)
        for (int64_t b = 0; b < n_blocks; ++b) {
            int bad;
            #if BONSAI_CAN_BUILD_AVX2
            if (m->use_avx2) {
                bad = prepare_q1_block_direct32(
                    x + (size_t) b * 128u,
                    m->lut32 + (size_t) b * 128u);
            } else
            #endif
            {
                bad = prepare_q1_block_lut32(
                    x + (size_t) b * 128u, &m->totals[b],
                    m->lut32 + (size_t) b * 16u * 256u);
            }
            if (bad) {
                #pragma omp atomic write
                m->range_bad = 1;
            }
        }
    }
    #pragma omp single
    {
        // Test/debug control for exercising the exact uint64 replay path.
        // No projection output (and therefore no dependent cache state) has
        // been written when this guard is raised.  Account for the selected
        // path in this same synchronization point rather than waking the team
        // for a second stats-only `single` region.
        if (m->use_lut32 && m->force_lut_fallback) m->range_bad = 1;
        if (m->range_bad == 1) m->stats.lut32_fallbacks++;
        else if (m->range_bad == 2) m->stats.lut64_groups++;
        else m->stats.lut32_hits++;
    }
    if (m->range_bad) {
        #pragma omp for schedule(static)
        for (int64_t b = 0; b < n_blocks; ++b) {
            prepare_q1_block_lut(x + (size_t) b * 128u,
                                 &m->totals[b],
                                 m->lut64 + (size_t) b * 16u * 256u);
        }
    }
    for (int item = 0; item < n_items; ++item) {
        const bonsai35_q1_desc *w = weights[item];
        int64_t *out = outputs[item];
        if (w->n_blocks != n_blocks) {
            #pragma omp single
            { bonsai35_set_error(m, 1); }
        }
        // Same-input projections are mutually independent.  Let workers
        // advance directly from one output tensor to the next, then publish
        // the complete group with the single barrier below.
        #if BONSAI_CAN_BUILD_AVX2
        if (!m->range_bad && m->use_avx2) {
            const int64_t pairs = (w->out_features + 1) / 2;
            #pragma omp for schedule(static) nowait
            for (int64_t pair = 0; pair < pairs; ++pair) {
                const int64_t o = pair * 2;
                const int have_second = o + 1 < w->out_features;
                const size_t bits_stride = (size_t) n_blocks * 16u;
                const size_t scale_stride = (size_t) n_blocks;
                int64_t values[2] = {0, 0};
                q1_elements2_direct32_s32_avx2_q16(
                    w->bits + (size_t) o * bits_stride,
                    w->scale + (size_t) o * scale_stride,
                    w->bits + (size_t) (o + have_second) * bits_stride,
                    w->scale + (size_t) (o + have_second) * scale_stride,
                    have_second, n_blocks, 0,
                    m->lut32, values);
                out[o] = values[0];
                if (have_second) out[o + 1] = values[1];
            }
        } else
        #endif
        {
            #pragma omp for schedule(static) nowait
            for (int64_t o = 0; o < w->out_features; ++o) {
                const uint8_t *bits = w->bits +
                    (size_t) o * (size_t) n_blocks * 16u;
                const int32_t *scale = w->scale +
                    (size_t) o * (size_t) n_blocks;
                if (m->range_bad) {
                    out[o] = q1_element_s32(
                        bits, scale, n_blocks, m->d.frac,
                        0, m->totals, m->lut64);
                } else {
                    out[o] = q1_element_lut32_s32(
                        bits, scale, n_blocks, m->d.frac,
                        0, m->totals, m->lut32);
                }
            }
        }
    }
    #pragma omp barrier
}

// Apply one same-input projection group to a bounded prompt tile. Each row
// retains the exact M=1 arithmetic; tiling changes only traversal order and
// keeps the active layer's weights hot.  A physical one-row AVX2 tile uses the
// resident compact direct32 representation, while portable, uint64, and true
// multi-row tiles retain the established activation-LUT paths below.
static void bonsai35_q1_group_rows(bonsai35_model *m, const int64_t *x,
                                   int64_t rows, int64_t x_stride,
                                   const bonsai35_q1_desc *const *weights,
                                   int64_t *const *outputs,
                                   const int64_t *output_strides,
                                   int n_items) {
    const int64_t n_blocks = weights[0]->n_blocks;
    const int64_t width_tile = n_blocks == 40 ? m->stats.prefill_tile_40
        : n_blocks == 48 ? m->stats.prefill_tile_48
        : n_blocks == 136 ? m->stats.prefill_tile_136
        : bonsai35_prefill_tile_for_blocks(m, n_blocks);
    for (int64_t base = 0; base < rows; base += width_tile) {
        int64_t count = rows - base;
        if (count > width_tile) count = width_tile;
        #if BONSAI_CAN_BUILD_AVX2
        const int direct32_one_row =
            count == 1 && m->use_lut32 && m->use_avx2;
        #endif
        if ((size_t) count * (size_t) n_blocks > m->tile_lut_blocks_capacity) {
            #pragma omp single
            { bonsai35_set_error(m, 2); }
            return;
        }
        bonsai35_count_prefill_tile(m, n_blocks);
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < count; ++r)
            m->tile_range_bad[r] = m->use_lut32 ? 0 : 2;
        if (m->use_lut32) {
            #pragma omp for schedule(static)
            for (int64_t idx = 0; idx < count * n_blocks; ++idx) {
                const int64_t r = idx / n_blocks;
                const int64_t b = idx % n_blocks;
                const size_t block_index =
                    (size_t) r * (size_t) n_blocks + (size_t) b;
                const int64_t *xb =
                    x + (size_t) (base + r) * (size_t) x_stride +
                        (size_t) b * 128u;
                int bad;
                #if BONSAI_CAN_BUILD_AVX2
                if (direct32_one_row) {
                    bad = prepare_q1_block_direct32(
                        xb, m->tile_lut32 + block_index * 128u);
                } else
                #endif
                {
                    bad = prepare_q1_block_lut32(
                        xb, &m->tile_totals[block_index],
                        m->tile_lut32 + block_index * 16u * 256u);
                }
                if (bad) {
                    #pragma omp atomic write
                    m->tile_range_bad[r] = 1;
                }
            }
            #pragma omp single
            {
                if (m->force_lut_fallback) {
                    for (int64_t r = 0; r < count; ++r)
                        m->tile_range_bad[r] = 1;
                }
            }
        }
        #pragma omp single
        {
            m->stats.q1_groups += (uint64_t) count;
            for (int64_t r = 0; r < count; ++r) {
                if (m->tile_range_bad[r] == 1) m->stats.lut32_fallbacks++;
                else if (m->tile_range_bad[r] == 2) m->stats.lut64_groups++;
                else m->stats.lut32_hits++;
            }
        }
        #pragma omp for schedule(static)
        for (int64_t idx = 0; idx < count * n_blocks; ++idx) {
            const int64_t r = idx / n_blocks;
            const int64_t b = idx % n_blocks;
            if (m->tile_range_bad[r]) {
                prepare_q1_block_lut(
                    x + (size_t) (base + r) * (size_t) x_stride +
                        (size_t) b * 128u,
                    &m->tile_totals[(size_t) r * (size_t) n_blocks +
                                    (size_t) b],
                    m->tile_lut64 +
                        ((size_t) r * (size_t) n_blocks + (size_t) b) *
                            16u * 256u);
            }
        }
        for (int item = 0; item < n_items; ++item) {
            const bonsai35_q1_desc *w = weights[item];
            if (w->n_blocks != n_blocks) {
                #pragma omp single
                { bonsai35_set_error(m, 1); }
            }
            #if BONSAI_CAN_BUILD_AVX2
            if (direct32_one_row && !m->tile_range_bad[0]) {
                const int64_t pairs = (w->out_features + 1) / 2;
                const size_t bits_stride = (size_t) n_blocks * 16u;
                const size_t scale_stride = (size_t) n_blocks;
                int64_t *out = outputs[item] +
                    (size_t) base * (size_t) output_strides[item];
                #pragma omp for schedule(static)
                for (int64_t pair = 0; pair < pairs; ++pair) {
                    const int64_t o = pair * 2;
                    const int have_second = o + 1 < w->out_features;
                    int64_t values[2] = {0, 0};
                    q1_elements2_direct32_s32_avx2_q16(
                        w->bits + (size_t) o * bits_stride,
                        w->scale + (size_t) o * scale_stride,
                        w->bits + (size_t) (o + have_second) * bits_stride,
                        w->scale + (size_t) (o + have_second) * scale_stride,
                        have_second, n_blocks, 0, m->tile_lut32, values);
                    out[o] = values[0];
                    if (have_second) out[o + 1] = values[1];
                }
                continue;
            }
            #endif
            #pragma omp for schedule(static)
            for (int64_t o = 0; o < w->out_features; ++o) {
                const uint8_t *bits = w->bits +
                    (size_t) o * (size_t) n_blocks * 16u;
                const int32_t *scale = w->scale +
                    (size_t) o * (size_t) n_blocks;
                if (count == 1) {
                    int64_t value;
                    if (m->tile_range_bad[0]) {
                        value = q1_element_s32(
                            bits, scale, n_blocks, m->d.frac, 0,
                            m->tile_totals, m->tile_lut64);
                    } else {
                        #if BONSAI_CAN_BUILD_AVX2
                        if (m->use_avx2 && n_blocks >= 128) {
                            value = q1_element_lut32_s32_avx2(
                                bits, scale, n_blocks, m->d.frac, 0,
                                m->tile_totals, m->tile_lut32);
                        } else
                        #endif
                        {
                            value = q1_element_lut32_s32(
                                bits, scale, n_blocks, m->d.frac, 0,
                                m->tile_totals, m->tile_lut32);
                        }
                    }
                    outputs[item][(size_t) base * (size_t) output_strides[item] +
                                  (size_t) o] = value;
                    continue;
                }

                // Block-major multi-row accumulation reuses each packed
                // weight block while touching only count*16 KiB of activation
                // LUT at once. Per-row block order and uint64 accumulation are
                // unchanged, so the result is identical to count M=1 calls.
                uint64_t row_total[32] = {0};
                for (int64_t b = 0; b < n_blocks; ++b) {
                    const uint8_t *bb = bits + (size_t) b * 16u;
                    const uint64_t scale_b = (uint64_t) (int64_t) scale[b];
                    for (int64_t r = 0; r < count; ++r) {
                        const size_t block_index =
                            (size_t) r * (size_t) n_blocks + (size_t) b;
                        uint64_t signed_sum;
                        if (m->tile_range_bad[r]) {
                            uint64_t pos_sum = 0;
                            const uint64_t *block_lut = m->tile_lut64 +
                                block_index * 16u * 256u;
                            for (int byte_i = 0; byte_i < 16; ++byte_i) {
                                pos_sum += block_lut[
                                    (size_t) byte_i * 256u + (size_t) bb[byte_i]];
                            }
                            signed_sum = 2u * pos_sum - m->tile_totals[block_index];
                        } else {
                            int64_t pos_sum = 0;
                            const int32_t *block_lut = m->tile_lut32 +
                                block_index * 16u * 256u;
                            for (int byte_i = 0; byte_i < 16; ++byte_i) {
                                pos_sum += (int64_t) block_lut[
                                    (size_t) byte_i * 256u + (size_t) bb[byte_i]];
                            }
                            signed_sum = 2u * (uint64_t) pos_sum -
                                         m->tile_totals[block_index];
                        }
                        row_total[r] += (uint64_t) arshift_i64(
                            u64_to_i64(signed_sum * scale_b), m->d.frac);
                    }
                }
                for (int64_t r = 0; r < count; ++r) {
                    outputs[item][
                        (size_t) (base + r) * (size_t) output_strides[item] +
                        (size_t) o] = u64_to_i64(row_total[r]);
                }
            }
        }
    }
}

// Final output projection with deterministic lowest-index tie breaking.  This
// runs inside the same persistent team as the 64 transformer layers, avoiding
// a second native/OpenMP entry for greedy decode.
static void bonsai35_q1_argmax(bonsai35_model *m, const int64_t *x,
                               const bonsai35_q1_desc *w, int64_t *out) {
    const int64_t n_blocks = w->n_blocks;
    #pragma omp single
    {
        m->range_bad = m->use_lut32 ? 0 : 2;
        m->argmax_index = 0;
        m->argmax_value = INT64_MIN;
        m->stats.q1_groups++;
    }
    if (m->use_lut32) {
        #pragma omp for schedule(static)
        for (int64_t b = 0; b < n_blocks; ++b) {
            int bad;
            #if BONSAI_CAN_BUILD_AVX2
            if (m->use_avx2) {
                bad = prepare_q1_block_direct32(
                    x + (size_t) b * 128u,
                    m->lut32 + (size_t) b * 128u);
            } else
            #endif
            {
                bad = prepare_q1_block_lut32(
                    x + (size_t) b * 128u, &m->totals[b],
                    m->lut32 + (size_t) b * 16u * 256u);
            }
            if (bad) {
                #pragma omp atomic write
                m->range_bad = 1;
            }
        }
    }
    #pragma omp single
    {
        if (m->use_lut32 && m->force_lut_fallback) m->range_bad = 1;
        if (m->range_bad == 1) m->stats.lut32_fallbacks++;
        else if (m->range_bad == 2) m->stats.lut64_groups++;
        else m->stats.lut32_hits++;
    }
    if (m->range_bad) {
        #pragma omp for schedule(static)
        for (int64_t b = 0; b < n_blocks; ++b) {
            prepare_q1_block_lut(x + (size_t) b * 128u,
                                 &m->totals[b],
                                 m->lut64 + (size_t) b * 16u * 256u);
        }
    }

    int64_t local_index = -1;
    int64_t local_value = INT64_MIN;
    #if BONSAI_CAN_BUILD_AVX2
    if (!m->range_bad && m->use_avx2) {
        const int64_t pairs = (w->out_features + 1) / 2;
        #pragma omp for schedule(static)
        for (int64_t pair = 0; pair < pairs; ++pair) {
            const int64_t o = pair * 2;
            const int have_second = o + 1 < w->out_features;
            const size_t bits_stride = (size_t) n_blocks * 16u;
            const size_t scale_stride = (size_t) n_blocks;
            int64_t values[2] = {0, 0};
            q1_elements2_direct32_s32_avx2_q16(
                w->bits + (size_t) o * bits_stride,
                w->scale + (size_t) o * scale_stride,
                w->bits + (size_t) (o + have_second) * bits_stride,
                w->scale + (size_t) (o + have_second) * scale_stride,
                have_second, n_blocks, 0,
                m->lut32, values);
            for (int lane = 0; lane < 1 + have_second; ++lane) {
                const int64_t index = o + lane;
                const int64_t value = values[lane];
                q1_argmax_consider(index, value, &local_index, &local_value);
            }
        }
    } else
    #endif
    {
        #pragma omp for schedule(static)
        for (int64_t o = 0; o < w->out_features; ++o) {
            const uint8_t *bits = w->bits +
                (size_t) o * (size_t) n_blocks * 16u;
            const int32_t *scale = w->scale +
                (size_t) o * (size_t) n_blocks;
            const int64_t value = m->range_bad
                ? q1_element_s32(bits, scale, n_blocks, m->d.frac,
                                 0, m->totals, m->lut64)
                : q1_element_lut32_s32(bits, scale, n_blocks, m->d.frac,
                                       0, m->totals, m->lut32);
            q1_argmax_consider(o, value, &local_index, &local_value);
        }
    }
    #pragma omp critical(bonsai35_argmax_reduce)
    {
        if (local_index >= 0)
            q1_argmax_consider(local_index, local_value,
                               &m->argmax_index, &m->argmax_value);
    }
    #pragma omp barrier
    #pragma omp single
    { *out = m->argmax_index; }
}

// Match fixedpoint.py's pre-shift RMS gain refusal exactly.  The oracle first
// forms normalized Q(frac) values, then refuses the whole row when
// max|normalized| * max|gain| would overflow int64 before the final shift.
static int bonsai35_rms_gain_ok(const int64_t *x, int64_t cols,
                                int64_t frac, uint64_t rms,
                                const int64_t *gain) {
    if (!gain) return 1;
    const __int128 fp = (__int128) 1 << frac;
    int64_t minimum = 0, maximum = 0;
    unsigned __int128 max_gain = 0;
    for (int64_t c = 0; c < cols; ++c) {
        if (x[c] < minimum) minimum = x[c];
        if (x[c] > maximum) maximum = x[c];
        const int64_t gc = gain[c];
        const unsigned __int128 ag = gc < 0
            ? (unsigned __int128) (-(__int128) gc)
            : (unsigned __int128) gc;
        if (ag > max_gain) max_gain = ag;
    }
    // floor(x*FP/rms) is monotone for positive rms.  Its largest absolute
    // value over a row must therefore occur at the row's minimum or maximum;
    // normalizing every element again only repeated the expensive 128-bit
    // division already performed by the output pass.  Including zero in the
    // extrema handles one-sided rows without a special case and preserves the
    // exact negative floor behavior.
    const __int128 normalized_minimum = floor_div_i128_u64(
        (__int128) minimum * fp, rms);
    const __int128 normalized_maximum = floor_div_i128_u64(
        (__int128) maximum * fp, rms);
    const unsigned __int128 abs_minimum = normalized_minimum < 0
        ? (unsigned __int128) (-normalized_minimum)
        : (unsigned __int128) normalized_minimum;
    const unsigned __int128 abs_maximum = normalized_maximum < 0
        ? (unsigned __int128) (-normalized_maximum)
        : (unsigned __int128) normalized_maximum;
    const unsigned __int128 max_norm =
        abs_minimum > abs_maximum ? abs_minimum : abs_maximum;
    return max_norm == 0 ||
        max_gain <= (unsigned __int128) INT64_MAX / max_norm;
}

static void bonsai35_rms(bonsai35_model *m, const int64_t *x, int64_t rows,
                         int64_t cols, int64_t stride, const int64_t *gain,
                         int64_t eps, int64_t *out) {
    if (rows == 1 && cols == m->d.d_model &&
        m->rms_partial_capacity >= omp_get_num_threads()) {
        const int tid = omp_get_thread_num();
        const int team = omp_get_num_threads();
        const int64_t lo = cols * (int64_t) tid / (int64_t) team;
        const int64_t hi = cols * (int64_t) (tid + 1) / (int64_t) team;
        unsigned __int128 local = 0;
        int ok = 1;
        for (int64_t c = lo; c < hi; ++c) {
            if (!add_square_u128(&local, x[c])) { ok = 0; break; }
        }
        m->rms_partials[tid] = local;
        m->rms_partial_ok[tid] = (unsigned char) ok;
        #pragma omp barrier
        #pragma omp single
        {
            const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
            unsigned __int128 ssq = 0;
            m->rms_ok = 1;
            for (int i = 0; i < team; ++i) {
                if (!m->rms_partial_ok[i] || ssq > max_u128 - m->rms_partials[i]) {
                    m->rms_ok = 0;
                    break;
                }
                ssq += m->rms_partials[i];
            }
            if (m->rms_ok) {
                const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
                unsigned __int128 mean = ssq / (unsigned __int128) cols;
                if ((unsigned __int128) (uint64_t) eps > max_u128 - mean) {
                    m->rms_ok = 0;
                } else {
                    mean += (unsigned __int128) (uint64_t) eps;
                    m->rms_value = isqrt_u128(mean);
                }
                if (!m->rms_value) m->rms_ok = 0;
            }
            if (m->rms_ok && !bonsai35_rms_gain_ok(
                    x, cols, m->d.frac, m->rms_value, gain)) {
                m->rms_ok = 0;
            }
            m->stats.parallel_rms_calls++;
            if (!m->rms_ok) bonsai35_set_error(m, 4);
        }
        #pragma omp barrier
        if (m->rms_ok) {
            const uint64_t rms = m->rms_value;
            #pragma omp for schedule(static)
            for (int64_t c = 0; c < cols; ++c) {
                int64_t normalized;
                if (floor_scaled_i64_u64(
                        x[c], m->d.frac, rms, &normalized)) {
                    // bonsai35_rms_gain_ok proved this product is inside the
                    // signed-int64 envelope before any worker reaches here.
                    out[c] = gain ? arshift_i64(
                        normalized * gain[c], m->d.frac) : normalized;
                    continue;
                }
                const __int128 num = (__int128) x[c] *
                    ((__int128) 1 << m->d.frac);
                __int128 value = floor_div_i128_u64(num, rms);
                if (gain) value = floor_shift_i128(
                    value * (__int128) gain[c], m->d.frac);
                if (!i128_to_i64(value, &out[c])) bonsai35_set_error(m, 4);
            }
        }
        return;
    }
    #pragma omp for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t *xr = x + (size_t) r * (size_t) stride;
        int64_t *yr = out + (size_t) r * (size_t) cols;
        unsigned __int128 ssq = 0;
        int ok = 1;
        for (int64_t c = 0; c < cols; ++c) {
            if (!add_square_u128(&ssq, xr[c])) { ok = 0; break; }
        }
        if (!ok) { bonsai35_set_error(m, 4); continue; }
        const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
        unsigned __int128 mean = ssq / (unsigned __int128) cols;
        if ((unsigned __int128) (uint64_t) eps > max_u128 - mean) {
            bonsai35_set_error(m, 4); continue;
        }
        mean += (unsigned __int128) (uint64_t) eps;
        const uint64_t rms = isqrt_u128(mean);
        if (rms == 0) { bonsai35_set_error(m, 1); continue; }
        if (!bonsai35_rms_gain_ok(xr, cols, m->d.frac, rms, gain)) {
            bonsai35_set_error(m, 4); continue;
        }
        for (int64_t c = 0; c < cols; ++c) {
            int64_t normalized;
            if (floor_scaled_i64_u64(
                    xr[c], m->d.frac, rms, &normalized)) {
                yr[c] = gain ? arshift_i64(
                    normalized * gain[c], m->d.frac) : normalized;
                continue;
            }
            const __int128 num = (__int128) xr[c] *
                ((__int128) 1 << m->d.frac);
            __int128 value = floor_div_i128_u64(num, rms);
            if (gain) value = floor_shift_i128(value * (__int128) gain[c], m->d.frac);
            if (!i128_to_i64(value, &yr[c])) { bonsai35_set_error(m, 4); break; }
        }
    }
}

// Fuse the first residual update with the following RMSNorm. The wrapped
// int64 residual is written before it contributes to the exact u128 sum, so
// this is byte-identical to the former add loop followed by bonsai35_rms.
static void bonsai35_residual_rms(bonsai35_model *m, int64_t *x,
                                  const int64_t *branch, int64_t rows,
                                  int64_t cols, const int64_t *gain,
                                  int64_t eps, int64_t *out) {
    #pragma omp single
    { m->stats.fused_residual_rms_calls++; }
    if (rows == 1 && cols == m->d.d_model &&
        m->rms_partial_capacity >= omp_get_num_threads()) {
        const int tid = omp_get_thread_num();
        const int team = omp_get_num_threads();
        const int64_t lo = cols * (int64_t) tid / (int64_t) team;
        const int64_t hi = cols * (int64_t) (tid + 1) / (int64_t) team;
        unsigned __int128 local = 0;
        int ok = 1;
        for (int64_t c = lo; c < hi; ++c) {
            const int64_t value = u64_to_i64(
                (uint64_t) x[c] + (uint64_t) branch[c]);
            x[c] = value;
            if (!add_square_u128(&local, value)) { ok = 0; break; }
        }
        m->rms_partials[tid] = local;
        m->rms_partial_ok[tid] = (unsigned char) ok;
        #pragma omp barrier
        #pragma omp single
        {
            const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
            unsigned __int128 ssq = 0;
            m->rms_ok = 1;
            for (int i = 0; i < team; ++i) {
                if (!m->rms_partial_ok[i] || ssq > max_u128 - m->rms_partials[i]) {
                    m->rms_ok = 0;
                    break;
                }
                ssq += m->rms_partials[i];
            }
            if (m->rms_ok) {
                const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
                unsigned __int128 mean = ssq / (unsigned __int128) cols;
                if ((unsigned __int128) (uint64_t) eps > max_u128 - mean) {
                    m->rms_ok = 0;
                } else {
                    mean += (unsigned __int128) (uint64_t) eps;
                    m->rms_value = isqrt_u128(mean);
                }
                if (!m->rms_value) m->rms_ok = 0;
            }
            if (m->rms_ok && !bonsai35_rms_gain_ok(
                    x, cols, m->d.frac, m->rms_value, gain)) {
                m->rms_ok = 0;
            }
            m->stats.parallel_rms_calls++;
            if (!m->rms_ok) bonsai35_set_error(m, 4);
        }
        #pragma omp barrier
        if (m->rms_ok) {
            const uint64_t rms = m->rms_value;
            #pragma omp for schedule(static)
            for (int64_t c = 0; c < cols; ++c) {
                int64_t normalized;
                if (floor_scaled_i64_u64(
                        x[c], m->d.frac, rms, &normalized)) {
                    out[c] = gain ? arshift_i64(
                        normalized * gain[c], m->d.frac) : normalized;
                    continue;
                }
                const __int128 num = (__int128) x[c] *
                    ((__int128) 1 << m->d.frac);
                __int128 value = floor_div_i128_u64(num, rms);
                if (gain) value = floor_shift_i128(
                    value * (__int128) gain[c], m->d.frac);
                if (!i128_to_i64(value, &out[c])) bonsai35_set_error(m, 4);
            }
        }
        return;
    }
    #pragma omp for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        int64_t *xr = x + (size_t) r * (size_t) cols;
        const int64_t *br = branch + (size_t) r * (size_t) cols;
        int64_t *yr = out + (size_t) r * (size_t) cols;
        unsigned __int128 ssq = 0;
        int ok = 1;
        for (int64_t c = 0; c < cols; ++c) {
            xr[c] = u64_to_i64((uint64_t) xr[c] + (uint64_t) br[c]);
            if (!add_square_u128(&ssq, xr[c])) { ok = 0; break; }
        }
        if (!ok) { bonsai35_set_error(m, 4); continue; }
        const unsigned __int128 max_u128 = ~(unsigned __int128) 0;
        unsigned __int128 mean = ssq / (unsigned __int128) cols;
        if ((unsigned __int128) (uint64_t) eps > max_u128 - mean) {
            bonsai35_set_error(m, 4); continue;
        }
        mean += (unsigned __int128) (uint64_t) eps;
        const uint64_t rms = isqrt_u128(mean);
        if (!rms) { bonsai35_set_error(m, 1); continue; }
        if (!bonsai35_rms_gain_ok(xr, cols, m->d.frac, rms, gain)) {
            bonsai35_set_error(m, 4); continue;
        }
        for (int64_t c = 0; c < cols; ++c) {
            int64_t normalized;
            if (floor_scaled_i64_u64(
                    xr[c], m->d.frac, rms, &normalized)) {
                yr[c] = gain ? arshift_i64(
                    normalized * gain[c], m->d.frac) : normalized;
                continue;
            }
            const __int128 num = (__int128) xr[c] *
                ((__int128) 1 << m->d.frac);
            __int128 value = floor_div_i128_u64(num, rms);
            if (gain) value = floor_shift_i128(
                value * (__int128) gain[c], m->d.frac);
            if (!i128_to_i64(value, &yr[c])) {
                bonsai35_set_error(m, 4);
                break;
            }
        }
    }
}

static void bonsai35_l2(bonsai35_model *m, const int64_t *x, int64_t rows,
                        int64_t cols, int64_t *out) {
    #pragma omp for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t *xr = x + (size_t) r * (size_t) cols;
        int64_t *yr = out + (size_t) r * (size_t) cols;
        unsigned __int128 ssq = 0;
        int ok = 1;
        for (int64_t c = 0; c < cols; ++c) {
            if (!add_square_u128(&ssq, xr[c])) { ok = 0; break; }
        }
        if (!ok) { bonsai35_set_error(m, 4); continue; }
        const uint64_t norm = isqrt_u128(ssq);
        for (int64_t c = 0; c < cols; ++c) {
            if (!norm) yr[c] = 0;
            else {
                if (floor_scaled_i64_u64(
                        xr[c], m->d.frac, norm, &yr[c])) continue;
                const __int128 num = (__int128) xr[c] *
                    ((__int128) 1 << m->d.frac);
                const __int128 val = floor_div_i128_u64(num, norm);
                if (!i128_to_i64(val, &yr[c])) { bonsai35_set_error(m, 4); break; }
            }
        }
    }
}

static void bonsai35_trace_internal_copy(bonsai35_model *m, int64_t li,
                                         int slot, const int64_t *source,
                                         int64_t count);

static void bonsai35_recurrent_layer(bonsai35_model *m,
                                     const bonsai35_layer_desc *l, int64_t li) {
    const int64_t frac = m->d.frac;
    const int64_t dim = m->d.ssm_state_size;
    const int64_t groups = m->d.ssm_group_count;
    const int64_t heads = m->d.ssm_value_heads;
    const int64_t inner = m->d.ssm_inner_size;
    const int64_t key_dim = groups * dim;
    const int64_t conv_dim = 2 * key_dim + inner;
    const int64_t conv_k = m->d.ssm_conv_kernel;
    const bonsai35_q1_desc *wg[4] = {&l->wqkv, &l->wz, &l->walpha, &l->wbeta};
    int64_t *og[4] = {m->qkv, m->z, m->alpha, m->beta};
    bonsai35_q1_group(m, m->n1, wg, og, 4);
    const int capture = m->trace_layer == li && m->pos == m->trace_target_pos;
    if (capture) {
        bonsai35_trace_internal_copy(m, li, 0, m->qkv, conv_dim);
        bonsai35_trace_internal_copy(m, li, 1, m->z, inner);
        bonsai35_trace_internal_copy(m, li, 2, m->alpha, heads);
        bonsai35_trace_internal_copy(m, li, 3, m->beta, heads);
    }

    int64_t *hist = m->conv[li];
    #pragma omp for schedule(static)
    for (int64_t c = 0; c < conv_dim; ++c) {
        uint64_t acc = 0;
        for (int64_t j = 0; j < conv_k - 1; ++j) {
            acc += (uint64_t) hist[(size_t) j * (size_t) conv_dim + (size_t) c] *
                   (uint64_t) l->conv_weight[(size_t) c * (size_t) conv_k + (size_t) j];
        }
        acc += (uint64_t) m->qkv[c] *
               (uint64_t) l->conv_weight[(size_t) c * (size_t) conv_k + (size_t) (conv_k - 1)];
        const int64_t shifted = arshift_i64(u64_to_i64(acc), frac);
        m->conv_out[c] = bonsai35_silu_one(shifted, frac);
        for (int64_t j = 0; j < conv_k - 2; ++j) {
            hist[(size_t) j * (size_t) conv_dim + (size_t) c] =
                hist[(size_t) (j + 1) * (size_t) conv_dim + (size_t) c];
        }
        hist[(size_t) (conv_k - 2) * (size_t) conv_dim + (size_t) c] = m->qkv[c];
    }
    if (capture)
        bonsai35_trace_internal_copy(m, li, 4, m->conv_out, conv_dim);
    #pragma omp single
    {
        // Fault injection deliberately fires after conv-cache mutation.  The
        // public Python executor must discard/replay this poisoned handle.
        if (m->debug_error_mode == 1) {
            bonsai35_set_error(m, 6);
            m->debug_error_mode = 0;
        }
    }

    bonsai35_l2(m, m->conv_out, groups, dim, m->qmap);
    bonsai35_l2(m, m->conv_out + key_dim, groups, dim, m->kmap);
    #pragma omp for schedule(static)
    for (int64_t h = groups; h < heads; ++h) {
        const int64_t src = h % groups;
        for (int64_t j = 0; j < dim; ++j) {
            m->qmap[(size_t) h * (size_t) dim + (size_t) j] =
                m->qmap[(size_t) src * (size_t) dim + (size_t) j];
            m->kmap[(size_t) h * (size_t) dim + (size_t) j] =
                m->kmap[(size_t) src * (size_t) dim + (size_t) j];
        }
    }
    if (capture) {
        bonsai35_trace_internal_copy(m, li, 5, m->qmap, inner);
        bonsai35_trace_internal_copy(m, li, 6, m->kmap, inner);
    }

    #pragma omp for schedule(static)
    for (int64_t h = 0; h < heads; ++h) {
        m->beta[h] = bonsai35_sigmoid_one(m->beta[h], frac);
        const int64_t sp = bonsai35_softplus(
            m, u64_to_i64((uint64_t) m->alpha[h] + (uint64_t) l->dt_bias[h]));
        const int64_t gate = arshift_i64(
            u64_to_i64((uint64_t) sp * (uint64_t) l->ssm_a[h]), frac);
        if (gate > 0) bonsai35_set_error(m, 1);
        m->alpha[h] = bonsai35_exp_negative(m, gate);
    }
    if (capture) {
        bonsai35_trace_internal_copy(m, li, 7, m->alpha, heads);
        bonsai35_trace_internal_copy(m, li, 8, m->beta, heads);
    }

    int64_t *state = m->state[li];
    const int64_t outer_shift = 2 * frac - m->d.ssm_state_frac;
    const int64_t *v = m->conv_out + 2 * key_dim;
    #pragma omp for schedule(static)
    for (int64_t h = 0; h < heads; ++h) {
        int64_t *sh = state + (size_t) h * (size_t) dim * (size_t) dim;
        const int64_t *qh = m->qmap + (size_t) h * (size_t) dim;
        const int64_t *kh = m->kmap + (size_t) h * (size_t) dim;
        const int64_t *vh = v + (size_t) h * (size_t) dim;
        int64_t *oh = m->gdn + (size_t) h * (size_t) dim;
        const int64_t decay = m->alpha[h], beta = m->beta[h];
        for (int64_t i = 0; i < dim; ++i) {
            int64_t *row = sh + (size_t) i * (size_t) dim;
            for (int64_t j = 0; j < dim; ++j) {
                row[j] = arshift_i64(
                    u64_to_i64((uint64_t) row[j] * (uint64_t) decay), frac);
            }
        }
        for (int64_t j = 0; j < dim; ++j) oh[j] = 0;
        for (int64_t i = 0; i < dim; ++i) {
            const int64_t ki = kh[i];
            const int64_t *row = sh + (size_t) i * (size_t) dim;
            for (int64_t j = 0; j < dim; ++j) {
                oh[j] = u64_to_i64((uint64_t) oh[j] +
                                    (uint64_t) row[j] * (uint64_t) ki);
            }
        }
        for (int64_t j = 0; j < dim; ++j) {
            const int64_t pred = arshift_i64(oh[j], m->d.ssm_state_frac);
            const int64_t delta = arshift_i64(u64_to_i64(
                ((uint64_t) vh[j] - (uint64_t) pred) * (uint64_t) beta), frac);
            if (capture) {
                m->trace_internal[9][(size_t) h * (size_t) dim + (size_t) j] = pred;
                m->trace_internal[10][(size_t) h * (size_t) dim + (size_t) j] = delta;
            }
            oh[j] = delta;
        }
        for (int64_t i = 0; i < dim; ++i) {
            int64_t *row = sh + (size_t) i * (size_t) dim;
            for (int64_t j = 0; j < dim; ++j) {
                const int64_t update = arshift_i64(u64_to_i64(
                    (uint64_t) kh[i] * (uint64_t) oh[j]), outer_shift);
                row[j] = u64_to_i64((uint64_t) row[j] + (uint64_t) update);
            }
        }
        if (capture) {
            int64_t *trace_state = m->trace_internal[11] +
                (size_t) h * (size_t) dim * (size_t) dim;
            for (int64_t i = 0; i < dim; ++i) {
                const int64_t *row = sh + (size_t) i * (size_t) dim;
                for (int64_t j = 0; j < dim; ++j)
                    trace_state[(size_t) i * (size_t) dim + (size_t) j] = row[j];
            }
        }
        for (int64_t j = 0; j < dim; ++j) oh[j] = 0;
        for (int64_t i = 0; i < dim; ++i) {
            const int64_t qi = qh[i];
            const int64_t *row = sh + (size_t) i * (size_t) dim;
            for (int64_t j = 0; j < dim; ++j) {
                oh[j] = u64_to_i64((uint64_t) oh[j] +
                                    (uint64_t) row[j] * (uint64_t) qi);
            }
        }
        for (int64_t j = 0; j < dim; ++j) {
            const int64_t score = arshift_i64(oh[j], frac);
            oh[j] = arshift_i64(u64_to_i64(
                (uint64_t) score * (uint64_t) m->d.gdn_scale), frac);
        }
    }

    bonsai35_rms(m, m->gdn, heads, dim, dim, l->ssm_norm_gain,
                 m->d.ssm_rms_eps, m->qmap);
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < inner; ++i) {
        const int64_t zg = bonsai35_silu_one(m->z[i], frac);
        m->qmap[i] = arshift_i64(u64_to_i64(
            (uint64_t) m->qmap[i] * (uint64_t) zg), frac);
    }
    if (capture)
        bonsai35_trace_internal_copy(m, li, 12, m->qmap, inner);
    const bonsai35_q1_desc *wo[1] = {&l->wout};
    int64_t *bo[1] = {m->branch};
    bonsai35_q1_group(m, m->qmap, wo, bo, 1);
}

static void bonsai35_attention_layer(bonsai35_model *m,
                                     const bonsai35_layer_desc *l, int64_t li) {
    const int64_t frac = m->d.frac, H = m->d.n_heads, Hkv = m->d.n_heads_kv;
    const int64_t hd = m->d.head_dim, half = m->d.rope_rot_dim / 2;
    const int64_t qg_width = 2 * H * hd, kv_width = Hkv * hd;
    const bonsai35_q1_desc *wg[3] = {&l->wqg, &l->wk, &l->wv};
    int64_t *og[3] = {m->qkv, m->kproj, m->vproj};
    bonsai35_q1_group(m, m->n1, wg, og, 3);
    const int capture = m->trace_layer == li && m->pos == m->trace_target_pos;
    if (capture) {
        bonsai35_trace_internal_copy(m, li, 0, m->qkv, qg_width);
        bonsai35_trace_internal_copy(m, li, 1, m->kproj, kv_width);
        bonsai35_trace_internal_copy(m, li, 2, m->vproj, kv_width);
    }

    bonsai35_rms(m, m->qkv, H, hd, 2 * hd, l->q_norm_gain,
                 m->d.rms_eps, m->qmap);
    bonsai35_rms(m, m->kproj, Hkv, hd, hd, l->k_norm_gain,
                 m->d.rms_eps, m->kmap);
    const int64_t *cos = m->d.cos + (size_t) m->pos * (size_t) half;
    const int64_t *sin = m->d.sin + (size_t) m->pos * (size_t) half;
    #pragma omp for schedule(static)
    for (int64_t idx = 0; idx < (H + Hkv) * half; ++idx) {
        const int64_t h = idx / half, j = idx % half;
        int64_t *row = h < H ? m->qmap + (size_t) h * (size_t) hd
                             : m->kmap + (size_t) (h - H) * (size_t) hd;
        const int64_t x0 = row[j], x1 = row[half + j];
        row[j] = arshift_i64(u64_to_i64(
            (uint64_t) x0 * (uint64_t) cos[j] -
            (uint64_t) x1 * (uint64_t) sin[j]), frac);
        row[half + j] = arshift_i64(u64_to_i64(
            (uint64_t) x0 * (uint64_t) sin[j] +
            (uint64_t) x1 * (uint64_t) cos[j]), frac);
    }
    if (capture) {
        bonsai35_trace_internal_copy(m, li, 3, m->qmap, H * hd);
        bonsai35_trace_internal_copy(m, li, 4, m->kmap, kv_width);
    }

    int64_t *kc = m->kcache[li], *vc = m->vcache[li];
    #pragma omp for schedule(static)
    for (int64_t idx = 0; idx < Hkv * hd; ++idx) {
        const int64_t h = idx / hd, j = idx % hd;
        const size_t dst = ((size_t) h * (size_t) m->d.context_len +
                            (size_t) m->pos) * (size_t) hd + (size_t) j;
        kc[dst] = m->kmap[idx];
        vc[dst] = m->vproj[idx];
    }
    #pragma omp single
    {
        // Same contract as the recurrent injection, but after a KV append.
        if (m->debug_error_mode == 2) {
            bonsai35_set_error(m, 6);
            m->debug_error_mode = 0;
        }
    }

    const int64_t length = m->pos + 1, rep = H / Hkv;
    const int64_t log2e = bonsai_scaled_log2e(frac);
    const int64_t da = ((frac + 2) << (2 * frac)) / log2e;
    const int64_t db = ((int64_t) 1 << 62) / log2e;
    const int64_t dclip = da < db ? da : db;
    const unsigned __int128 i64max = (unsigned __int128) INT64_MAX;
    #pragma omp for schedule(static)
    for (int64_t h = 0; h < H; ++h) {
        const int64_t kv = h / rep;
        const int64_t *qh = m->qmap + (size_t) h * (size_t) hd;
        const int64_t *kh = kc + (size_t) kv * (size_t) m->d.context_len * (size_t) hd;
        const int64_t *vh = vc + (size_t) kv * (size_t) m->d.context_len * (size_t) hd;
        int64_t *sc = m->scores + (size_t) h * (size_t) m->d.context_len;
        const uint64_t maxq = bonsai_maxabs_u64(qh, (size_t) hd);
        const uint64_t maxk = bonsai_maxabs_u64(kh, (size_t) length * (size_t) hd);
        if ((unsigned __int128) maxq * maxk > i64max / (unsigned __int128) hd) {
            bonsai35_set_error(m, 2); continue;
        }
        int64_t mx = INT64_MIN;
        for (int64_t t = 0; t < length; ++t) {
            const int64_t *kt = kh + (size_t) t * (size_t) hd;
            int64_t dot = 0;
            for (int64_t j = 0; j < hd; ++j) dot += qh[j] * kt[j];
            int64_t value = arshift_i64(dot, frac);
            value = arshift_i64(value * m->d.attention_scale, frac);
            sc[t] = value; if (value > mx) mx = value;
        }
        if (capture) {
            int64_t *trace_scores = m->trace_internal[5] +
                (size_t) h * (size_t) length;
            for (int64_t t = 0; t < length; ++t) trace_scores[t] = sc[t];
        }
        int64_t Z = 0;
        for (int64_t t = 0; t < length; ++t) {
            int64_t delta = mx - sc[t]; if (delta > dclip) delta = dclip;
            sc[t] = bonsai_exp2_neg_fixed((delta * log2e) >> frac, frac);
            Z += sc[t];
        }
        for (int64_t t = 0; t < length; ++t) sc[t] = Z
            ? (int64_t) (((__int128) sc[t] * ((__int128) 1 << frac)) / Z)
            : 0;
        if (capture) {
            int64_t *trace_probs = m->trace_internal[6] +
                (size_t) h * (size_t) length;
            for (int64_t t = 0; t < length; ++t) trace_probs[t] = sc[t];
        }
        const uint64_t maxp = bonsai_maxabs_u64(sc, (size_t) length);
        const uint64_t maxv = bonsai_maxabs_u64(vh, (size_t) length * (size_t) hd);
        if ((unsigned __int128) maxp * maxv > i64max / (unsigned __int128) length) {
            bonsai35_set_error(m, 2); continue;
        }
        int64_t *oh = m->gdn + (size_t) h * (size_t) hd;
        for (int64_t j = 0; j < hd; ++j) {
            int64_t acc = 0;
            for (int64_t t = 0; t < length; ++t) {
                acc += sc[t] * vh[(size_t) t * (size_t) hd + (size_t) j];
            }
            const int64_t att = arshift_i64(acc, frac);
            const int64_t gate = m->qkv[(size_t) h * 2u * (size_t) hd +
                                        (size_t) hd + (size_t) j];
            oh[j] = arshift_i64(u64_to_i64(
                (uint64_t) att * (uint64_t) bonsai35_sigmoid_one(gate, frac)), frac);
        }
    }
    if (capture)
        bonsai35_trace_internal_copy(m, li, 7, m->gdn, H * hd);
    const bonsai35_q1_desc *wo[1] = {&l->wo};
    int64_t *bo[1] = {m->branch};
    bonsai35_q1_group(m, m->gdn, wo, bo, 1);
}

static void bonsai35_recurrent_tile(bonsai35_model *m,
                                    const bonsai35_layer_desc *l, int64_t li,
                                    int64_t absolute_start, int64_t rows) {
    const int64_t frac = m->d.frac;
    const int64_t dim = m->d.ssm_state_size;
    const int64_t groups = m->d.ssm_group_count;
    const int64_t heads = m->d.ssm_value_heads;
    const int64_t inner = m->d.ssm_inner_size;
    const int64_t key_dim = groups * dim;
    const int64_t conv_dim = 2 * key_dim + inner;
    const int64_t conv_k = m->d.ssm_conv_kernel;
    const bonsai35_q1_desc *wg[4] = {&l->wqkv, &l->wz, &l->walpha, &l->wbeta};
    int64_t *og[4] = {m->tile_qkv, m->tile_z, m->tile_alpha, m->tile_beta};
    const int64_t os[4] = {conv_dim, inner, heads, heads};
    bonsai35_q1_group_rows(
        m, m->tile_n1, rows, m->d.d_model, wg, og, os, 4);

    const int64_t trace_r = m->trace_layer == li
        ? m->trace_target_pos - absolute_start : -1;
    if (trace_r >= 0 && trace_r < rows) {
        bonsai35_trace_internal_copy(
            m, li, 0, m->tile_qkv + (size_t) trace_r * (size_t) conv_dim,
            conv_dim);
        bonsai35_trace_internal_copy(
            m, li, 1, m->tile_z + (size_t) trace_r * (size_t) inner, inner);
        bonsai35_trace_internal_copy(
            m, li, 2, m->tile_alpha + (size_t) trace_r * (size_t) heads, heads);
        bonsai35_trace_internal_copy(
            m, li, 3, m->tile_beta + (size_t) trace_r * (size_t) heads, heads);
    }

    int64_t *hist = m->conv[li];
    int64_t *state = m->state[li];
    const int64_t outer_shift = 2 * frac - m->d.ssm_state_frac;
    for (int64_t r = 0; r < rows; ++r) {
        int64_t *qkv = m->tile_qkv + (size_t) r * (size_t) conv_dim;
        int64_t *z = m->tile_z + (size_t) r * (size_t) inner;
        int64_t *alpha = m->tile_alpha + (size_t) r * (size_t) heads;
        int64_t *beta = m->tile_beta + (size_t) r * (size_t) heads;
        int64_t *conv_out = m->tile_conv_out + (size_t) r * (size_t) conv_dim;
        int64_t *qmap = m->tile_qmap + (size_t) r * (size_t) inner;
        int64_t *kmap = m->tile_kmap + (size_t) r * (size_t) inner;
        int64_t *gdn = m->tile_gdn + (size_t) r * (size_t) inner;
        const int capture = r == trace_r;
        #pragma omp for schedule(static)
        for (int64_t c = 0; c < conv_dim; ++c) {
            uint64_t acc = 0;
            for (int64_t j = 0; j < conv_k - 1; ++j) {
                acc += (uint64_t) hist[(size_t) j * (size_t) conv_dim + (size_t) c] *
                       (uint64_t) l->conv_weight[(size_t) c * (size_t) conv_k + (size_t) j];
            }
            acc += (uint64_t) qkv[c] *
                   (uint64_t) l->conv_weight[(size_t) c * (size_t) conv_k +
                                              (size_t) (conv_k - 1)];
            conv_out[c] = bonsai35_silu_one(
                arshift_i64(u64_to_i64(acc), frac), frac);
            for (int64_t j = 0; j < conv_k - 2; ++j) {
                hist[(size_t) j * (size_t) conv_dim + (size_t) c] =
                    hist[(size_t) (j + 1) * (size_t) conv_dim + (size_t) c];
            }
            hist[(size_t) (conv_k - 2) * (size_t) conv_dim + (size_t) c] = qkv[c];
        }
        if (capture)
            bonsai35_trace_internal_copy(m, li, 4, conv_out, conv_dim);
        #pragma omp single
        {
            if (m->debug_error_mode == 1) {
                bonsai35_set_error(m, 6);
                m->debug_error_mode = 0;
            }
        }
        bonsai35_l2(m, conv_out, groups, dim, qmap);
        bonsai35_l2(m, conv_out + key_dim, groups, dim, kmap);
        #pragma omp for schedule(static)
        for (int64_t h = groups; h < heads; ++h) {
            const int64_t src = h % groups;
            for (int64_t j = 0; j < dim; ++j) {
                qmap[(size_t) h * (size_t) dim + (size_t) j] =
                    qmap[(size_t) src * (size_t) dim + (size_t) j];
                kmap[(size_t) h * (size_t) dim + (size_t) j] =
                    kmap[(size_t) src * (size_t) dim + (size_t) j];
            }
        }
        if (capture) {
            bonsai35_trace_internal_copy(m, li, 5, qmap, inner);
            bonsai35_trace_internal_copy(m, li, 6, kmap, inner);
        }
        #pragma omp for schedule(static)
        for (int64_t h = 0; h < heads; ++h) {
            beta[h] = bonsai35_sigmoid_one(beta[h], frac);
            const int64_t sp = bonsai35_softplus(
                m, u64_to_i64((uint64_t) alpha[h] + (uint64_t) l->dt_bias[h]));
            const int64_t gate = arshift_i64(
                u64_to_i64((uint64_t) sp * (uint64_t) l->ssm_a[h]), frac);
            if (gate > 0) bonsai35_set_error(m, 1);
            alpha[h] = bonsai35_exp_negative(m, gate);
        }
        if (capture) {
            bonsai35_trace_internal_copy(m, li, 7, alpha, heads);
            bonsai35_trace_internal_copy(m, li, 8, beta, heads);
        }
        const int64_t *v = conv_out + 2 * key_dim;
        #pragma omp for schedule(static)
        for (int64_t h = 0; h < heads; ++h) {
            int64_t *sh = state + (size_t) h * (size_t) dim * (size_t) dim;
            const int64_t *qh = qmap + (size_t) h * (size_t) dim;
            const int64_t *kh = kmap + (size_t) h * (size_t) dim;
            const int64_t *vh = v + (size_t) h * (size_t) dim;
            int64_t *oh = gdn + (size_t) h * (size_t) dim;
            const int64_t decay = alpha[h], b = beta[h];
            for (int64_t i = 0; i < dim; ++i) {
                int64_t *row = sh + (size_t) i * (size_t) dim;
                for (int64_t j = 0; j < dim; ++j) {
                    row[j] = arshift_i64(
                        u64_to_i64((uint64_t) row[j] * (uint64_t) decay), frac);
                }
            }
            for (int64_t j = 0; j < dim; ++j) oh[j] = 0;
            for (int64_t i = 0; i < dim; ++i) {
                const int64_t ki = kh[i];
                const int64_t *row = sh + (size_t) i * (size_t) dim;
                for (int64_t j = 0; j < dim; ++j) {
                    oh[j] = u64_to_i64(
                        (uint64_t) oh[j] + (uint64_t) row[j] * (uint64_t) ki);
                }
            }
            for (int64_t j = 0; j < dim; ++j) {
                const int64_t pred = arshift_i64(oh[j], m->d.ssm_state_frac);
                const int64_t delta = arshift_i64(u64_to_i64(
                    ((uint64_t) vh[j] - (uint64_t) pred) * (uint64_t) b), frac);
                if (capture) {
                    m->trace_internal[9][(size_t) h * (size_t) dim +
                                          (size_t) j] = pred;
                    m->trace_internal[10][(size_t) h * (size_t) dim +
                                           (size_t) j] = delta;
                }
                oh[j] = delta;
            }
            for (int64_t i = 0; i < dim; ++i) {
                int64_t *row = sh + (size_t) i * (size_t) dim;
                for (int64_t j = 0; j < dim; ++j) {
                    const int64_t update = arshift_i64(u64_to_i64(
                        (uint64_t) kh[i] * (uint64_t) oh[j]), outer_shift);
                    row[j] = u64_to_i64((uint64_t) row[j] + (uint64_t) update);
                }
            }
            if (capture) {
                int64_t *trace_state = m->trace_internal[11] +
                    (size_t) h * (size_t) dim * (size_t) dim;
                for (int64_t i = 0; i < dim; ++i) {
                    const int64_t *row = sh + (size_t) i * (size_t) dim;
                    for (int64_t j = 0; j < dim; ++j)
                        trace_state[(size_t) i * (size_t) dim + (size_t) j] = row[j];
                }
            }
            for (int64_t j = 0; j < dim; ++j) oh[j] = 0;
            for (int64_t i = 0; i < dim; ++i) {
                const int64_t qi = qh[i];
                const int64_t *row = sh + (size_t) i * (size_t) dim;
                for (int64_t j = 0; j < dim; ++j) {
                    oh[j] = u64_to_i64(
                        (uint64_t) oh[j] + (uint64_t) row[j] * (uint64_t) qi);
                }
            }
            for (int64_t j = 0; j < dim; ++j) {
                const int64_t score = arshift_i64(oh[j], frac);
                oh[j] = arshift_i64(u64_to_i64(
                    (uint64_t) score * (uint64_t) m->d.gdn_scale), frac);
            }
        }
        bonsai35_rms(m, gdn, heads, dim, dim, l->ssm_norm_gain,
                     m->d.ssm_rms_eps, qmap);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < inner; ++i) {
            qmap[i] = arshift_i64(u64_to_i64(
                (uint64_t) qmap[i] *
                (uint64_t) bonsai35_silu_one(z[i], frac)), frac);
        }
        if (capture)
            bonsai35_trace_internal_copy(m, li, 12, qmap, inner);
    }
    const bonsai35_q1_desc *wo[1] = {&l->wout};
    int64_t *bo[1] = {m->tile_branch};
    const int64_t bs[1] = {m->d.d_model};
    bonsai35_q1_group_rows(
        m, m->tile_qmap, rows, inner, wo, bo, bs, 1);
}

static void bonsai35_attention_tile(bonsai35_model *m,
                                    const bonsai35_layer_desc *l, int64_t li,
                                    int64_t absolute_start, int64_t rows) {
    const int64_t frac = m->d.frac, H = m->d.n_heads, Hkv = m->d.n_heads_kv;
    const int64_t hd = m->d.head_dim, half = m->d.rope_rot_dim / 2;
    const int64_t qg_width = 2 * H * hd;
    const int64_t kv_width = Hkv * hd;
    const int64_t inner = H * hd;
    const bonsai35_q1_desc *wg[3] = {&l->wqg, &l->wk, &l->wv};
    int64_t *og[3] = {m->tile_qkv, m->tile_kproj, m->tile_vproj};
    const int64_t os[3] = {qg_width, kv_width, kv_width};
    bonsai35_q1_group_rows(
        m, m->tile_n1, rows, m->d.d_model, wg, og, os, 3);
    const int64_t trace_r = m->trace_layer == li
        ? m->trace_target_pos - absolute_start : -1;
    if (trace_r >= 0 && trace_r < rows) {
        bonsai35_trace_internal_copy(
            m, li, 0, m->tile_qkv + (size_t) trace_r * (size_t) qg_width,
            qg_width);
        bonsai35_trace_internal_copy(
            m, li, 1, m->tile_kproj + (size_t) trace_r * (size_t) kv_width,
            kv_width);
        bonsai35_trace_internal_copy(
            m, li, 2, m->tile_vproj + (size_t) trace_r * (size_t) kv_width,
            kv_width);
    }
    bonsai35_rms(m, m->tile_qkv, rows * H, hd, 2 * hd,
                 l->q_norm_gain, m->d.rms_eps, m->tile_qmap);
    bonsai35_rms(m, m->tile_kproj, rows * Hkv, hd, hd,
                 l->k_norm_gain, m->d.rms_eps, m->tile_kmap);

    const int64_t rope_work = rows * (H + Hkv) * half;
    #pragma omp for schedule(static)
    for (int64_t idx = 0; idx < rope_work; ++idx) {
        const int64_t per_token = (H + Hkv) * half;
        const int64_t r = idx / per_token;
        const int64_t rem = idx % per_token;
        const int64_t h = rem / half, j = rem % half;
        int64_t *row = h < H
            ? m->tile_qmap + ((size_t) r * (size_t) H + (size_t) h) * (size_t) hd
            : m->tile_kmap + ((size_t) r * (size_t) Hkv + (size_t) (h - H)) *
                               (size_t) hd;
        const int64_t *cos = m->d.cos +
            (size_t) (absolute_start + r) * (size_t) half;
        const int64_t *sin = m->d.sin +
            (size_t) (absolute_start + r) * (size_t) half;
        const int64_t x0 = row[j], x1 = row[half + j];
        row[j] = arshift_i64(u64_to_i64(
            (uint64_t) x0 * (uint64_t) cos[j] -
            (uint64_t) x1 * (uint64_t) sin[j]), frac);
        row[half + j] = arshift_i64(u64_to_i64(
            (uint64_t) x0 * (uint64_t) sin[j] +
            (uint64_t) x1 * (uint64_t) cos[j]), frac);
    }
    if (trace_r >= 0 && trace_r < rows) {
        bonsai35_trace_internal_copy(
            m, li, 3, m->tile_qmap + (size_t) trace_r * (size_t) inner,
            inner);
        bonsai35_trace_internal_copy(
            m, li, 4, m->tile_kmap + (size_t) trace_r * (size_t) kv_width,
            kv_width);
    }

    int64_t *kc = m->kcache[li], *vc = m->vcache[li];
    #pragma omp for schedule(static)
    for (int64_t idx = 0; idx < rows * Hkv * hd; ++idx) {
        const int64_t r = idx / (Hkv * hd);
        const int64_t rem = idx % (Hkv * hd);
        const int64_t h = rem / hd, j = rem % hd;
        const size_t dst = ((size_t) h * (size_t) m->d.context_len +
                            (size_t) (absolute_start + r)) * (size_t) hd +
                           (size_t) j;
        kc[dst] = m->tile_kmap[idx];
        vc[dst] = m->tile_vproj[idx];
    }
    #pragma omp single
    {
        if (m->debug_error_mode == 2) {
            bonsai35_set_error(m, 6);
            m->debug_error_mode = 0;
        }
    }

    const int64_t rep = H / Hkv;
    const int64_t log2e = bonsai_scaled_log2e(frac);
    const int64_t da = ((frac + 2) << (2 * frac)) / log2e;
    const int64_t db = ((int64_t) 1 << 62) / log2e;
    const int64_t dclip = da < db ? da : db;
    const unsigned __int128 i64max = (unsigned __int128) INT64_MAX;
    #pragma omp for schedule(static)
    for (int64_t idx = 0; idx < rows * H; ++idx) {
        const int64_t r = idx / H, h = idx % H;
        const int64_t position = absolute_start + r;
        const int64_t length = position + 1;
        const int64_t kv = h / rep;
        const int64_t *qh = m->tile_qmap +
            ((size_t) r * (size_t) H + (size_t) h) * (size_t) hd;
        const int64_t *kh = kc +
            (size_t) kv * (size_t) m->d.context_len * (size_t) hd;
        const int64_t *vh = vc +
            (size_t) kv * (size_t) m->d.context_len * (size_t) hd;
        int64_t *sc = m->tile_scores +
            ((size_t) r * (size_t) H + (size_t) h) * (size_t) m->d.context_len;
        const uint64_t maxq = bonsai_maxabs_u64(qh, (size_t) hd);
        const uint64_t maxk = bonsai_maxabs_u64(kh, (size_t) length * (size_t) hd);
        if ((unsigned __int128) maxq * maxk > i64max / (unsigned __int128) hd) {
            bonsai35_set_error(m, 2);
            continue;
        }
        int64_t mx = INT64_MIN;
        for (int64_t t = 0; t < length; ++t) {
            const int64_t *kt = kh + (size_t) t * (size_t) hd;
            int64_t dot = 0;
            for (int64_t j = 0; j < hd; ++j) dot += qh[j] * kt[j];
            int64_t value = arshift_i64(dot, frac);
            value = arshift_i64(value * m->d.attention_scale, frac);
            sc[t] = value;
            if (value > mx) mx = value;
        }
        if (r == trace_r) {
            int64_t *trace_scores = m->trace_internal[5] +
                (size_t) h * (size_t) length;
            for (int64_t t = 0; t < length; ++t) trace_scores[t] = sc[t];
        }
        int64_t Z = 0;
        for (int64_t t = 0; t < length; ++t) {
            int64_t delta = mx - sc[t];
            if (delta > dclip) delta = dclip;
            sc[t] = bonsai_exp2_neg_fixed((delta * log2e) >> frac, frac);
            Z += sc[t];
        }
        for (int64_t t = 0; t < length; ++t)
            sc[t] = Z
                ? (int64_t) (((__int128) sc[t] * ((__int128) 1 << frac)) / Z)
                : 0;
        if (r == trace_r) {
            int64_t *trace_probs = m->trace_internal[6] +
                (size_t) h * (size_t) length;
            for (int64_t t = 0; t < length; ++t) trace_probs[t] = sc[t];
        }
        const uint64_t maxp = bonsai_maxabs_u64(sc, (size_t) length);
        const uint64_t maxv = bonsai_maxabs_u64(vh, (size_t) length * (size_t) hd);
        if ((unsigned __int128) maxp * maxv >
            i64max / (unsigned __int128) length) {
            bonsai35_set_error(m, 2);
            continue;
        }
        int64_t *oh = m->tile_gdn +
            ((size_t) r * (size_t) H + (size_t) h) * (size_t) hd;
        for (int64_t j = 0; j < hd; ++j) {
            int64_t acc = 0;
            for (int64_t t = 0; t < length; ++t) {
                acc += sc[t] * vh[(size_t) t * (size_t) hd + (size_t) j];
            }
            const int64_t att = arshift_i64(acc, frac);
            const int64_t gate = m->tile_qkv[
                (size_t) r * (size_t) qg_width +
                (size_t) h * 2u * (size_t) hd + (size_t) hd + (size_t) j];
            oh[j] = arshift_i64(u64_to_i64(
                (uint64_t) att * (uint64_t) bonsai35_sigmoid_one(gate, frac)), frac);
        }
    }
    if (trace_r >= 0 && trace_r < rows)
        bonsai35_trace_internal_copy(
            m, li, 7, m->tile_gdn + (size_t) trace_r * (size_t) inner,
            inner);
    const bonsai35_q1_desc *wo[1] = {&l->wo};
    int64_t *bo[1] = {m->tile_branch};
    const int64_t bs[1] = {m->d.d_model};
    bonsai35_q1_group_rows(
        m, m->tile_gdn, rows, inner, wo, bo, bs, 1);
}

static void bonsai35_ffn_rows(bonsai35_model *m,
                              const bonsai35_layer_desc *l, int64_t rows) {
    const bonsai35_q1_desc *wg[2] = {&l->w1, &l->wu};
    int64_t *og[2] = {m->tile_ff_gate, m->tile_ff_up};
    const int64_t os[2] = {m->d.d_ff, m->d.d_ff};
    bonsai35_q1_group_rows(
        m, m->tile_n2, rows, m->d.d_model, wg, og, os, 2);
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < rows * m->d.d_ff; ++i) {
        m->tile_ff_hidden[i] = arshift_i64(u64_to_i64(
            (uint64_t) bonsai35_silu_one(m->tile_ff_gate[i], m->d.frac) *
            (uint64_t) m->tile_ff_up[i]), m->d.frac);
    }
    const bonsai35_q1_desc *wd[1] = {&l->w2};
    int64_t *bo[1] = {m->tile_branch};
    const int64_t bs[1] = {m->d.d_model};
    bonsai35_q1_group_rows(
        m, m->tile_ff_hidden, rows, m->d.d_ff, wd, bo, bs, 1);
}

static void bonsai35_ffn(bonsai35_model *m, const bonsai35_layer_desc *l) {
    const bonsai35_q1_desc *wg[2] = {&l->w1, &l->wu};
    int64_t *og[2] = {m->ff_gate, m->ff_up};
    bonsai35_q1_group(m, m->n2, wg, og, 2);
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < m->d.d_ff; ++i) {
        m->ff_hidden[i] = arshift_i64(u64_to_i64(
            (uint64_t) bonsai35_silu_one(m->ff_gate[i], m->d.frac) *
            (uint64_t) m->ff_up[i]), m->d.frac);
    }
    const bonsai35_q1_desc *wd[1] = {&l->w2};
    int64_t *bo[1] = {m->branch};
    bonsai35_q1_group(m, m->ff_hidden, wd, bo, 1);
}

static void bonsai35_trace_copy(bonsai35_model *m, int64_t li, int slot,
                                int64_t dst_row, const int64_t *source,
                                int64_t rows) {
    if (m->trace_layer != li || !m->trace[slot]) return;
    const int64_t count = rows * m->d.d_model;
    int64_t *target = m->trace[slot] +
        (size_t) dst_row * (size_t) m->d.d_model;
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < count; ++i) target[i] = source[i];
}

static void bonsai35_trace_internal_copy(bonsai35_model *m, int64_t li,
                                         int slot, const int64_t *source,
                                         int64_t count) {
    if (m->trace_layer != li || slot < 0 || slot >= 16 ||
        !m->trace_internal[slot] || count <= 0) return;
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < count; ++i)
        m->trace_internal[slot][i] = source[i];
}

static void bonsai35_token_body(bonsai35_model *m, int64_t token, int64_t *out) {
    const int64_t blocks = m->d.embed.n_blocks;
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < m->d.d_model; ++i) {
        const int64_t b = i / 128, within = i % 128;
        const uint8_t byte = m->d.embed.bits[
            ((size_t) token * (size_t) blocks + (size_t) b) * 16u + (size_t) (within / 8)];
        const int64_t sign = ((byte >> (within % 8)) & 1u) ? 1 : -1;
        m->x[i] = sign * (int64_t) m->d.embed.scale[
            (size_t) token * (size_t) blocks + (size_t) b];
    }
    for (int64_t li = 0; li < m->d.n_layers; ++li) {
        const bonsai35_layer_desc *l = &m->layers[li];
        bonsai35_rms(m, m->x, 1, m->d.d_model, m->d.d_model,
                     l->n1_gain, m->d.rms_eps, m->n1);
        bonsai35_trace_copy(m, li, 0, 0, m->n1, 1);
        if (l->kind == 0) bonsai35_recurrent_layer(m, l, li);
        else bonsai35_attention_layer(m, l, li);
        bonsai35_trace_copy(m, li, 1, 0, m->branch, 1);
        bonsai35_residual_rms(m, m->x, m->branch, 1, m->d.d_model,
                              l->n2_gain, m->d.rms_eps, m->n2);
        bonsai35_trace_copy(m, li, 2, 0, m->x, 1);
        bonsai35_trace_copy(m, li, 3, 0, m->n2, 1);
        bonsai35_ffn(m, l);
        bonsai35_trace_copy(m, li, 4, 0, m->branch, 1);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < m->d.d_model; ++i) {
            m->x[i] = u64_to_i64((uint64_t) m->x[i] + (uint64_t) m->branch[i]);
        }
        bonsai35_trace_copy(m, li, 5, 0, m->x, 1);
    }
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < m->d.d_model; ++i) out[i] = m->x[i];
    #pragma omp single
    { m->pos++; }
}

static void bonsai35_prefill_layer_major_body(bonsai35_model *m,
                                               const int64_t *tokens,
                                               int64_t count, int output_mode,
                                               int64_t *out) {
    const int64_t d_model = m->d.d_model;
    const int64_t blocks = m->d.embed.n_blocks;
    #pragma omp single
    {
        m->stats.layer_major_prefills++;
        m->stats.layer_major_rows += (uint64_t) count;
    }
    #pragma omp for schedule(static)
    for (int64_t idx = 0; idx < count * d_model; ++idx) {
        const int64_t t = idx / d_model;
        const int64_t i = idx % d_model;
        const int64_t b = i / 128, within = i % 128;
        const uint8_t byte = m->d.embed.bits[
            ((size_t) tokens[t] * (size_t) blocks + (size_t) b) * 16u +
            (size_t) (within / 8)];
        const int64_t sign = ((byte >> (within % 8)) & 1u) ? 1 : -1;
        m->prefill_x[idx] = sign * (int64_t) m->d.embed.scale[
            (size_t) tokens[t] * (size_t) blocks + (size_t) b];
    }

    const int64_t outer_tile = m->prefill_tile_rows;
    for (int64_t li = 0; li < m->d.n_layers; ++li) {
        const bonsai35_layer_desc *l = &m->layers[li];
        for (int64_t base = 0; base < count; base += outer_tile) {
            int64_t rows = count - base;
            if (rows > outer_tile) rows = outer_tile;
            int64_t *x = m->prefill_x + (size_t) base * (size_t) d_model;
            bonsai35_rms(m, x, rows, d_model, d_model,
                         l->n1_gain, m->d.rms_eps, m->tile_n1);
            bonsai35_trace_copy(m, li, 0, base, m->tile_n1, rows);
            if (l->kind == 0)
                bonsai35_recurrent_tile(m, l, li, base, rows);
            else
                bonsai35_attention_tile(m, l, li, base, rows);
            bonsai35_trace_copy(m, li, 1, base, m->tile_branch, rows);
            bonsai35_residual_rms(
                m, x, m->tile_branch, rows, d_model,
                l->n2_gain, m->d.rms_eps, m->tile_n2);
            bonsai35_trace_copy(m, li, 2, base, x, rows);
            bonsai35_trace_copy(m, li, 3, base, m->tile_n2, rows);
            bonsai35_ffn_rows(m, l, rows);
            bonsai35_trace_copy(m, li, 4, base, m->tile_branch, rows);
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < rows * d_model; ++i) {
                x[i] = u64_to_i64(
                    (uint64_t) x[i] + (uint64_t) m->tile_branch[i]);
            }
            bonsai35_trace_copy(m, li, 5, base, x, rows);
        }
    }
    if (output_mode == 0) {
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < count * d_model; ++i) out[i] = m->prefill_x[i];
    }
    const int64_t *last = m->prefill_x + (size_t) (count - 1) * (size_t) d_model;
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < d_model; ++i) m->x[i] = last[i];
    #pragma omp single
    {
        m->pos = count;
        m->trace_rows = m->trace_layer >= 0 ? count : 0;
    }
}

static int bonsai35_ensure_prefill_rows(bonsai35_model *m, int64_t rows) {
    if (rows <= 0 || rows > m->d.context_len) return 1;
    if ((size_t) rows <= m->prefill_capacity_rows) return 0;
    size_t capacity = m->prefill_capacity_rows ? m->prefill_capacity_rows : 16u;
    while (capacity < (size_t) rows) {
        if (capacity > (size_t) m->d.context_len / 2u) {
            capacity = (size_t) m->d.context_len;
            break;
        }
        capacity *= 2u;
    }
    size_t count = 0;
    if (!checked_mul_size(capacity, (size_t) m->d.d_model, &count) ||
        count > SIZE_MAX / sizeof(int64_t)) return 1;
    int64_t *grown = (int64_t *) realloc(m->prefill_x, count * sizeof(int64_t));
    if (!grown) return 2;
    m->prefill_x = grown;
    m->prefill_capacity_rows = capacity;
    return 0;
}

static int bonsai35_ensure_trace_rows(bonsai35_model *m, int64_t rows) {
    if (m->trace_layer < 0) return 0;
    if (rows <= 0 || rows > m->d.context_len) return 1;
    if ((size_t) rows > m->trace_capacity_rows) {
        size_t count = 0;
        if (!checked_mul_size((size_t) rows, (size_t) m->d.d_model, &count) ||
            count > SIZE_MAX / sizeof(int64_t)) return 1;
        int64_t *fresh[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
        for (int slot = 0; slot < 6; ++slot) {
            fresh[slot] = (int64_t *) bonsai_calloc_array(count, sizeof(int64_t));
            if (!fresh[slot]) {
                for (int i = 0; i < 6; ++i) free(fresh[i]);
                return 2;
            }
        }
        for (int slot = 0; slot < 6; ++slot) {
            free(m->trace[slot]);
            m->trace[slot] = fresh[slot];
        }
        m->trace_capacity_rows = (size_t) rows;
    }

    size_t wanted[16] = {0};
    const bonsai35_layer_desc *l = &m->layers[m->trace_layer];
    if (l->kind == 0) {
        const size_t heads = (size_t) m->d.ssm_value_heads;
        const size_t inner = (size_t) m->d.ssm_inner_size;
        wanted[0] = m->conv_dim; wanted[1] = inner;
        wanted[2] = heads; wanted[3] = heads; wanted[4] = m->conv_dim;
        wanted[5] = inner; wanted[6] = inner;
        wanted[7] = heads; wanted[8] = heads;
        wanted[9] = inner; wanted[10] = inner;
        wanted[11] = m->state_n; wanted[12] = inner;
    } else {
        wanted[0] = m->qg_width; wanted[1] = m->kv_width;
        wanted[2] = m->kv_width;
        wanted[3] = m->attention_inner; wanted[4] = m->kv_width;
        wanted[5] = m->scores_n;
        wanted[6] = wanted[5]; wanted[7] = m->attention_inner;
    }
    for (int slot = 0; slot < 16; ++slot) {
        if (wanted[slot] > m->trace_internal_capacity[slot]) {
            if (wanted[slot] > SIZE_MAX / sizeof(int64_t)) return 1;
            int64_t *grown = (int64_t *) realloc(
                m->trace_internal[slot], wanted[slot] * sizeof(int64_t));
            if (!grown) return 2;
            m->trace_internal[slot] = grown;
            m->trace_internal_capacity[slot] = wanted[slot];
        }
        m->trace_internal_count[slot] = 0;
    }
    return 0;
}

static void bonsai35_model_release(bonsai35_model *m) {
    if (!m) return;
    if (m->state) for (int64_t i = 0; i < m->d.n_layers; ++i) free(m->state[i]);
    if (m->conv) for (int64_t i = 0; i < m->d.n_layers; ++i) free(m->conv[i]);
    if (m->kcache) for (int64_t i = 0; i < m->d.n_layers; ++i) free(m->kcache[i]);
    if (m->vcache) for (int64_t i = 0; i < m->d.n_layers; ++i) free(m->vcache[i]);
    free(m->state); free(m->conv); free(m->kcache); free(m->vcache);
    free(m->layers);
    free(m->x); free(m->n1); free(m->n2); free(m->branch);
    free(m->qkv); free(m->z); free(m->alpha); free(m->beta);
    free(m->kproj); free(m->vproj); free(m->conv_out);
    free(m->qmap); free(m->kmap); free(m->gdn);
    free(m->ff_gate); free(m->ff_up); free(m->ff_hidden);
    free(m->scores); free(m->totals); free(m->lut64); free(m->lut32);
    free(m->prefill_x);
    free(m->tile_n1); free(m->tile_n2); free(m->tile_branch);
    free(m->tile_qkv); free(m->tile_z); free(m->tile_alpha); free(m->tile_beta);
    free(m->tile_kproj); free(m->tile_vproj); free(m->tile_conv_out);
    free(m->tile_qmap); free(m->tile_kmap); free(m->tile_gdn);
    free(m->tile_ff_gate); free(m->tile_ff_up); free(m->tile_ff_hidden);
    free(m->tile_scores);
    free(m->tile_range_bad); free(m->tile_totals);
    free(m->tile_lut64); free(m->tile_lut32);
    free(m->rms_partials); free(m->rms_partial_ok);
    for (int slot = 0; slot < 6; ++slot) free(m->trace[slot]);
    for (int slot = 0; slot < 16; ++slot) free(m->trace_internal[slot]);
    free(m);
}

int bonsai35_model_create(const bonsai35_model_desc *d, void **handle_out) {
    if (!d || !handle_out || !d->layers || !d->final_norm_gain || !d->cos ||
        !d->sin || !d->softplus_lut || !d->exp_lut ||
        d->n_layers != 64 || d->frac != 16 || d->d_model != 5120 ||
        d->d_ff != 17408 || d->n_heads != 24 || d->n_heads_kv != 4 ||
        d->head_dim != 256 || d->rope_rot_dim != 64 ||
        d->ssm_state_size != 128 || d->ssm_group_count != 16 ||
        d->ssm_inner_size != 6144 || d->ssm_value_heads != 48 ||
        d->ssm_conv_kernel != 4 || d->ssm_state_frac != 30 ||
        d->context_len <= 0 || d->vocab <= 0 || d->lut_step <= 0 ||
        d->softplus_count < 2 || d->exp_count < 2 ||
        (d->isa_mode != 1 && d->isa_mode != 2) ||
        (d->isa_mode == 2 && !q1_hardware_has_avx2()) ||
        !bonsai35_q1_shape(&d->embed, d->vocab, d->d_model) ||
        !bonsai35_q1_shape(&d->output, d->vocab, d->d_model)) {
        return 5; // unsupported descriptor; Python keeps the canonical fallback
    }
    size_t state_n = 0, key_dim = 0, doubled_key = 0, conv_dim = 0;
    size_t conv_n = 0, kv_per_head = 0, kv_n = 0;
    size_t qg_width = 0, kv_width = 0, attention_inner = 0, scores_n = 0;
    if (!checked_mul3_size((size_t) d->ssm_value_heads,
                           (size_t) d->ssm_state_size,
                           (size_t) d->ssm_state_size, &state_n) ||
        !checked_mul_size((size_t) d->ssm_group_count,
                          (size_t) d->ssm_state_size, &key_dim) ||
        !checked_mul_size(key_dim, 2u, &doubled_key) ||
        !checked_add_size(doubled_key, (size_t) d->ssm_inner_size, &conv_dim) ||
        !checked_mul_size((size_t) (d->ssm_conv_kernel - 1), conv_dim, &conv_n) ||
        !checked_mul_size((size_t) d->context_len,
                          (size_t) d->head_dim, &kv_per_head) ||
        !checked_mul_size((size_t) d->n_heads_kv, kv_per_head, &kv_n) ||
        !checked_mul3_size(2u, (size_t) d->n_heads,
                           (size_t) d->head_dim, &qg_width) ||
        !checked_mul_size((size_t) d->n_heads_kv,
                          (size_t) d->head_dim, &kv_width) ||
        !checked_mul_size((size_t) d->n_heads,
                          (size_t) d->head_dim, &attention_inner) ||
        !checked_mul_size((size_t) d->n_heads,
                          (size_t) d->context_len, &scores_n) ||
        state_n > SIZE_MAX / sizeof(int64_t) ||
        conv_n > SIZE_MAX / sizeof(int64_t) ||
        kv_n > SIZE_MAX / sizeof(int64_t) ||
        scores_n > SIZE_MAX / sizeof(int64_t) ||
        state_n > (size_t) INT64_MAX || conv_dim > (size_t) INT64_MAX ||
        qg_width > (size_t) INT64_MAX || kv_width > (size_t) INT64_MAX ||
        attention_inner > (size_t) INT64_MAX || scores_n > (size_t) INT64_MAX) {
        return 5;
    }
    bonsai35_model *m = (bonsai35_model *) calloc(1, sizeof(*m));
    if (!m) return 2;
    m->d = *d;
    m->state_n = state_n; m->conv_dim = conv_dim; m->conv_n = conv_n;
    m->kv_per_head = kv_per_head; m->kv_n = kv_n;
    m->qg_width = qg_width; m->kv_width = kv_width;
    m->attention_inner = attention_inner; m->scores_n = scores_n;
    const char *lut_mode = getenv("TRINOTE_BONSAI35_Q1_LUT32");
    m->use_lut32 = !(lut_mode && (!strcmp(lut_mode, "0") ||
        !strcmp(lut_mode, "false") || !strcmp(lut_mode, "no") ||
        !strcmp(lut_mode, "off")));
    m->use_avx2 = d->isa_mode == 2;
    m->stats.selected_isa = m->use_avx2 ? 2 : 1;
    m->stats.selected_lut_bits = m->use_lut32 ? 32 : 64;
    m->stats.cache_width_bits = 64;
    m->stats.prefill_tile_40 = bonsai35_prefill_tile_for_blocks(m, 40);
    m->stats.prefill_tile_48 = bonsai35_prefill_tile_for_blocks(m, 48);
    m->stats.prefill_tile_136 = bonsai35_prefill_tile_for_blocks(m, 136);
    m->prefill_tile_rows = m->stats.prefill_tile_40;
    if (m->stats.prefill_tile_48 > m->prefill_tile_rows)
        m->prefill_tile_rows = m->stats.prefill_tile_48;
    if (m->stats.prefill_tile_136 > m->prefill_tile_rows)
        m->prefill_tile_rows = m->stats.prefill_tile_136;
    m->trace_layer = -1;
    m->rms_partial_capacity = omp_get_max_threads();
    if (m->rms_partial_capacity < 1) m->rms_partial_capacity = 1;
    m->layers = (bonsai35_layer_desc *) bonsai_calloc_array(
        (size_t) d->n_layers, sizeof(*m->layers));
    if (!m->layers) { bonsai35_model_release(m); return 2; }
    for (int64_t i = 0; i < d->n_layers; ++i) {
        m->layers[i] = d->layers[i];
        const bonsai35_layer_desc *l = &m->layers[i];
        if (!l->n1_gain || !l->n2_gain ||
            !bonsai35_q1_shape(&l->w1, d->d_ff, d->d_model) ||
            !bonsai35_q1_shape(&l->wu, d->d_ff, d->d_model) ||
            !bonsai35_q1_shape(&l->w2, d->d_model, d->d_ff) ||
            (l->kind != 0 && l->kind != 1)) {
            bonsai35_model_release(m); return 5;
        }
        const int64_t key_dim = d->ssm_group_count * d->ssm_state_size;
        const int64_t conv_dim = 2 * key_dim + d->ssm_inner_size;
        if (l->kind == 0 && (
            !bonsai35_q1_shape(&l->wqkv, conv_dim, d->d_model) ||
            !bonsai35_q1_shape(&l->wz, d->ssm_inner_size, d->d_model) ||
            !bonsai35_q1_shape(&l->walpha, d->ssm_value_heads, d->d_model) ||
            !bonsai35_q1_shape(&l->wbeta, d->ssm_value_heads, d->d_model) ||
            !bonsai35_q1_shape(&l->wout, d->d_model, d->ssm_inner_size) ||
            !l->conv_weight || !l->dt_bias || !l->ssm_a || !l->ssm_norm_gain)) {
            bonsai35_model_release(m); return 5;
        }
        if (l->kind == 1 && (
            !bonsai35_q1_shape(&l->wqg, 2 * d->n_heads * d->head_dim, d->d_model) ||
            !bonsai35_q1_shape(&l->wk, d->n_heads_kv * d->head_dim, d->d_model) ||
            !bonsai35_q1_shape(&l->wv, d->n_heads_kv * d->head_dim, d->d_model) ||
            !bonsai35_q1_shape(&l->wo, d->d_model, d->n_heads * d->head_dim) ||
            !l->q_norm_gain || !l->k_norm_gain)) {
            bonsai35_model_release(m); return 5;
        }
    }
    m->d.layers = m->layers;
    const size_t nl = (size_t) d->n_layers;
    m->state = (int64_t **) bonsai_calloc_array(nl, sizeof(int64_t *));
    m->conv = (int64_t **) bonsai_calloc_array(nl, sizeof(int64_t *));
    m->kcache = (int64_t **) bonsai_calloc_array(nl, sizeof(int64_t *));
    m->vcache = (int64_t **) bonsai_calloc_array(nl, sizeof(int64_t *));
    if (!m->state || !m->conv || !m->kcache || !m->vcache) {
        bonsai35_model_release(m); return 2;
    }
    for (int64_t i = 0; i < d->n_layers; ++i) {
        if (m->layers[i].kind == 0) {
            m->state[i] = (int64_t *) bonsai_calloc_array(state_n, sizeof(int64_t));
            m->conv[i] = (int64_t *) bonsai_calloc_array(conv_n, sizeof(int64_t));
            if (!m->state[i] || !m->conv[i]) { bonsai35_model_release(m); return 2; }
        } else {
            m->kcache[i] = (int64_t *) bonsai_calloc_array(kv_n, sizeof(int64_t));
            m->vcache[i] = (int64_t *) bonsai_calloc_array(kv_n, sizeof(int64_t));
            if (!m->kcache[i] || !m->vcache[i]) { bonsai35_model_release(m); return 2; }
        }
    }
    #define B35_ALLOC(NAME, COUNT) do { \
        m->NAME = (int64_t *) bonsai_calloc_array((size_t) (COUNT), sizeof(int64_t)); \
        if (!m->NAME) { bonsai35_model_release(m); return 2; } \
    } while (0)
    B35_ALLOC(x, d->d_model); B35_ALLOC(n1, d->d_model);
    B35_ALLOC(n2, d->d_model); B35_ALLOC(branch, d->d_model);
    B35_ALLOC(qkv, qg_width); B35_ALLOC(z, d->ssm_inner_size);
    B35_ALLOC(alpha, d->ssm_value_heads); B35_ALLOC(beta, d->ssm_value_heads);
    B35_ALLOC(kproj, kv_width);
    B35_ALLOC(vproj, kv_width);
    B35_ALLOC(conv_out, conv_dim); B35_ALLOC(qmap, d->ssm_inner_size);
    B35_ALLOC(kmap, d->ssm_inner_size); B35_ALLOC(gdn, d->ssm_inner_size);
    B35_ALLOC(ff_gate, d->d_ff); B35_ALLOC(ff_up, d->d_ff);
    B35_ALLOC(ff_hidden, d->d_ff);
    B35_ALLOC(scores, scores_n);
    #undef B35_ALLOC
    #define B35_TILE_ALLOC(NAME, WIDTH) do { \
        size_t b35_count = 0; \
        if (!checked_mul_size((size_t) m->prefill_tile_rows, \
                              (size_t) (WIDTH), &b35_count)) { \
            bonsai35_model_release(m); return 2; \
        } \
        m->NAME = (int64_t *) bonsai_calloc_array(b35_count, sizeof(int64_t)); \
        if (!m->NAME) { bonsai35_model_release(m); return 2; } \
    } while (0)
    B35_TILE_ALLOC(tile_n1, d->d_model);
    B35_TILE_ALLOC(tile_n2, d->d_model);
    B35_TILE_ALLOC(tile_branch, d->d_model);
    B35_TILE_ALLOC(tile_qkv, qg_width);
    B35_TILE_ALLOC(tile_z, d->ssm_inner_size);
    B35_TILE_ALLOC(tile_alpha, d->ssm_value_heads);
    B35_TILE_ALLOC(tile_beta, d->ssm_value_heads);
    B35_TILE_ALLOC(tile_kproj, kv_width);
    B35_TILE_ALLOC(tile_vproj, kv_width);
    B35_TILE_ALLOC(tile_conv_out, conv_dim);
    B35_TILE_ALLOC(tile_qmap, d->ssm_inner_size);
    B35_TILE_ALLOC(tile_kmap, d->ssm_inner_size);
    B35_TILE_ALLOC(tile_gdn, d->ssm_inner_size);
    B35_TILE_ALLOC(tile_ff_gate, d->d_ff);
    B35_TILE_ALLOC(tile_ff_up, d->d_ff);
    B35_TILE_ALLOC(tile_ff_hidden, d->d_ff);
    B35_TILE_ALLOC(tile_scores, scores_n);
    #undef B35_TILE_ALLOC
    size_t tile_blocks = 0, blocks48 = 0, blocks136 = 0;
    if (!checked_mul_size((size_t) m->stats.prefill_tile_40, 40u, &tile_blocks) ||
        !checked_mul_size((size_t) m->stats.prefill_tile_48, 48u, &blocks48) ||
        !checked_mul_size((size_t) m->stats.prefill_tile_136, 136u, &blocks136)) {
        bonsai35_model_release(m); return 2;
    }
    if (blocks48 > tile_blocks) tile_blocks = blocks48;
    if (blocks136 > tile_blocks) tile_blocks = blocks136;
    size_t tile_lut_count = 0;
    if (!checked_mul_size(tile_blocks, 16u * 256u, &tile_lut_count)) {
        bonsai35_model_release(m); return 2;
    }
    m->tile_lut_blocks_capacity = tile_blocks;
    m->tile_range_bad = (int *) bonsai_calloc_array(
        (size_t) m->prefill_tile_rows, sizeof(int));
    m->tile_totals = (uint64_t *) bonsai_calloc_array(
        tile_blocks, sizeof(uint64_t));
    m->tile_lut64 = (uint64_t *) bonsai_calloc_array(
        tile_lut_count, sizeof(uint64_t));
    m->tile_lut32 = (int32_t *) bonsai_calloc_array(
        tile_lut_count, sizeof(int32_t));
    if (!m->tile_range_bad || !m->tile_totals ||
        !m->tile_lut64 || !m->tile_lut32) {
        bonsai35_model_release(m); return 2;
    }
    m->rms_partials = (unsigned __int128 *) bonsai_calloc_array(
        (size_t) m->rms_partial_capacity, sizeof(unsigned __int128));
    m->rms_partial_ok = (unsigned char *) bonsai_calloc_array(
        (size_t) m->rms_partial_capacity, sizeof(unsigned char));
    if (!m->rms_partials || !m->rms_partial_ok) {
        bonsai35_model_release(m); return 2;
    }
    const int64_t max_blocks = d->d_ff / 128;
    if (!checked_mul3_size((size_t) max_blocks, 16u, 256u, &m->lut_count)) {
        bonsai35_model_release(m); return 2;
    }
    m->totals = (uint64_t *) bonsai_calloc_array((size_t) max_blocks, sizeof(uint64_t));
    m->lut64 = (uint64_t *) bonsai_calloc_array(m->lut_count, sizeof(uint64_t));
    m->lut32 = (int32_t *) bonsai_calloc_array(m->lut_count, sizeof(int32_t));
    if (!m->totals || !m->lut64 || !m->lut32) {
        bonsai35_model_release(m); return 2;
    }
    *handle_out = m;
    return 0;
}

void bonsai35_model_free(void *handle) {
    bonsai35_model_release((bonsai35_model *) handle);
}

int bonsai35_model_reset(void *handle) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m) return 1;
    for (int64_t i = 0; i < m->d.n_layers; ++i) {
        if (m->layers[i].kind == 0) {
            memset(m->state[i], 0, m->state_n * sizeof(int64_t));
            memset(m->conv[i], 0, m->conv_n * sizeof(int64_t));
        }
    }
    m->pos = 0; m->error = 0; m->trace_rows = 0;
    return 0;
}

// output_mode: 0 writes every hidden row, 1 writes final-token logits, and 2
// writes the final-token greedy argmax.  Every mode owns exactly one OpenMP
// region for embedding, all 64 layers, final norm, and (when requested) the
// output projection.
static int bonsai35_model_run(bonsai35_model *m, const int64_t *tokens,
                              int64_t count, int reset_first,
                              int output_mode, int64_t *out) {
    if (reset_first) {
        int rc = bonsai35_model_reset(m);
        if (rc) return rc;
        m->stats.prefill_calls++;
    } else {
        m->stats.decode_calls++;
    }
    const int layer_major = reset_first && count > 1;
    if (layer_major) {
        const int rc = bonsai35_ensure_prefill_rows(m, count);
        if (rc) return rc;
    }
    {
        const int rc = bonsai35_ensure_trace_rows(m, count);
        if (rc) return rc;
    }
    if (m->trace_layer >= 0) {
        const int64_t trace_length = reset_first ? count : m->pos + count;
        m->trace_target_pos = trace_length - 1;
        const bonsai35_layer_desc *tl = &m->layers[m->trace_layer];
        if (tl->kind == 0) {
            const int64_t heads = m->d.ssm_value_heads;
            const int64_t inner = m->d.ssm_inner_size;
            m->trace_internal_count[0] = (int64_t) m->conv_dim;
            m->trace_internal_count[1] = inner;
            m->trace_internal_count[2] = heads;
            m->trace_internal_count[3] = heads;
            m->trace_internal_count[4] = (int64_t) m->conv_dim;
            m->trace_internal_count[5] = inner;
            m->trace_internal_count[6] = inner;
            m->trace_internal_count[7] = heads;
            m->trace_internal_count[8] = heads;
            m->trace_internal_count[9] = inner;
            m->trace_internal_count[10] = inner;
            m->trace_internal_count[11] = (int64_t) m->state_n;
            m->trace_internal_count[12] = inner;
        } else {
            size_t trace_scores = 0;
            if (!checked_mul_size((size_t) m->d.n_heads,
                                  (size_t) trace_length, &trace_scores) ||
                trace_scores > (size_t) INT64_MAX) return 1;
            m->trace_internal_count[0] = (int64_t) m->qg_width;
            m->trace_internal_count[1] = (int64_t) m->kv_width;
            m->trace_internal_count[2] = (int64_t) m->kv_width;
            m->trace_internal_count[3] = (int64_t) m->attention_inner;
            m->trace_internal_count[4] = (int64_t) m->kv_width;
            m->trace_internal_count[5] = (int64_t) trace_scores;
            m->trace_internal_count[6] = (int64_t) trace_scores;
            m->trace_internal_count[7] = (int64_t) m->attention_inner;
        }
    }
    m->trace_rows = 0;
    m->error = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            m->stats.team_entries++;
            m->stats.last_team_size = omp_get_num_threads();
        }
        if (layer_major) {
            bonsai35_prefill_layer_major_body(
                m, tokens, count, output_mode, out);
        } else {
            for (int64_t t = 0; t < count; ++t) {
                int64_t *hidden = output_mode == 0
                    ? out + (size_t) t * (size_t) m->d.d_model
                    : m->n2; // disposable copy; final norm reads resident x
                bonsai35_token_body(m, tokens[t], hidden);
            }
            #pragma omp single
            { m->trace_rows = m->trace_layer >= 0 ? count : 0; }
        }
        if (output_mode != 0) {
            bonsai35_rms(m, m->x, 1, m->d.d_model, m->d.d_model,
                         m->d.final_norm_gain, m->d.rms_eps, m->n1);
            if (output_mode == 1) {
                const bonsai35_q1_desc *weights[1] = {&m->d.output};
                int64_t *outputs[1] = {out};
                bonsai35_q1_group(m, m->n1, weights, outputs, 1);
            } else {
                bonsai35_q1_argmax(m, m->n1, &m->d.output, out);
            }
        }
    }
    return m->error;
}

int bonsai35_model_decode(void *handle, int64_t token, int64_t *hidden_out) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || !hidden_out || token < 0 || token >= m->d.vocab ||
        m->pos >= m->d.context_len) return 1;
    return bonsai35_model_run(m, &token, 1, 0, 0, hidden_out);
}

int bonsai35_model_decode_logits(void *handle, int64_t token, int64_t *logits_out) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || !logits_out || token < 0 || token >= m->d.vocab ||
        m->pos >= m->d.context_len) return 1;
    return bonsai35_model_run(m, &token, 1, 0, 1, logits_out);
}

int bonsai35_model_decode_argmax(void *handle, int64_t token, int64_t *argmax_out) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || !argmax_out || token < 0 || token >= m->d.vocab ||
        m->pos >= m->d.context_len) return 1;
    return bonsai35_model_run(m, &token, 1, 0, 2, argmax_out);
}

int bonsai35_model_prefill(void *handle, const int64_t *tokens, int64_t count,
                           int64_t *hidden_out) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || !tokens || !hidden_out || count <= 0 || count > m->d.context_len) return 1;
    for (int64_t i = 0; i < count; ++i) {
        if (tokens[i] < 0 || tokens[i] >= m->d.vocab) return 1;
    }
    return bonsai35_model_run(m, tokens, count, 1, 0, hidden_out);
}

int bonsai35_model_prefill_logits(void *handle, const int64_t *tokens,
                                  int64_t count, int64_t *logits_out) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || !tokens || !logits_out || count <= 0 ||
        count > m->d.context_len) return 1;
    for (int64_t i = 0; i < count; ++i) {
        if (tokens[i] < 0 || tokens[i] >= m->d.vocab) return 1;
    }
    return bonsai35_model_run(m, tokens, count, 1, 1, logits_out);
}

int bonsai35_model_prefill_argmax(void *handle, const int64_t *tokens,
                                  int64_t count, int64_t *argmax_out) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || !tokens || !argmax_out || count <= 0 ||
        count > m->d.context_len) return 1;
    for (int64_t i = 0; i < count; ++i) {
        if (tokens[i] < 0 || tokens[i] >= m->d.vocab) return 1;
    }
    return bonsai35_model_run(m, tokens, count, 1, 2, argmax_out);
}

int bonsai35_model_force_lut_fallback(void *handle, int enabled) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || (enabled != 0 && enabled != 1)) return 1;
    m->force_lut_fallback = enabled;
    return 0;
}

int bonsai35_model_debug_fail_after_mutation(void *handle, int mode) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || mode < 0 || mode > 2) return 1;
    m->debug_error_mode = mode;
    return 0;
}

int64_t bonsai35_model_position(void *handle) {
    bonsai35_model *m = (bonsai35_model *) handle;
    return m ? m->pos : -1;
}

static uint64_t bonsai35_fingerprint_words(uint64_t hash,
                                           const int64_t *values, size_t count) {
    // Diagnostic only (not a receipt/cryptographic commitment): stable FNV-1a
    // over exact int64 bit patterns, used to prove fallback cache parity.
    const uint64_t prime = UINT64_C(1099511628211);
    for (size_t i = 0; i < count; ++i) {
        hash ^= (uint64_t) values[i];
        hash *= prime;
    }
    hash ^= (uint64_t) count;
    return hash * prime;
}

int bonsai35_model_cache_fingerprints(void *handle, uint64_t out[4]) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || !out) return 1;
    const uint64_t offset = UINT64_C(1469598103934665603);
    uint64_t hs = offset, hc = offset, hk = offset, hv = offset;
    size_t valid = 0;
    if (!checked_mul_size((size_t) m->pos,
                          (size_t) m->d.head_dim, &valid) ||
        valid > m->kv_per_head) return 1;
    for (int64_t li = 0; li < m->d.n_layers; ++li) {
        if (m->layers[li].kind == 0) {
            hs = bonsai35_fingerprint_words(hs, m->state[li], m->state_n);
            hc = bonsai35_fingerprint_words(hc, m->conv[li], m->conv_n);
        } else {
            for (int64_t h = 0; h < m->d.n_heads_kv; ++h) {
                size_t base = 0;
                if (!checked_mul_size((size_t) h, m->kv_per_head, &base) ||
                    base > m->kv_n) return 1;
                hk = bonsai35_fingerprint_words(hk, m->kcache[li] + base, valid);
                hv = bonsai35_fingerprint_words(hv, m->vcache[li] + base, valid);
            }
        }
    }
    out[0] = hs ^ (uint64_t) m->pos;
    out[1] = hc ^ (uint64_t) m->pos;
    out[2] = hk ^ (uint64_t) m->pos;
    out[3] = hv ^ (uint64_t) m->pos;
    return 0;
}

int bonsai35_model_debug_trace_layer(void *handle, int64_t layer) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || layer < -1 || layer >= m->d.n_layers) return 1;
    m->trace_layer = layer;
    m->trace_rows = 0;
    return 0;
}

int64_t bonsai35_model_debug_trace_rows(void *handle) {
    bonsai35_model *m = (bonsai35_model *) handle;
    return m ? m->trace_rows : -1;
}

int bonsai35_model_export_trace(void *handle, int64_t kind,
                                int64_t *out, int64_t count) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || kind < 0 || kind >= 6 || m->trace_layer < 0 ||
        m->trace_rows <= 0 || !out) return 1;
    size_t expected = 0, bytes = 0;
    if (!checked_mul_size((size_t) m->trace_rows,
                          (size_t) m->d.d_model, &expected) ||
        !checked_mul_size(expected, sizeof(int64_t), &bytes)) return 1;
    if (expected > (size_t) INT64_MAX || count != (int64_t) expected ||
        !m->trace[kind]) return 1;
    memcpy(out, m->trace[kind], bytes);
    return 0;
}

int64_t bonsai35_model_debug_internal_count(void *handle, int64_t kind) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || kind < 0 || kind >= 16 || m->trace_layer < 0 ||
        m->trace_rows <= 0) return -1;
    return m->trace_internal_count[kind];
}

int bonsai35_model_export_internal(void *handle, int64_t kind,
                                   int64_t *out, int64_t count) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || kind < 0 || kind >= 16 || m->trace_layer < 0 ||
        m->trace_rows <= 0 || count <= 0 || !out ||
        count != m->trace_internal_count[kind] ||
        !m->trace_internal[kind]) return 1;
    size_t bytes = 0;
    if (!checked_mul_size((size_t) count, sizeof(int64_t), &bytes)) return 1;
    memcpy(out, m->trace_internal[kind], bytes);
    return 0;
}

// Export one logical cache tensor for producer-independent SHA-256
// commitments in Python.  The resident attention allocation is
// [Hkv, context, head_dim]; only [Hkv, position, head_dim] is semantic, so K/V
// are compacted head-by-head.  Kinds: 0=state, 1=conv, 2=K, 3=V,
// 4=last residual (layer must be -1).  This diagnostic path runs outside the
// timed graph and deliberately preserves exact int64 values rather than the
// lossy FNV cache fingerprints above.
int bonsai35_model_export_tensor(void *handle, int64_t layer, int64_t kind,
                                 int64_t *out, int64_t count) {
    bonsai35_model *m = (bonsai35_model *) handle;
    if (!m || count < 0) return 1;
    if (kind == 4) {
        if (layer != -1 || count != m->d.d_model || !out || m->pos <= 0) return 1;
        size_t bytes = 0;
        if (!checked_mul_size((size_t) count, sizeof(int64_t), &bytes)) return 1;
        memcpy(out, m->x, bytes);
        return 0;
    }
    if (layer < 0 || layer >= m->d.n_layers) return 1;

    const int recurrent = m->layers[layer].kind == 0;
    size_t expected = 0;
    const int64_t *source = NULL;
    if (kind == 0 && recurrent) {
        expected = m->state_n;
        source = m->state[layer];
    } else if (kind == 1 && recurrent) {
        expected = m->conv_n;
        source = m->conv[layer];
    } else if ((kind == 2 || kind == 3) && !recurrent) {
        size_t valid_per_head = 0;
        if (!checked_mul_size((size_t) m->pos,
                              (size_t) m->d.head_dim, &valid_per_head) ||
            valid_per_head > m->kv_per_head ||
            !checked_mul_size((size_t) m->d.n_heads_kv,
                              valid_per_head, &expected)) return 1;
        source = kind == 2 ? m->kcache[layer] : m->vcache[layer];
    } else {
        return 1;
    }
    if (expected > (size_t) INT64_MAX || count != (int64_t) expected ||
        (expected && (!source || !out)) ||
        expected > SIZE_MAX / sizeof(int64_t)) return 1;
    if (kind == 2 || kind == 3) {
        size_t valid_per_head = 0;
        if (!checked_mul_size((size_t) m->pos,
                              (size_t) m->d.head_dim, &valid_per_head)) return 1;
        const size_t resident_per_head = m->kv_per_head;
        size_t copy_bytes = 0;
        if (!checked_mul_size(valid_per_head, sizeof(int64_t), &copy_bytes)) return 1;
        for (int64_t head = 0; head < m->d.n_heads_kv; ++head) {
            size_t out_base = 0, source_base = 0;
            if (!checked_mul_size((size_t) head, valid_per_head, &out_base) ||
                !checked_mul_size((size_t) head, resident_per_head, &source_base))
                return 1;
            memcpy(out + out_base, source + source_base, copy_bytes);
        }
    } else if (expected) {
        memcpy(out, source, expected * sizeof(int64_t));
    }
    return 0;
}

int bonsai35_model_get_stats(void *handle, bonsai35_exec_stats *out) {
    if (!handle || !out) return 1;
    *out = ((bonsai35_model *) handle)->stats;
    return 0;
}

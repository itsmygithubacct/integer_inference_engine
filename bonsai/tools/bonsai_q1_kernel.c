#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
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
        H <= 0 || Hkv <= 0 || hd <= 0 || L <= 0 || frac < 1 || frac > 29 || H % Hkv != 0 ||
        k_kv_stride < L * hd || v_kv_stride < L * hd) {
        return 1;
    }
    size_t need = 0;
    if (!checked_mul_size((size_t) H, (size_t) L, &need)) {
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
    uint64_t *maxk = (uint64_t *) malloc((size_t) Hkv * sizeof(uint64_t));
    uint64_t *maxv = (uint64_t *) malloc((size_t) Hkv * sizeof(uint64_t));
    if (!maxk || !maxv) {
        free(maxk);
        free(maxv);
        return 2;
    }
    for (int64_t kv = 0; kv < Hkv; ++kv) {
        maxk[kv] = bonsai_maxabs_u64(k + (size_t) kv * k_kv_stride, (size_t) L * hd);
        maxv[kv] = bonsai_maxabs_u64(v + (size_t) kv * v_kv_stride, (size_t) L * hd);
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
        const int64_t *qh = q + (size_t) h * hd;
        const int64_t *kh = k + (size_t) kv * k_kv_stride;
        const int64_t *vh = v + (size_t) kv * v_kv_stride;
        int64_t *sc = scratch + (size_t) h * L;

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
            const int64_t *kj = kh + (size_t) j * hd;
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
            sc[j] = Z ? ((sc[j] << frac) / Z) : 0;
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
        int64_t *oh = out + (size_t) h * hd;
        for (int64_t d = 0; d < hd; ++d) {
            int64_t acc = 0;
            for (int64_t j = 0; j < L; ++j) {
                acc += sc[j] * vh[(size_t) j * hd + d];
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
        frac < 1 || frac > 29 || H % Hkv != 0 || L != start + M) {
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

    uint64_t *maxk = (uint64_t *) malloc((size_t) Hkv * sizeof(uint64_t));
    uint64_t *maxv = (uint64_t *) malloc((size_t) Hkv * sizeof(uint64_t));
    if (!maxk || !maxv) {
        free(maxk);
        free(maxv);
        return 2;
    }
    for (int64_t kv = 0; kv < Hkv; ++kv) {
        maxk[kv] = bonsai_maxabs_u64(k + (size_t) kv * L * hd, (size_t) L * hd);
        maxv[kv] = bonsai_maxabs_u64(v + (size_t) kv * L * hd, (size_t) L * hd);
    }
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) {
        nthreads = 1;
    }
    size_t scr_count = 0;
    if (!checked_mul_size((size_t) nthreads, (size_t) L, &scr_count)) {
        free(maxk);
        free(maxv);
        return 1;
    }
    int64_t *scratch = (int64_t *) malloc(scr_count * sizeof(int64_t));   // per-thread (L) score/prob buffer
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
            int64_t *sc = scratch + (size_t) tid * L;
            int64_t kv = h / rep;
            int64_t Lv = start + m + 1;                       // causal: query m sees keys [0, Lv)
            const int64_t *qh = q + ((size_t) h * M + (size_t) m) * hd;
            const int64_t *kh = k + (size_t) kv * L * hd;
            const int64_t *vh = v + (size_t) kv * L * hd;

            uint64_t maxq = bonsai_maxabs_u64(qh, (size_t) hd);   // q@K bound (contract hd), 128-bit, no wrap
            unsigned __int128 qk = (unsigned __int128) maxq * (unsigned __int128) maxk[kv];
            if (qk > i64max / (unsigned __int128) hd) {
                #pragma omp atomic write
                overflow = 1;
                continue;
            }
            int64_t mx = INT64_MIN;
            for (int64_t j = 0; j < Lv; ++j) {
                const int64_t *kj = kh + (size_t) j * hd;
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
                sc[j] = Z ? ((sc[j] << frac) / Z) : 0;
            }
            uint64_t maxp = bonsai_maxabs_u64(sc, (size_t) Lv);   // probs@V bound (contract Lv)
            unsigned __int128 pv = (unsigned __int128) maxp * (unsigned __int128) maxv[kv];
            if (pv > i64max / (unsigned __int128) Lv) {
                #pragma omp atomic write
                overflow = 1;
                continue;
            }
            int64_t *oh = out + ((size_t) h * M + (size_t) m) * hd;
            for (int64_t d = 0; d < hd; ++d) {
                int64_t acc = 0;
                for (int64_t j = 0; j < Lv; ++j) {
                    acc += sc[j] * vh[(size_t) j * hd + d];
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
        B <= 0 || H <= 0 || Hkv <= 0 || hd <= 0 || frac < 1 || frac > 29 || H % Hkv != 0) {
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

    int64_t Lmax = 0;
    for (int64_t b = 0; b < B; ++b) {
        if (!k_ptrs[b] || !v_ptrs[b] || lengths[b] <= 0 ||
            k_kv_strides[b] < lengths[b] * hd || v_kv_strides[b] < lengths[b] * hd) {
            return 1;
        }
        if (lengths[b] > Lmax) {
            Lmax = lengths[b];
        }
    }
    size_t nbk = 0;
    if (!checked_mul_size((size_t) B, (size_t) Hkv, &nbk)) {
        return 1;
    }
    uint64_t *maxk = (uint64_t *) malloc(nbk * sizeof(uint64_t));   // per (b, kv) max|k|, max|v| for the bound
    uint64_t *maxv = (uint64_t *) malloc(nbk * sizeof(uint64_t));
    if (!maxk || !maxv) {
        free(maxk);
        free(maxv);
        return 2;
    }
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t kv = 0; kv < Hkv; ++kv) {
            maxk[b * Hkv + kv] = bonsai_maxabs_u64(k_ptrs[b] + (size_t) kv * k_kv_strides[b],
                                                   (size_t) lengths[b] * hd);
            maxv[b * Hkv + kv] = bonsai_maxabs_u64(v_ptrs[b] + (size_t) kv * v_kv_strides[b],
                                                   (size_t) lengths[b] * hd);
        }
    }
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) {
        nthreads = 1;
    }
    size_t scr_count = 0;
    if (!checked_mul_size((size_t) nthreads, (size_t) Lmax, &scr_count)) {
        free(maxk);
        free(maxv);
        return 1;
    }
    int64_t *scratch = (int64_t *) malloc(scr_count * sizeof(int64_t));
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
            int64_t *sc = scratch + (size_t) tid * Lmax;
            int64_t kv = h / rep;
            int64_t L = lengths[b];
            const int64_t *qh = q + ((size_t) b * H + (size_t) h) * hd;
            const int64_t *kh = k_ptrs[b] + (size_t) kv * k_kv_strides[b];
            const int64_t *vh = v_ptrs[b] + (size_t) kv * v_kv_strides[b];

            uint64_t maxq = bonsai_maxabs_u64(qh, (size_t) hd);
            unsigned __int128 qk = (unsigned __int128) maxq * (unsigned __int128) maxk[b * Hkv + kv];
            if (qk > i64max / (unsigned __int128) hd) {
                #pragma omp atomic write
                overflow = 1;
                continue;
            }
            int64_t mx = INT64_MIN;
            for (int64_t j = 0; j < L; ++j) {
                const int64_t *kj = kh + (size_t) j * hd;
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
                sc[j] = Z ? ((sc[j] << frac) / Z) : 0;
            }
            uint64_t maxp = bonsai_maxabs_u64(sc, (size_t) L);
            unsigned __int128 pv = (unsigned __int128) maxp * (unsigned __int128) maxv[b * Hkv + kv];
            if (pv > i64max / (unsigned __int128) L) {
                #pragma omp atomic write
                overflow = 1;
                continue;
            }
            int64_t *oh = out + ((size_t) b * H + (size_t) h) * hd;
            for (int64_t d = 0; d < hd; ++d) {
                int64_t acc = 0;
                for (int64_t j = 0; j < L; ++j) {
                    acc += sc[j] * vh[(size_t) j * hd + d];
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
// uses the fixedpoint.py::fixed_point_sigmoid d_clip form — ((frac+2)<<(2*frac))/log2e, with NO
// (1<<62)//log2e cap (that cap is softmax-only). The final (x*sig)>>frac WRAPS mod 2^64 like the NumPy
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
    const int64_t d_clip = ((frac + 2) << (2 * frac)) / log2e;   // sigmoid form (no 1<<62 cap)
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
        int64_t sig = z ? ((e1 << frac) / z) : 0;    // e1<<frac >= 0, z > 0 -> trunc == floor
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
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            if (prepare_q1_block_lut32(xb, &totals[idx], lut + idx * 16u * 256u)) {
                range_bad = 1;
            }
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
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            if (prepare_q1_block_lut32(xb, &totals[idx], lut + idx * 16u * 256u)) {
                range_bad = 1;
            }
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
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t *xb = x + (size_t) t * x_width + (size_t) b * 128u;
            const size_t idx = (size_t) t * (size_t) n_blocks + (size_t) b;
            if (prepare_q1_block_lut32(xb, &totals[idx], lut + idx * 16u * 256u)) {
                range_bad = 1;
            }
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

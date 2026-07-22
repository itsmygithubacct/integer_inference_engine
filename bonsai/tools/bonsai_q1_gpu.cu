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
#include <mma.h>
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

// Qwen3.5 artifacts commit Q1 scales as int32 after the importer has proved the
// narrowing lossless.  Keep those scales narrow on device instead of silently
// expanding roughly 800 MiB of the 27B artifact back to int64.  The multiply
// still explicitly widens the scale to int64 before the modulo-2^64 product,
// so this kernel has exactly the same arithmetic contract as q1_linear_kernel.
__global__ void q1_linear_scale32_kernel(
        const long long* __restrict__ x,
        const unsigned char* __restrict__ bits,
        const int* __restrict__ scale,
        long long tokens, long long out_f, long long n_blocks, long long frac,
        long long* __restrict__ out)
{
    const long long gtid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long warp_id = gtid >> 5;
    const int lane = (int)(threadIdx.x & 31);
    if (warp_id >= tokens * out_f) return;

    const long long t = warp_id / out_f;
    const long long o = warp_id % out_f;
    const long long* xrow = x + t * (n_blocks * 128);
    const unsigned char* brow = bits + o * (n_blocks * 16);
    const int* srow = scale + o * n_blocks;
    unsigned long long total = 0ULL;
    for (long long b = 0; b < n_blocks; ++b) {
        const long long* xb = xrow + b * 128;
        const unsigned char* bb = brow + b * 16;
        long long lane_partial = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = lane * 4 + j;
            const int sbit = (bb[i >> 3] >> (i & 7)) & 1;
            lane_partial += (long long)(2 * sbit - 1) * xb[i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            lane_partial += __shfl_down_sync(0xffffffffu, lane_partial, off);
        if (lane == 0) {
            const unsigned long long prod =
                (unsigned long long)lane_partial * (unsigned long long)(long long)srow[b];
            total += (unsigned long long)arshift_i64_floor((long long)prod, frac);
        }
    }
    if (lane == 0) out[t * out_f + o] = (long long)total;
}

// GPU analogue of the CPU activation-LUT kernel.  For each 8-activation byte
// lane, commit all 256 exact subset sums once; every output row then performs
// 16 gathers per 128-wide block instead of 128 sign/activation operations (or
// 128 DP4A limb instructions).  The table is reused by grouped projections.
__global__ void q1_lut_build_kernel(const long long* x, long long n_blocks,
                                    long long* lut) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long total=n_blocks*16*256;
    if(i>=total)return;
    const int mask=(int)(i&255);const long long lane=(i>>8)&15;const long long b=i>>12;
    const long long* src=x+b*128+lane*8;unsigned long long sum=0;
    #pragma unroll
    for(int j=0;j<8;++j)if(mask&(1<<j))sum+=(unsigned long long)src[j];
    lut[i]=(long long)sum;
}

__global__ void q1_lut32_build_kernel(const long long* x,long long n_blocks,int* lut,int* overflow){
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,total=n_blocks*16*256;if(i>=total)return;
    const int mask=(int)(i&255);const long long lane=(i>>8)&15,b=i>>12;const long long* src=x+b*128+lane*8;
    unsigned long long us=0;for(int j=0;j<8;++j)if(mask&(1<<j))us+=(unsigned long long)src[j];
    const long long s=(long long)us;if(s<(long long)INT_MIN||s>(long long)INT_MAX){atomicOr(overflow,1);lut[i]=0;}
    else lut[i]=(int)s;
}

__global__ void q1_lut_apply_scale32_kernel(
        const long long* lut,const unsigned char* bits,const int* scale,
        long long out_f,long long n_blocks,long long frac,long long* out){
    const long long o=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(o>=out_f)return;
    const unsigned char* br=bits+o*n_blocks*16;const int* sr=scale+o*n_blocks;
    unsigned long long total=0;
    for(long long b=0;b<n_blocks;++b){
        unsigned long long selected=0,all=0;const long long* lb=lut+b*16*256;
        #pragma unroll
        for(int lane=0;lane<16;++lane){selected+=(unsigned long long)lb[lane*256+br[b*16+lane]];all+=(unsigned long long)lb[lane*256+255];}
        const long long signed_sum=(long long)(selected*2ULL-all);
        const unsigned long long prod=(unsigned long long)signed_sum*(unsigned long long)(long long)sr[b];
        total+=(unsigned long long)arshift_i64_floor((long long)prod,frac);
    }
    out[o]=(long long)total;
}

__global__ void q1_lut_apply_scale64_kernel(
        const long long* lut,const unsigned char* bits,const long long* scale,
        long long out_f,long long n_blocks,long long frac,long long* out){
    const long long o=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(o>=out_f)return;
    const unsigned char* br=bits+o*n_blocks*16;const long long* sr=scale+o*n_blocks;
    unsigned long long total=0;
    for(long long b=0;b<n_blocks;++b){
        unsigned long long selected=0,all=0;const long long* lb=lut+b*16*256;
        #pragma unroll
        for(int lane=0;lane<16;++lane){selected+=(unsigned long long)lb[lane*256+br[b*16+lane]];all+=(unsigned long long)lb[lane*256+255];}
        const long long signed_sum=(long long)(selected*2ULL-all);
        const unsigned long long prod=(unsigned long long)signed_sum*(unsigned long long)sr[b];
        total+=(unsigned long long)arshift_i64_floor((long long)prod,frac);
    }
    out[o]=(long long)total;
}

__global__ void q1_transpose_bits_kernel(const unsigned char* src,unsigned char* dst,
                                         long long out_f,long long n_blocks){
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long n=out_f*n_blocks*16;if(i>=n)return;
    const long long lane=i%16,b=(i/16)%n_blocks,o=i/(16*n_blocks);
    dst[(b*16+lane)*out_f+o]=src[i];
}
// BMMA layout: fixed block, then output row, then its 16 packed K bits.
__global__ void q1_repack_bits_bmma_kernel(const unsigned char* src,unsigned char* dst,
                                           long long out_f,long long n_blocks){
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=out_f*n_blocks*16;if(i>=n)return;
    const long long lane=i%16,b=(i/16)%n_blocks,o=i/(16*n_blocks);
    dst[(b*out_f+o)*16+lane]=src[i];
}
__global__ void q1_transpose_scale32_kernel(const int* src,int* dst,long long out_f,long long n_blocks){
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=out_f*n_blocks)return;
    const long long b=i%n_blocks,o=i/n_blocks;dst[b*out_f+o]=src[i];
}

__global__ void q1_lut_apply_scale32_transposed_kernel(
        const long long* lut,const unsigned char* bits,const int* scale,
        long long out_f,long long n_blocks,long long frac,long long* out){
    const long long o=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(o>=out_f)return;
    unsigned long long total=0;
    for(long long b=0;b<n_blocks;++b){
        unsigned long long selected=0,all=0;const long long* lb=lut+b*16*256;
        #pragma unroll
        for(int lane=0;lane<16;++lane){
            const unsigned char wb=bits[(b*16+lane)*out_f+o];
            selected+=(unsigned long long)lb[lane*256+wb];all+=(unsigned long long)lb[lane*256+255];
        }
        const long long signed_sum=(long long)(selected*2ULL-all);
        const unsigned long long prod=(unsigned long long)signed_sum*(unsigned long long)(long long)scale[b*out_f+o];
        total+=(unsigned long long)arshift_i64_floor((long long)prod,frac);
    }
    out[o]=(long long)total;
}

// Four output rows per thread exposes independent gather chains and removes
// three quarters of loop/control overhead.  The int32 LUT is lossless only
// after q1_lut32_build_kernel's device guard; all sums/products remain int64.
__global__ void q1_lut32_apply_scale32_transposed_x4_kernel(
        const int* lut,const unsigned char* bits,const int* scale,long long out_f,
        long long n_blocks,long long frac,long long* out){
    const long long base=((long long)blockIdx.x*blockDim.x+threadIdx.x)*4;if(base>=out_f)return;
    unsigned long long total[4]={0,0,0,0};
    for(long long b=0;b<n_blocks;++b){
        unsigned long long sel[4]={0,0,0,0},all=0;const int* lb=lut+b*16*256;
        #pragma unroll
        for(int lane=0;lane<16;++lane){all+=(unsigned long long)(long long)lb[lane*256+255];
            #pragma unroll
            for(int r=0;r<4;++r)if(base+r<out_f){const unsigned char wb=bits[(b*16+lane)*out_f+base+r];
                sel[r]+=(unsigned long long)(long long)lb[lane*256+wb];}}
        #pragma unroll
        for(int r=0;r<4;++r)if(base+r<out_f){const long long ss=(long long)(sel[r]*2ULL-all);
            const unsigned long long p=(unsigned long long)ss*(unsigned long long)(long long)scale[b*out_f+base+r];
            total[r]+=(unsigned long long)arshift_i64_floor((long long)p,frac);}
    }
    #pragma unroll
    for(int r=0;r<4;++r)if(base+r<out_f)out[base+r]=(long long)total[r];
}

__global__ void q1_linear_scale32_transposed_kernel(
        const long long* x,const unsigned char* bits,const int* scale,long long tokens,
        long long out_f,long long n_blocks,long long frac,long long* out){
    const long long gtid=(long long)blockIdx.x*blockDim.x+threadIdx.x,warp=gtid>>5;
    const int lane=threadIdx.x&31;if(warp>=tokens*out_f)return;
    const long long t=warp/out_f,o=warp%out_f;const long long* xr=x+t*n_blocks*128;
    unsigned long long total=0;
    for(long long b=0;b<n_blocks;++b){long long part=0;
        #pragma unroll
        for(int j=0;j<4;++j){const int e=lane*4+j;const unsigned char wb=bits[(b*16+(e>>3))*out_f+o];
            part+=(long long)(2*((wb>>(e&7))&1)-1)*xr[b*128+e];}
        #pragma unroll
        for(int off=16;off;off>>=1)part+=__shfl_down_sync(0xffffffffu,part,off);
        if(lane==0){const unsigned long long prod=(unsigned long long)part*(unsigned long long)(long long)scale[b*out_f+o];
            total+=(unsigned long long)arshift_i64_floor((long long)prod,frac);}
    }if(lane==0)out[t*out_f+o]=(long long)total;
}

// Pack one int64 activation block into four 128x8 column-major b1 tiles: tile
// g contains bitplanes 8g..8g+7.  Two's-complement bit 31 is handled with a
// negative coefficient during reconstruction, so the mapping is exact for the
// entire signed int32 envelope (guarded by the caller's LUT/range policy).
__global__ void q1_bmma_activation_kernel(const long long* x,long long n_blocks,
                                          unsigned int* planes,int* overflow){
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,total=n_blocks*32*4;if(i>=total)return;
    const int word=i&3,bit=(i>>2)&31;const long long b=i>>7;unsigned int packed=0;
    for(int j=0;j<32;++j){const long long v=x[b*128+word*32+j];
        if(v<(long long)INT_MIN||v>(long long)INT_MAX){atomicOr(overflow,1);return;}
        packed|=((unsigned int)(((int)v>>bit)&1))<<j;}
    const int group=bit>>3,col=bit&7;
    planes[((b*4+group)*8+col)*4+word]=packed;
}

// Arithmetic for one eight-row BMMA output tile.  Both the single-projection
// and grouped-projection launchers call this exact routine, so grouping changes
// only CUDA scheduling: weight bytes, bitplane reads, block order, scale
// multiply, per-block floor, and modulo-2^64 accumulation remain identical.
__device__ __forceinline__ void q1_bmma_apply_scale32_tile(
        const unsigned int* planes,const unsigned char* bits,const int* scale,
        long long out_f,long long n_blocks,long long frac,long long* out,
        long long out_base,int lane,int* tile){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    unsigned long long total=0;
    for(long long b=0;b<n_blocks;++b){
        wmma::fragment<wmma::matrix_a,8,8,128,wmma::experimental::precision::b1,wmma::row_major> af;
        wmma::load_matrix_sync(af,bits+(b*out_f+out_base)*16,128);
        int popw=0;if(lane<8){const unsigned int* wr=reinterpret_cast<const unsigned int*>(bits+(b*out_f+out_base+lane)*16);
            popw=__popc(wr[0])+__popc(wr[1])+__popc(wr[2])+__popc(wr[3]);}
        long long signed_sum=0;
        #pragma unroll
        for(int group=0;group<4;++group){
            wmma::fragment<wmma::matrix_b,8,8,128,wmma::experimental::precision::b1,wmma::col_major> bf;
            wmma::fragment<wmma::accumulator,8,8,128,int> cf,df;wmma::fill_fragment(cf,0);
            wmma::load_matrix_sync(bf,planes+(b*4+group)*32,128);
            wmma::bmma_sync(df,af,bf,cf,wmma::experimental::bmmaBitOpXOR,
                            wmma::experimental::bmmaAccumulateOpPOPC);
            wmma::store_matrix_sync(tile,df,8,wmma::mem_row_major);__syncwarp();
            if(lane<8){
                #pragma unroll
                for(int p=0;p<8;++p){const long long term=(long long)popw-(long long)tile[lane*8+p];
                    const int bit=group*8+p;if(bit==31)signed_sum-=term*(1LL<<31);else signed_sum+=term*(1LL<<bit);}
            }__syncwarp();
        }
        if(lane<8){const unsigned long long prod=(unsigned long long)signed_sum*
                (unsigned long long)(long long)scale[b*out_f+out_base+lane];
            total+=(unsigned long long)arshift_i64_floor((long long)prod,frac);}
    }
    if(lane<8)out[out_base+lane]=(long long)total;
#endif
}

__global__ void q1_bmma_apply_scale32_kernel(
        const unsigned int* planes,const unsigned char* bits,const int* scale,
        long long out_f,long long n_blocks,long long frac,long long* out){
#if __CUDA_ARCH__ >= 750
    const int lane=threadIdx.x&31,warp_in_block=threadIdx.x>>5;
    const long long warp=(long long)blockIdx.x*(blockDim.x/32)+warp_in_block;
    const long long out_base=warp*8;if(out_base>=out_f)return;
    __shared__ int smem[4][64];
    q1_bmma_apply_scale32_tile(
        planes,bits,scale,out_f,n_blocks,frac,out,out_base,lane,smem[warp_in_block]);
#endif
}

// Same prepared activation, up to four independent resident weights.  A warp
// is assigned one ordinary eight-row output tile and then executes the same
// tile routine as the legacy one-projection kernel above.  Combining grids
// removes launch/graph nodes without concatenating weights or outputs and
// without changing any reduction order.
__global__ void q1_bmma_apply_scale32_group4_kernel(
        const unsigned int* planes,
        const unsigned char* bits0,const int* scale0,long long out_f0,long long* out0,
        const unsigned char* bits1,const int* scale1,long long out_f1,long long* out1,
        const unsigned char* bits2,const int* scale2,long long out_f2,long long* out2,
        const unsigned char* bits3,const int* scale3,long long out_f3,long long* out3,
        long long n_blocks,long long frac){
#if __CUDA_ARCH__ >= 750
    const int lane=threadIdx.x&31,warp_in_block=threadIdx.x>>5;
    long long tile_index=(long long)blockIdx.x*(blockDim.x/32)+warp_in_block;
    const long long tiles0=out_f0/8,tiles1=out_f1/8,tiles2=out_f2/8,tiles3=out_f3/8;
    if(tile_index>=tiles0+tiles1+tiles2+tiles3)return;
    const unsigned char* bits=bits0;const int* scale=scale0;
    long long out_f=out_f0;long long* out=out0;
    if(tile_index>=tiles0){
        tile_index-=tiles0;bits=bits1;scale=scale1;out_f=out_f1;out=out1;
        if(tile_index>=tiles1){
            tile_index-=tiles1;bits=bits2;scale=scale2;out_f=out_f2;out=out2;
            if(tile_index>=tiles2){
                tile_index-=tiles2;bits=bits3;scale=scale3;out_f=out_f3;out=out3;
            }
        }
    }
    __shared__ int smem[4][64];
    q1_bmma_apply_scale32_tile(
        planes,bits,scale,out_f,n_blocks,frac,out,tile_index*8,lane,smem[warp_in_block]);
#endif
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

// DP4A resident apply for the losslessly narrowed Qwen3.5 scale layout.  This
// is intentionally a separate kernel rather than a device-side dtype branch:
// all warps in a launch read one known scale representation and the integer
// instruction stream remains deterministic.
__global__ void q1_dp4a_warp_scale32_kernel(
        const signed char* __restrict__ d_limb, const unsigned char* __restrict__ bits,
        const int* __restrict__ scale, long long tokens, long long out_f,
        long long n_blocks, long long frac, int L, long long* __restrict__ out) {
    const long long gtid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long warp_id = gtid >> 5;
    const int lane = (int)(threadIdx.x & 31);
    if (warp_id >= tokens * out_f) return;
    const long long t = warp_id / out_f, o = warp_id % out_f;
    const long long K = n_blocks * 128, TK = tokens * K;
    const unsigned char* brow = bits + o * (n_blocks * 16);
    const int* srow = scale + o * n_blocks;
    unsigned long long total = 0ULL;
    for (long long b = 0; b < n_blocks; ++b) {
        const unsigned char wbyte = brow[b * 16 + (lane >> 1)];
        const int bb = (lane & 1) * 4;
        const int w0 = ((wbyte >> (bb + 0)) & 1) ? 1 : -1;
        const int w1 = ((wbyte >> (bb + 1)) & 1) ? 1 : -1;
        const int w2 = ((wbyte >> (bb + 2)) & 1) ? 1 : -1;
        const int w3 = ((wbyte >> (bb + 3)) & 1) ? 1 : -1;
        const int wpk = (w0 & 0xFF) | ((w1 & 0xFF) << 8) |
                        ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
        const long long off = t * K + b * 128 + (long long)lane * 4;
        long long lane_partial = 0;
        for (int l = 0; l < L; ++l) {
            const int dpk = *reinterpret_cast<const int*>(d_limb + (long long)l * TK + off);
            lane_partial += ((long long)__dp4a(wpk, dpk, 0)) << (8 * l);
        }
        #pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1)
            lane_partial += __shfl_down_sync(0xffffffffu, lane_partial, o2);
        if (lane == 0) {
            const unsigned long long prod =
                (unsigned long long)lane_partial * (unsigned long long)(long long)srow[b];
            total += (unsigned long long)arshift_i64_floor((long long)prod, frac);
        }
    }
    if (lane == 0) out[t * out_f + o] = (long long)total;
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

__device__ __forceinline__ unsigned long long g_isqrt_u64_fast(unsigned long long n){
    unsigned long long res=0,bit=1ULL<<62;while(bit>n)bit>>=2;
    while(bit){if(n>=res+bit){n-=res+bit;res=(res>>1)+bit;}else res>>=1;bit>>=2;}return res;
}
__device__ __forceinline__ long long g_floor_div_i64_u64(long long n,unsigned long long d){
    if(n>=0)return (long long)((unsigned long long)n/d);
    const unsigned long long mag=(~(unsigned long long)n)+1ULL;
    const unsigned long long q=mag/d+(mag%d!=0);return q==(1ULL<<63)?(long long)0x8000000000000000LL:-(long long)q;
}

// Block-parallel committed-envelope RMSNorm.  It first proves every input and
// gain fits int32 and max(x)^2*cols fits uint64; only then uses native
// 32x32->64 products and 64-bit division.  A failed proof sets overflow and the
// resident producer is discarded, preserving the big-int CPU oracle fallback.
__global__ void rmsnorm_fast_i32_kernel(const long long* x,long long rows,long long cols,
                                        long long frac,unsigned long long eps,const long long* gain,
                                        long long* out,int* overflow){
    const long long r=blockIdx.x;if(r>=rows)return;const int tid=threadIdx.x;
    __shared__ unsigned long long ss[256],mm[256];__shared__ unsigned long long rms;__shared__ int bad;
    unsigned long long local_s=0,local_m=0;int local_bad=0;const long long* row=x+r*cols;
    for(long long i=tid;i<cols;i+=blockDim.x){const long long v=row[i];
        if(v<INT_MIN||v>INT_MAX)local_bad=1;const long long a=v<0?-v:v;if((unsigned long long)a>local_m)local_m=(unsigned long long)a;}
    ss[tid]=0;mm[tid]=local_m;if(tid==0)bad=0;__syncthreads();if(local_bad)atomicOr(&bad,1);
    for(int off=blockDim.x/2;off;off>>=1){__syncthreads();if(tid<off&&mm[tid+off]>mm[tid])mm[tid]=mm[tid+off];}
    __syncthreads();if(tid==0&&mm[0]&&mm[0]*mm[0]>ULLONG_MAX/(unsigned long long)cols)bad=1;__syncthreads();
    if(!bad){for(long long i=tid;i<cols;i+=blockDim.x){const long long v=row[i];local_s+=(unsigned long long)(v*v);}ss[tid]=local_s;}
    __syncthreads();for(int off=blockDim.x/2;off;off>>=1){if(tid<off)ss[tid]+=ss[tid+off];__syncthreads();}
    if(tid==0&&!bad){unsigned long long mean=ss[0]/(unsigned long long)cols;
        if(mean>ULLONG_MAX-eps)bad=1;else{mean+=eps;rms=g_isqrt_u64_fast(mean);if(!rms)bad=1;}}
    __syncthreads();if(bad){if(tid==0)atomicOr(overflow,1);return;}
    const long long fp=1LL<<frac;long long* dst=out+r*cols;
    for(long long i=tid;i<cols;i+=blockDim.x){const long long n=g_floor_div_i64_u64(row[i]*fp,rms);long long y=n;
        if(gain){const long long gg=gain[i];if(n<INT_MIN||n>INT_MAX||gg<INT_MIN||gg>INT_MAX){atomicOr(overflow,1);continue;}
            y=arshift_i64_floor(n*gg,frac);}dst[i]=y;}
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

// ---- Qwen3.5 recurrent primitive parity rung -----------------------------------------------------------
// These kernels implement one M=1 Gated DeltaNet update while keeping the
// canonical Q30 state on device.  They are also used as the arithmetic core of
// the resident hybrid executor; the standalone host ABI below is a small,
// independently testable parity rung.

__device__ __forceinline__ long long g_sigmoid_fixed(long long xi, long long frac,
                                                      long long log2e, long long d_clip) {
    const long long m = xi > 0 ? xi : 0;
    long long d0 = m;
    long long d1 = g_u64_to_i64((unsigned long long)m - (unsigned long long)xi);
    if (d0 > d_clip) d0 = d_clip;
    if (d1 > d_clip) d1 = d_clip;
    const long long e0 = g_exp2_neg_fixed((d0 * log2e) >> frac, frac);
    const long long e1 = g_exp2_neg_fixed((d1 * log2e) >> frac, frac);
    const long long denom = e0 + e1;
    return denom ? ((e1 << frac) / denom) : 0;
}

__device__ __forceinline__ long long g_lut_interp(const long long* lut, long long n,
                                                   long long x, long long minimum,
                                                   long long step) {
    const long long maximum = minimum + step * (n - 1);
    if (x <= minimum) return lut[0];
    if (x >= maximum) return lut[n - 1];
    const long long pos = x - minimum;
    long long idx = pos / step;
    if (idx > n - 2) idx = n - 2;
    const long long rem = pos - idx * step;
    return lut[idx] + ((lut[idx + 1] - lut[idx]) * rem) / step;
}

// Exact integer L2 norm, one CUDA thread per row.  Qwen3.5 key/state width is
// 128; a u128 accumulator covers the committed envelope and reports overflow.
__global__ void bonsai35_l2norm_kernel(const long long* x, long long rows, long long cols,
                                       long long frac, long long* out, int* overflow) {
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || *overflow) return;
    const long long* src = x + r * cols;
    long long* dst = out + r * cols;
    unsigned __int128 ssq = 0;
    for (long long i = 0; i < cols; ++i) {
        if (!g_add_square_u128(&ssq, src[i])) { atomicOr(overflow, 1); return; }
    }
    const unsigned long long norm = g_isqrt_u128(ssq);
    if (!norm) {
        for (long long i = 0; i < cols; ++i) dst[i] = 0;
        return;
    }
    const __int128 fp = (__int128)1 << frac;
    for (long long i = 0; i < cols; ++i) {
        const __int128 q = g_floor_div_i128_u64((__int128)src[i] * fp, norm);
        if (!g_i128_to_i64(q, &dst[i])) { atomicOr(overflow, 1); return; }
    }
}

__global__ void bonsai35_controls_kernel(
        const long long* alpha, const long long* beta, const long long* dt_bias,
        const long long* ssm_a, const long long* soft_lut, long long soft_n,
        const long long* exp_lut, long long exp_n, long long value_heads,
        long long frac, long long log2e, long long d_clip,
        long long soft_min, long long soft_step, long long soft_max,
        long long exp_min, long long exp_step,
        long long* beta_out, long long* decay_out, int* overflow) {
    const long long h = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= value_heads || *overflow) return;
    beta_out[h] = g_sigmoid_fixed(beta[h], frac, log2e, d_clip);
    const long long ax = g_u64_to_i64((unsigned long long)alpha[h] + (unsigned long long)dt_bias[h]);
    long long soft;
    if (ax <= soft_min) soft = 0;
    else if (ax >= soft_max) soft = ax;
    else soft = g_lut_interp(soft_lut, soft_n, ax, soft_min, soft_step);
    const long long gate = arshift_i64_floor(
        g_u64_to_i64((unsigned long long)soft * (unsigned long long)ssm_a[h]), frac);
    if (gate > 0) { atomicOr(overflow, 1); return; }
    const long long fp = (long long)1 << frac;
    long long decay;
    if (gate <= exp_min) decay = 0;
    else if (gate >= 0) decay = fp;
    else decay = g_lut_interp(exp_lut, exp_n, gate, exp_min, exp_step);
    decay_out[h] = decay;
}

// One block per value head, 128 threads.  Each thread owns one state/output
// column j, so the two synchronization points exactly separate prediction,
// outer update, and output contraction without atomics or changed shifts.
__global__ void bonsai35_state_step_kernel(
        const long long* q_key, const long long* k_key, const long long* v,
        const long long* beta, const long long* decay, long long* state,
        long long value_heads, long long key_heads, long long state_size,
        long long frac, long long state_frac, long long gdn_scale,
        long long* output) {
    const long long h = blockIdx.x;
    const long long j = threadIdx.x;
    __shared__ long long delta[128];
    if (h >= value_heads) return;
    const bool active = j < state_size;
    const long long kh = h % key_heads;
    const long long* q = q_key + kh * state_size;
    const long long* k = k_key + kh * state_size;
    long long* st = state + h * state_size * state_size;
    long long pred = 0;
    if (active) {
        unsigned long long acc = 0;
        for (long long i = 0; i < state_size; ++i) {
            const size_t off = (size_t)i * state_size + j;
            st[off] = arshift_i64_floor(
                g_u64_to_i64((unsigned long long)st[off] * (unsigned long long)decay[h]), frac);
            acc += (unsigned long long)st[off] * (unsigned long long)k[i];
        }
        pred = arshift_i64_floor((long long)acc, state_frac);
        const long long diff = g_u64_to_i64((unsigned long long)v[h * state_size + j] -
                                            (unsigned long long)pred);
        delta[j] = arshift_i64_floor(
            g_u64_to_i64((unsigned long long)diff * (unsigned long long)beta[h]), frac);
    }
    __syncthreads();
    if (active) {
        const long long outer_shift = 2 * frac - state_frac;
        for (long long i = 0; i < state_size; ++i) {
            const long long add = arshift_i64_floor(
                g_u64_to_i64((unsigned long long)k[i] * (unsigned long long)delta[j]), outer_shift);
            const size_t off = (size_t)i * state_size + j;
            st[off] = g_u64_to_i64((unsigned long long)st[off] + (unsigned long long)add);
        }
    }
    __syncthreads();
    if (active) {
        unsigned long long acc = 0;
        for (long long i = 0; i < state_size; ++i)
            acc += (unsigned long long)st[(size_t)i * state_size + j] * (unsigned long long)q[i];
        const long long score = arshift_i64_floor((long long)acc, frac);
        output[h * state_size + j] = arshift_i64_floor(
            g_u64_to_i64((unsigned long long)score * (unsigned long long)gdn_scale), frac);
    }
}

// Guarded narrow-storage variant for sm_86.  The canonical Q30 state is stored
// as int32 only while every value fits; each multiply is still an exact signed
// 32x32->64 product followed by the canonical floor shift.  This avoids the
// RTX 3070's very slow emulated 64x64 integer multiply.  Any escape sets the
// poison flag and the whole GPU context is discarded/replayed on CPU.
__global__ void bonsai35_state_step_i32_kernel(
        const long long* q_key,const long long* k_key,const long long* v,
        const long long* beta,const long long* decay,int* state,
        long long value_heads,long long key_heads,long long state_size,
        long long frac,long long state_frac,long long gdn_scale,
        long long* output,int* overflow){
    const long long h=blockIdx.x,j=threadIdx.x;__shared__ int delta[128];
    if(h>=value_heads)return;const bool active=j<state_size;const long long kh=h%key_heads;
    const long long* q=q_key+kh*state_size;const long long* k=k_key+kh*state_size;
    int* st=state+(size_t)h*state_size*state_size;unsigned long long acc=0;
    if(active){
        const long long kv=k[j]; // touch one element for a uniform head-range guard below
        if(kv<INT_MIN||kv>INT_MAX||decay[h]<INT_MIN||decay[h]>INT_MAX)atomicOr(overflow,1);
        const int dec=(int)decay[h];
        for(long long i=0;i<state_size;++i){const size_t off=(size_t)i*state_size+j;
            const long long ki=k[i];if(ki<INT_MIN||ki>INT_MAX){atomicOr(overflow,1);continue;}
            const long long nv=arshift_i64_floor((long long)st[off]*(long long)dec,frac);
            if(nv<INT_MIN||nv>INT_MAX){atomicOr(overflow,1);continue;}st[off]=(int)nv;
            acc+=(unsigned long long)((long long)st[off]*(long long)(int)ki);
        }
        const long long pred=arshift_i64_floor((long long)acc,state_frac);
        const long long vv=v[h*state_size+j],bb=beta[h];
        if(vv<INT_MIN||vv>INT_MAX||bb<INT_MIN||bb>INT_MAX){atomicOr(overflow,1);delta[j]=0;}
        else{const long long diff=g_u64_to_i64((unsigned long long)vv-(unsigned long long)pred);
            if(diff<INT_MIN||diff>INT_MAX){atomicOr(overflow,1);delta[j]=0;}
            else{const long long dd=arshift_i64_floor(diff*(long long)(int)bb,frac);
                if(dd<INT_MIN||dd>INT_MAX){atomicOr(overflow,1);delta[j]=0;}else delta[j]=(int)dd;}}
    }
    __syncthreads();
    if(active){const long long outer_shift=2*frac-state_frac;
        for(long long i=0;i<state_size;++i){const long long ki=k[i];if(ki<INT_MIN||ki>INT_MAX)continue;
            const long long add=arshift_i64_floor((long long)(int)ki*(long long)delta[j],outer_shift);
            const long long nv=(long long)st[(size_t)i*state_size+j]+add;
            if(nv<INT_MIN||nv>INT_MAX){atomicOr(overflow,1);continue;}st[(size_t)i*state_size+j]=(int)nv;}}
    __syncthreads();
    if(active){acc=0;for(long long i=0;i<state_size;++i){const long long qi=q[i];
            if(qi<INT_MIN||qi>INT_MAX){atomicOr(overflow,1);continue;}
            acc+=(unsigned long long)((long long)st[(size_t)i*state_size+j]*(long long)(int)qi);}
        const long long score=arshift_i64_floor((long long)acc,frac);
        output[h*state_size+j]=arshift_i64_floor(
            g_u64_to_i64((unsigned long long)score*(unsigned long long)gdn_scale),frac);}
}

// Exact modulo-2^64 signed i64*i32 using only native 32-bit multiplies.  This
// retains the canonical int64 Q30 state when it grows beyond int32 without
// falling back to sm_86's expensive general 64x64 multiply sequence.
__device__ __forceinline__ long long g_mul_i64_i32_wrap(long long a,int b){
    const unsigned long long ua=(unsigned long long)a;const unsigned int alo=(unsigned int)ua;
    const unsigned int ahi=(unsigned int)(ua>>32),blo=(unsigned int)b;
    unsigned long long p=(unsigned long long)alo*(unsigned long long)blo;
    unsigned int cross=ahi*blo;if(b<0)cross+=(unsigned int)(0U-alo);
    p+=(unsigned long long)cross<<32;return (long long)p;
}

__global__ void bonsai35_state_step_wide32_kernel(
        const long long* q_key,const long long* k_key,const long long* v,
        const long long* beta,const long long* decay,long long* state,
        long long value_heads,long long key_heads,long long state_size,
        long long frac,long long state_frac,long long gdn_scale,
        long long* output,int* overflow){
    const long long h=blockIdx.x,j=threadIdx.x;__shared__ long long delta[128];
    if(h>=value_heads)return;const bool active=j<state_size;const long long kh=h%key_heads;
    const long long* q=q_key+kh*state_size;const long long* k=k_key+kh*state_size;
    long long* st=state+(size_t)h*state_size*state_size;unsigned long long acc=0;
    if(active){
        if(decay[h]<INT_MIN||decay[h]>INT_MAX||beta[h]<INT_MIN||beta[h]>INT_MAX)atomicOr(overflow,1);
        const int dec=(int)decay[h];
        for(long long i=0;i<state_size;++i){const size_t off=(size_t)i*state_size+j;const long long ki=k[i];
            if(ki<INT_MIN||ki>INT_MAX){atomicOr(overflow,1);continue;}
            st[off]=arshift_i64_floor(g_mul_i64_i32_wrap(st[off],dec),frac);
            acc+=(unsigned long long)g_mul_i64_i32_wrap(st[off],(int)ki);}
        const long long pred=arshift_i64_floor((long long)acc,state_frac);
        const long long diff=g_u64_to_i64((unsigned long long)v[h*state_size+j]-(unsigned long long)pred);
        delta[j]=arshift_i64_floor(g_mul_i64_i32_wrap(diff,(int)beta[h]),frac);
    }
    __syncthreads();
    if(active){const long long outer_shift=2*frac-state_frac;
        for(long long i=0;i<state_size;++i){const long long ki=k[i];if(ki<INT_MIN||ki>INT_MAX)continue;
            const long long add=arshift_i64_floor(g_mul_i64_i32_wrap(delta[j],(int)ki),outer_shift);
            st[(size_t)i*state_size+j]=g_u64_to_i64((unsigned long long)st[(size_t)i*state_size+j]+(unsigned long long)add);}}
    __syncthreads();
    if(active){acc=0;for(long long i=0;i<state_size;++i){const long long qi=q[i];
            if(qi<INT_MIN||qi>INT_MAX){atomicOr(overflow,1);continue;}
            acc+=(unsigned long long)g_mul_i64_i32_wrap(st[(size_t)i*state_size+j],(int)qi);}
        const long long score=arshift_i64_floor((long long)acc,frac);
        output[h*state_size+j]=arshift_i64_floor(
            g_mul_i64_i32_wrap(score,(int)gdn_scale),frac);}
}

__global__ void bonsai35_split_qgate_kernel(const long long* qg, long long* q,
                                             long long* gate, long long H, long long hd) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * hd) return;
    const long long h = i / hd, e = i % hd;
    q[i] = qg[(h * 2) * hd + e];
    gate[i] = qg[(h * 2 + 1) * hd + e];
}

// Qwen3.5 text IMRoPE is a NeoX rotate-half over only the first n_rot
// channels.  cos/sin contain n_rot/2 entries for this absolute position.
__global__ void bonsai35_partial_rope_kernel(long long* x, const long long* cos,
                                              const long long* sin, long long rows,
                                              long long hd, long long n_rot, long long frac) {
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const long long half = n_rot / 2;
    long long* row = x + r * hd;
    for (long long e = 0; e < half; ++e) {
        const long long x0 = row[e], x1 = row[half + e];
        row[e] = arshift_i64_floor(g_u64_to_i64(
            (unsigned long long)x0 * (unsigned long long)cos[e] -
            (unsigned long long)x1 * (unsigned long long)sin[e]), frac);
        row[half + e] = arshift_i64_floor(g_u64_to_i64(
            (unsigned long long)x0 * (unsigned long long)sin[e] +
            (unsigned long long)x1 * (unsigned long long)cos[e]), frac);
    }
}

__global__ void bonsai35_sigmoid_gate_kernel(const long long* x, const long long* gate,
                                              long long* out, long long n, long long frac,
                                              long long log2e, long long d_clip) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long sig = g_sigmoid_fixed(gate[i], frac, log2e, d_clip);
    out[i] = arshift_i64_floor(g_u64_to_i64(
        (unsigned long long)x[i] * (unsigned long long)sig), frac);
}

__global__ void bonsai35_partial_rope_pos_kernel(long long* x, const long long* pos,
                                                  const long long* cos_table,
                                                  const long long* sin_table,
                                                  long long rows, long long hd,
                                                  long long n_rot, long long frac) {
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const long long half = n_rot / 2;
    const long long* c = cos_table + (*pos) * half;
    const long long* s = sin_table + (*pos) * half;
    long long* row = x + r * hd;
    for (long long e = 0; e < half; ++e) {
        const long long x0=row[e],x1=row[half+e];
        row[e]=arshift_i64_floor(g_u64_to_i64(
            (unsigned long long)x0*(unsigned long long)c[e]-
            (unsigned long long)x1*(unsigned long long)s[e]),frac);
        row[half+e]=arshift_i64_floor(g_u64_to_i64(
            (unsigned long long)x0*(unsigned long long)s[e]+
            (unsigned long long)x1*(unsigned long long)c[e]),frac);
    }
}

// Depthwise width-k convolution for a single recurrent decode token.  History
// is [slot,k-1,conv_dim], oldest first; update and output are independent per
// channel, so one thread performs both without a global synchronization.
__global__ void bonsai35_conv_decode_kernel(
        const long long* qkv, long long* history, const long long* weight,
        long long slot, long long conv_dim, long long conv_k, long long frac,
        long long* out) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=conv_dim)return;
    long long* hist=history+(size_t)slot*(conv_k-1)*conv_dim;
    unsigned long long acc=0;
    for(long long j=0;j<conv_k-1;++j)
        acc+=(unsigned long long)hist[j*conv_dim+i]*(unsigned long long)weight[i*conv_k+j];
    acc+=(unsigned long long)qkv[i]*(unsigned long long)weight[i*conv_k+conv_k-1];
    out[i]=arshift_i64_floor((long long)acc,frac);
    for(long long j=0;j<conv_k-2;++j)hist[j*conv_dim+i]=hist[(j+1)*conv_dim+i];
    hist[(conv_k-2)*conv_dim+i]=qkv[i];
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

// Exact block-parallel M=1 decode attention for Qwen3.5.  One block owns one
// query head for every phase, so score/max/Z/probability/output ordering has no
// cross-block race.  The existing triangle guards prove every qK and pV sum
// fits signed int64; integer addition is therefore associative and the warp /
// block reduction order is byte-identical to the serial oracle.  Scores and V
// sums still traverse their contracted dimensions in exact integer space—no
// float, rescaling, or changed floor point is introduced.
static constexpr long long Q35_ATTN_PARALLEL_MIN_L = 32;
static constexpr int Q35_ATTN_TPB = 256;

__device__ __forceinline__ unsigned long long q35_abs_i64(long long value) {
    return value < 0
        ? ((unsigned long long)(~(unsigned long long)value) + 1ULL)
        : (unsigned long long)value;
}

// Parallel full-cache maxabs for the standalone parity ABI.  The resident
// graph does not call this: its monotone max is updated from only the appended
// K/V row below.
__global__ void maxabs_bkv_parallel_kernel(
        const long long* __restrict__ cache,
        const long long* __restrict__ lengths,
        long long B, long long Hkv, long long hd, long long cap,
        unsigned long long* __restrict__ out) {
    const long long gid = (long long)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (gid >= B * Hkv) return;
    const long long b = gid / Hkv, kv = gid % Hkv, Lv = lengths[b];
    const long long* base = cache + (b * Hkv + kv) * cap * hd;
    __shared__ unsigned long long reduction[Q35_ATTN_TPB];
    unsigned long long local = 0;
    const long long count = Lv * hd;
    for (long long i = tid; i < count; i += blockDim.x) {
        const unsigned long long value = q35_abs_i64(base[i]);
        if (value > local) local = value;
    }
    reduction[tid] = local;
    __syncthreads();
    for (int stride = Q35_ATTN_TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride && reduction[tid + stride] > reduction[tid])
            reduction[tid] = reduction[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[gid] = reduction[0];
}

// Resident cache guard maxima are monotone: each decode appends exactly one
// row per attention layer/KV head.  Updating from that row avoids rescanning
// roughly a GiB of K/V cache data per 4K token across the 16 full layers.
__global__ void q35_update_maxabs_rows_kernel(
        const long long* __restrict__ rows, long long Hkv, long long hd,
        unsigned long long* __restrict__ persistent_max,
        const int* __restrict__ overflow) {
    const long long kv = (long long)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (kv >= Hkv) return;
    __shared__ int abort_block;
    if (tid == 0) abort_block = atomicAdd((int*)overflow, 0);
    __syncthreads();
    if (abort_block) return;
    __shared__ unsigned long long reduction[Q35_ATTN_TPB];
    unsigned long long local = 0;
    const long long* row = rows + kv * hd;
    for (long long d = tid; d < hd; d += blockDim.x) {
        const unsigned long long value = q35_abs_i64(row[d]);
        if (value > local) local = value;
    }
    reduction[tid] = local;
    __syncthreads();
    for (int stride = Q35_ATTN_TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride && reduction[tid + stride] > reduction[tid])
            reduction[tid] = reduction[tid + stride];
        __syncthreads();
    }
    if (tid == 0 && reduction[0] > persistent_max[kv])
        persistent_max[kv] = reduction[0];
}

template <typename CacheT>
__global__ void attention_decode_m1_parallel_kernel(
        const long long* __restrict__ q,
        const CacheT* __restrict__ Kc,
        const CacheT* __restrict__ Vc,
        const long long* __restrict__ lengths,
        long long B, long long H, long long Hkv, long long hd, long long cap,
        long long frac, long long inv_sqrt_fp, long long log2e, long long d_clip,
        const unsigned long long* __restrict__ maxk,
        const unsigned long long* __restrict__ maxv,
        long long* __restrict__ out,
        long long* __restrict__ scratch,
        int* __restrict__ overflow) {
    const long long gid = (long long)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (gid >= B * H) return;
    const long long b = gid / H, h = gid % H;
    const long long rep = H / Hkv, kv = h / rep, Lv = lengths[b];
    const long long* qh = q + gid * hd;
    const CacheT* Kb = Kc + (b * Hkv + kv) * cap * hd;
    const CacheT* Vb = Vc + (b * Hkv + kv) * cap * hd;
    long long* sc = scratch + gid * cap;
    long long* oh = out + gid * hd;
    const unsigned __int128 i64max =
        (unsigned __int128)0x7fffffffffffffffULL;

    __shared__ unsigned long long reduction_u[Q35_ATTN_TPB];
    __shared__ long long warp_max[Q35_ATTN_TPB / 32];
    __shared__ long long score_max;
    __shared__ unsigned long long Z_shared;
    __shared__ int abort_block;
    if (tid == 0) abort_block = (atomicAdd(overflow, 0) != 0);
    __syncthreads();
    if (abort_block) return;
    if (Lv <= 0) {
        for (long long d = tid; d < hd; d += blockDim.x) oh[d] = 0;
        return;
    }

    // The tiny-cache path avoids reduction barriers.  All lanes take the same
    // branch; only lane zero executes the original serial arithmetic.
    if (Lv < Q35_ATTN_PARALLEL_MIN_L) {
        if (tid != 0) return;
        unsigned long long maxq = 0;
        for (long long d = 0; d < hd; ++d) {
            const unsigned long long value = q35_abs_i64(qh[d]);
            if (value > maxq) maxq = value;
        }
        if ((unsigned __int128)maxq * (unsigned __int128)maxk[b * Hkv + kv]
                > i64max / (unsigned __int128)hd) {
            atomicOr(overflow, 1); return;
        }
        long long mx = (long long)0x8000000000000000ULL;
        for (long long j = 0; j < Lv; ++j) {
            long long dot = 0;
            for (long long d = 0; d < hd; ++d) dot += qh[d] * Kb[j * hd + d];
            long long score = arshift_i64_floor(dot, frac);
            score = arshift_i64_floor(score * inv_sqrt_fp, frac);
            sc[j] = score;
            if (score > mx) mx = score;
        }
        long long Z = 0;
        for (long long j = 0; j < Lv; ++j) {
            long long delta = mx - sc[j];
            if (delta > d_clip) delta = d_clip;
            const long long u = (delta * log2e) >> frac;
            const long long e = g_exp2_neg_fixed(u, frac);
            sc[j] = e; Z += e;
        }
        for (long long j = 0; j < Lv; ++j)
            sc[j] = Z ? ((sc[j] << frac) / Z) : 0;
        unsigned long long maxp = 0;
        for (long long j = 0; j < Lv; ++j) {
            const unsigned long long value = q35_abs_i64(sc[j]);
            if (value > maxp) maxp = value;
        }
        if ((unsigned __int128)maxp * (unsigned __int128)maxv[b * Hkv + kv]
                > i64max / (unsigned __int128)Lv) {
            atomicOr(overflow, 1); return;
        }
        for (long long d = 0; d < hd; ++d) {
            long long acc = 0;
            for (long long j = 0; j < Lv; ++j) acc += sc[j] * Vb[j * hd + d];
            oh[d] = arshift_i64_floor(acc, frac);
        }
        return;
    }

    // q maxabs and the exact pre-dot guard.
    unsigned long long local_maxq = 0;
    for (long long d = tid; d < hd; d += blockDim.x) {
        const unsigned long long value = q35_abs_i64(qh[d]);
        if (value > local_maxq) local_maxq = value;
    }
    reduction_u[tid] = local_maxq;
    __syncthreads();
    for (int stride = Q35_ATTN_TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride && reduction_u[tid + stride] > reduction_u[tid])
            reduction_u[tid] = reduction_u[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        abort_block = ((unsigned __int128)reduction_u[0]
            * (unsigned __int128)maxk[b * Hkv + kv]
            > i64max / (unsigned __int128)hd);
        if (abort_block) atomicOr(overflow, 1);
    }
    __syncthreads();
    if (abort_block) return;

    // Eight warps compute eight positions concurrently.  Lanes cover head
    // dimensions, giving coalesced K loads; every per-score dot is an exact
    // in-range integer tree reduction under the guard above.
    const int lane = tid & 31, warp = tid >> 5;
    long long local_score_max = (long long)0x8000000000000000ULL;
    for (long long j = warp; j < Lv; j += Q35_ATTN_TPB / 32) {
        long long dot = 0;
        for (long long d = lane; d < hd; d += 32)
            dot += qh[d] * Kb[j * hd + d];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        if (lane == 0) {
            long long score = arshift_i64_floor(dot, frac);
            score = arshift_i64_floor(score * inv_sqrt_fp, frac);
            sc[j] = score;
            if (score > local_score_max) local_score_max = score;
        }
    }
    if (lane == 0) warp_max[warp] = local_score_max;
    __syncthreads();
    if (tid == 0) {
        long long mx = warp_max[0];
        for (int w = 1; w < Q35_ATTN_TPB / 32; ++w)
            if (warp_max[w] > mx) mx = warp_max[w];
        score_max = mx;
    }
    __syncthreads();

    // Integer exp polynomial and positive exact normalization.  Z is bounded
    // by L*2^frac (<=2^41 for the supported frac<=29/L<=4096), so its tree sum
    // is exact and independent of reduction order.
    unsigned long long local_Z = 0;
    for (long long j = tid; j < Lv; j += blockDim.x) {
        long long delta = score_max - sc[j];
        if (delta > d_clip) delta = d_clip;
        const long long u = (delta * log2e) >> frac;
        const long long e = g_exp2_neg_fixed(u, frac);
        sc[j] = e;
        local_Z += (unsigned long long)e;
    }
    reduction_u[tid] = local_Z;
    __syncthreads();
    for (int stride = Q35_ATTN_TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduction_u[tid] += reduction_u[tid + stride];
        __syncthreads();
    }
    if (tid == 0) Z_shared = reduction_u[0];
    __syncthreads();

    unsigned long long local_maxp = 0;
    for (long long j = tid; j < Lv; j += blockDim.x) {
        const unsigned long long e = (unsigned long long)sc[j];
        const long long probability = Z_shared
            ? (long long)((e << frac) / Z_shared) : 0;
        sc[j] = probability;
        const unsigned long long value = q35_abs_i64(probability);
        if (value > local_maxp) local_maxp = value;
    }
    reduction_u[tid] = local_maxp;
    __syncthreads();
    for (int stride = Q35_ATTN_TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride && reduction_u[tid + stride] > reduction_u[tid])
            reduction_u[tid] = reduction_u[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        abort_block = ((unsigned __int128)reduction_u[0]
            * (unsigned __int128)maxv[b * Hkv + kv]
            > i64max / (unsigned __int128)Lv);
        if (abort_block) atomicOr(overflow, 1);
    }
    __syncthreads();
    if (abort_block) return;

    // Dimensions are independent and coalesced across a warp.  Each thread
    // retains ascending-position accumulation, exactly matching the oracle.
    for (long long d = tid; d < hd; d += blockDim.x) {
        long long acc = 0;
        for (long long j = 0; j < Lv; ++j)
            acc += sc[j] * Vb[j * hd + d];
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

// Qwen3.5 K/V values are Q16 and normally narrow by orders of magnitude, but
// narrowing is never assumed.  The preflight and commit are separate kernels
// in one stream: preflight examines BOTH complete rows and performs no writes;
// commit observes its completed flag before writing either cache.  Thus one
// unsafe lane cannot leave safe lanes partially appended.  The poisoned graph
// is discarded and replayed on the canonical int64 CPU path.
__global__ void q35_kv_i32_preflight_pair_kernel(
        const long long* __restrict__ k, const long long* __restrict__ v,
        long long n, int* __restrict__ overflow) {
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const long long kval = k[gid], vval = v[gid];
    if (kval < (long long)INT_MIN || kval > (long long)INT_MAX ||
        vval < (long long)INT_MIN || vval > (long long)INT_MAX)
        atomicOr(overflow, 1);
}

__global__ void q35_kv_i32_commit_pair_kernel(
        const long long* __restrict__ k, const long long* __restrict__ v,
        int* __restrict__ K, int* __restrict__ V,
        const long long* __restrict__ pos, long long Hkv, long long hd,
        long long cap, const int* __restrict__ overflow) {
    __shared__ int abort_block;
    if (threadIdx.x == 0) abort_block = atomicAdd((int*)overflow, 0);
    __syncthreads();
    if (abort_block) return;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= Hkv * hd) return;
    const long long kv = gid / hd, d = gid % hd;
    const size_t dst = ((size_t)kv * cap + *pos) * hd + d;
    K[dst] = (int)k[gid];
    V[dst] = (int)v[gid];
}

__global__ void q35_narrow_i32_guard_kernel(
        const long long* __restrict__ src, int* __restrict__ dst,
        long long n, int* __restrict__ overflow) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long value = src[i];
    if (value < (long long)INT_MIN || value > (long long)INT_MAX) {
        atomicOr(overflow, 1);
        return;
    }
    dst[i] = (int)value;
}

__global__ void q35_widen_i32_kernel(
        const int* __restrict__ src, long long* __restrict__ dst, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (long long)src[i];
}

// Select one row from a resident BMMA-layout Q1 embedding table.  The only
// token-dependent host input is the 8-byte ID.  Bits are laid out
// (block,out,row-byte) and int32 scales (block,out), exactly as registered by
// bonsai_q1_register_weight_i32_bmma.  Dequantization is integer sign*scale,
// identical to reference_bonsai.q1_rows_fp and independent of frac.
__global__ void q35_embedding_row_bmma_kernel(
        const unsigned char* __restrict__ bits,
        const int* __restrict__ scales,
        long long vocab, long long n_blocks,
        const long long* __restrict__ token,
        long long* __restrict__ out, int* __restrict__ overflow) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long d = n_blocks * 128;
    if (e >= d) return;
    const long long tok = *token;
    if (tok < 0 || tok >= vocab) {
        if (e == 0) atomicOr(overflow, 1);
        return;
    }
    const long long block = e >> 7, within = e & 127;
    const unsigned char packed = bits[(block * vocab + tok) * 16 + (within >> 3)];
    const long long scale = (long long)scales[block * vocab + tok];
    out[e] = ((packed >> (within & 7)) & 1) ? scale : -scale;
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
    int* dscale32;
    long long out_f;
    long long n_blocks;
    int scale_bits;
    int layout;                         // 0 output-major artifact, 1 GPU-coalesced block/lane/output
};
std::vector<ResidentWeight> g_weights;   // handle = index into this registry (process-global)
size_t g_weight_bytes = 0;               // exact live device bytes, exposed for feasibility reporting

// Resident int64 device buffers (gains, cos/sin tables) for the M3 monolith — uploaded once, reused.
struct ResidentBuf { long long* ptr; size_t n; };
std::vector<ResidentBuf> g_buffers;       // handle = index

// Exact-allocation feasibility reservations.  A reservation is deliberately
// split into the same logical allocations as the future graph (state, K, V,
// conv history, scratch arenas) instead of one optimistic monolithic malloc;
// this catches allocator fragmentation after ~1,000 resident weight uploads.
struct GpuReservation { std::vector<void*> ptrs; size_t bytes; bool alive; };
std::vector<GpuReservation> g_reservations;

// Stable all-int64 ctypes ABI for the Qwen3.5 hybrid context.
struct Bonsai35Config {
    long long n_layers,d,dff,H,Hkv,hd,vocab,cap,frac,eps,n_rot;
    long long key_heads,value_heads,state_size,state_frac,inner,conv_k;
    long long gdn_scale,attn_scale,ssm_eps;
    long long soft_min,soft_step,soft_max,exp_min,exp_step;
    long long embed,final_gain,out_head,cos_buf,sin_buf,soft_buf,soft_n,exp_buf,exp_n;
};
struct Bonsai35LayerDesc {
    long long kind,slot,n1,n2,w1,wu,w2;
    long long wqkv,wz,walpha,wbeta,wout,conv,dt_bias,ssm_a,ssm_norm;
    long long wqg,wk,wv,wo,q_norm,k_norm;
};
struct Bonsai35Ctx {
    Bonsai35Config c; std::vector<Bonsai35LayerDesc> layers; bool alive,poisoned;
    long long t,nrec,natt,conv_dim,max_k,graph_launches,input_mode;
    long long token_submissions,embedded_submissions,model_input_host_bytes;
    // Post-layer residual snapshots are a parity/debug facility, not part of
    // production inference.  Keeping this off removes one D2D memcpy graph
    // node for the embedding plus one after every transformer layer.
    bool capture_trace,group_projections;
    long long graph_nodes,graph_kernel_nodes,graph_memcpy_nodes;
    long long graph_memset_nodes,graph_other_nodes;
    long long *state,*conv_hist;
    int *K,*V;  // guarded Q16 cache: every append proves exact int32 narrowing
    long long *x,*norm,*tmp,*qkv,*z,*alpha,*beta,*conv;
    long long *qn,*kn,*rv,*rout,*rnorm,*zs,*rgated,*ctl_beta,*ctl_decay;
    long long *qg,*aq,*agate,*ak,*av,*aout,*agated,*ffg,*ffu,*ffh,*scores,*digits,*logits,*trace;
    long long *pos,*len,*token; unsigned long long *maxk,*maxv; int *overflow;
    long long *h_x,*h_logits,*h_pos,*h_len,*h_token; int *h_overflow;
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t graph_exec; bool graph_ready;
};
std::vector<Bonsai35Ctx> g_bonsai35;

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

// Query allocator-visible memory after forcing CUDA context creation.  This is
// used by the 27B feasibility proof; unlike nvidia-smi it is scoped to the
// active device and includes the context/allocator state seen by this library.
int bonsai_gpu_mem_info(unsigned long long* free_bytes, unsigned long long* total_bytes) {
    if (!free_bytes || !total_bytes) return 1;
    size_t f = 0, t = 0;
    if (cudaFree(0) != cudaSuccess || cudaMemGetInfo(&f, &t) != cudaSuccess) return 2;
    *free_bytes = (unsigned long long)f;
    *total_bytes = (unsigned long long)t;
    return 0;
}

unsigned long long bonsai_q1_resident_weight_bytes(void) {
    return (unsigned long long)g_weight_bytes;
}

// Test/debug ABI for the exact KV-cache storage contract.  Values at both
// int32 endpoints must survive narrowing and widening byte-exactly; any value
// outside that closed interval returns the same fail-loud status used by the
// resident graph before an output is exposed.
int bonsai35_kv_i32_roundtrip_gpu(
        const long long* src_host, long long n, long long* dst_host) {
    if (!src_host || !dst_host || n <= 0) return 1;
    long long *src = nullptr, *dst = nullptr;
    int *cache = nullptr, *overflow = nullptr;
    bool ok = true;
    int host_overflow = 0;
    ok = ok && (cudaMalloc(&src, (size_t)n * sizeof(long long)) == cudaSuccess);
    ok = ok && (cudaMalloc(&cache, (size_t)n * sizeof(int)) == cudaSuccess);
    ok = ok && (cudaMalloc(&dst, (size_t)n * sizeof(long long)) == cudaSuccess);
    ok = ok && (cudaMalloc(&overflow, sizeof(int)) == cudaSuccess);
    if (ok) {
        ok = ok && (cudaMemcpy(src, src_host, (size_t)n * sizeof(long long),
                              cudaMemcpyHostToDevice) == cudaSuccess);
        ok = ok && (cudaMemset(overflow, 0, sizeof(int)) == cudaSuccess);
    }
    if (ok) {
        const int threads = 128;
        const unsigned blocks = (unsigned)((n + threads - 1) / threads);
        q35_narrow_i32_guard_kernel<<<blocks, threads>>>(src, cache, n, overflow);
        ok = ok && (cudaGetLastError() == cudaSuccess);
        ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
        ok = ok && (cudaMemcpy(&host_overflow, overflow, sizeof(int),
                              cudaMemcpyDeviceToHost) == cudaSuccess);
        if (ok && !host_overflow) {
            q35_widen_i32_kernel<<<blocks, threads>>>(cache, dst, n);
            ok = ok && (cudaGetLastError() == cudaSuccess);
            ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
            ok = ok && (cudaMemcpy(dst_host, dst, (size_t)n * sizeof(long long),
                                  cudaMemcpyDeviceToHost) == cudaSuccess);
        }
    }
    cudaFree(src); cudaFree(cache); cudaFree(dst); cudaFree(overflow);
    if (!ok) return 2;
    return host_overflow ? 4 : 0;
}

// Adversarial transaction ABI: initialize two cache rows to caller-provided
// sentinels, run the SAME paired preflight/commit kernels as resident decode,
// and return the final rows even on rc=4.  Tests use a single unsafe lane to
// prove that neither K nor V receives any partial writes.
int bonsai35_kv_i32_transaction_gpu(
        const long long* k_host, const long long* v_host,
        const int* k_initial, const int* v_initial,
        long long n, long long* k_out_host, long long* v_out_host) {
    if(!k_host||!v_host||!k_initial||!v_initial||!k_out_host||!v_out_host||n<=0)return 1;
    long long *k=nullptr,*v=nullptr,*kout=nullptr,*vout=nullptr,*pos=nullptr;
    int *K=nullptr,*V=nullptr,*overflow=nullptr;bool ok=true;int hov=0;const long long zero=0;
    const size_t wide=(size_t)n*8,narrow=(size_t)n*4;const int T=128;
    ok=ok&&(cudaMalloc(&k,wide)==cudaSuccess);ok=ok&&(cudaMalloc(&v,wide)==cudaSuccess);
    ok=ok&&(cudaMalloc(&K,narrow)==cudaSuccess);ok=ok&&(cudaMalloc(&V,narrow)==cudaSuccess);
    ok=ok&&(cudaMalloc(&kout,wide)==cudaSuccess);ok=ok&&(cudaMalloc(&vout,wide)==cudaSuccess);
    ok=ok&&(cudaMalloc(&pos,8)==cudaSuccess);ok=ok&&(cudaMalloc(&overflow,sizeof(int))==cudaSuccess);
    if(ok){
        ok=ok&&(cudaMemcpy(k,k_host,wide,cudaMemcpyHostToDevice)==cudaSuccess);
        ok=ok&&(cudaMemcpy(v,v_host,wide,cudaMemcpyHostToDevice)==cudaSuccess);
        ok=ok&&(cudaMemcpy(K,k_initial,narrow,cudaMemcpyHostToDevice)==cudaSuccess);
        ok=ok&&(cudaMemcpy(V,v_initial,narrow,cudaMemcpyHostToDevice)==cudaSuccess);
        ok=ok&&(cudaMemcpy(pos,&zero,8,cudaMemcpyHostToDevice)==cudaSuccess);
        ok=ok&&(cudaMemset(overflow,0,sizeof(int))==cudaSuccess);
    }
    if(ok){
        const unsigned blocks=(unsigned)((n+T-1)/T);
        q35_kv_i32_preflight_pair_kernel<<<blocks,T>>>(k,v,n,overflow);
        q35_kv_i32_commit_pair_kernel<<<blocks,T>>>(k,v,K,V,pos,1,n,1,overflow);
        q35_widen_i32_kernel<<<blocks,T>>>(K,kout,n);
        q35_widen_i32_kernel<<<blocks,T>>>(V,vout,n);
        ok=ok&&(cudaGetLastError()==cudaSuccess);ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);
        ok=ok&&(cudaMemcpy(&hov,overflow,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
        ok=ok&&(cudaMemcpy(k_out_host,kout,wide,cudaMemcpyDeviceToHost)==cudaSuccess);
        ok=ok&&(cudaMemcpy(v_out_host,vout,wide,cudaMemcpyDeviceToHost)==cudaSuccess);
    }
    cudaFree(k);cudaFree(v);cudaFree(K);cudaFree(V);cudaFree(kout);cudaFree(vout);
    cudaFree(pos);cudaFree(overflow);if(!ok)return 2;return hov?4:0;
}

// Atomically reserve a list of device allocations.  On any failure all
// allocations made by this call are freed before returning -1, so callers can
// cleanly fall back to CPU without poisoning the long-lived process.
long long bonsai_gpu_reservation_create(const unsigned long long* sizes, long long count,
                                        unsigned long long* allocated_bytes) {
    if (!sizes || count <= 0 || count > 1024) return -1;
    GpuReservation r; r.bytes = 0; r.alive = false;
    r.ptrs.reserve((size_t)count);
    for (long long i = 0; i < count; ++i) {
        const size_t size_max = ~(size_t)0;
        if (sizes[i] == 0 || sizes[i] > (unsigned long long)size_max ||
            r.bytes > size_max - (size_t)sizes[i]) {
            for (void* p : r.ptrs) cudaFree(p);
            return -1;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, (size_t)sizes[i]) != cudaSuccess) {
            // Clear CUDA's sticky allocation error before returning control.
            cudaGetLastError();
            for (void* q : r.ptrs) cudaFree(q);
            return -1;
        }
        r.ptrs.push_back(p);
        r.bytes += (size_t)sizes[i];
    }
    r.alive = true;
    g_reservations.push_back(std::move(r));
    const long long h = (long long)g_reservations.size() - 1;
    if (allocated_bytes) *allocated_bytes = (unsigned long long)g_reservations[(size_t)h].bytes;
    return h;
}

void bonsai_gpu_reservation_free(long long handle) {
    if (handle < 0 || (size_t)handle >= g_reservations.size()) return;
    GpuReservation& r = g_reservations[(size_t)handle];
    if (!r.alive) return;
    for (void* p : r.ptrs) cudaFree(p);
    r.ptrs.clear(); r.bytes = 0; r.alive = false;
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
    int* d_ovf=nullptr; int h_ovf=0;
    bool ok=true; int rc=1;
    ok=ok&&(cudaMalloc(&dx,xsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&db,bsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&ds,ssz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dout,osz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&dd,dsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&d_ovf,sizeof(int))==cudaSuccess);
    ok=ok&&(cudaMemcpy(dx,x,xsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(db,bits,bsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(ds,scale,ssz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemset(d_ovf,0,sizeof(int))==cudaSuccess);
    if (ok) {
        const int TPB=128;
        // L=4 envelope guard (was defined but never launched → a silent-wrap hazard for an out-of-envelope
        // L=4). Run it on the raw activations; if ANY |x| leaves the balanced base-256 L=4 range the flag is
        // set and we return rc 4 (CPU fallback) instead of computing non-byte-exact digits. Never fires for
        // committed models (|x|~2^25 « 2.14e9); L=8 is always exact so it is skipped.
        if (LL == 4) {
            range_guard_l4_kernel<<<(unsigned)((tokens*K+TPB-1)/TPB),TPB>>>(dx,tokens*K,d_ovf);
            ok=ok&&(cudaGetLastError()==cudaSuccess);
            ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);
            ok=ok&&(cudaMemcpy(&h_ovf,d_ovf,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
        }
    }
    if (ok && !h_ovf) {
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
    if (ok && !h_ovf) ok=ok&&(cudaMemcpy(out,dout,osz,cudaMemcpyDeviceToHost)==cudaSuccess);
    if (h_ovf) rc=4;              // out of L=4 envelope: caller falls back to the always-exact path (no wrap)
    else if (ok) rc=0; else rc=2;
    if (dx) cudaFree(dx); if (db) cudaFree(db); if (ds) cudaFree(ds);
    if (dout) cudaFree(dout); if (dd) cudaFree(dd); if (d_ovf) cudaFree(d_ovf);
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
    g_weights.push_back(ResidentWeight{dbits, dscale, nullptr, out_f, n_blocks, 64, 0});
    g_weight_bytes += bsz + ssz;
    return (long long)(g_weights.size() - 1);
}

// Upload one projection while preserving its committed int32 scale storage.
// The Qwen3.5 importer rejects any scale that cannot be narrowed losslessly;
// this ABI therefore performs no saturation or reinterpretation.
long long bonsai_q1_register_weight_i32(const unsigned char* bits, const int* scale,
                                        long long out_f, long long n_blocks) {
    if (!bits || !scale || out_f <= 0 || n_blocks <= 0) return -1;
    const size_t bsz = (size_t)out_f * (size_t)n_blocks * 16;
    const size_t ssz = (size_t)out_f * (size_t)n_blocks * sizeof(int);
    unsigned char* dbits = nullptr;
    int* dscale = nullptr;
    if (cudaMalloc(&dbits, bsz) != cudaSuccess) return -1;
    if (cudaMalloc(&dscale, ssz) != cudaSuccess) { cudaFree(dbits); return -1; }
    if (cudaMemcpy(dbits, bits, bsz, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dscale, scale, ssz, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dbits); cudaFree(dscale); return -1;
    }
    g_weights.push_back(ResidentWeight{dbits, nullptr, dscale, out_f, n_blocks, 32, 0});
    g_weight_bytes += bsz + ssz;
    return (long long)(g_weights.size() - 1);
}

// Register int32-scale Q1 in a runtime-only coalesced CUDA layout.  A temporary
// upload is transposed on device and freed before return, so resident bytes are
// unchanged and the committed artifact/digest remain untouched.
long long bonsai_q1_register_weight_i32_gpu_layout(const unsigned char* bits,const int* scale,
                                                   long long out_f,long long n_blocks){
    if(!bits||!scale||out_f<=0||n_blocks<=0)return -1;
    const size_t bsz=(size_t)out_f*n_blocks*16,ssz=(size_t)out_f*n_blocks*4;
    unsigned char *srcb=nullptr,*dstb=nullptr;int *srcs=nullptr,*dsts=nullptr;bool ok=true;
    ok=ok&&(cudaMalloc(&srcb,bsz)==cudaSuccess);ok=ok&&(cudaMalloc(&dstb,bsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&srcs,ssz)==cudaSuccess);ok=ok&&(cudaMalloc(&dsts,ssz)==cudaSuccess);
    ok=ok&&(cudaMemcpy(srcb,bits,bsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(srcs,scale,ssz,cudaMemcpyHostToDevice)==cudaSuccess);
    if(ok){const int T=128;q1_transpose_bits_kernel<<<(unsigned)((bsz+T-1)/T),T>>>(srcb,dstb,out_f,n_blocks);
        q1_transpose_scale32_kernel<<<(unsigned)(((size_t)out_f*n_blocks+T-1)/T),T>>>(srcs,dsts,out_f,n_blocks);
        ok=ok&&(cudaGetLastError()==cudaSuccess);ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);}
    if(srcb)cudaFree(srcb);if(srcs)cudaFree(srcs);
    if(!ok){if(dstb)cudaFree(dstb);if(dsts)cudaFree(dsts);cudaGetLastError();return -1;}
    g_weights.push_back(ResidentWeight{dstb,nullptr,dsts,out_f,n_blocks,32,1});g_weight_bytes+=bsz+ssz;
    return (long long)g_weights.size()-1;
}

long long bonsai_q1_register_weight_i32_bmma(const unsigned char* bits,const int* scale,
                                             long long out_f,long long n_blocks){
    if(!bits||!scale||out_f<=0||n_blocks<=0||out_f%8)return -1;
    const size_t bsz=(size_t)out_f*n_blocks*16,ssz=(size_t)out_f*n_blocks*4;
    unsigned char *srcb=nullptr,*dstb=nullptr;int *srcs=nullptr,*dsts=nullptr;bool ok=true;
    ok=ok&&(cudaMalloc(&srcb,bsz)==cudaSuccess);ok=ok&&(cudaMalloc(&dstb,bsz)==cudaSuccess);
    ok=ok&&(cudaMalloc(&srcs,ssz)==cudaSuccess);ok=ok&&(cudaMalloc(&dsts,ssz)==cudaSuccess);
    ok=ok&&(cudaMemcpy(srcb,bits,bsz,cudaMemcpyHostToDevice)==cudaSuccess);
    ok=ok&&(cudaMemcpy(srcs,scale,ssz,cudaMemcpyHostToDevice)==cudaSuccess);
    if(ok){const int T=128;q1_repack_bits_bmma_kernel<<<(unsigned)((bsz+T-1)/T),T>>>(srcb,dstb,out_f,n_blocks);
        q1_transpose_scale32_kernel<<<(unsigned)(((size_t)out_f*n_blocks+T-1)/T),T>>>(srcs,dsts,out_f,n_blocks);
        ok=ok&&(cudaGetLastError()==cudaSuccess);ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);}
    if(srcb)cudaFree(srcb);if(srcs)cudaFree(srcs);
    if(!ok){if(dstb)cudaFree(dstb);if(dsts)cudaFree(dsts);cudaGetLastError();return -1;}
    g_weights.push_back(ResidentWeight{dstb,nullptr,dsts,out_f,n_blocks,32,2});g_weight_bytes+=bsz+ssz;
    return (long long)g_weights.size()-1;
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
        if (w.scale_bits == 32 && w.layout == 1)
            q1_linear_scale32_transposed_kernel<<<(unsigned int)nblk, threads>>>(
                dx,w.dbits,w.dscale32,tokens,out_f,n_blocks,frac,dout);
        else if (w.scale_bits == 32)
            q1_linear_scale32_kernel<<<(unsigned int)nblk, threads>>>(
                dx, w.dbits, w.dscale32, tokens, out_f, n_blocks, frac, dout);
        else
            q1_linear_kernel<<<(unsigned int)nblk, threads>>>(
                dx, w.dbits, w.dscale, tokens, out_f, n_blocks, frac, dout);
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

// Standalone Qwen3.5 M=1 Gated DeltaNet parity rung.  All graph values remain
// integer; state is copied back only because this ABI is for oracle comparison.
// The resident executor invokes the same kernels against persistent state.
int bonsai35_recurrent_step_gpu(
        const long long* q, const long long* k, const long long* v,
        const long long* z, const long long* alpha, const long long* beta,
        long long* state_host, const long long* dt_bias, const long long* ssm_a,
        const long long* norm_gain, const long long* soft_lut, long long soft_n,
        const long long* exp_lut, long long exp_n,
        long long key_heads, long long value_heads, long long state_size,
        long long frac, long long state_frac, long long soft_min,
        long long soft_step, long long soft_max, long long exp_min,
        long long exp_step, long long gdn_scale, long long ssm_eps,
        long long* gated_host) {
    if (!q || !k || !v || !z || !alpha || !beta || !state_host || !dt_bias || !ssm_a ||
        !norm_gain || !soft_lut || !exp_lut || !gated_host || key_heads <= 0 ||
        value_heads <= 0 || state_size <= 0 || state_size > 128 || soft_n < 2 || exp_n < 2 ||
        frac < 1 || frac > 29 || state_frac < frac || 2 * frac < state_frac ||
        soft_step <= 0 || exp_step <= 0) return 1;
    const size_t K = (size_t)key_heads * state_size;
    const size_t V = (size_t)value_heads * state_size;
    const size_t S = (size_t)value_heads * state_size * state_size;
    long long *dq=nullptr,*dk=nullptr,*dv=nullptr,*dz=nullptr,*da=nullptr,*db=nullptr;
    long long *dstate=nullptr,*ddt=nullptr,*dA=nullptr,*dng=nullptr,*dsl=nullptr,*del=nullptr;
    long long *dqn=nullptr,*dkn=nullptr,*dbeta=nullptr,*ddecay=nullptr,*dout=nullptr,*dnorm=nullptr,*dzs=nullptr,*dgated=nullptr;
    int* dov=nullptr;
    bool ok = true; int hov = 0;
    #define MALLOC(P,N) do { ok = ok && (cudaMalloc(&(P),(N)) == cudaSuccess); } while (0)
    #define H2D(P,H,N) do { ok = ok && (cudaMemcpy((P),(H),(N),cudaMemcpyHostToDevice) == cudaSuccess); } while (0)
    MALLOC(dq,K*8); MALLOC(dk,K*8); MALLOC(dv,V*8); MALLOC(dz,V*8);
    MALLOC(da,(size_t)value_heads*8); MALLOC(db,(size_t)value_heads*8); MALLOC(dstate,S*8);
    MALLOC(ddt,(size_t)value_heads*8); MALLOC(dA,(size_t)value_heads*8); MALLOC(dng,(size_t)state_size*8);
    MALLOC(dsl,(size_t)soft_n*8); MALLOC(del,(size_t)exp_n*8); MALLOC(dqn,K*8); MALLOC(dkn,K*8);
    MALLOC(dbeta,(size_t)value_heads*8); MALLOC(ddecay,(size_t)value_heads*8);
    MALLOC(dout,V*8); MALLOC(dnorm,V*8); MALLOC(dzs,V*8); MALLOC(dgated,V*8); MALLOC(dov,sizeof(int));
    if (ok) {
        H2D(dq,q,K*8); H2D(dk,k,K*8); H2D(dv,v,V*8); H2D(dz,z,V*8);
        H2D(da,alpha,(size_t)value_heads*8); H2D(db,beta,(size_t)value_heads*8); H2D(dstate,state_host,S*8);
        H2D(ddt,dt_bias,(size_t)value_heads*8); H2D(dA,ssm_a,(size_t)value_heads*8);
        H2D(dng,norm_gain,(size_t)state_size*8); H2D(dsl,soft_lut,(size_t)soft_n*8);
        H2D(del,exp_lut,(size_t)exp_n*8); ok=ok&&(cudaMemset(dov,0,sizeof(int))==cudaSuccess);
    }
    const long long LOG2E_Q16=94548; const long long sh=16-frac;
    const long long log2e=sh>=0?(LOG2E_Q16>>sh):(LOG2E_Q16<<(-sh));
    const long long dca=((frac+2)<<(2*frac))/log2e, dcb=((long long)1<<62)/log2e;
    const long long dclip=dca<dcb?dca:dcb;
    if (ok) {
        const int T=64;
        bonsai35_l2norm_kernel<<<(unsigned)((key_heads+T-1)/T),T>>>(dq,key_heads,state_size,frac,dqn,dov);
        bonsai35_l2norm_kernel<<<(unsigned)((key_heads+T-1)/T),T>>>(dk,key_heads,state_size,frac,dkn,dov);
        bonsai35_controls_kernel<<<(unsigned)((value_heads+T-1)/T),T>>>(
            da,db,ddt,dA,dsl,soft_n,del,exp_n,value_heads,frac,log2e,dclip,
            soft_min,soft_step,soft_max,exp_min,exp_step,dbeta,ddecay,dov);
        bonsai35_state_step_kernel<<<(unsigned)value_heads,128>>>(
            dqn,dkn,dv,dbeta,ddecay,dstate,value_heads,key_heads,state_size,
            frac,state_frac,gdn_scale,dout);
        rmsnorm_kernel<<<(unsigned)((value_heads+T-1)/T),T>>>(
            dout,value_heads,state_size,frac,ssm_eps,dng,dnorm,nullptr,dov);
        silu_kernel<<<(unsigned)((V+T-1)/T),T>>>(dz,dzs,(long long)V,frac,log2e,dclip);
        mulshift_kernel<<<(unsigned)((V+T-1)/T),T>>>(dnorm,dzs,dgated,(long long)V,frac);
        ok=ok&&(cudaGetLastError()==cudaSuccess); ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);
        ok=ok&&(cudaMemcpy(&hov,dov,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
        if (ok && !hov) {
            ok=ok&&(cudaMemcpy(state_host,dstate,S*8,cudaMemcpyDeviceToHost)==cudaSuccess);
            ok=ok&&(cudaMemcpy(gated_host,dgated,V*8,cudaMemcpyDeviceToHost)==cudaSuccess);
        }
    }
    #undef MALLOC
    #undef H2D
    cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dz);cudaFree(da);cudaFree(db);
    cudaFree(dstate);cudaFree(ddt);cudaFree(dA);cudaFree(dng);cudaFree(dsl);cudaFree(del);
    cudaFree(dqn);cudaFree(dkn);cudaFree(dbeta);cudaFree(ddecay);cudaFree(dout);cudaFree(dnorm);
    cudaFree(dzs);cudaFree(dgated);cudaFree(dov);
    if (!ok) return 2;
    return hov ? 4 : 0;
}

// Standalone Qwen3.5 gated full-attention M=1 parity rung.  K/V prefix inputs
// are read-only; only the transformed new K row and gated attention output are
// returned, so a failed GPU attempt cannot partially mutate a caller cache.
int bonsai35_attention_decode_gpu(
        const long long* qg, const long long* k_new, const long long* v_new,
        const long long* k_prefix, const long long* v_prefix, long long prefix_len,
        long long H, long long Hkv, long long hd, long long n_rot,
        long long frac, long long eps, long long inv_sqrt,
        const long long* q_gain, const long long* k_gain,
        const long long* cos, const long long* sin,
        long long* gated_host, long long* k_row_host) {
    if (!qg || !k_new || !v_new || !q_gain || !k_gain || !cos || !sin ||
        !gated_host || !k_row_host || prefix_len < 0 || H <= 0 || Hkv <= 0 ||
        H % Hkv || hd <= 0 || n_rot <= 0 || n_rot > hd || n_rot % 2 ||
        frac < 1 || frac > 29 || eps < 0 || (prefix_len && (!k_prefix || !v_prefix))) return 1;
    const long long L = prefix_len + 1;
    const size_t Q=(size_t)H*hd, KV=(size_t)Hkv*hd, CACHE=(size_t)Hkv*L*hd;
    long long *dqg=nullptr,*dq=nullptr,*dgate=nullptr,*dk=nullptr,*dv=nullptr,*dK=nullptr,*dV=nullptr;
    long long *dqgains=nullptr,*dkgains=nullptr,*dc=nullptr,*ds=nullptr,*dout=nullptr,*dgated=nullptr;
    long long *dscores=nullptr,*dlen=nullptr,*dpos=nullptr;
    unsigned long long *dmk=nullptr,*dmv=nullptr; int* dov=nullptr;
    bool ok=true; int hov=0;
    #define MALLOC2(P,N) do { ok=ok&&(cudaMalloc(&(P),(N))==cudaSuccess); } while(0)
    #define H2D2(P,S,N) do { ok=ok&&(cudaMemcpy((P),(S),(N),cudaMemcpyHostToDevice)==cudaSuccess); } while(0)
    MALLOC2(dqg,Q*2*8);MALLOC2(dq,Q*8);MALLOC2(dgate,Q*8);MALLOC2(dk,KV*8);MALLOC2(dv,KV*8);
    MALLOC2(dK,CACHE*8);MALLOC2(dV,CACHE*8);MALLOC2(dqgains,(size_t)hd*8);MALLOC2(dkgains,(size_t)hd*8);
    MALLOC2(dc,(size_t)(n_rot/2)*8);MALLOC2(ds,(size_t)(n_rot/2)*8);MALLOC2(dout,Q*8);MALLOC2(dgated,Q*8);
    MALLOC2(dscores,(size_t)H*L*8);MALLOC2(dlen,8);MALLOC2(dpos,8);
    MALLOC2(dmk,(size_t)Hkv*8);MALLOC2(dmv,(size_t)Hkv*8);MALLOC2(dov,sizeof(int));
    if (ok) {
        H2D2(dqg,qg,Q*2*8);H2D2(dk,k_new,KV*8);H2D2(dv,v_new,KV*8);
        H2D2(dqgains,q_gain,(size_t)hd*8);H2D2(dkgains,k_gain,(size_t)hd*8);
        H2D2(dc,cos,(size_t)(n_rot/2)*8);H2D2(ds,sin,(size_t)(n_rot/2)*8);
        H2D2(dlen,&L,8);H2D2(dpos,&prefix_len,8);ok=ok&&(cudaMemset(dov,0,sizeof(int))==cudaSuccess);
        if (prefix_len) {
            // Prefix rows are contiguous per KV head; the destination has one
            // extra row, so copy head-by-head rather than as one flat block.
            for (long long h=0; ok && h<Hkv; ++h) {
                ok=ok&&(cudaMemcpy(dK+(size_t)h*L*hd,k_prefix+(size_t)h*prefix_len*hd,
                                   (size_t)prefix_len*hd*8,cudaMemcpyHostToDevice)==cudaSuccess);
                ok=ok&&(cudaMemcpy(dV+(size_t)h*L*hd,v_prefix+(size_t)h*prefix_len*hd,
                                   (size_t)prefix_len*hd*8,cudaMemcpyHostToDevice)==cudaSuccess);
            }
        }
    }
    const long long LOG2E_Q16=94548, sh=16-frac;
    const long long log2e=sh>=0?(LOG2E_Q16>>sh):(LOG2E_Q16<<(-sh));
    const long long dca=((frac+2)<<(2*frac))/log2e,dcb=((long long)1<<62)/log2e;
    const long long dclip=dca<dcb?dca:dcb;
    if (ok) {
        const int T=64;
        bonsai35_split_qgate_kernel<<<(unsigned)((Q+T-1)/T),T>>>(dqg,dq,dgate,H,hd);
        rmsnorm_kernel<<<(unsigned)((H+T-1)/T),T>>>(dq,H,hd,frac,eps,dqgains,dq,nullptr,dov);
        rmsnorm_kernel<<<(unsigned)((Hkv+T-1)/T),T>>>(dk,Hkv,hd,frac,eps,dkgains,dk,nullptr,dov);
        bonsai35_partial_rope_kernel<<<(unsigned)((H+T-1)/T),T>>>(dq,dc,ds,H,hd,n_rot,frac);
        bonsai35_partial_rope_kernel<<<(unsigned)((Hkv+T-1)/T),T>>>(dk,dc,ds,Hkv,hd,n_rot,frac);
        kv_append_kernel<<<(unsigned)((KV+T-1)/T),T>>>(dk,dK,dpos,1,Hkv,hd,L);
        kv_append_kernel<<<(unsigned)((KV+T-1)/T),T>>>(dv,dV,dpos,1,Hkv,hd,L);
        maxabs_bkv_parallel_kernel<<<(unsigned)Hkv,Q35_ATTN_TPB>>>(dK,dlen,1,Hkv,hd,L,dmk);
        maxabs_bkv_parallel_kernel<<<(unsigned)Hkv,Q35_ATTN_TPB>>>(dV,dlen,1,Hkv,hd,L,dmv);
        attention_decode_m1_parallel_kernel<long long><<<(unsigned)H,Q35_ATTN_TPB>>>(
            dq,dK,dV,dlen,1,H,Hkv,hd,L,frac,inv_sqrt,log2e,dclip,dmk,dmv,dout,dscores,dov);
        bonsai35_sigmoid_gate_kernel<<<(unsigned)((Q+T-1)/T),T>>>(
            dout,dgate,dgated,(long long)Q,frac,log2e,dclip);
        ok=ok&&(cudaGetLastError()==cudaSuccess);ok=ok&&(cudaDeviceSynchronize()==cudaSuccess);
        ok=ok&&(cudaMemcpy(&hov,dov,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
        if (ok&&!hov) {
            ok=ok&&(cudaMemcpy(gated_host,dgated,Q*8,cudaMemcpyDeviceToHost)==cudaSuccess);
            ok=ok&&(cudaMemcpy(k_row_host,dk,KV*8,cudaMemcpyDeviceToHost)==cudaSuccess);
        }
    }
    #undef MALLOC2
    #undef H2D2
    cudaFree(dqg);cudaFree(dq);cudaFree(dgate);cudaFree(dk);cudaFree(dv);cudaFree(dK);cudaFree(dV);
    cudaFree(dqgains);cudaFree(dkgains);cudaFree(dc);cudaFree(ds);cudaFree(dout);cudaFree(dgated);
    cudaFree(dscores);cudaFree(dlen);cudaFree(dpos);cudaFree(dmk);cudaFree(dmv);cudaFree(dov);
    if(!ok)return 2;return hov?4:0;
}

// Grid helpers shared by the monolith + the batched-decode step (one-thread-per-item vs one-warp-per-output).
static const int MONO_TPB = 64;
static inline unsigned int mono_blocks(long long nthreads) { return (unsigned int)((nthreads + MONO_TPB - 1) / MONO_TPB); }
static inline unsigned int mono_wblocks(long long nwarps) { return (unsigned int)((nwarps * 32 + MONO_TPB - 1) / MONO_TPB); }

static inline bool q35_weight_ok(long long h) {
    return h >= 0 && (size_t)h < g_weights.size() && g_weights[(size_t)h].dbits;
}
static inline bool q35_buf_ok(long long h) {
    return h >= 0 && (size_t)h < g_buffers.size() && g_buffers[(size_t)h].ptr;
}
static inline long long* q35_buf(long long h) { return g_buffers[(size_t)h].ptr; }

static void q35_prepare_digits(Bonsai35Ctx& c, const long long* x, long long K) {
    const long long words=(K/128)*32*4;
    q1_bmma_activation_kernel<<<mono_blocks(words),MONO_TPB,0,c.stream>>>(
        x,K/128,reinterpret_cast<unsigned int*>(c.digits),c.overflow);
}
static void q35_apply_prepared(Bonsai35Ctx& c, long long wh, long long* out) {
    const ResidentWeight& w=g_weights[(size_t)wh];
    const unsigned blocks=(unsigned)((w.out_f/8+3)/4); // four warps/block
    if(w.scale_bits==32 && w.layout==2)
        q1_bmma_apply_scale32_kernel<<<blocks,128,0,c.stream>>>(
            reinterpret_cast<unsigned int*>(c.digits),w.dbits,w.dscale32,w.out_f,w.n_blocks,c.c.frac,out);
    else if(w.scale_bits==32 && w.layout==1)
        q1_lut32_apply_scale32_transposed_x4_kernel<<<blocks,MONO_TPB,0,c.stream>>>(
            reinterpret_cast<int*>(c.digits),w.dbits,w.dscale32,w.out_f,w.n_blocks,c.c.frac,out);
    else if(w.scale_bits==32)
        q1_lut_apply_scale32_kernel<<<blocks,MONO_TPB,0,c.stream>>>(
            c.digits,w.dbits,w.dscale32,w.out_f,w.n_blocks,c.c.frac,out);
    else
        q1_lut_apply_scale64_kernel<<<blocks,MONO_TPB,0,c.stream>>>(
            c.digits,w.dbits,w.dscale,w.out_f,w.n_blocks,c.c.frac,out);
}

static void q35_apply_prepared_group4(
        Bonsai35Ctx& c,
        long long wh0,long long* out0,long long wh1,long long* out1,
        long long wh2,long long* out2,long long wh3,long long* out3) {
    const ResidentWeight& w0=g_weights[(size_t)wh0];
    const ResidentWeight& w1=wh1>=0?g_weights[(size_t)wh1]:w0;
    const ResidentWeight& w2=wh2>=0?g_weights[(size_t)wh2]:w0;
    const ResidentWeight& w3=wh3>=0?g_weights[(size_t)wh3]:w0;
    const long long f1=wh1>=0?w1.out_f:0,f2=wh2>=0?w2.out_f:0,f3=wh3>=0?w3.out_f:0;
    const long long tiles=(w0.out_f+f1+f2+f3)/8;
    const unsigned blocks=(unsigned)((tiles+3)/4); // four warps/block
    q1_bmma_apply_scale32_group4_kernel<<<blocks,128,0,c.stream>>>(
        reinterpret_cast<unsigned int*>(c.digits),
        w0.dbits,w0.dscale32,w0.out_f,out0,
        w1.dbits,w1.dscale32,f1,out1?out1:out0,
        w2.dbits,w2.dscale32,f2,out2?out2:out0,
        w3.dbits,w3.dscale32,f3,out3?out3:out0,
        w0.n_blocks,c.c.frac);
}

static bool q35_enqueue_decode(Bonsai35Ctx& c, bool token_input) {
    const Bonsai35Config& g=c.c; const long long T=MONO_TPB;
    const long long LOG2E_Q16=94548,sh=16-g.frac;
    const long long log2e=sh>=0?(LOG2E_Q16>>sh):(LOG2E_Q16<<(-sh));
    const long long dca=((g.frac+2)<<(2*g.frac))/log2e,dcb=((long long)1<<62)/log2e;
    const long long dclip=dca<dcb?dca:dcb;
    cudaMemsetAsync(c.overflow,0,sizeof(int),c.stream);
    if(token_input){
        const ResidentWeight& embed=g_weights[(size_t)g.embed];
        cudaMemcpyAsync(c.token,c.h_token,8,cudaMemcpyHostToDevice,c.stream);
        q35_embedding_row_bmma_kernel<<<mono_blocks(g.d),T,0,c.stream>>>(
            embed.dbits,embed.dscale32,embed.out_f,embed.n_blocks,c.token,c.x,c.overflow);
        if(c.capture_trace)
            cudaMemcpyAsync(c.trace,c.x,(size_t)g.d*8,cudaMemcpyDeviceToDevice,c.stream);
    }else{
        cudaMemcpyAsync(c.x,c.h_x,(size_t)g.d*8,cudaMemcpyHostToDevice,c.stream);
        if(c.capture_trace)
            cudaMemcpyAsync(c.trace,c.h_x,(size_t)g.d*8,cudaMemcpyHostToDevice,c.stream);
    }
    cudaMemcpyAsync(c.pos,c.h_pos,8,cudaMemcpyHostToDevice,c.stream);
    cudaMemcpyAsync(c.len,c.h_len,8,cudaMemcpyHostToDevice,c.stream);
    for(long long li=0;li<g.n_layers;++li){
        const Bonsai35LayerDesc& l=c.layers[(size_t)li];
        rmsnorm_fast_i32_kernel<<<1,256,0,c.stream>>>(c.x,1,g.d,g.frac,g.eps,q35_buf(l.n1),c.norm,c.overflow);
        if(l.kind==0){ // recurrent
            q35_prepare_digits(c,c.norm,g.d);
            if(c.group_projections)
                q35_apply_prepared_group4(c,l.wqkv,c.qkv,l.wz,c.z,l.walpha,c.alpha,l.wbeta,c.beta);
            else{
                q35_apply_prepared(c,l.wqkv,c.qkv);q35_apply_prepared(c,l.wz,c.z);
                q35_apply_prepared(c,l.walpha,c.alpha);q35_apply_prepared(c,l.wbeta,c.beta);
            }
            bonsai35_conv_decode_kernel<<<mono_blocks(c.conv_dim),T,0,c.stream>>>(
                c.qkv,c.conv_hist,q35_buf(l.conv),l.slot,c.conv_dim,g.conv_k,g.frac,c.conv);
            silu_kernel<<<mono_blocks(c.conv_dim),T,0,c.stream>>>(
                c.conv,c.conv,c.conv_dim,g.frac,log2e,dclip);
            const long long key_width=g.key_heads*g.state_size;
            bonsai35_l2norm_kernel<<<mono_blocks(g.key_heads),T,0,c.stream>>>(
                c.conv,g.key_heads,g.state_size,g.frac,c.qn,c.overflow);
            bonsai35_l2norm_kernel<<<mono_blocks(g.key_heads),T,0,c.stream>>>(
                c.conv+key_width,g.key_heads,g.state_size,g.frac,c.kn,c.overflow);
            cudaMemcpyAsync(c.rv,c.conv+2*key_width,(size_t)g.value_heads*g.state_size*8,
                            cudaMemcpyDeviceToDevice,c.stream);
            bonsai35_controls_kernel<<<mono_blocks(g.value_heads),T,0,c.stream>>>(
                c.alpha,c.beta,q35_buf(l.dt_bias),q35_buf(l.ssm_a),q35_buf(g.soft_buf),g.soft_n,
                q35_buf(g.exp_buf),g.exp_n,g.value_heads,g.frac,log2e,dclip,
                g.soft_min,g.soft_step,g.soft_max,g.exp_min,g.exp_step,c.ctl_beta,c.ctl_decay,c.overflow);
            long long* st=c.state+(size_t)l.slot*g.value_heads*g.state_size*g.state_size;
            bonsai35_state_step_wide32_kernel<<<(unsigned)g.value_heads,128,0,c.stream>>>(
                c.qn,c.kn,c.rv,c.ctl_beta,c.ctl_decay,st,g.value_heads,g.key_heads,g.state_size,
                g.frac,g.state_frac,g.gdn_scale,c.rout,c.overflow);
            rmsnorm_fast_i32_kernel<<<(unsigned)g.value_heads,256,0,c.stream>>>(
                c.rout,g.value_heads,g.state_size,g.frac,g.ssm_eps,q35_buf(l.ssm_norm),
                c.rnorm,c.overflow);
            silu_kernel<<<mono_blocks(g.inner),T,0,c.stream>>>(c.z,c.zs,g.inner,g.frac,log2e,dclip);
            mulshift_kernel<<<mono_blocks(g.inner),T,0,c.stream>>>(c.rnorm,c.zs,c.rgated,g.inner,g.frac);
            q35_prepare_digits(c,c.rgated,g.inner);q35_apply_prepared(c,l.wout,c.tmp);
        }else{ // full attention
            q35_prepare_digits(c,c.norm,g.d);
            if(c.group_projections)
                q35_apply_prepared_group4(c,l.wqg,c.qg,l.wk,c.ak,l.wv,c.av,-1,nullptr);
            else{
                q35_apply_prepared(c,l.wqg,c.qg);q35_apply_prepared(c,l.wk,c.ak);q35_apply_prepared(c,l.wv,c.av);
            }
            bonsai35_split_qgate_kernel<<<mono_blocks(g.H*g.hd),T,0,c.stream>>>(c.qg,c.aq,c.agate,g.H,g.hd);
            rmsnorm_fast_i32_kernel<<<(unsigned)g.H,256,0,c.stream>>>(
                c.aq,g.H,g.hd,g.frac,g.eps,q35_buf(l.q_norm),c.aq,c.overflow);
            rmsnorm_fast_i32_kernel<<<(unsigned)g.Hkv,256,0,c.stream>>>(
                c.ak,g.Hkv,g.hd,g.frac,g.eps,q35_buf(l.k_norm),c.ak,c.overflow);
            bonsai35_partial_rope_pos_kernel<<<mono_blocks(g.H),T,0,c.stream>>>(
                c.aq,c.pos,q35_buf(g.cos_buf),q35_buf(g.sin_buf),g.H,g.hd,g.n_rot,g.frac);
            bonsai35_partial_rope_pos_kernel<<<mono_blocks(g.Hkv),T,0,c.stream>>>(
                c.ak,c.pos,q35_buf(g.cos_buf),q35_buf(g.sin_buf),g.Hkv,g.hd,g.n_rot,g.frac);
            int* Kl=c.K+(size_t)l.slot*g.Hkv*g.cap*g.hd;
            int* Vl=c.V+(size_t)l.slot*g.Hkv*g.cap*g.hd;
            unsigned long long* maxKl=c.maxk+(size_t)l.slot*g.Hkv;
            unsigned long long* maxVl=c.maxv+(size_t)l.slot*g.Hkv;
            q35_kv_i32_preflight_pair_kernel<<<mono_blocks(g.Hkv*g.hd),T,0,c.stream>>>(
                c.ak,c.av,g.Hkv*g.hd,c.overflow);
            q35_kv_i32_commit_pair_kernel<<<mono_blocks(g.Hkv*g.hd),T,0,c.stream>>>(
                c.ak,c.av,Kl,Vl,c.pos,g.Hkv,g.hd,g.cap,c.overflow);
            q35_update_maxabs_rows_kernel<<<(unsigned)g.Hkv,Q35_ATTN_TPB,0,c.stream>>>(
                c.ak,g.Hkv,g.hd,maxKl,c.overflow);
            q35_update_maxabs_rows_kernel<<<(unsigned)g.Hkv,Q35_ATTN_TPB,0,c.stream>>>(
                c.av,g.Hkv,g.hd,maxVl,c.overflow);
            attention_decode_m1_parallel_kernel<int><<<(unsigned)g.H,Q35_ATTN_TPB,0,c.stream>>>(
                c.aq,Kl,Vl,c.len,1,g.H,g.Hkv,g.hd,g.cap,g.frac,g.attn_scale,log2e,dclip,
                maxKl,maxVl,c.aout,c.scores,c.overflow);
            bonsai35_sigmoid_gate_kernel<<<mono_blocks(g.H*g.hd),T,0,c.stream>>>(
                c.aout,c.agate,c.agated,g.H*g.hd,g.frac,log2e,dclip);
            q35_prepare_digits(c,c.agated,g.H*g.hd);q35_apply_prepared(c,l.wo,c.tmp);
        }
        add_kernel<<<mono_blocks(g.d),T,0,c.stream>>>(c.x,c.tmp,c.x,g.d);
        rmsnorm_fast_i32_kernel<<<1,256,0,c.stream>>>(c.x,1,g.d,g.frac,g.eps,q35_buf(l.n2),c.norm,c.overflow);
        q35_prepare_digits(c,c.norm,g.d);
        if(c.group_projections)
            q35_apply_prepared_group4(c,l.w1,c.ffg,l.wu,c.ffu,-1,nullptr,-1,nullptr);
        else{q35_apply_prepared(c,l.w1,c.ffg);q35_apply_prepared(c,l.wu,c.ffu);}
        silu_kernel<<<mono_blocks(g.dff),T,0,c.stream>>>(c.ffg,c.ffg,g.dff,g.frac,log2e,dclip);
        mulshift_kernel<<<mono_blocks(g.dff),T,0,c.stream>>>(c.ffg,c.ffu,c.ffh,g.dff,g.frac);
        q35_prepare_digits(c,c.ffh,g.dff);q35_apply_prepared(c,l.w2,c.tmp);
        add_kernel<<<mono_blocks(g.d),T,0,c.stream>>>(c.x,c.tmp,c.x,g.d);
        if(c.capture_trace)
            cudaMemcpyAsync(c.trace+(size_t)(li+1)*g.d,c.x,(size_t)g.d*8,
                            cudaMemcpyDeviceToDevice,c.stream);
    }
    rmsnorm_fast_i32_kernel<<<1,256,0,c.stream>>>(c.x,1,g.d,g.frac,g.eps,q35_buf(g.final_gain),c.norm,c.overflow);
    q35_prepare_digits(c,c.norm,g.d);q35_apply_prepared(c,g.out_head,c.logits);
    cudaMemcpyAsync(c.h_logits,c.logits,(size_t)g.vocab*8,cudaMemcpyDeviceToHost,c.stream);
    cudaMemcpyAsync(c.h_overflow,c.overflow,sizeof(int),cudaMemcpyDeviceToHost,c.stream);
    return cudaGetLastError()==cudaSuccess;
}

// Record the instantiated schedule shape without putting timers or callbacks
// into the hot graph.  Unknown/new CUDA node kinds are deliberately grouped as
// "other" so this ABI remains useful across CUDA toolkit versions.
static void q35_measure_graph(Bonsai35Ctx& c){
    c.graph_nodes=c.graph_kernel_nodes=c.graph_memcpy_nodes=0;
    c.graph_memset_nodes=c.graph_other_nodes=0;
    size_t count=0;
    if(!c.graph||cudaGraphGetNodes(c.graph,nullptr,&count)!=cudaSuccess){
        c.graph_nodes=c.graph_kernel_nodes=c.graph_memcpy_nodes=-1;
        c.graph_memset_nodes=c.graph_other_nodes=-1;
        cudaGetLastError();return;
    }
    std::vector<cudaGraphNode_t> nodes;
    try{nodes.resize(count);}catch(...){
        c.graph_nodes=c.graph_kernel_nodes=c.graph_memcpy_nodes=-1;
        c.graph_memset_nodes=c.graph_other_nodes=-1;return;
    }
    if(count&&cudaGraphGetNodes(c.graph,nodes.data(),&count)!=cudaSuccess){
        c.graph_nodes=c.graph_kernel_nodes=c.graph_memcpy_nodes=-1;
        c.graph_memset_nodes=c.graph_other_nodes=-1;
        cudaGetLastError();return;
    }
    c.graph_nodes=(long long)count;
    for(size_t i=0;i<count;++i){
        cudaGraphNodeType type;
        if(cudaGraphNodeGetType(nodes[i],&type)!=cudaSuccess){
            c.graph_other_nodes++;cudaGetLastError();continue;
        }
        if(type==cudaGraphNodeTypeKernel)c.graph_kernel_nodes++;
        else if(type==cudaGraphNodeTypeMemcpy)c.graph_memcpy_nodes++;
        else if(type==cudaGraphNodeTypeMemset)c.graph_memset_nodes++;
        else c.graph_other_nodes++;
    }
}

static void q35_free_ctx(Bonsai35Ctx& c){
    if(c.graph_exec)cudaGraphExecDestroy(c.graph_exec);if(c.graph)cudaGraphDestroy(c.graph);
    if(c.stream)cudaStreamDestroy(c.stream);
    #define F(P) do{if(c.P)cudaFree(c.P);c.P=nullptr;}while(0)
    F(state);F(conv_hist);F(K);F(V);F(x);F(norm);F(tmp);F(qkv);F(z);F(alpha);F(beta);F(conv);
    F(qn);F(kn);F(rv);F(rout);F(rnorm);F(zs);F(rgated);F(ctl_beta);F(ctl_decay);F(qg);F(aq);F(agate);
    F(ak);F(av);F(aout);F(agated);F(ffg);F(ffu);F(ffh);F(scores);F(digits);F(logits);F(pos);F(len);F(token);
    F(maxk);F(maxv);F(overflow);F(trace);
    #undef F
    if(c.h_x)cudaFreeHost(c.h_x);if(c.h_logits)cudaFreeHost(c.h_logits);if(c.h_pos)cudaFreeHost(c.h_pos);
    if(c.h_len)cudaFreeHost(c.h_len);if(c.h_token)cudaFreeHost(c.h_token);if(c.h_overflow)cudaFreeHost(c.h_overflow);
    c.h_x=c.h_logits=c.h_pos=c.h_len=c.h_token=nullptr;c.h_overflow=nullptr;c.alive=false;
}

// CUDA 12.8 with GCC 13 can internalize this function while lowering the
// generic allocation lambda, despite the surrounding extern "C" block.  The
// ctypes ABI then disappears from .dynsym even though the build succeeds.
// Keep the entry point externally visible across supported nvcc host compilers.
#if defined(__clang__)
__attribute__((visibility("default"), used))
#elif defined(__GNUC__)
__attribute__((visibility("default"), used, externally_visible))
#endif
long long bonsai35_ctx_create(const Bonsai35Config* cfg,const Bonsai35LayerDesc* layers){
    if(!cfg||!layers||cfg->n_layers<=0||cfg->d<=0||cfg->dff<=0||cfg->H<=0||cfg->Hkv<=0||
       cfg->H%cfg->Hkv||cfg->hd<=0||cfg->cap<=0||cfg->state_size<=0||cfg->state_size>128||
       cfg->conv_k<2||cfg->inner!=cfg->value_heads*cfg->state_size)return -1;
    Bonsai35Ctx c{};c.c=*cfg;c.alive=false;c.poisoned=false;c.t=0;c.graph_launches=0;c.graph_ready=false;
    c.capture_trace=false;c.group_projections=true;
    c.graph_nodes=c.graph_kernel_nodes=c.graph_memcpy_nodes=0;
    c.graph_memset_nodes=c.graph_other_nodes=0;
    c.input_mode=0;c.token_submissions=0;c.embedded_submissions=0;c.model_input_host_bytes=0;
    c.layers.assign(layers,layers+cfg->n_layers);c.nrec=0;c.natt=0;
    for(const auto& l:c.layers){if(l.kind==0)c.nrec++;else if(l.kind==1)c.natt++;else return -1;}
    c.conv_dim=2*cfg->key_heads*cfg->state_size+cfg->inner;c.max_k=cfg->d;
    if(cfg->dff>c.max_k)c.max_k=cfg->dff;if(cfg->inner>c.max_k)c.max_k=cfg->inner;
    auto vw=[](long long h){return q35_weight_ok(h)&&g_weights[(size_t)h].scale_bits==32&&g_weights[(size_t)h].layout==2;};
    auto ww=[&](long long h,long long out_f,long long in_f){
        return vw(h)&&g_weights[(size_t)h].out_f==out_f&&g_weights[(size_t)h].n_blocks*128==in_f;
    };
    auto vb=[](long long h){return q35_buf_ok(h);};
    if(!ww(cfg->embed,cfg->vocab,cfg->d)||!ww(cfg->out_head,cfg->vocab,cfg->d)||
       !vb(cfg->final_gain)||!vb(cfg->cos_buf)||!vb(cfg->sin_buf)||
       !vb(cfg->soft_buf)||!vb(cfg->exp_buf))return -1;
    for(const auto& l:c.layers){
        if(!vb(l.n1)||!vb(l.n2)||!ww(l.w1,cfg->dff,cfg->d)||
           !ww(l.wu,cfg->dff,cfg->d)||!ww(l.w2,cfg->d,cfg->dff))return -1;
        if(l.kind==0&&(!ww(l.wqkv,c.conv_dim,cfg->d)||!ww(l.wz,cfg->inner,cfg->d)||
           !ww(l.walpha,cfg->value_heads,cfg->d)||!ww(l.wbeta,cfg->value_heads,cfg->d)||
           !ww(l.wout,cfg->d,cfg->inner)||
           !vb(l.conv)||!vb(l.dt_bias)||!vb(l.ssm_a)||!vb(l.ssm_norm)))return -1;
        if(l.kind==1&&(!ww(l.wqg,2*cfg->H*cfg->hd,cfg->d)||
           !ww(l.wk,cfg->Hkv*cfg->hd,cfg->d)||!ww(l.wv,cfg->Hkv*cfg->hd,cfg->d)||
           !ww(l.wo,cfg->d,cfg->H*cfg->hd)||!vb(l.q_norm)||!vb(l.k_norm)))return -1;
    }
    bool ok=true;auto cm=[&](auto** p,size_t n){if(ok)ok=(cudaMalloc((void**)p,n)==cudaSuccess);};
    const size_t S=(size_t)c.nrec*cfg->value_heads*cfg->state_size*cfg->state_size;
    const size_t CH=(size_t)c.nrec*(cfg->conv_k-1)*c.conv_dim;
    const size_t KV=(size_t)c.natt*cfg->Hkv*cfg->cap*cfg->hd;
    cm(&c.state,S*8);cm(&c.conv_hist,CH*8);cm(&c.K,KV*4);cm(&c.V,KV*4);
    cm(&c.x,(size_t)cfg->d*8);cm(&c.norm,(size_t)cfg->d*8);cm(&c.tmp,(size_t)cfg->d*8);
    cm(&c.qkv,(size_t)c.conv_dim*8);cm(&c.z,(size_t)cfg->inner*8);cm(&c.alpha,(size_t)cfg->value_heads*8);
    cm(&c.beta,(size_t)cfg->value_heads*8);cm(&c.conv,(size_t)c.conv_dim*8);
    const size_t KW=(size_t)cfg->key_heads*cfg->state_size,RW=(size_t)cfg->value_heads*cfg->state_size;
    cm(&c.qn,KW*8);cm(&c.kn,KW*8);cm(&c.rv,RW*8);cm(&c.rout,RW*8);cm(&c.rnorm,RW*8);
    cm(&c.zs,RW*8);cm(&c.rgated,RW*8);cm(&c.ctl_beta,(size_t)cfg->value_heads*8);cm(&c.ctl_decay,(size_t)cfg->value_heads*8);
    const size_t Q=(size_t)cfg->H*cfg->hd,QG=2*Q,AK=(size_t)cfg->Hkv*cfg->hd;
    cm(&c.qg,QG*8);cm(&c.aq,Q*8);cm(&c.agate,Q*8);cm(&c.ak,AK*8);cm(&c.av,AK*8);
    cm(&c.aout,Q*8);cm(&c.agated,Q*8);cm(&c.ffg,(size_t)cfg->dff*8);cm(&c.ffu,(size_t)cfg->dff*8);
    cm(&c.ffh,(size_t)cfg->dff*8);cm(&c.scores,(size_t)cfg->H*cfg->cap*8);
    cm(&c.digits,(size_t)(c.max_k/128)*4*128);cm(&c.logits,(size_t)cfg->vocab*8);
    cm(&c.pos,8);cm(&c.len,8);cm(&c.token,8);
    cm(&c.maxk,(size_t)c.natt*cfg->Hkv*8);cm(&c.maxv,(size_t)c.natt*cfg->Hkv*8);cm(&c.overflow,sizeof(int));
    if(ok)ok=(cudaHostAlloc(&c.h_x,(size_t)cfg->d*8,cudaHostAllocPortable)==cudaSuccess);
    if(ok)ok=(cudaHostAlloc(&c.h_logits,(size_t)cfg->vocab*8,cudaHostAllocPortable)==cudaSuccess);
    if(ok)ok=(cudaHostAlloc(&c.h_pos,8,cudaHostAllocPortable)==cudaSuccess);
    if(ok)ok=(cudaHostAlloc(&c.h_len,8,cudaHostAllocPortable)==cudaSuccess);
    if(ok)ok=(cudaHostAlloc(&c.h_token,8,cudaHostAllocPortable)==cudaSuccess);
    if(ok)ok=(cudaHostAlloc(&c.h_overflow,sizeof(int),cudaHostAllocPortable)==cudaSuccess);
    if(ok)ok=(cudaStreamCreateWithFlags(&c.stream,cudaStreamNonBlocking)==cudaSuccess);
    if(ok){cudaMemset(c.state,0,S*8);cudaMemset(c.conv_hist,0,CH*8);cudaMemset(c.K,0,KV*4);cudaMemset(c.V,0,KV*4);
        cudaMemset(c.maxk,0,(size_t)c.natt*cfg->Hkv*8);cudaMemset(c.maxv,0,(size_t)c.natt*cfg->Hkv*8);}
    if(!ok){q35_free_ctx(c);cudaGetLastError();return -1;}
    c.alive=true;g_bonsai35.push_back(std::move(c));return (long long)g_bonsai35.size()-1;
}

// Trace mode must be selected before the first decode captures the graph.  It
// cannot be toggled after capture because that would make the reported graph
// shape and its data dependencies disagree with the caller's expectation.
int bonsai35_ctx_set_trace(long long handle,int enabled){
    if(handle<0||(size_t)handle>=g_bonsai35.size())return 1;
    Bonsai35Ctx& c=g_bonsai35[(size_t)handle];
    if(!c.alive||c.graph_ready||c.t!=0)return 1;
    if(enabled){
        if(!c.trace&&cudaMalloc(&c.trace,(size_t)(c.c.n_layers+1)*c.c.d*8)!=cudaSuccess){
            cudaGetLastError();return 2;
        }
        c.capture_trace=true;
    }else{
        if(c.trace&&cudaFree(c.trace)!=cudaSuccess){cudaGetLastError();return 2;}
        c.trace=nullptr;c.capture_trace=false;
    }
    return 0;
}

// Projection grouping is a pre-capture scheduling choice.  Disabling it is
// retained as an exact same-binary control for parity and same-host AB tests;
// production defaults to the grouped graph.
int bonsai35_ctx_set_projection_grouping(long long handle,int enabled){
    if(handle<0||(size_t)handle>=g_bonsai35.size())return 1;
    Bonsai35Ctx& c=g_bonsai35[(size_t)handle];
    if(!c.alive||c.graph_ready||c.t!=0)return 1;
    c.group_projections=enabled!=0;return 0;
}

int bonsai35_decode_step(long long handle,const long long* x_host,long long pos,long long* logits_host){
    if(handle<0||(size_t)handle>=g_bonsai35.size()||!x_host||!logits_host)return 1;
    Bonsai35Ctx& c=g_bonsai35[(size_t)handle];if(!c.alive||c.poisoned||pos!=c.t||pos>=c.c.cap)return 1;
    if(c.graph_ready&&c.input_mode!=1)return 3;
    memcpy(c.h_x,x_host,(size_t)c.c.d*8);*c.h_pos=pos;*c.h_len=pos+1;*c.h_overflow=0;
    if(!c.graph_ready){
        if(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal)!=cudaSuccess)return 2;
        bool ok=q35_enqueue_decode(c,false);
        if(!ok||cudaStreamEndCapture(c.stream,&c.graph)!=cudaSuccess)return 2;
        q35_measure_graph(c);
        if(cudaGraphInstantiate(&c.graph_exec,c.graph,nullptr,nullptr,0)!=cudaSuccess)return 2;
        c.graph_ready=true;c.input_mode=1;
    }
    if(cudaGraphLaunch(c.graph_exec,c.stream)!=cudaSuccess||cudaStreamSynchronize(c.stream)!=cudaSuccess)return 2;
    c.graph_launches++;c.embedded_submissions++;c.model_input_host_bytes+=(long long)c.c.d*8;
    if(*c.h_overflow){c.poisoned=true;return 4;}
    memcpy(logits_host,c.h_logits,(size_t)c.c.vocab*8);c.t++;return 0;
}

// Production resident decode.  The captured graph transfers one token ID and
// expands its resident packed-Q1 embedding row on device; no d_model-sized
// host activation crosses PCIe.  A context is deliberately locked to the
// input mode of its first capture so only one large CUDA graph is resident.
int bonsai35_decode_token(long long handle,long long token,long long pos,long long* logits_host){
    if(handle<0||(size_t)handle>=g_bonsai35.size()||!logits_host)return 1;
    Bonsai35Ctx& c=g_bonsai35[(size_t)handle];
    if(!c.alive||c.poisoned||pos!=c.t||pos>=c.c.cap||token<0||token>=c.c.vocab)return 1;
    if(c.graph_ready&&c.input_mode!=2)return 3;
    *c.h_token=token;*c.h_pos=pos;*c.h_len=pos+1;*c.h_overflow=0;
    if(!c.graph_ready){
        if(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal)!=cudaSuccess)return 2;
        bool ok=q35_enqueue_decode(c,true);
        if(!ok||cudaStreamEndCapture(c.stream,&c.graph)!=cudaSuccess)return 2;
        q35_measure_graph(c);
        if(cudaGraphInstantiate(&c.graph_exec,c.graph,nullptr,nullptr,0)!=cudaSuccess)return 2;
        c.graph_ready=true;c.input_mode=2;
    }
    if(cudaGraphLaunch(c.graph_exec,c.stream)!=cudaSuccess||cudaStreamSynchronize(c.stream)!=cudaSuccess)return 2;
    c.graph_launches++;c.token_submissions++;c.model_input_host_bytes+=8;
    if(*c.h_overflow){c.poisoned=true;return 4;}
    memcpy(logits_host,c.h_logits,(size_t)c.c.vocab*8);c.t++;return 0;
}

int bonsai35_ctx_reset(long long handle){
    if(handle<0||(size_t)handle>=g_bonsai35.size())return 1;Bonsai35Ctx& c=g_bonsai35[(size_t)handle];
    if(!c.alive)return 1;const auto& g=c.c;
    cudaMemset(c.state,0,(size_t)c.nrec*g.value_heads*g.state_size*g.state_size*8);
    cudaMemset(c.conv_hist,0,(size_t)c.nrec*(g.conv_k-1)*c.conv_dim*8);
    cudaMemset(c.K,0,(size_t)c.natt*g.Hkv*g.cap*g.hd*4);cudaMemset(c.V,0,(size_t)c.natt*g.Hkv*g.cap*g.hd*4);
    cudaMemset(c.maxk,0,(size_t)c.natt*g.Hkv*8);cudaMemset(c.maxv,0,(size_t)c.natt*g.Hkv*8);
    c.t=0;c.poisoned=false;return cudaDeviceSynchronize()==cudaSuccess?0:2;
}
void bonsai35_ctx_free(long long handle){
    if(handle<0||(size_t)handle>=g_bonsai35.size())return;Bonsai35Ctx& c=g_bonsai35[(size_t)handle];
    if(c.alive)q35_free_ctx(c);
}

// Debug/parity export after a completed step.  KV is compacted to
// (attention_layers,Hkv,t,hd), omitting unused 4K capacity.
int bonsai35_ctx_export(long long handle,long long* state_host,long long* conv_host,
                        long long* k_host,long long* v_host,long long* trace_host){
    if(handle<0||(size_t)handle>=g_bonsai35.size())return 1;Bonsai35Ctx& c=g_bonsai35[(size_t)handle];
    if(!c.alive||c.poisoned||(!c.capture_trace&&trace_host))return 1;const auto& g=c.c;bool ok=true;
    const size_t S=(size_t)c.nrec*g.value_heads*g.state_size*g.state_size;
    const size_t CH=(size_t)c.nrec*(g.conv_k-1)*c.conv_dim;
    if(state_host)ok=ok&&(cudaMemcpy(state_host,c.state,S*8,cudaMemcpyDeviceToHost)==cudaSuccess);
    if(conv_host)ok=ok&&(cudaMemcpy(conv_host,c.conv_hist,CH*8,cudaMemcpyDeviceToHost)==cudaSuccess);
    if(trace_host)ok=ok&&(cudaMemcpy(trace_host,c.trace,(size_t)(g.n_layers+1)*g.d*8,cudaMemcpyDeviceToHost)==cudaSuccess);
    if((k_host||v_host)&&c.t>0){
        const size_t row_elems=(size_t)c.t*g.hd,bytes=row_elems*sizeof(int);
        std::vector<int> narrow;
        try{narrow.resize(row_elems);}catch(...){return 2;}
        for(long long a=0;ok&&a<c.natt;++a)for(long long h=0;ok&&h<g.Hkv;++h){
            const size_t src=((size_t)a*g.Hkv+h)*g.cap*g.hd;
            const size_t dst=((size_t)a*g.Hkv+h)*c.t*g.hd;
            if(k_host){ok=ok&&(cudaMemcpy(narrow.data(),c.K+src,bytes,cudaMemcpyDeviceToHost)==cudaSuccess);
                if(ok)for(size_t i=0;i<row_elems;++i)k_host[dst+i]=(long long)narrow[i];}
            if(v_host){ok=ok&&(cudaMemcpy(narrow.data(),c.V+src,bytes,cudaMemcpyDeviceToHost)==cudaSuccess);
                if(ok)for(size_t i=0;i<row_elems;++i)v_host[dst+i]=(long long)narrow[i];}
        }
    }
    return ok?0:2;
}

int bonsai35_ctx_stats(long long handle,long long* launches,long long* position,int* graph_ready,int* poisoned){
    if(handle<0||(size_t)handle>=g_bonsai35.size())return 1;const Bonsai35Ctx& c=g_bonsai35[(size_t)handle];
    if(!c.alive)return 1;if(launches)*launches=c.graph_launches;if(position)*position=c.t;
    if(graph_ready)*graph_ready=c.graph_ready?1:0;if(poisoned)*poisoned=c.poisoned?1:0;return 0;
}

int bonsai35_ctx_input_stats(long long handle,long long* input_mode,long long* token_submissions,
                             long long* embedded_submissions,long long* model_input_host_bytes){
    if(handle<0||(size_t)handle>=g_bonsai35.size())return 1;const Bonsai35Ctx& c=g_bonsai35[(size_t)handle];
    if(!c.alive)return 1;if(input_mode)*input_mode=c.input_mode;
    if(token_submissions)*token_submissions=c.token_submissions;
    if(embedded_submissions)*embedded_submissions=c.embedded_submissions;
    if(model_input_host_bytes)*model_input_host_bytes=c.model_input_host_bytes;return 0;
}

int bonsai35_ctx_graph_stats(long long handle,long long* total_nodes,long long* kernel_nodes,
                             long long* memcpy_nodes,long long* memset_nodes,
                             long long* other_nodes,int* trace_enabled){
    if(handle<0||(size_t)handle>=g_bonsai35.size())return 1;
    const Bonsai35Ctx& c=g_bonsai35[(size_t)handle];if(!c.alive)return 1;
    if(total_nodes)*total_nodes=c.graph_nodes;
    if(kernel_nodes)*kernel_nodes=c.graph_kernel_nodes;
    if(memcpy_nodes)*memcpy_nodes=c.graph_memcpy_nodes;
    if(memset_nodes)*memset_nodes=c.graph_memset_nodes;
    if(other_nodes)*other_nodes=c.graph_other_nodes;
    if(trace_enabled)*trace_enabled=c.capture_trace?1:0;
    return 0;
}

int bonsai35_ctx_projection_stats(long long handle,int* grouping_enabled,
                                  long long* logical_applies,long long* kernel_nodes){
    if(handle<0||(size_t)handle>=g_bonsai35.size())return 1;
    const Bonsai35Ctx& c=g_bonsai35[(size_t)handle];if(!c.alive)return 1;
    const long long logical=8*c.nrec+7*c.natt+1; // includes the output head
    const long long scheduled=c.group_projections?4*(c.nrec+c.natt)+1:logical;
    if(grouping_enabled)*grouping_enabled=c.group_projections?1:0;
    if(logical_applies)*logical_applies=logical;
    if(kernel_nodes)*kernel_nodes=scheduled;
    return 0;
}

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
    long long d_clip = dca<dcb?dca:dcb, d_clip_silu=d_clip;  // SiLU mirrors softmax's (1<<62)//log2e cap (oracle parity)
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
    for (size_t i = 0; i < g_reservations.size(); ++i) {
        if (!g_reservations[i].alive) continue;
        for (void* p : g_reservations[i].ptrs) cudaFree(p);
        g_reservations[i].ptrs.clear();
        g_reservations[i].alive = false;
        g_reservations[i].bytes = 0;
    }
    g_reservations.clear();
    for (size_t i = 0; i < g_weights.size(); ++i) {
        if (g_weights[i].dbits) cudaFree(g_weights[i].dbits);
        if (g_weights[i].dscale) cudaFree(g_weights[i].dscale);
        if (g_weights[i].dscale32) cudaFree(g_weights[i].dscale32);
    }
    g_weights.clear();
    g_weight_bytes = 0;
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
    long long d_clip_silu = d_clip_attn;   // SiLU mirrors the (1<<62)//log2e cap for oracle parity (was uncapped)

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

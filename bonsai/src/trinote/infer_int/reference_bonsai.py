"""Pure NumPy integer reference engine for Bonsai-8B Qwen3 Q1_0 artifacts.

This is a separate canonical path from BitNet `ReferenceModelV2`. Bonsai weights are packed Q1_0:
one sign bit per weight plus one FP16-derived scale per 128-weight group. The reference path stores the
sign bits packed and the group scales in fixed-point, then evaluates linear layers by group-wise integer
signed sums followed by fixed-point scale application.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..determinism.fixedpoint import (
    fixed_point_matmul,
    fixed_point_rmsnorm,
    fixed_point_sigmoid,
    fixed_point_softmax,
)
from ..model.rope_v2 import apply_rope_fixed_neox, build_rope_tables
from .q1_native import (
    attention_decode_batched_native,
    attention_decode_native,
    attention_prefill_native,
    q1_argmax_native,
    q1_linear_native,
    q1_linear_prepared_many_native,
    q1_linear_prepared_native,
    q1_native_available,
    q1_prepare_native,
    rmsnorm_native,
    silu_native,
)
from .gpu_native import (             # per-host opt-in GPU kernels (return None when absent -> CPU fallback)
    attention_prefill_gpu,
    batched_decode_available,
    buf_upload_i64,
    decode_ctx_create,
    decode_ctx_free,
    decode_ctx_seed_seq,
    decode_step,
    monolith_available,
    prefill_forward_gpu,
    q1_apply_gpu,
    q1_apply_resident,
    q1_register_weight,
    rmsnorm_gpu,
)
from .sampler import apply_rep_penalty

_GROUP = 128
_NEG_INF_SHIFT = 30
_BYTE_SUBSET_MASKS = (
    (np.arange(256, dtype=np.uint16)[:, None] >> np.arange(8, dtype=np.uint16)) & 1
).astype(np.int64)
_BYTE_SUBSET_MASKS.flags.writeable = False


def _oracle_q1_worker_count() -> int:
    raw = os.environ.get("TRINOTE_ORACLE_Q1_THREADS", "4")
    try:
        return max(1, min(32, int(raw)))
    except ValueError:
        return 4


_ORACLE_Q1_WORKERS = _oracle_q1_worker_count()
_ORACLE_Q1_POOL = (
    ThreadPoolExecutor(max_workers=_ORACLE_Q1_WORKERS, thread_name_prefix="bonsai-q1-oracle")
    if _ORACLE_Q1_WORKERS > 1 else None
)


def oracle_q1_worker_count() -> int:
    """Return the immutable worker count selected when this module loaded."""
    return _ORACLE_Q1_WORKERS


def _q1_subset_output_chunk(
    subset_lut: np.ndarray,
    totals: np.ndarray,
    block_index: np.ndarray,
    byte_index: np.ndarray,
    bits: np.ndarray,
    scales: np.ndarray,
    frac: int,
    lo: int,
    hi: int,
) -> tuple[int, np.ndarray]:
    """Evaluate independent output rows for the pure packed-byte oracle."""
    selected = subset_lut[
        block_index, byte_index, bits[lo:hi]
    ].sum(axis=2, dtype=np.int64)
    acc = (selected << np.int64(1)) - totals[None, :]
    values = ((acc * scales[lo:hi]) >> frac).sum(axis=1, dtype=np.int64)
    return lo, values


def _q1_expanded_output_chunk(
    x_groups: np.ndarray,
    bits: np.ndarray,
    scales: np.ndarray,
    frac: int,
    lo: int,
    hi: int,
) -> tuple[int, np.ndarray]:
    """Evaluate independent multi-token output rows with the original equation."""
    signs = _unpack_q1_signs(bits[lo:hi]).astype(np.int64)
    acc = np.einsum("tbi,obi->tob", x_groups, signs, optimize=True)
    values = ((acc * scales[lo:hi][None, :, :]) >> frac).sum(
        axis=2, dtype=np.int64
    )
    return lo, values

# OVERFLOW POLICY for the Q1_0 linear (decision #q1-wrap) — deliberately DIFFERENT from the attention
# `fixed_point_matmul`, which fails loud (`_assert_no_int64_overflow`). The dominant Q1_0 product
# `acc * scale` (|acc| <= max|x_fp|*128 over a 128-wide {-1,+1} group) stays far inside int64 for the
# RMSNorm-bounded shipped model, so no live overflow occurs. But where it *would* overflow, this path
# intentionally lets the int64 multiply/accumulate WRAP rather than raise: the numpy reference and the
# native C kernel (tools/bonsai_q1_kernel.c, via unsigned mod-2^64 arithmetic) wrap BIT-IDENTICALLY, so
# the determinism property receipts rely on — both producer and verifier compute the same bytes — is
# preserved even at overflow. That parity is locked by
# `tests/test_bonsai_smoke.py::test_bonsai_native_q1_kernel_matches_oracle_at_overflow_boundary_if_present`.
# A fail-loud guard here would break that tested invariant; the attention path raises only because its
# acts×acts products have no such kernel-parity contract. (Review L8 / I8.)


def _unpack_q1_signs(bits: np.ndarray) -> np.ndarray:
    """Packed Q1_0 bits `(out, n_blocks, 16)` -> signs `(out, n_blocks, 128)` in {-1,+1}."""
    b = np.asarray(bits, dtype=np.uint8)
    u = np.unpackbits(b, axis=-1, bitorder="little").astype(np.int8)
    return (u * 2 - 1).astype(np.int8)


def q1_rows_fp(bits: np.ndarray, scale_fp: np.ndarray, rows, frac: int) -> np.ndarray:
    """Dequant selected Q1_0 rows to fixed-point `(len(rows), in_features)`.

    Used for token embedding lookup. It expands only the requested token rows.
    """
    idx = np.asarray(rows, dtype=np.int64)
    signs = _unpack_q1_signs(bits[idx]).astype(np.int64)
    scales = np.asarray(scale_fp[idx], dtype=np.int64)
    return (signs * scales[:, :, None]).reshape(idx.shape[0], -1)


def q1_linear_ref(x_fp: np.ndarray, bits: np.ndarray, scale_fp: np.ndarray, frac: int,
                  *, out_chunk: int = 256) -> np.ndarray:
    """Fixed-point linear `x @ W.T` for packed Q1_0 weights.

    `bits`/`scale_fp` describe rows of W `(out_features, in_features)`. The computation is:
      sum_blocks ((sum_128 x_i * sign_i) * scale_block) >> frac
    which is exactly the dequantized binary-weight matmul under the committed fixed-point scales.
    """
    x = np.atleast_2d(np.asarray(x_fp, dtype=np.int64))
    b = np.asarray(bits, dtype=np.uint8)
    s = np.asarray(scale_fp, dtype=np.int64)
    out_f, n_blocks = s.shape
    if x.shape[1] != n_blocks * _GROUP:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * _GROUP}")
    # Decode is overwhelmingly the canonical 27B receipt oracle's bottleneck.
    # For one activation row, evaluate each packed byte through a 256-entry
    # subset-sum table instead of expanding every weight bit to an int64 sign:
    #
    #   sum(x_i * sign_i) = 2 * sum(x_i where bit_i=1) - sum(x_i)
    #
    # This is the same identity in Z/(2^64): all NumPy int64 additions,
    # subtractions, shifts, and products retain the oracle's deliberate wrap
    # semantics even at adversarial extrema. It also keeps the verifier a pure
    # NumPy implementation, independent of every native producer handle.
    if x.shape[0] == 1:
        if out_chunk <= 0:
            raise ValueError("out_chunk must be positive")
        xb = x.reshape(n_blocks, 16, 8)
        subset_lut = np.einsum(
            "bqe,me->bqm", xb, _BYTE_SUBSET_MASKS, optimize=True
        )
        totals = xb.sum(axis=(1, 2), dtype=np.int64)
        block_index = np.arange(n_blocks)[None, :, None]
        byte_index = np.arange(16)[None, None, :]
        out = np.empty((1, out_f), dtype=np.int64)
        ranges = [
            (lo, min(lo + out_chunk, out_f))
            for lo in range(0, out_f, out_chunk)
        ]
        def apply_chunk(bounds):
            return _q1_subset_output_chunk(
                subset_lut, totals, block_index, byte_index, b, s, frac, *bounds
            )
        if _ORACLE_Q1_POOL is not None and len(ranges) > 1:
            chunks = _ORACLE_Q1_POOL.map(apply_chunk, ranges)
        else:
            chunks = map(apply_chunk, ranges)
        for lo, values in chunks:
            out[0, lo:lo + values.size] = values
        return out

    # int64 wrap is permitted here and bit-identical to the native kernel — see the OVERFLOW POLICY note above.
    xg = x.reshape(x.shape[0], n_blocks, _GROUP)
    if out_chunk <= 0:
        raise ValueError("out_chunk must be positive")
    out = np.empty((x.shape[0], out_f), dtype=np.int64)
    ranges = [
        (lo, min(lo + out_chunk, out_f))
        for lo in range(0, out_f, out_chunk)
    ]

    def apply_expanded(bounds):
        return _q1_expanded_output_chunk(xg, b, s, frac, *bounds)

    if _ORACLE_Q1_POOL is not None and len(ranges) > 1:
        chunks = _ORACLE_Q1_POOL.map(apply_expanded, ranges)
    else:
        chunks = map(apply_expanded, ranges)
    for lo, values in chunks:
        out[:, lo:lo + values.shape[1]] = values
    return out


def q1_linear_signs_ref(x_fp: np.ndarray, signs_i8: np.ndarray, scale_fp: np.ndarray, frac: int,
                        *, out_chunk: int = 256) -> np.ndarray:
    """Same Q1_0 linear as `q1_linear_ref`, but with signs unpacked once and cached as int8."""
    x = np.atleast_2d(np.asarray(x_fp, dtype=np.int64))
    signs_all = np.asarray(signs_i8, dtype=np.int8)
    s = np.asarray(scale_fp, dtype=np.int64)
    out_f, n_blocks = s.shape
    if signs_all.shape != (out_f, n_blocks, _GROUP):
        raise ValueError(
            f"Q1_0 sign cache shape {signs_all.shape} does not match {(out_f, n_blocks, _GROUP)}"
        )
    if x.shape[1] != n_blocks * _GROUP:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * _GROUP}")
    # int64 wrap is permitted here and bit-identical to the native kernel — see the OVERFLOW POLICY note above.
    xg = x.reshape(x.shape[0], n_blocks, _GROUP)
    out = np.empty((x.shape[0], out_f), dtype=np.int64)
    for lo in range(0, out_f, out_chunk):
        hi = min(lo + out_chunk, out_f)
        acc = np.einsum("tbi,obi->tob", xg, signs_all[lo:hi], optimize=True)
        out[:, lo:hi] = ((acc * s[lo:hi][None, :, :]) >> frac).sum(axis=2)
    return out


def fixed_point_silu(x_fp: np.ndarray, frac: int, *, native: bool = False) -> np.ndarray:
    """Deterministic fixed-point SiLU: x * sigmoid(x), with sigmoid from the integer softmax kernel.

    With `native=True` (the native engine path) and `TRINOTE_NATIVE_SILU` not opted out, uses the byte-
    identical C kernel; otherwise the NumPy oracle. The default `native=False` keeps `forward()` on the pure
    NumPy path so it stays a faithful comparison oracle for the native paths."""
    x = np.asarray(x_fp, dtype=np.int64)
    if native and _native_silu_enabled():
        try:
            out = silu_native(x, frac)
        except (MemoryError, RuntimeError):
            out = None
        if out is not None:
            return out
    sig = fixed_point_sigmoid(x, frac)
    # x*sig is NOT fail-loud-guarded: the SiLU activation path is a tested wrap-by-construction exception to
    # §3.4 (like the Q1 apply), byte-identical to silu_native at the int64 extremes
    # (test_bonsai_native_silu_matches_oracle_if_present). A raise here would desync the oracle from the C/GPU
    # producers. Real activations stay in-envelope (RMSNorm-bounded); an out-of-envelope x wraps consistently.
    return (x * sig) >> frac


def _rmsnorm(x_fp: np.ndarray, frac: int, gain: np.ndarray, *, native: bool = False,
             eps: int = 1) -> np.ndarray:
    if native and _gpu_full_enabled():
        try:
            g = rmsnorm_gpu(x_fp, frac, eps, gain)        # byte-identical; None on overflow/unavailable -> CPU
        except (MemoryError, RuntimeError):
            g = None
        if g is not None:
            return g
    if native and _native_rmsnorm_enabled():
        try:
            out = rmsnorm_native(x_fp, frac, eps=eps, gain_q=gain)
        except (MemoryError, RuntimeError):
            out = None
        if out is not None:
            return out
    return fixed_point_rmsnorm(x_fp, frac, eps=eps, gain_q=gain)


def _head_rmsnorm(
    x_heads: np.ndarray,
    frac: int,
    gain: np.ndarray,
    *,
    native: bool = False,
    eps: int = 1,
) -> np.ndarray:
    """RMSNorm over the head_dim axis for `(H,T,hd)` head tensors."""
    h, t, d = x_heads.shape
    y = _rmsnorm(x_heads.reshape(h * t, d), frac, gain, native=native, eps=eps)
    return y.reshape(h, t, d)


def _q1(layer: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    return layer[f"{name}_bits"], layer[f"{name}_scale_fp"]


def _scale_fp(owner: dict, name: str) -> np.ndarray:
    """Return the int32 scale cache for this weight if the loader built one (narrow native reproducer
    path, Recommendation 7), else the committed int64 scale. The native wrapper dispatches on dtype, and
    the int64 oracle fallback always reads the committed `{name}_scale_fp`, so output IDs are unchanged."""
    narrow = owner.get(f"{name}_scale_fp_i32")
    return narrow if narrow is not None else owner[f"{name}_scale_fp"]


def _q1_bl_ref(x_fp, layer, name: str, frac: int) -> np.ndarray:
    return q1_linear_ref(x_fp, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac)


def _q1_bl_fast(x_fp, layer, name: str, frac: int) -> np.ndarray:
    signs = layer.get(f"{name}_signs_i8")
    if signs is None:
        return _q1_bl_ref(x_fp, layer, name, frac)
    return q1_linear_signs_ref(x_fp, signs, layer[f"{name}_scale_fp"], frac)


def _q1_bl_native(x_fp, layer, name: str, frac: int) -> np.ndarray:
    if _gpu_enabled():
        # Per-host opt-in GPU path (resident weights): byte-identical to the CPU path below. Returns None when
        # the GPU .so/GPU is absent or a tile overflows the kernel envelope -> fall through to native/oracle.
        try:
            g = _gpu_apply(layer, name, x_fp, frac)
        except (MemoryError, RuntimeError):
            g = None
        if g is not None:
            return g
    lut32 = _lut32_enabled()
    scale = layer[f"{name}_scale_fp"] if lut32 else _scale_fp(layer, name)
    try:
        out = q1_linear_native(x_fp, layer[f"{name}_bits"], scale, frac, lut32=lut32)
    except (MemoryError, RuntimeError):
        out = None
    if out is None:
        return _q1_bl_fast(x_fp, layer, name, frac)
    return out


def _q1_bl_native_prepared_many(x_fp, layer, names: tuple[str, ...], frac: int) -> tuple[np.ndarray, ...] | None:
    """Apply same-input native Q1_0 projections after preparing the activation LUT once."""
    if not names:
        return ()
    if _gpu_enabled():
        # The GPU path has no activation-LUT to amortize; returning None makes the caller fall back to per-name
        # q1(...) = _q1_bl_native, which routes each projection through the GPU (or to CPU if a call returns None).
        return None
    try:
        n_blocks = int(layer[f"{names[0]}_scale_fp"].shape[1])
        lut32 = _lut32_enabled()
        prep = q1_prepare_native(x_fp, n_blocks, lut32=lut32)
        if prep is None and lut32:
            prep = q1_prepare_native(x_fp, n_blocks)   # int32 LUT out of envelope -> uint64 LUT
        if prep is None:
            return None
        # lut32 uses the committed int64 scales; otherwise the (possibly int32) scale-cache scale.
        scale_for = (lambda nm: layer[f"{nm}_scale_fp"]) if lut32 else (lambda nm: _scale_fp(layer, nm))
        if _prepared_multi_enabled():
            weights = tuple((layer[f"{name}_bits"], scale_for(name)) for name in names)
            many = q1_linear_prepared_many_native(prep, weights, frac)
            if many is not None:
                return many
        out = []
        for name in names:
            y = q1_linear_prepared_native(prep, layer[f"{name}_bits"], scale_for(name), frac)
            if y is None:
                return None
            out.append(y)
        return tuple(out)
    except (MemoryError, RuntimeError):
        return None


def _prepared_multi_enabled() -> bool:
    """Default-enable grouped native Q1 projections; allow an env escape hatch for benchmarking/debug."""
    v = os.environ.get("TRINOTE_Q1_PREPARED_MULTI")
    if v is None:
        return True
    return v.strip().lower() not in {"0", "false", "no", "off", ""}


def _native_rmsnorm_enabled() -> bool:
    """Default-enable native integer RMSNorm; allow an env escape hatch for oracle comparison."""
    v = os.environ.get("TRINOTE_NATIVE_RMSNORM")
    if v is None:
        return True
    return v.strip().lower() not in {"0", "false", "no", "off", ""}


def _scale_cache_enabled() -> bool:
    """Opt-in narrow int32 Q1 scale cache (Recommendation 7 prototype; default OFF). Enable with
    TRINOTE_Q1_SCALE_CACHE=1 to halve Q1 scale-array bandwidth via a native-only reproducer."""
    v = os.environ.get("TRINOTE_Q1_SCALE_CACHE")
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _lut32_enabled() -> bool:
    """Opt-in int32 activation-LUT-entry kernels (optimization-scopes/INT32-LUT-ENTRY.md; default OFF).
    Enable with TRINOTE_Q1_LUT32=1 to halve the Q1 gather data. Byte-identical to the uint64-LUT path for
    in-envelope blocks; falls back to the int64 LUT per-block on out-of-envelope (rc 5). Uses the committed
    int64 scales (takes precedence over the int32 scale cache, which it does not combine with)."""
    v = os.environ.get("TRINOTE_Q1_LUT32")
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _native_attn_enabled() -> bool:
    """Default-enable the native M=1 cached-decode attention kernel; TRINOTE_NATIVE_ATTN=0 opts out (e.g.
    to compare against the NumPy oracle). Byte-identical to the NumPy path; falls back to it on overflow."""
    v = os.environ.get("TRINOTE_NATIVE_ATTN")
    if v is None:
        return True
    return v.strip().lower() not in {"0", "false", "no", "off", ""}


def _native_silu_enabled() -> bool:
    """Default-enable native fixed-point SiLU; TRINOTE_NATIVE_SILU=0 opts out (oracle comparison).
    Byte-identical to fixed_point_silu; falls back to the NumPy path when unavailable/disabled."""
    v = os.environ.get("TRINOTE_NATIVE_SILU")
    if v is None:
        return True
    return v.strip().lower() not in {"0", "false", "no", "off", ""}


def _gpu_enabled() -> bool:
    """Per-host opt-in GPU Q1 kernel (default OFF). Enable with TRINOTE_GPU=1. Byte-identical to the CPU
    native/oracle path; falls back to it whenever the GPU .so is absent, the GPU is missing, or a tile overflows
    the kernel's integer envelope (q1_apply_gpu returns None). NEVER the committed default — the CPU oracle stays
    the canonical verifier, so a GPU-produced receipt re-executes on a CPU-only host. See
    research/bonsai-notary/IMPLEMENT-GPU-MODE.md."""
    v = os.environ.get("TRINOTE_GPU")
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _verify_gpu_enabled() -> bool:
    """Opt-in (default OFF) GPU byte-exactness self-check. Enable with BONSAI_VERIFY_GPU=1.

    SCOPE (be precise — this is a load-bearing security claim): when set, every DISCRETE GPU Q1 apply routed
    through `_gpu_apply`/`_gpu_oracle_check` re-runs the CPU int64 oracle (`q1_linear_ref`) for the same inputs
    and asserts byte-equality, raising on any mismatch. It does NOT cover the fused on-device PREFILL MONOLITH
    (`prefill_forward_gpu`) or the RESIDENT-DECODE path, which compute Q1 applies + RMSNorm + RoPE + SiLU +
    attention entirely on-device without per-op oracle re-checks — so this flag alone does not prove those
    paths byte-exact. Those are gated instead by the standalone kernel↔oracle PARITY TESTS (run with GPU
    hardware present); the runtime flag is a per-apply spot-check, not a whole-forward guarantee. OFF by
    default (re-running the oracle defeats the GPU perf win). The GPU .so is gitignored/per-host so default CI
    cannot exercise either mechanism. For a fully oracle-checked run, use the CPU/native path (the canonical
    verifier) — a GPU-produced receipt is verified by re-executing it on that CPU oracle regardless."""
    v = os.environ.get("BONSAI_VERIFY_GPU")
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _gpu_oracle_check(out, x_fp, bits, scale_fp, frac, *, where: str):
    """When BONSAI_VERIFY_GPU=1, assert `out` (a GPU Q1 result) is BYTE-IDENTICAL to the CPU int64 oracle
    `q1_linear_ref` over the same inputs; raise AssertionError on any divergence. No-op when the check is off
    or `out` is None (the GPU already fell back to CPU, so there is nothing GPU-produced to trust). Returns
    `out` unchanged so callers can wrap a return value inline."""
    if out is None or not _verify_gpu_enabled():
        return out
    oracle = q1_linear_ref(x_fp, bits, scale_fp, frac)
    g = np.asarray(out)
    if g.shape != oracle.shape or not np.array_equal(g, oracle):
        raise AssertionError(
            f"BONSAI_VERIFY_GPU: GPU Q1 result for {where} is NOT byte-identical to the CPU oracle "
            f"(gpu shape={g.shape}, oracle shape={oracle.shape}); refusing to trust the GPU output")
    return out


def _gpu_resident_batch_enabled() -> bool:
    """Opt-in (default OFF): use the FULLY-RESIDENT M=B batched decode (KV + RMSNorm/RoPE/attention on device)
    instead of the default GPU batch path (M=B applies on GPU, RMSNorm/attention on CPU). Both are byte-exact,
    but the resident path REGRESSES on sm_86 at decode batch sizes (measured B=16: 7.4 vs 12.2 tok/s) — its
    per-step on-device RMSNorm is one-thread-per-row (B threads = poor occupancy) + ~500 tiny kernel launches +
    sync per step, which the existing CPU-RMSNorm/attention hybrid beats. Kept as a verified foundation for
    block-per-row RMSNorm + kernel fusion / larger B / a datacenter GPU. Enable with TRINOTE_GPU_RESIDENT_BATCH=1."""
    v = os.environ.get("TRINOTE_GPU_RESIDENT_BATCH")
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _gpu_full_enabled() -> bool:
    """Opt-in (default OFF): also route RMSNorm + prefill-attention to the GPU, not just the Q1 applies.
    Both kernels are byte-exact and standalone-tested, but wiring them via PER-OP dispatch (operands transfer
    host↔device each call, and the v1 kernels are one-thread-per-row / per-(h,m)) currently REGRESSES prefill
    vs applies-only (measured 7.8s vs 4.9s @ T=64) — the transfers + unoptimized kernels outweigh the CPU work
    they remove. They only pay off once `x` stays device-resident across the layer (M3 true-residency, TODO).
    So `TRINOTE_GPU` alone = the proven applies-only win; `TRINOTE_GPU_FULL=1` opts into the full path (for the
    end-to-end byte-exact gate and as the residency foundation)."""
    v = os.environ.get("TRINOTE_GPU_FULL")
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _gpu_apply(owner: dict, name: str, x_fp: np.ndarray, frac: int):
    """Resident GPU Q1 apply for projection `name` on `owner` (a layer dict or the artifact). Registers the
    weight on the device ONCE (caching the handle on `owner`), then reuses it so only activations move per
    call — the residency win. Degrades cleanly: resident -> per-call upload -> None (CPU fallback). All paths
    are byte-identical to q1_linear_ref."""
    hkey = f"_gpu_h_{name}"
    h = owner.get(hkey)
    bits = owner[f"{name}_bits"]
    scale = owner[f"{name}_scale_fp"]
    if h is None:
        h = q1_register_weight(bits, scale)
        if h is None:                                    # residency unavailable -> per-call upload (or None->CPU)
            out = q1_apply_gpu(x_fp, bits, scale, frac)
            return _gpu_oracle_check(out, x_fp, bits, scale, frac, where=f"{name} (per-call)")
        owner[hkey] = h
        owner[f"{hkey}_shape"] = (int(scale.shape[0]), int(scale.shape[1]))
    out_f, n_blocks = owner[f"{hkey}_shape"]
    g = q1_apply_resident(h, x_fp, out_f, n_blocks, frac)
    if g is None:                                        # resident apply failed (e.g. overflow) -> per-call -> CPU
        out = q1_apply_gpu(x_fp, bits, scale, frac)
        return _gpu_oracle_check(out, x_fp, bits, scale, frac, where=f"{name} (resident-fallback)")
    return _gpu_oracle_check(g, x_fp, bits, scale, frac, where=f"{name} (resident)")


def _avail_ram_bytes():
    """Best-effort available RAM on Linux. Used only to avoid overcommitting the sign cache."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _attention_ref(x_fp, layer, cfg, cos, sin, frac, q1=_q1_bl_ref):
    T = x_fp.shape[0]
    H, hd = int(cfg["n_heads"]), int(cfg["head_dim"])
    Hkv = int(cfg.get("n_heads_kv", H))
    rep = H // Hkv
    native = q1 is _q1_bl_native
    qkv = _q1_bl_native_prepared_many(x_fp, layer, ("wq", "wk", "wv"), frac) if q1 is _q1_bl_native else None
    if qkv is None:
        q = q1(x_fp, layer, "wq", frac)
        k = q1(x_fp, layer, "wk", frac)
        v = q1(x_fp, layer, "wv", frac)
    else:
        q, k, v = qkv
    qh = q.reshape(T, H, hd).transpose(1, 0, 2)
    kh = k.reshape(T, Hkv, hd).transpose(1, 0, 2)
    vh = v.reshape(T, Hkv, hd).transpose(1, 0, 2)
    qh = _head_rmsnorm(qh, frac, layer["q_norm_gain_fp"], native=native)
    kh = _head_rmsnorm(kh, frac, layer["k_norm_gain_fp"], native=native)
    # The ONLY per-token float in this path. It depends solely on the constant head_dim (not on the data),
    # is a single correctly-rounded IEEE op feeding round() — cross-platform stable — and there is no float
    # REDUCTION here, which is the actual determinism guarantee. Everything downstream is pure integer.
    inv_sqrt_fp = round((1.0 / np.sqrt(hd)) * (1 << frac))
    out_heads = None
    # GPU M=N prefill attention (M3): same pre-RoPE-all-heads + causal kernel as the native path, byte-identical;
    # None -> falls through to the native CPU kernel, then the NumPy oracle.
    if native and T > 1 and _gpu_full_enabled():
        try:
            ga = attention_prefill_gpu(apply_rope_fixed_neox(qh, cos, sin, frac),
                                       apply_rope_fixed_neox(kh, cos, sin, frac), vh, 0, frac, inv_sqrt_fp)
        except (MemoryError, RuntimeError):
            ga = None
        if ga is not None:
            out_heads = ga
    # Native M=N prefill attention (deep-dive L5): pre-RoPE all heads (vectorized — identical to the per-head
    # rope in the loop below) and run the causal kernel. Byte-identical; None -> NumPy fallback on
    # overflow/unavailable. start=0 (full forward), so query i attends keys [0, i] like np.triu(k=1).
    if out_heads is None and native and T > 1 and _native_attn_enabled():
        try:
            na = attention_prefill_native(apply_rope_fixed_neox(qh, cos, sin, frac),
                                          apply_rope_fixed_neox(kh, cos, sin, frac), vh, 0, frac, inv_sqrt_fp)
        except (MemoryError, RuntimeError):
            na = None
        if na is not None:
            out_heads = na
    if out_heads is None:
        neg = -(1 << (frac + _NEG_INF_SHIFT))
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        out_heads = np.empty((H, T, hd), dtype=np.int64)
        for h in range(H):
            kv = h // rep
            qr = apply_rope_fixed_neox(qh[h], cos, sin, frac)
            kr = apply_rope_fixed_neox(kh[kv], cos, sin, frac)
            scores = fixed_point_matmul(qr, kr.T, frac)
            scores = (scores * inv_sqrt_fp) >> frac
            scores = np.where(causal, neg, scores)
            probs = fixed_point_softmax(scores, frac)
            out_heads[h] = fixed_point_matmul(probs, vh[kv], frac)
    attn = out_heads.transpose(1, 0, 2).reshape(T, H * hd)
    return q1(attn, layer, "wo", frac)


def _ffn_ref(x_fp, layer, frac, q1=_q1_bl_ref):
    native = q1 is _q1_bl_native
    gate_up = _q1_bl_native_prepared_many(x_fp, layer, ("w1", "wu"), frac) if native else None
    if gate_up is None:
        gate = q1(x_fp, layer, "w1", frac)
        up = q1(x_fp, layer, "wu", frac)
    else:
        gate, up = gate_up
    silu = fixed_point_silu(gate, frac, native=native)
    # silu*up shares the SiLU path's wrap-by-construction contract (the same shared NumPy line runs for the
    # oracle and native FFN, so it can never desync them); real gate/up are RMSNorm-bounded and in-envelope.
    h = (silu * up) >> frac
    return q1(h, layer, "w2", frac)


class _BonsaiKVCache:
    """Per-layer fixed-point K/V for positions [0, t), valid while the context window does not slide."""
    __slots__ = ("k", "v", "k_buf", "v_buf", "lengths", "t")

    def __init__(self, n_layers: int):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.k_buf = [None] * n_layers
        self.v_buf = [None] * n_layers
        self.lengths = [0] * n_layers
        self.t = 0

    def extend(self, li: int, kh: np.ndarray, vh: np.ndarray) -> None:
        old = self.lengths[li]
        add = int(kh.shape[1])
        need = old + add
        kb = self.k_buf[li]
        vb = self.v_buf[li]
        if kb is None:
            cap = max(need, 16)
            kb = np.empty((kh.shape[0], cap, kh.shape[2]), dtype=kh.dtype)
            vb = np.empty((vh.shape[0], cap, vh.shape[2]), dtype=vh.dtype)
        elif need > kb.shape[1]:
            cap = max(need, kb.shape[1] * 2)
            nk = np.empty((kb.shape[0], cap, kb.shape[2]), dtype=kb.dtype)
            nv = np.empty((vb.shape[0], cap, vb.shape[2]), dtype=vb.dtype)
            nk[:, :old, :] = kb[:, :old, :]
            nv[:, :old, :] = vb[:, :old, :]
            kb, vb = nk, nv
        kb[:, old:need, :] = kh
        vb[:, old:need, :] = vh
        self.k_buf[li] = kb
        self.v_buf[li] = vb
        self.lengths[li] = need
        self.k[li] = kb[:, :need, :]
        self.v[li] = vb[:, :need, :]


def _attention_with_cache_bonsai(x_fp, layer, cfg, cos, sin, frac, cache, li, start, q1=_q1_bl_ref):
    """KV-cached Qwen3 attention for new positions `[start, start + M)`, byte-identical to full attention."""
    M = x_fp.shape[0]
    H, hd = int(cfg["n_heads"]), int(cfg["head_dim"])
    Hkv = int(cfg.get("n_heads_kv", H))
    rep = H // Hkv
    native = q1 is _q1_bl_native
    qkv = _q1_bl_native_prepared_many(x_fp, layer, ("wq", "wk", "wv"), frac) if q1 is _q1_bl_native else None
    if qkv is None:
        q = q1(x_fp, layer, "wq", frac)
        k = q1(x_fp, layer, "wk", frac)
        v = q1(x_fp, layer, "wv", frac)
    else:
        q, k, v = qkv
    qh = q.reshape(M, H, hd).transpose(1, 0, 2)
    kh = k.reshape(M, Hkv, hd).transpose(1, 0, 2)
    vh = v.reshape(M, Hkv, hd).transpose(1, 0, 2)
    qh = _head_rmsnorm(qh, frac, layer["q_norm_gain_fp"], native=native)
    kh = _head_rmsnorm(kh, frac, layer["k_norm_gain_fp"], native=native)
    cpos, spos = cos[start:start + M], sin[start:start + M]
    qh = apply_rope_fixed_neox(qh, cpos, spos, frac)
    kh = apply_rope_fixed_neox(kh, cpos, spos, frac)
    cache.extend(li, kh, vh)
    L = start + M
    # Same single per-token float as _attention_ref: depends only on the constant head_dim, correctly
    # rounded, cross-platform stable; no float reduction (the true determinism guarantee). See above.
    inv_sqrt_fp = round((1.0 / np.sqrt(hd)) * (1 << frac))
    out_heads = None
    # Native M=1 decode attention: the single new query attends to all L cached positions, so the causal
    # mask is all-false and can be skipped. Byte-identical to the NumPy path below; on overflow it returns
    # None and we fall through to the NumPy path (which fails loud), preserving the no-silent-wrap contract.
    if native and M == 1 and _native_attn_enabled():
        # Pass the cache K/V slices directly: the native wrapper reads each head's contiguous (L,hd) block
        # using the buffer's inter-head stride, so no per-token copy of the growing cache is made. Wrap the
        # call like every other native dispatch site: overflow already returns None, but an out-of-envelope
        # arg (rc 1) raises — fall back to the loud NumPy path instead of crashing the decode.
        try:
            na = attention_decode_native(qh[:, 0, :], cache.k[li], cache.v[li], frac, inv_sqrt_fp)
        except (MemoryError, RuntimeError):
            na = None
        if na is not None:
            out_heads = na.reshape(H, 1, hd)
    # Native M=N PREFILL attention (deep-dive L5): byte-identical to the causal NumPy loop below; returns
    # None (-> NumPy fallback) on overflow or when unavailable. qh is (H,M,hd) post q-norm+RoPE; the cache
    # holds all L=start+M positions.
    if out_heads is None and native and M > 1 and _native_attn_enabled():
        try:
            na = attention_prefill_native(qh, cache.k[li], cache.v[li], start, frac, inv_sqrt_fp)
        except (MemoryError, RuntimeError):
            na = None
        if na is not None:
            out_heads = na
    if out_heads is None:
        neg = -(1 << (frac + _NEG_INF_SHIFT))
        mask = np.arange(L)[None, :] > (start + np.arange(M))[:, None]
        out_heads = np.empty((H, M, hd), dtype=np.int64)
        for h in range(H):
            kv = h // rep
            scores = fixed_point_matmul(qh[h], cache.k[li][kv].T, frac)
            scores = (scores * inv_sqrt_fp) >> frac
            scores = np.where(mask, neg, scores)
            probs = fixed_point_softmax(scores, frac)
            out_heads[h] = fixed_point_matmul(probs, cache.v[li][kv], frac)
    attn = out_heads.transpose(1, 0, 2).reshape(M, H * hd)
    return q1(attn, layer, "wo", frac)


def _attention_batched_decode(x_fp, layer, cfg, cos, sin, frac, caches, li, starts, q1=_q1_bl_ref):
    """One M=1 decode step for B sequences at once (request-batching). QKV + wo projections run as M=B (the
    throughput lever); q/k-norm + RoPE are batched (each sequence's own absolute position via cos[starts]); the
    B per-sequence attentions run in ONE native call over the ragged caches, with a NumPy fallback on the
    ALREADY-EXTENDED caches (so no double-extend). BYTE-IDENTICAL to running each sequence's M=1
    `_attention_with_cache_bonsai` (rows independent; per-sequence cache/position). Returns (B, d)."""
    B = x_fp.shape[0]
    H, hd = int(cfg["n_heads"]), int(cfg["head_dim"])
    Hkv = int(cfg.get("n_heads_kv", H))
    rep = H // Hkv
    half = hd // 2
    native = q1 is _q1_bl_native
    qkv = _q1_bl_native_prepared_many(x_fp, layer, ("wq", "wk", "wv"), frac) if native else None
    if qkv is None:
        q = q1(x_fp, layer, "wq", frac)
        k = q1(x_fp, layer, "wk", frac)
        v = q1(x_fp, layer, "wv", frac)
    else:
        q, k, v = qkv
    qh = _head_rmsnorm(q.reshape(B, H, hd), frac, layer["q_norm_gain_fp"], native=native)
    kh = _head_rmsnorm(k.reshape(B, Hkv, hd), frac, layer["k_norm_gain_fp"], native=native)
    vh = v.reshape(B, Hkv, hd)
    pos = np.asarray(starts, dtype=np.int64)             # each sequence's absolute position for this token
    c = cos[pos][:, None, :]                             # (B, 1, half) — broadcast over heads
    s = sin[pos][:, None, :]

    def _rope(t):                                        # rotate-half NeoX, == apply_rope_fixed_neox for M=1
        t0, t1 = t[..., :half], t[..., half:]
        o = np.empty_like(t)
        o[..., :half] = (t0 * c - t1 * s) >> frac
        o[..., half:] = (t0 * s + t1 * c) >> frac
        return o

    qh = _rope(qh)
    kh = _rope(kh)
    for b in range(B):
        caches[b].extend(li, kh[b][:, None, :], vh[b][:, None, :])
    inv_sqrt_fp = round((1.0 / np.sqrt(hd)) * (1 << frac))
    out_heads = None
    if native and _native_attn_enabled():
        try:
            out_heads = attention_decode_batched_native(
                qh, [caches[b].k[li] for b in range(B)], [caches[b].v[li] for b in range(B)],
                [int(caches[b].lengths[li]) for b in range(B)], frac, inv_sqrt_fp)
        except (MemoryError, RuntimeError):
            out_heads = None
    if out_heads is None:                                # NumPy fallback on the already-extended caches
        out_heads = np.empty((B, H, hd), dtype=np.int64)
        for b in range(B):
            kc, vc = caches[b].k[li], caches[b].v[li]    # (Hkv, L_b, hd) — single query attends all L_b keys
            for h in range(H):
                kv = h // rep
                scores = fixed_point_matmul(qh[b, h:h + 1], kc[kv].T, frac)
                scores = (scores * inv_sqrt_fp) >> frac
                out_heads[b, h] = fixed_point_matmul(fixed_point_softmax(scores, frac), vc[kv], frac)[0]
    return q1(out_heads.reshape(B, H * hd), layer, "wo", frac)


@dataclass
class BonsaiReferenceModel:
    receipt_verify_cached_threshold: ClassVar[int] = 64
    artifact: dict

    @property
    def cfg(self) -> dict:
        return self.artifact["config"]

    def _output_linear(self, x_fp: np.ndarray, frac: int, *, fast: bool) -> np.ndarray:
        a = self.artifact
        if getattr(self, "_native", False):
            if _gpu_enabled():                                     # output head is a Q1 apply too (resident)
                try:
                    g = _gpu_apply(a, "output", x_fp, frac)
                except (MemoryError, RuntimeError):
                    g = None
                if g is not None:
                    return g
            lut32 = _lut32_enabled()
            scale = a["output_scale_fp"] if lut32 else _scale_fp(a, "output")
            try:
                out = q1_linear_native(x_fp, a["output_bits"], scale, frac, lut32=lut32)
            except (MemoryError, RuntimeError):
                out = None
            if out is not None:
                return out
        if fast and "output_signs_i8" in a:
            return q1_linear_signs_ref(x_fp, a["output_signs_i8"], a["output_scale_fp"], frac)
        return q1_linear_ref(x_fp, a["output_bits"], a["output_scale_fp"], frac)

    def _output_argmax(self, x_fp: np.ndarray, frac: int, *, fast: bool) -> np.ndarray:
        a = self.artifact
        if getattr(self, "_native", False):
            if _gpu_enabled():
                # No fused GPU argmax kernel yet (the P4 lowest-index-tie kernel is milestone M4); use the GPU
                # logits + np.argmax, whose first-max tie-break is the committed lowest-index rule.
                return np.asarray(self._output_linear(x_fp, frac, fast=fast).argmax(axis=1), dtype=np.int64)
            lut32 = _lut32_enabled()
            scale = a["output_scale_fp"] if lut32 else _scale_fp(a, "output")
            try:
                ids = q1_argmax_native(x_fp, a["output_bits"], scale, frac, lut32=lut32)
            except (MemoryError, RuntimeError):
                ids = None
            if ids is not None:
                return ids
        return np.asarray(self._output_linear(x_fp, frac, fast=fast).argmax(axis=1), dtype=np.int64)

    def _forward_impl(self, token_ids: list[int] | np.ndarray, *, last_only: bool,
                      q1=_q1_bl_ref, output_fast: bool = False) -> np.ndarray:
        a, cfg = self.artifact, self.cfg
        frac = int(cfg["frac"])
        ids = np.asarray(token_ids, dtype=np.int64)
        T = ids.shape[0]
        rope_rows = a["cos_fp"].shape[0]
        if T > rope_rows:
            raise ValueError(f"sequence length {T} exceeds committed RoPE rows {rope_rows}")
        cos, sin = a["cos_fp"][:T], a["sin_fp"][:T]
        x = q1_rows_fp(a["embed_bits"], a["embed_scale_fp"], ids, frac)
        native = q1 is _q1_bl_native
        for layer in a["layers"]:
            n1 = _rmsnorm(x, frac, layer["n1_gain_fp"], native=native)
            x = x + _attention_ref(n1, layer, cfg, cos, sin, frac, q1=q1)
            n2 = _rmsnorm(x, frac, layer["n2_gain_fp"], native=native)
            x = x + _ffn_ref(n2, layer, frac, q1=q1)
        x = _rmsnorm(x, frac, a["final_norm_gain_fp"], native=native)
        return self._output_linear(x[-1:] if last_only else x, frac, fast=output_fast)

    def forward(self, token_ids: list[int] | np.ndarray, *, last_only: bool = False) -> np.ndarray:
        return self._forward_impl(token_ids, last_only=last_only, q1=_q1_bl_ref, output_fast=False)

    def forward_fast(self, token_ids: list[int] | np.ndarray, *, last_only: bool = False) -> np.ndarray:
        """Fast sign-cache forward. Byte-identical to `forward` when `enable_fast()` has populated caches."""
        if not getattr(self, "_fast", False) and not getattr(self, "_native", False):
            return self.forward(token_ids, last_only=last_only)
        q1 = _q1_bl_native if getattr(self, "_native", False) else _q1_bl_fast
        return self._forward_impl(token_ids, last_only=last_only, q1=q1, output_fast=True)

    def teacher_forced_logits(self, token_ids: list[int] | np.ndarray) -> np.ndarray:
        """Verifier hook: use byte-identical fast kernels when enabled, otherwise the packed oracle."""
        return self.forward_fast(token_ids) if getattr(self, "_fast", False) else self.forward(token_ids)

    def generate_greedy(self, token_ids: list[int], n_new: int,
                        *, rep_penalty_fp: int = 0, no_repeat_ngram: int = 0) -> list[int]:
        seq = list(token_ids)
        ctx = int(self.cfg["context_len"])
        for _ in range(n_new):
            row = self.forward(seq[-ctx:], last_only=True)[0]
            if rep_penalty_fp or no_repeat_ngram:
                row = apply_rep_penalty(row, seq, rep_penalty_fp, no_repeat_ngram, int(self.cfg["frac"]))
            seq.append(int(row.argmax()))
        return seq

    def enable_fast(self, *, check_ram: bool = True, cache_output: bool = True) -> bool:
        """Cache unpacked Q1_0 signs as int8 for the generation/verification fast path.

        This is a pure hoist of `_unpack_q1_signs(constant_bits)`: no arithmetic changes. It is RAM-gated
        because caching one byte per Q1 weight costs several GB for the real 8B artifact.
        """
        names = ("wq", "wk", "wv", "wo", "w1", "wu", "w2")
        layers = self.artifact["layers"]
        need = 0
        for layer in layers:
            for n in names:
                if f"{n}_signs_i8" not in layer:
                    need += int(layer[f"{n}_scale_fp"].size) * _GROUP
        if cache_output and "output_signs_i8" not in self.artifact:
            need += int(self.artifact["output_scale_fp"].size) * _GROUP
        if check_ram and need:
            avail = _avail_ram_bytes()
            if avail is not None and avail < int(need * 1.25):
                self._fast = False
                return False
        for layer in layers:
            for n in names:
                key = f"{n}_signs_i8"
                if key not in layer:
                    layer[key] = _unpack_q1_signs(layer[f"{n}_bits"])
        if cache_output and "output_signs_i8" not in self.artifact:
            self.artifact["output_signs_i8"] = _unpack_q1_signs(self.artifact["output_bits"])
        self._fast = True
        return True

    def enable_native(self) -> bool:
        """Enable the optional packed-Q1 C kernel when the shared library is built."""
        if not q1_native_available():
            self._native = False
            return False
        self._native = True
        self._fast = True
        if _scale_cache_enabled():
            self.enable_scale_cache()
        return True

    def enable_scale_cache(self, *, dtype=np.int32) -> bool:
        """Build a native-only narrow (int32) cache of the Q1 scales (Recommendation 7).

        The committed int64 artifact is unchanged; this is a runtime reproducer that roughly halves Q1
        scale-array bandwidth. It is byte-identical to the int64 kernel for in-range scales because the
        native `*_scale32` kernels promote each int32 scale to the same 64-bit multiply operand, and any
        native miss falls back to the committed int64 oracle. Returns False (leaving int64 scales in
        place) if ANY committed scale would lose precision in `dtype` — the cache is all-or-nothing so a
        weight never silently mixes representations.
        """
        info = np.iinfo(dtype)
        names = ("wq", "wk", "wv", "wo", "w1", "wu", "w2")
        targets = [(layer, f"{n}_scale_fp", f"{n}_scale_fp_i32")
                   for layer in self.artifact["layers"] for n in names]
        targets.append((self.artifact, "output_scale_fp", "output_scale_fp_i32"))
        for owner, src, _dst in targets:
            s = owner[src]
            if s.size and (int(s.min()) < info.min or int(s.max()) > info.max):
                self._scale_cache = False
                return False
        for owner, src, dst in targets:
            if dst not in owner:
                owner[dst] = np.ascontiguousarray(owner[src].astype(dtype))
        self._scale_cache = True
        return True

    def _run_layers(self, new_ids, cache: "_BonsaiKVCache") -> np.ndarray:
        """Run new positions through all layers while appending per-layer K/V to `cache`."""
        ids = list(new_ids)
        if not ids:
            raise ValueError("Bonsai cached decode requires at least one token")
        a, cfg = self.artifact, self.cfg
        frac = int(cfg["frac"])
        start = cache.t
        if start + len(ids) > int(a["cos_fp"].shape[0]):
            raise ValueError("cached Bonsai run exceeds committed RoPE rows")
        if getattr(self, "_native", False):
            q1 = _q1_bl_native
        else:
            q1 = _q1_bl_fast if getattr(self, "_fast", False) else _q1_bl_ref
        native = q1 is _q1_bl_native
        x = q1_rows_fp(a["embed_bits"], a["embed_scale_fp"], np.asarray(ids, dtype=np.int64), frac)
        cos, sin = a["cos_fp"], a["sin_fp"]
        for li, layer in enumerate(a["layers"]):
            n1 = _rmsnorm(x, frac, layer["n1_gain_fp"], native=native)
            x = x + _attention_with_cache_bonsai(
                n1, layer, cfg, cos, sin, frac, cache, li, start, q1=q1
            )
            n2 = _rmsnorm(x, frac, layer["n2_gain_fp"], native=native)
            x = x + _ffn_ref(n2, layer, frac, q1=q1)
        cache.t = start + len(ids)
        return x

    def prefill_logits(self, token_ids) -> np.ndarray:
        """Last-position logits via the KV-cache prefill. Byte-identical to `forward(..., last_only=True)`.
        Under TRINOTE_GPU, uses the M3 TRUE-RESIDENCY monolith (whole forward on-device, no per-op transfer)
        when available; falls back byte-identically to the CPU KV-cache path on None."""
        a = self.artifact
        frac = int(self.cfg["frac"])
        if getattr(self, "_native", False) and _gpu_enabled():
            g = self.prefill_logits_gpu_resident(token_ids)
            if g is not None:
                return g
        cache = _BonsaiKVCache(len(a["layers"]))
        x = self._run_layers(list(token_ids), cache)
        last = _rmsnorm(x[-1:], frac, a["final_norm_gain_fp"], native=getattr(self, "_native", False))
        return self._output_linear(last, frac, fast=getattr(self, "_fast", False))

    def _gpu_mono_handles(self):
        """Register all weights + upload all gains/cos/sin to the device ONCE (cached on the artifact), returning
        the handle bundle for the M3 resident prefill monolith. None if residency/buffer upload is unavailable."""
        a = self.artifact
        cached = a.get("_gpu_mono")
        if cached is not None:
            return cached
        if not monolith_available():
            return None
        layers = a["layers"]
        def regw(owner, name):
            return q1_register_weight(owner[f"{name}_bits"], owner[f"{name}_scale_fp"])
        def upg(arr):
            return buf_upload_i64(arr)
        w = {nm: [] for nm in ("wq", "wk", "wv", "wo", "w1", "wu", "w2")}
        g = {nm: [] for nm in ("n1g", "n2g", "qng", "kng")}
        gain_key = {"n1g": "n1_gain_fp", "n2g": "n2_gain_fp", "qng": "q_norm_gain_fp", "kng": "k_norm_gain_fp"}
        for layer in layers:
            for nm in w:
                h = regw(layer, nm)
                if h is None:
                    return None
                w[nm].append(h)
            for nm in g:
                h = upg(layer[gain_key[nm]])
                if h is None:
                    return None
                g[nm].append(h)
        out_head = regw(a, "output")
        finalg = upg(a["final_norm_gain_fp"])
        cos_h = upg(a["cos_fp"])                              # full table; the kernel indexes [:T]
        sin_h = upg(a["sin_fp"])
        if None in (out_head, finalg, cos_h, sin_h):
            return None
        bundle = {
            "w": {**{k: np.asarray(v, dtype=np.int64) for k, v in w.items()},
                  "out_head": out_head, "out_head_vocab": int(a["output_scale_fp"].shape[0])},
            "g": {**{k: np.asarray(v, dtype=np.int64) for k, v in g.items()},
                  "finalg": finalg, "cos_h": cos_h, "sin_h": sin_h},
        }
        a["_gpu_mono"] = bundle
        return bundle

    def prefill_logits_gpu_resident(self, token_ids):
        """M3 TRUE-RESIDENCY prefill: the whole forward runs on the GPU (residual never leaves the device),
        returning last-position logits (1, vocab) byte-identical to `forward(..., last_only=True)`. Returns None
        if the monolith/residency is unavailable or any kernel overflows (caller falls back to the CPU path)."""
        out = self._gpu_prefill(token_ids, want_kv=False)
        return None if out is None else out

    def _gpu_prefill(self, token_ids, *, want_kv: bool):
        """Run the M3 resident prefill monolith. want_kv=False -> last-position logits (1,vocab), or None.
        want_kv=True -> (logits (1,vocab), seeded _BonsaiKVCache) so generative-decode can continue on CPU
        byte-identically (the exported K/V are bit-for-bit what CPU prefill would have cached), or None."""
        a, cfg = self.artifact, self.cfg
        frac = int(cfg["frac"])
        ids = list(token_ids)
        T = len(ids)
        if T < 1:
            return None
        bundle = self._gpu_mono_handles()
        if bundle is None:
            return None
        H, hd = int(cfg["n_heads"]), int(cfg["head_dim"])
        Hkv = int(cfg.get("n_heads_kv", H))
        n_layers = len(a["layers"])
        x_embed = q1_rows_fp(a["embed_bits"], a["embed_scale_fp"], np.asarray(ids, dtype=np.int64), frac)
        d = int(x_embed.shape[1])
        dff = int(a["layers"][0]["w1_scale_fp"].shape[0])
        inv_sqrt_fp = round((1.0 / np.sqrt(hd)) * (1 << frac))
        dims = (T, d, n_layers, H, Hkv, hd, dff)
        if not want_kv:
            logits = prefill_forward_gpu(x_embed, dims, bundle["w"], bundle["g"], frac, 1, inv_sqrt_fp)
            return None if logits is None else logits.reshape(1, -1)
        logits, k_arr, v_arr = prefill_forward_gpu(x_embed, dims, bundle["w"], bundle["g"], frac, 1,
                                                   inv_sqrt_fp, export_kv=True)
        if logits is None:
            return None
        cache = _BonsaiKVCache(n_layers)
        for li in range(n_layers):
            cache.extend(li, k_arr[li], v_arr[li])         # seeds k/v/lengths; byte-identical to CPU prefill
        cache.t = T
        return logits.reshape(1, -1), cache

    def generate_cached(self, token_ids, n_new, pick, *, eos=None, on_token=None) -> list[int]:
        """KV-cached decode returning new tokens. Hidden/logit bytes match `forward` while the window holds."""
        a = self.artifact
        frac = int(self.cfg["frac"])
        window = min(int(self.cfg["context_len"]), int(a["cos_fp"].shape[0]))
        prompt = list(token_ids)
        if not prompt:
            return []
        if len(prompt) + int(n_new) > window:
            return self._generate_uncached(prompt, n_new, pick, eos, on_token)
        seq = list(prompt)
        out: list[int] = []
        n_steps = int(n_new)
        # GPU resident-prefill path: seed the KV cache on-device + get the last-position logits, then decode on
        # CPU from the byte-identical cache. Falls back to the CPU KV-prefill on None (unavailable/overflow).
        row0 = None
        cache = None
        if getattr(self, "_native", False) and _gpu_enabled():
            gp = self._gpu_prefill(seq, want_kv=True)
            if gp is not None:
                logits0, cache = gp
                row0 = logits0[0]
        if cache is None:                                  # CPU prefill (the committed path)
            cache = _BonsaiKVCache(len(a["layers"]))
            x = self._run_layers(seq, cache)
            row0 = self._output_linear(
                _rmsnorm(x[-1:], frac, a["final_norm_gain_fp"], native=getattr(self, "_native", False)),
                frac, fast=getattr(self, "_fast", False))[0]
        for step in range(n_steps):
            row = row0 if step == 0 else self._output_linear(
                _rmsnorm(self._run_layers([seq[-1]], cache)[-1:], frac, a["final_norm_gain_fp"],
                         native=getattr(self, "_native", False)),
                frac, fast=getattr(self, "_fast", False))[0]
            tok = int(pick(row, len(seq), seq))
            seq.append(tok)
            out.append(tok)
            if eos is not None and tok == eos:
                break
            if on_token is not None:
                on_token(tok)
            # next step's decode (run_layers([seq[-1]])) happens at the top of the loop
        return out

    def generate_greedy_tokens_cached(self, token_ids, n_new, *, eos=None, on_token=None) -> list[int]:
        """KV-cached plain greedy decode using native output-argmax when available.

        This is only for unpenalized greedy. Sampling and repetition controls need the full logits row and
        continue through `generate_cached`.
        """
        a = self.artifact
        frac = int(self.cfg["frac"])
        window = min(int(self.cfg["context_len"]), int(a["cos_fp"].shape[0]))
        prompt = list(token_ids)
        if not prompt:
            return []
        if len(prompt) + int(n_new) > window:
            return self._generate_uncached(
                prompt,
                n_new,
                lambda row, _pos, _hist: int(np.asarray(row).argmax()),
                eos,
                on_token,
            )
        cache = _BonsaiKVCache(len(a["layers"]))
        seq = list(prompt)
        x = self._run_layers(seq, cache)
        out: list[int] = []
        n_steps = int(n_new)
        for step in range(n_steps):
            last = _rmsnorm(x[-1:], frac, a["final_norm_gain_fp"], native=getattr(self, "_native", False))
            tok = int(self._output_argmax(last, frac, fast=getattr(self, "_fast", False))[0])
            seq.append(tok)
            out.append(tok)
            if eos is not None and tok == eos:
                break
            if on_token is not None:
                on_token(tok)
            if step + 1 < n_steps:
                x = self._run_layers([tok], cache)
        return out

    def _generate_uncached(self, token_ids, n_new, pick, eos, on_token) -> list[int]:
        ctx = int(self.cfg["context_len"])
        seq, out = list(token_ids), []
        for _ in range(int(n_new)):
            row = self.forward(seq[-ctx:], last_only=True)[0]
            tok = int(pick(row, len(seq), seq))
            seq.append(tok)
            out.append(tok)
            if eos is not None and tok == eos:
                break
            if on_token is not None:
                on_token(tok)
        return out

    def generate_greedy_cached(self, token_ids: list[int], n_new: int,
                               *, rep_penalty_fp: int = 0, no_repeat_ngram: int = 0) -> list[int]:
        if not rep_penalty_fp and not no_repeat_ngram:
            return list(token_ids) + self.generate_greedy_tokens_cached(token_ids, n_new)
        frac = int(self.cfg["frac"])

        def _pick(row, _pos, hist):
            if rep_penalty_fp or no_repeat_ngram:
                row = apply_rep_penalty(row, hist, rep_penalty_fp, no_repeat_ngram, frac)
            return int(np.asarray(row).argmax())

        return list(token_ids) + self.generate_cached(token_ids, n_new, _pick)

    def _run_layers_batched(self, new_tokens, caches) -> np.ndarray:
        """One M=B decode step over B independent sequences (one new token each). The QKV/FFN/output gathers
        run as M=B rows (the request-batching throughput win); attention is per-sequence (each uses its own
        KV cache + position). BYTE-IDENTICAL to running each sequence's `_run_layers([tok])` standalone — every
        gather is per-row independent, RMSNorm/SiLU are per-row, and each cache/position is private. Returns
        x (B, d) and advances each cache by 1."""
        a, cfg = self.artifact, self.cfg
        frac = int(cfg["frac"])
        if getattr(self, "_native", False):
            q1 = _q1_bl_native
        else:
            q1 = _q1_bl_fast if getattr(self, "_fast", False) else _q1_bl_ref
        native = q1 is _q1_bl_native
        cos, sin = a["cos_fp"], a["sin_fp"]
        starts = [c.t for c in caches]                       # each sequence's position at this step's start
        x = q1_rows_fp(a["embed_bits"], a["embed_scale_fp"], np.asarray(new_tokens, dtype=np.int64), frac)
        B = len(new_tokens)
        for li, layer in enumerate(a["layers"]):
            n1 = _rmsnorm(x, frac, layer["n1_gain_fp"], native=native)            # (B, d)
            x = x + _attention_batched_decode(n1, layer, cfg, cos, sin, frac, caches, li, starts, q1=q1)
            n2 = _rmsnorm(x, frac, layer["n2_gain_fp"], native=native)            # (B, d)
            x = x + _ffn_ref(n2, layer, frac, q1=q1)                              # (B, d) BATCHED gather
        for b in range(B):
            caches[b].t = starts[b] + 1
        return x

    def _batched_decode_resident(self, seqs, x_prefill, caches, n_new, picks, eos):
        """Fully-resident M=B batched decode: KV cache + RMSNorm/RoPE/attention live on the GPU across all
        steps (only (B,d) in / (B,vocab) logits out per step). Byte-identical to the CPU batched/sequential
        path. Returns the B new-token lists, or None to fall back (unavailable / a step overflows). Does NOT
        mutate the caller's `seqs` (works on copies) so the fallback path stays clean."""
        a, cfg = self.artifact, self.cfg
        frac = int(cfg["frac"])
        B = len(seqs)
        n_layers = len(a["layers"])
        H, hd = int(cfg["n_heads"]), int(cfg["head_dim"])
        Hkv = int(cfg.get("n_heads_kv", H))
        d = int(x_prefill.shape[1])
        dff = int(a["layers"][0]["w1_scale_fp"].shape[0])
        vocab = int(a["output_scale_fp"].shape[0])
        inv_sqrt_fp = round((1.0 / np.sqrt(hd)) * (1 << frac))
        bundle = self._gpu_mono_handles()
        if bundle is None:
            return None
        cap = max(len(s) for s in seqs) + int(n_new) + 1
        ctx = decode_ctx_create((B, n_layers, H, Hkv, hd, d, dff, vocab),
                                bundle["w"], bundle["g"], frac, 1, inv_sqrt_fp, cap)
        if ctx is None:
            return None
        try:
            for b in range(B):                                   # seed each sequence's prefilled KV
                k_arr = np.stack([caches[b].k[li] for li in range(n_layers)])   # (n_layers,Hkv,Lb,hd)
                v_arr = np.stack([caches[b].v[li] for li in range(n_layers)])
                if not decode_ctx_seed_seq(ctx, b, k_arr, v_arr):
                    return None
            work = [list(s) for s in seqs]                       # copies; never touch the caller's seqs
            outs = [[] for _ in range(B)]
            done = [False] * B
            clen = [len(s) for s in seqs]                        # cache length per seq (grows uniformly per step)
            # step-0 logits from the prefill hidden (same as the CPU loop's first iteration)
            logits = self._output_linear(_rmsnorm(x_prefill, frac, a["final_norm_gain_fp"], native=True),
                                         frac, fast=True)
            for step in range(int(n_new)):
                next_toks = []
                for b in range(B):
                    if done[b]:
                        next_toks.append(int(eos) if eos is not None else 0)
                        continue
                    tok = int(picks[b](logits[b], len(work[b]), work[b]))
                    work[b].append(tok)
                    outs[b].append(tok)
                    if eos is not None and tok == eos:
                        done[b] = True
                    next_toks.append(tok)
                if all(done) or step + 1 >= int(n_new):
                    break
                x_in = q1_rows_fp(a["embed_bits"], a["embed_scale_fp"], np.asarray(next_toks, dtype=np.int64), frac)
                logits = decode_step(ctx, x_in, np.asarray(clen, dtype=np.int64), vocab)   # appends KV at clen
                if logits is None:                              # overflow -> abandon, fall back to CPU path
                    return None
                clen = [c + 1 for c in clen]
            return outs
        finally:
            decode_ctx_free(ctx)

    def generate_batched(self, prompts, n_new, picks=None, *, eos=None) -> list[list[int]]:
        """Decode B prompts together (M=B steps) for higher served throughput — returns B lists of NEW tokens,
        BYTE-IDENTICAL to generating each prompt standalone via `generate_cached` (rows independent;
        per-sequence KV/position/sampler). `picks` is one `(row,pos,hist)->tok` callable for all, or a list of
        B (e.g. per-sequence samplers); default = greedy argmax. Finished (eos) sequences are masked but kept
        in the batch. Falls back to per-sequence `generate_cached` if any prompt+n_new exceeds the window."""
        a, cfg = self.artifact, self.cfg
        frac = int(cfg["frac"])
        n_layers = len(a["layers"])
        window = min(int(cfg["context_len"]), int(a["cos_fp"].shape[0]))
        B = len(prompts)
        seqs = [list(p) for p in prompts]
        if picks is None or callable(picks):
            one = picks if callable(picks) else (lambda row, _pos, _hist: int(np.asarray(row).argmax()))
            picks = [one] * B
        if any(len(seqs[b]) + int(n_new) > window for b in range(B)):             # ragged-too-long fallback
            return [self.generate_cached(seqs[b], n_new, picks[b], eos=eos) for b in range(B)]
        caches = [_BonsaiKVCache(n_layers) for _ in range(B)]
        x = np.stack([self._run_layers(seqs[b], caches[b])[-1] for b in range(B)])  # per-seq prefill -> (B, d)
        # Fully-resident M=B batched decode (KV + RMSNorm/RoPE/attention on device). Byte-identical but OFF by
        # default: it regresses vs the existing GPU batch path at decode batch sizes on sm_86 (tiny-row RMSNorm
        # occupancy + per-step launch overhead) — opt in with TRINOTE_GPU_RESIDENT_BATCH=1. The DEFAULT GPU
        # batch path is the M=B applies-on-GPU loop below (RMSNorm/attention on CPU), which is faster here.
        if (getattr(self, "_native", False) and _gpu_enabled() and _gpu_resident_batch_enabled()
                and batched_decode_available()):
            res = self._batched_decode_resident(seqs, x, caches, int(n_new), picks, eos)
            if res is not None:
                return res
        outs: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B
        n_steps = int(n_new)
        native = getattr(self, "_native", False)
        for step in range(n_steps):
            last = _rmsnorm(x, frac, a["final_norm_gain_fp"], native=native)      # (B, d)
            logits = self._output_linear(last, frac, fast=getattr(self, "_fast", False))   # (B, vocab)
            next_toks = []
            for b in range(B):
                if done[b]:
                    next_toks.append(int(eos) if eos is not None else 0)          # filler; its output ignored
                    continue
                tok = int(picks[b](logits[b], len(seqs[b]), seqs[b]))
                seqs[b].append(tok)
                outs[b].append(tok)
                if eos is not None and tok == eos:
                    done[b] = True
                next_toks.append(tok)
            if all(done):
                break
            if step + 1 < n_steps:
                x = self._run_layers_batched(next_toks, caches)                   # (B, d) — the batched step
        return outs


def random_bonsai_artifact(cfg_params: dict, *, seq_len: int = 32, seed: int = 0) -> dict:
    """Small random Bonsai-shaped artifact for smoke tests."""
    rng = np.random.default_rng(seed)
    d = int(cfg_params["dModel"])
    nh = int(cfg_params["nHeads"])
    nkv = int(cfg_params.get("nHeadsKv", nh))
    hd = int(cfg_params.get("headDim", d // nh))
    dff = int(cfg_params["dFfn"])
    vocab = int(cfg_params["vocab"])
    nl = int(cfg_params["nLayers"])
    frac = int(cfg_params["fpFracBits"])

    def gain(dim):
        return np.full(dim, 1 << frac, dtype=np.int64)

    def q1(out_f, in_f):
        assert in_f % _GROUP == 0
        n_blocks = in_f // _GROUP
        bits = rng.integers(0, 256, size=(out_f, n_blocks, _GROUP // 8), dtype=np.uint8)
        scale = np.full((out_f, n_blocks), int(round(0.03 * (1 << frac))), dtype=np.int64)
        return bits, scale

    layers = []
    for _ in range(nl):
        wq_b, wq_s = q1(d, d)
        wk_b, wk_s = q1(nkv * hd, d)
        wv_b, wv_s = q1(nkv * hd, d)
        wo_b, wo_s = q1(d, d)
        w1_b, w1_s = q1(dff, d)
        wu_b, wu_s = q1(dff, d)
        w2_b, w2_s = q1(d, dff)
        layers.append({
            "n1_gain_fp": gain(d), "n2_gain_fp": gain(d),
            "q_norm_gain_fp": gain(hd), "k_norm_gain_fp": gain(hd),
            "wq_bits": wq_b, "wq_scale_fp": wq_s,
            "wk_bits": wk_b, "wk_scale_fp": wk_s,
            "wv_bits": wv_b, "wv_scale_fp": wv_s,
            "wo_bits": wo_b, "wo_scale_fp": wo_s,
            "w1_bits": w1_b, "w1_scale_fp": w1_s,
            "wu_bits": wu_b, "wu_scale_fp": wu_s,
            "w2_bits": w2_b, "w2_scale_fp": w2_s,
        })
    embed_b, embed_s = q1(vocab, d)
    out_b, out_s = q1(vocab, d)
    cos, sin = build_rope_tables(seq_len, hd, int(cfg_params["ropeBase"]), frac)
    return {
        "config": {
            "architecture": "qwen3", "dModel": d, "nLayers": nl, "n_heads": nh,
            "n_heads_kv": nkv, "head_dim": hd, "dFfn": dff, "vocab": vocab,
            "context_len": seq_len, "frac": frac, "ropeBase": int(cfg_params["ropeBase"]),
            "ropeScalingType": cfg_params.get("ropeScalingType", "none"),
        },
        "embed_bits": embed_b, "embed_scale_fp": embed_s,
        "output_bits": out_b, "output_scale_fp": out_s,
        "final_norm_gain_fp": gain(d), "cos_fp": cos, "sin_fp": sin,
        "layers": layers,
    }

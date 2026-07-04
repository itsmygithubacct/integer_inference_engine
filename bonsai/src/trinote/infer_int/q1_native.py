"""Optional native packed-Q1_0 linear kernel for Bonsai."""
from __future__ import annotations

import ctypes
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..notary_paths import kernel_so


_ROOT = Path(__file__).resolve().parents[3]
_LIB = Path(kernel_so())   # prefers ~/.local/trinote/bin, falls back to <repo>/tools (back-compat)
_TLS = threading.local()
_DEFAULT_WORKSPACE_MAX_MB = 64


@lru_cache(maxsize=1)
def _load_lib():
    if not _LIB.exists():
        return None
    # A stale/partial/wrong-arch .so should degrade to "native unavailable" (oracle fallback), not crash the
    # whole engine. Catch the load error AND a missing core symbol and return None — @lru_cache memoizes the
    # None, so q1_native_available() stays cheap and honest instead of raising on every call.
    try:
        lib = ctypes.CDLL(str(_LIB))
    except OSError:
        return None
    try:
        fn = lib.bonsai_q1_linear_i64
    except AttributeError:
        return None
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    try:
        fnw = lib.bonsai_q1_linear_i64_workspace
    except AttributeError:
        pass
    else:
        fnw.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnw.restype = ctypes.c_int
    try:
        fna = lib.bonsai_q1_argmax_i64_workspace
    except AttributeError:
        pass
    else:
        fna.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fna.restype = ctypes.c_int
    try:
        fnp = lib.bonsai_q1_prepare_i64
    except AttributeError:
        pass
    else:
        fnp.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnp.restype = ctypes.c_int
    try:
        fnlp = lib.bonsai_q1_linear_i64_prepared
    except AttributeError:
        pass
    else:
        fnlp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnlp.restype = ctypes.c_int
    try:
        fnmp = lib.bonsai_q1_linear_i64_prepared_multi
    except AttributeError:
        pass
    else:
        fnmp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnmp.restype = ctypes.c_int
    try:
        fnr = lib.bonsai_rmsnorm_i64
    except AttributeError:
        pass
    else:
        fnr.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        fnr.restype = ctypes.c_int
    # Optional narrow int32 scale-cache variants (Recommendation 7). Same ctypes layout as their int64
    # twins — only the C-level scale element type differs — so the int64 argtype lists are reused.
    _scale32_argtypes = {
        "bonsai_q1_linear_i64_workspace_scale32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_argmax_i64_workspace_scale32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_prepared_scale32": [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_prepared_multi_scale32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ],
    }
    for _sym, _argtypes in _scale32_argtypes.items():
        try:
            _fn = getattr(lib, _sym)
        except AttributeError:
            continue
        _fn.argtypes = _argtypes
        _fn.restype = ctypes.c_int
    # int32 activation-LUT-entry variants (same ctypes layout as the uint64 twins — the LUT pointer is a
    # void_p either way; only the C element type differs).
    _lut32_argtypes = {
        "bonsai_q1_prepare_i64_lut32": [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_workspace_lut32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_prepared_lut32": [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_prepared_multi_lut32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_argmax_i64_workspace_lut32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
    }
    for _sym, _argtypes in _lut32_argtypes.items():
        try:
            _fn = getattr(lib, _sym)
        except AttributeError:
            continue
        _fn.argtypes = _argtypes
        _fn.restype = ctypes.c_int
    try:
        fnsilu = lib.bonsai_silu_i64
    except AttributeError:
        pass
    else:
        fnsilu.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        fnsilu.restype = ctypes.c_int
    try:
        fnattn = lib.bonsai_attention_decode_i64
    except AttributeError:
        pass
    else:
        fnattn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnattn.restype = ctypes.c_int
    try:
        fnattp = lib.bonsai_attention_prefill_i64
    except AttributeError:
        pass
    else:
        fnattp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                # q, k, v
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                   # H, Hkv, hd
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                   # M, L, start
            ctypes.c_int64, ctypes.c_int64,                                   # frac, inv_sqrt_fp
            ctypes.c_void_p,                                                  # out
        ]
        fnattp.restype = ctypes.c_int
    try:
        fnattb = lib.bonsai_attention_decode_batched_i64
    except AttributeError:
        pass
    else:
        fnattb.argtypes = [
            ctypes.c_void_p,                                                  # q (B,H,hd)
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                # k_ptrs[B], v_ptrs[B], lengths[B]
            ctypes.c_void_p, ctypes.c_void_p,                                 # k_kv_strides[B], v_kv_strides[B]
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,   # B, H, Hkv, hd
            ctypes.c_int64, ctypes.c_int64,                                   # frac, inv_sqrt_fp
            ctypes.c_void_p,                                                  # out (B,H,hd)
        ]
        fnattb.restype = ctypes.c_int
    return lib


def q1_native_available() -> bool:
    return _load_lib() is not None


def _workspace_max_bytes() -> int:
    try:
        mb = int(os.environ.get("TRINOTE_Q1_WORKSPACE_MAX_MB", str(_DEFAULT_WORKSPACE_MAX_MB)))
    except ValueError:
        mb = _DEFAULT_WORKSPACE_MAX_MB
    return max(0, mb) * 1024 * 1024


def _workspace_arrays(total_count: int, lut_count: int):
    need_bytes = (int(total_count) + int(lut_count)) * np.dtype(np.uint64).itemsize
    if need_bytes <= 0 or need_bytes > _workspace_max_bytes():
        return None
    totals, lut = getattr(_TLS, "q1_workspace", (None, None))
    if totals is None or totals.size < total_count:
        totals = np.empty(total_count, dtype=np.uint64)
    if lut is None or lut.size < lut_count:
        lut = np.empty(lut_count, dtype=np.uint64)
    _TLS.q1_workspace = (totals, lut)
    return totals, lut


def _workspace_arrays_lut32(total_count: int, lut_count: int):
    """Like _workspace_arrays but the activation LUT is int32 (half the bytes). Separate TLS slot so it does
    not alias the uint64 LUT workspace. Sized by the int32 LUT footprint against the same MB cap."""
    need_bytes = int(total_count) * np.dtype(np.uint64).itemsize + int(lut_count) * np.dtype(np.int32).itemsize
    if need_bytes <= 0 or need_bytes > _workspace_max_bytes():
        return None
    totals, lut = getattr(_TLS, "q1_workspace_lut32", (None, None))
    if totals is None or totals.size < total_count:
        totals = np.empty(total_count, dtype=np.uint64)
    if lut is None or lut.size < lut_count:
        lut = np.empty(lut_count, dtype=np.int32)
    _TLS.q1_workspace_lut32 = (totals, lut)
    return totals, lut


_DEFAULT_ATTN_SCRATCH_MAX_MB = 128


def _attn_scratch_max_bytes() -> int:
    try:
        mb = int(os.environ.get("TRINOTE_ATTN_SCRATCH_MAX_MB", str(_DEFAULT_ATTN_SCRATCH_MAX_MB)))
    except ValueError:
        mb = _DEFAULT_ATTN_SCRATCH_MAX_MB
    return max(0, mb) * 1024 * 1024


def _attn_scratch(count: int):
    """Thread-local per-head scores/probs scratch (H*L int64) for the native attention kernel."""
    need_bytes = int(count) * np.dtype(np.int64).itemsize
    if need_bytes <= 0 or need_bytes > _attn_scratch_max_bytes():
        return None
    s = getattr(_TLS, "attn_scratch", None)
    if s is None or s.size < count:
        s = np.empty(count, dtype=np.int64)
        _TLS.attn_scratch = s
    return s


class Q1Prepared:
    """Prepared activation LUT for applying multiple Q1_0 projections to the same input."""

    __slots__ = ("x", "tokens", "n_blocks", "totals", "lut")

    def __init__(self, x: np.ndarray, tokens: int, n_blocks: int,
                 totals: np.ndarray, lut: np.ndarray):
        self.x = x
        self.tokens = int(tokens)
        self.n_blocks = int(n_blocks)
        self.totals = totals
        self.lut = lut


def _contiguous_q1_weight(bits: np.ndarray, scale_fp: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    b = np.asarray(bits, dtype=np.uint8)
    src = np.asarray(scale_fp)
    # The scale is a fixed-point integer; a float (or otherwise non-integer) dtype here would be silently
    # truncated by ascontiguousarray(dtype=int64) and corrupt the result. Reject it loudly.
    if not np.issubdtype(src.dtype, np.integer):
        raise TypeError(f"Q1_0 scale must be an integer dtype, got {src.dtype}")
    # Preserve a narrow int32 scale cache (Recommendation 7) so the *_scale32 kernels can read it directly;
    # otherwise canonicalize to the committed int64 scale. int32 is lossless for any in-range scale and the
    # native math is byte-identical (q1_element_s32 promotes to the same 64-bit operand).
    dtype = np.int32 if src.dtype == np.int32 else np.int64
    s = np.ascontiguousarray(src, dtype=dtype)
    if not b.flags.c_contiguous:
        b = np.ascontiguousarray(b)
    out_f, n_blocks = s.shape
    if b.shape != (out_f, n_blocks, 16):
        raise ValueError(f"Q1_0 bits shape {b.shape} does not match {(out_f, n_blocks, 16)}")
    return b, s, int(out_f), int(n_blocks)


def q1_prepare_native(x_fp: np.ndarray, n_blocks: int, *, lut32: bool = False) -> Q1Prepared | None:
    """Prepare a native activation LUT reusable across Q1_0 weights with the same input width.

    With lut32=True the LUT entries are int32 (half the gather bytes); returns None if the int32 symbol is
    absent, the workspace is too large, or a block exceeds the int32 envelope (rc 5) — the caller then
    retries with the uint64 LUT. The returned Q1Prepared's `lut.dtype` (int32 vs uint64) tells the apply
    wrappers which kernel to dispatch."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_prepare_i64"):
        return None
    n_blocks = int(n_blocks)
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    if x.shape[1] != n_blocks * 128:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * 128}")
    total_count = int(x.shape[0]) * n_blocks
    lut_count = total_count * 16 * 256
    if lut32:
        fn = getattr(lib, "bonsai_q1_prepare_i64_lut32", None)
        if fn is None:
            return None
        workspace = _workspace_arrays_lut32(total_count, lut_count)
        if workspace is None:
            return None
        totals, lut = workspace
    else:
        fn = lib.bonsai_q1_prepare_i64
        workspace = _workspace_arrays(total_count, lut_count)
        if workspace is None:
            return None
        totals, lut = workspace
    rc = fn(
        x.ctypes.data,
        ctypes.c_int64(x.shape[0]),
        ctypes.c_int64(n_blocks),
        totals.ctypes.data,
        ctypes.c_size_t(totals.size),
        lut.ctypes.data,
        ctypes.c_size_t(lut.size),
    )
    if rc == 3 or rc == 5:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_prepare_i64 failed with code {rc}")
    return Q1Prepared(x, x.shape[0], n_blocks, totals, lut)


def q1_linear_prepared_native(prepared: Q1Prepared, bits: np.ndarray,
                              scale_fp: np.ndarray, frac: int) -> np.ndarray | None:
    """Apply a packed-Q1_0 linear layer using a prepared activation LUT."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_linear_i64_prepared"):
        return None
    b, s, out_f, n_blocks = _contiguous_q1_weight(bits, scale_fp)
    if n_blocks != prepared.n_blocks:
        raise ValueError(
            f"Q1_0 prepared n_blocks mismatch: prepared {prepared.n_blocks}, weight expects {n_blocks}"
        )
    out = np.empty((prepared.tokens, out_f), dtype=np.int64)
    if prepared.lut.dtype == np.int32:
        # int32-LUT prepared kernel reads an int64 scale; an int32-LUT prepare implies its apply symbol.
        fn = getattr(lib, "bonsai_q1_linear_i64_prepared_lut32", None)
        if fn is None:
            return None
        if s.dtype != np.int64:
            s = np.ascontiguousarray(s, dtype=np.int64)
    elif s.dtype == np.int32:
        fn = getattr(lib, "bonsai_q1_linear_i64_prepared_scale32", None)
        if fn is None:
            return None
    else:
        fn = lib.bonsai_q1_linear_i64_prepared
    rc = fn(
        b.ctypes.data,
        s.ctypes.data,
        ctypes.c_int64(prepared.tokens),
        ctypes.c_int64(out_f),
        ctypes.c_int64(n_blocks),
        ctypes.c_int64(int(frac)),
        out.ctypes.data,
        prepared.totals.ctypes.data,
        ctypes.c_size_t(prepared.totals.size),
        prepared.lut.ctypes.data,
        ctypes.c_size_t(prepared.lut.size),
    )
    if rc == 3:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_linear_i64_prepared failed with code {rc}")
    return out


def q1_linear_prepared_many_native(prepared: Q1Prepared, weights, frac: int) -> tuple[np.ndarray, ...] | None:
    """Apply multiple packed-Q1_0 linears against one prepared activation LUT in a single native call."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_linear_i64_prepared_multi"):
        return None
    packed = []
    out_features = []
    outs = []
    narrow = True
    for bits, scale_fp in weights:
        b, s, out_f, n_blocks = _contiguous_q1_weight(bits, scale_fp)
        if n_blocks != prepared.n_blocks:
            raise ValueError(
                f"Q1_0 prepared n_blocks mismatch: prepared {prepared.n_blocks}, weight expects {n_blocks}"
            )
        packed.append((b, s))
        out_features.append(out_f)
        outs.append(np.empty((prepared.tokens, out_f), dtype=np.int64))
        if s.dtype != np.int32:
            narrow = False
    if not packed:
        return ()
    if prepared.lut.dtype == np.int32:
        # int32-LUT multi kernel reads int64 scales; only it can read the int32 LUT, so a missing symbol
        # means fall back (None) rather than handing the int32 LUT to a uint64-LUT kernel.
        fn = getattr(lib, "bonsai_q1_linear_i64_prepared_multi_lut32", None)
        if fn is None:
            return None
        packed = [(b, s if s.dtype == np.int64 else np.ascontiguousarray(s, dtype=np.int64))
                  for b, s in packed]
    else:
        fn = getattr(lib, "bonsai_q1_linear_i64_prepared_multi_scale32", None) if narrow else None
        if fn is None:
            # int64-kernel path: canonicalize EVERY scale to int64 first so an int32 cache array is never
            # handed to the int64 kernel, which would read its 4-byte entries as 8-byte scales and silently
            # corrupt the logits (the worst determinism failure). This covers a mixed int32/int64 batch and
            # the case where the scale32 symbol is absent; genuine int64 arrays pass through untouched.
            packed = [(b, s if s.dtype == np.int64 else np.ascontiguousarray(s, dtype=np.int64))
                      for b, s in packed]
            fn = lib.bonsai_q1_linear_i64_prepared_multi
    ptr_array = ctypes.c_void_p * len(packed)
    bits_ptrs = ptr_array(*(b.ctypes.data for b, _s in packed))
    scale_ptrs = ptr_array(*(s.ctypes.data for _b, s in packed))
    out_ptrs = ptr_array(*(out.ctypes.data for out in outs))
    out_f_arr = np.ascontiguousarray(np.asarray(out_features, dtype=np.int64))
    rc = fn(
        ctypes.cast(bits_ptrs, ctypes.c_void_p),
        ctypes.cast(scale_ptrs, ctypes.c_void_p),
        out_f_arr.ctypes.data,
        ctypes.c_int64(len(packed)),
        ctypes.c_int64(prepared.tokens),
        ctypes.c_int64(prepared.n_blocks),
        ctypes.c_int64(int(frac)),
        ctypes.cast(out_ptrs, ctypes.c_void_p),
        prepared.totals.ctypes.data,
        ctypes.c_size_t(prepared.totals.size),
        prepared.lut.ctypes.data,
        ctypes.c_size_t(prepared.lut.size),
    )
    if rc == 3:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_linear_i64_prepared_multi failed with code {rc}")
    return tuple(outs)


def q1_linear_native(x_fp: np.ndarray, bits: np.ndarray, scale_fp: np.ndarray, frac: int,
                     *, lut32: bool = False) -> np.ndarray | None:
    """Return native packed-Q1_0 linear output, or None when the native library is unavailable.

    With lut32=True (and an int64 scale) the int32-LUT-entry workspace kernel is tried first; it falls
    through to the uint64-LUT path if the symbol is absent, the workspace is too large, or a block exceeds
    the int32 envelope (rc 5)."""
    lib = _load_lib()
    if lib is None:
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    b, s, out_f, n_blocks = _contiguous_q1_weight(bits, scale_fp)
    if x.shape[1] != n_blocks * 128:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * 128}")
    out = np.empty((x.shape[0], out_f), dtype=np.int64)
    if lut32 and s.dtype == np.int64:
        fn = getattr(lib, "bonsai_q1_linear_i64_workspace_lut32", None)
        if fn is not None:
            total_count = int(x.shape[0]) * int(n_blocks)
            lut_count = total_count * 16 * 256
            workspace = _workspace_arrays_lut32(total_count, lut_count)
            if workspace is not None:
                totals, lut = workspace
                rc = fn(
                    x.ctypes.data, b.ctypes.data, s.ctypes.data,
                    ctypes.c_int64(x.shape[0]), ctypes.c_int64(out_f),
                    ctypes.c_int64(n_blocks), ctypes.c_int64(int(frac)),
                    out.ctypes.data,
                    totals.ctypes.data, ctypes.c_size_t(totals.size),
                    lut.ctypes.data, ctypes.c_size_t(lut.size),
                )
                if rc == 0:
                    return out
                if rc not in (3, 5):    # 3 short ws / 5 out-of-int32 -> fall through to uint64 path
                    raise RuntimeError(f"bonsai_q1_linear_i64_workspace_lut32 failed with code {rc}")
    if s.dtype == np.int32:
        # Narrow scale cache: there is no base (non-workspace) int32 kernel, so require the workspace
        # variant and a workspace allocation, else signal None to fall back to the int64 oracle path.
        fn = getattr(lib, "bonsai_q1_linear_i64_workspace_scale32", None)
        if fn is None:
            return None
        total_count = int(x.shape[0]) * int(n_blocks)
        lut_count = total_count * 16 * 256
        workspace = _workspace_arrays(total_count, lut_count)
        if workspace is None:
            return None
        totals, lut = workspace
        rc = fn(
            x.ctypes.data,
            b.ctypes.data,
            s.ctypes.data,
            ctypes.c_int64(x.shape[0]),
            ctypes.c_int64(out_f),
            ctypes.c_int64(n_blocks),
            ctypes.c_int64(int(frac)),
            out.ctypes.data,
            totals.ctypes.data,
            ctypes.c_size_t(totals.size),
            lut.ctypes.data,
            ctypes.c_size_t(lut.size),
        )
        if rc == 3:
            return None
        if rc != 0:
            raise RuntimeError(f"bonsai_q1_linear_i64_workspace_scale32 failed with code {rc}")
        return out
    rc = None
    workspace_fn = getattr(lib, "bonsai_q1_linear_i64_workspace", None)
    if workspace_fn is not None:
        total_count = int(x.shape[0]) * int(n_blocks)
        lut_count = total_count * 16 * 256
        workspace = _workspace_arrays(total_count, lut_count)
        if workspace is not None:
            totals, lut = workspace
            rc = workspace_fn(
                x.ctypes.data,
                b.ctypes.data,
                s.ctypes.data,
                ctypes.c_int64(x.shape[0]),
                ctypes.c_int64(out_f),
                ctypes.c_int64(n_blocks),
                ctypes.c_int64(int(frac)),
                out.ctypes.data,
                totals.ctypes.data,
                ctypes.c_size_t(totals.size),
                lut.ctypes.data,
                ctypes.c_size_t(lut.size),
            )
    if rc is None or rc == 3:
        rc = lib.bonsai_q1_linear_i64(
            x.ctypes.data,
            b.ctypes.data,
            s.ctypes.data,
            ctypes.c_int64(x.shape[0]),
            ctypes.c_int64(out_f),
            ctypes.c_int64(n_blocks),
            ctypes.c_int64(int(frac)),
            out.ctypes.data,
        )
    if rc == 2:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_linear_i64 failed with code {rc}")
    return out


def q1_argmax_native(x_fp: np.ndarray, bits: np.ndarray, scale_fp: np.ndarray, frac: int,
                     *, lut32: bool = False) -> np.ndarray | None:
    """Return argmax ids for native packed-Q1_0 linear output without materializing the full logits row.

    With lut32=True (and an int64 scale) the int32-LUT-entry argmax kernel is tried first (the vocab head is
    the most LUT-reuse-bound gather); it falls through to the uint64-LUT argmax on rc 3/5."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_argmax_i64_workspace"):
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    b, s, out_f, n_blocks = _contiguous_q1_weight(bits, scale_fp)
    if x.shape[1] != n_blocks * 128:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * 128}")
    total_count = int(x.shape[0]) * int(n_blocks)
    lut_count = total_count * 16 * 256
    ids = np.empty(x.shape[0], dtype=np.int64)
    values = np.empty(x.shape[0], dtype=np.int64)
    if lut32 and s.dtype == np.int64:
        fn = getattr(lib, "bonsai_q1_argmax_i64_workspace_lut32", None)
        if fn is not None:
            ws32 = _workspace_arrays_lut32(total_count, lut_count)
            if ws32 is not None:
                totals32, lut32arr = ws32
                rc = fn(
                    x.ctypes.data, b.ctypes.data, s.ctypes.data,
                    ctypes.c_int64(x.shape[0]), ctypes.c_int64(out_f),
                    ctypes.c_int64(n_blocks), ctypes.c_int64(int(frac)),
                    ids.ctypes.data, values.ctypes.data,
                    totals32.ctypes.data, ctypes.c_size_t(totals32.size),
                    lut32arr.ctypes.data, ctypes.c_size_t(lut32arr.size),
                )
                if rc == 0:
                    return ids
                if rc not in (3, 5):
                    raise RuntimeError(f"bonsai_q1_argmax_i64_workspace_lut32 failed with code {rc}")
    workspace = _workspace_arrays(total_count, lut_count)
    if workspace is None:
        return None
    totals, lut = workspace
    if s.dtype == np.int32:
        fn = getattr(lib, "bonsai_q1_argmax_i64_workspace_scale32", None)
        if fn is None:
            return None
    else:
        fn = lib.bonsai_q1_argmax_i64_workspace
    rc = fn(
        x.ctypes.data,
        b.ctypes.data,
        s.ctypes.data,
        ctypes.c_int64(x.shape[0]),
        ctypes.c_int64(out_f),
        ctypes.c_int64(n_blocks),
        ctypes.c_int64(int(frac)),
        ids.ctypes.data,
        values.ctypes.data,
        totals.ctypes.data,
        ctypes.c_size_t(totals.size),
        lut.ctypes.data,
        ctypes.c_size_t(lut.size),
    )
    if rc == 3:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_argmax_i64_workspace failed with code {rc}")
    return ids


def rmsnorm_native(x_fp: np.ndarray, frac: int, *, eps: int = 1,
                   gain_q: np.ndarray | None = None) -> np.ndarray | None:
    """Return native fixed-point RMSNorm output, or None when unavailable/outside the fast envelope."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_rmsnorm_i64"):
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    gain = None
    gain_ptr = ctypes.c_void_p(0)
    if gain_q is not None:
        gain = np.ascontiguousarray(np.asarray(gain_q, dtype=np.int64))
        if gain.shape != (x.shape[1],):
            raise ValueError(f"RMSNorm gain shape {gain.shape} does not match {(x.shape[1],)}")
        gain_ptr = ctypes.c_void_p(gain.ctypes.data)
    out = np.empty(x.shape, dtype=np.int64)
    rc = lib.bonsai_rmsnorm_i64(
        x.ctypes.data,
        ctypes.c_int64(x.shape[0]),
        ctypes.c_int64(x.shape[1]),
        ctypes.c_int64(int(frac)),
        ctypes.c_int64(int(eps)),
        gain_ptr,
        out.ctypes.data,
    )
    if rc == 4:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_rmsnorm_i64 failed with code {rc}")
    return out


def _kv_for_native(arr: np.ndarray) -> np.ndarray:
    """Return a (Hkv, L, hd) int64 view whose per-head (L, hd) block is contiguous, WITHOUT copying when the
    input is a KV-cache buffer slice: only the inter-head stride may exceed L*hd (the cap*hd buffer stride),
    which the native kernel takes as a parameter. Falls back to a contiguous copy for any other layout."""
    a = np.asarray(arr)
    if a.dtype != np.int64:
        a = np.ascontiguousarray(a, dtype=np.int64)
    if a.ndim != 3:
        return a
    it = a.itemsize
    _Hkv, L, hd = a.shape
    if (a.strides[2] == it and a.strides[1] == hd * it
            and a.strides[0] % it == 0 and a.strides[0] >= L * hd * it):
        return a
    return np.ascontiguousarray(a)


def silu_native(x_fp: np.ndarray, frac: int) -> np.ndarray | None:
    """Native element-wise fixed-point SiLU, or None when unavailable. Byte-identical to
    reference_bonsai.fixed_point_silu; the caller falls back to the NumPy oracle on None."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_silu_i64"):
        return None
    x = np.ascontiguousarray(np.asarray(x_fp, dtype=np.int64))
    out = np.empty(x.shape, dtype=np.int64)
    rc = lib.bonsai_silu_i64(
        x.ctypes.data,
        ctypes.c_int64(x.size),
        ctypes.c_int64(int(frac)),
        out.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"bonsai_silu_i64 failed with code {rc}")
    return out


def attention_decode_native(q_fp: np.ndarray, k_fp: np.ndarray, v_fp: np.ndarray,
                            frac: int, inv_sqrt_fp: int) -> np.ndarray | None:
    """Native M=1 cached-decode attention. q:(H,hd), k/v:(Hkv,L,hd) int64 fixed-point (post q/k-norm+RoPE
    for q/k; cached for k/v). Returns (H,hd) int64, or None when unavailable / the workspace is too large /
    a head would overflow the int64 attention bound (the caller then falls back to the NumPy path, which
    fails loud, preserving attention's no-silent-wrap contract)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_attention_decode_i64"):
        return None
    q = np.ascontiguousarray(q_fp, dtype=np.int64)            # (H, hd) is contiguous already (no-op)
    k = _kv_for_native(k_fp)
    v = _kv_for_native(v_fp)
    if q.ndim != 2 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"attention shapes: q{q.shape} k{k.shape} v{v.shape}")
    H, hd = q.shape
    Hkv, L, hd_k = k.shape
    if hd_k != hd or v.shape != (Hkv, L, hd):
        raise ValueError(f"attention shape mismatch: q{q.shape} k{k.shape} v{v.shape}")
    it = k.itemsize
    k_kv_stride = k.strides[0] // it                          # cap*hd for a KV-cache buffer slice (no copy)
    v_kv_stride = v.strides[0] // it
    scratch = _attn_scratch(H * L)
    if scratch is None:
        return None
    out = np.empty((H, hd), dtype=np.int64)
    rc = lib.bonsai_attention_decode_i64(
        q.ctypes.data, k.ctypes.data, v.ctypes.data,
        ctypes.c_int64(H), ctypes.c_int64(Hkv), ctypes.c_int64(hd), ctypes.c_int64(L),
        ctypes.c_int64(int(k_kv_stride)), ctypes.c_int64(int(v_kv_stride)),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(inv_sqrt_fp)),
        out.ctypes.data, scratch.ctypes.data, ctypes.c_size_t(scratch.size),
    )
    if rc == 2 or rc == 3:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_attention_decode_i64 failed with code {rc}")
    return out


def attention_prefill_native(q_fp: np.ndarray, k_fp: np.ndarray, v_fp: np.ndarray,
                             start: int, frac: int, inv_sqrt_fp: int) -> np.ndarray | None:
    """Native M=N causal PREFILL attention. q:(H,M,hd) post q-norm+RoPE; k/v:(Hkv,L,hd) RoPE'd K / raw V,
    L == start+M. Returns (H,M,hd) int64 byte-identical to the NumPy causal path, or None when unavailable /
    a head would overflow the int64 bound (the caller then uses the loud NumPy path — no silent wrap)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_attention_prefill_i64"):
        return None
    q = np.ascontiguousarray(q_fp, dtype=np.int64)
    k = np.ascontiguousarray(k_fp, dtype=np.int64)        # contiguous copy: prefill is one call/layer, not hot
    v = np.ascontiguousarray(v_fp, dtype=np.int64)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"prefill attn shapes: q{q.shape} k{k.shape} v{v.shape}")
    H, M, hd = q.shape
    Hkv, L, hd_k = k.shape
    if hd_k != hd or v.shape != (Hkv, L, hd) or L != start + M:
        raise ValueError(f"prefill attn shape/length mismatch: q{q.shape} k{k.shape} v{v.shape} start={start}")
    out = np.empty((H, M, hd), dtype=np.int64)
    rc = lib.bonsai_attention_prefill_i64(
        q.ctypes.data, k.ctypes.data, v.ctypes.data,
        ctypes.c_int64(H), ctypes.c_int64(Hkv), ctypes.c_int64(hd),
        ctypes.c_int64(M), ctypes.c_int64(L), ctypes.c_int64(int(start)),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(inv_sqrt_fp)),
        out.ctypes.data,
    )
    if rc == 2:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_attention_prefill_i64 failed with code {rc}")
    return out


def attention_decode_batched_native(q_fp: np.ndarray, k_list, v_list, lengths,
                                    frac: int, inv_sqrt_fp: int) -> np.ndarray | None:
    """Native BATCHED M=1 decode attention: B independent decode attentions in ONE call. q:(B,H,hd);
    k_list/v_list: B cache arrays each (Hkv, L_b, hd) int64 (cache-buffer views — the buffer stride is passed,
    no copy); lengths: B ints L_b. Returns (B,H,hd) int64 byte-identical to B separate
    attention_decode_native(q[b], k_list[b], v_list[b]) calls, or None on overflow / unavailable (caller then
    uses the per-sequence NumPy/M=1 path — no silent wrap)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_attention_decode_batched_i64"):
        return None
    B = len(k_list)
    if B == 0 or len(v_list) != B or len(lengths) != B:
        raise ValueError("batched attn: k_list/v_list/lengths must have the same B > 0")
    q = np.ascontiguousarray(q_fp, dtype=np.int64)
    if q.ndim != 3 or q.shape[0] != B:
        raise ValueError(f"batched attn q shape {q.shape} (want (B={B}, H, hd))")
    H, hd = int(q.shape[1]), int(q.shape[2])
    k_addr = np.empty(B, dtype=np.uintp)
    v_addr = np.empty(B, dtype=np.uintp)
    klen = np.empty(B, dtype=np.int64)
    kstr = np.empty(B, dtype=np.int64)
    vstr = np.empty(B, dtype=np.int64)
    keep = []                                                # hold refs so addresses stay valid through the call
    Hkv = None
    for b in range(B):
        k = _kv_for_native(k_list[b])
        v = _kv_for_native(v_list[b])
        keep.append((k, v))
        hkv_b, Lb, hd_k = k.shape
        if Hkv is None:
            Hkv = int(hkv_b)
        if hd_k != hd or k.shape != v.shape or int(hkv_b) != Hkv or Lb != int(lengths[b]):
            raise ValueError(f"batched attn shape mismatch at b={b}: k{k.shape} v{v.shape} len={lengths[b]}")
        k_addr[b] = k.ctypes.data
        v_addr[b] = v.ctypes.data
        klen[b] = Lb
        kstr[b] = k.strides[0] // k.itemsize
        vstr[b] = v.strides[0] // v.itemsize
    out = np.empty((B, H, hd), dtype=np.int64)
    rc = lib.bonsai_attention_decode_batched_i64(
        q.ctypes.data, k_addr.ctypes.data, v_addr.ctypes.data, klen.ctypes.data,
        kstr.ctypes.data, vstr.ctypes.data,
        ctypes.c_int64(B), ctypes.c_int64(H), ctypes.c_int64(int(Hkv)), ctypes.c_int64(hd),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(inv_sqrt_fp)),
        out.ctypes.data,
    )
    del keep
    if rc == 2:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_attention_decode_batched_i64 failed with code {rc}")
    return out
